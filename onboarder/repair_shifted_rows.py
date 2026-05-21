"""One-shot repair for Sheet rows whose columns got shifted left.

Symptom: rows where columns I (`course_admin_url`) and J (`failure_reason`)
were skipped by the writer, so everything from K (timestamp) onwards is
shifted -2 (cols K-N) and then -1 (cols P+) relative to the canonical
LESSONS_HEADER layout.

Concretely, a broken row looks like:

  col  8 = '2026-05-20T14:12:22Z'   <- timestamp value, but in course_admin_url col
  col  9 = '2026-05-20T13-03-46'    <- run_id value, but in failure_reason col
  col 10 = 'KKjVvqZzTWQ'             <- video_id value, but in timestamp col
  col 11 = 'UC7oHw_Z-tLpYOpW6yyjdhIw'<- channel_id value, but in run_id col
  col 12 = ''                        <- empty (no value)
  col 13 = '3'                       <- course_idx value, but in channel_id col
  col 14 = '322'                     <- duration_sec value, but in course_idx col
  col 15 = 'A guided body...'        <- lesson_description, but in duration_sec col
  col 23 = transcript text           <- transcript_excerpt, but in author_expertise col
  col 24 = russian text              <- lesson_description_ru, but in transcript_excerpt col

This script finds such rows and rewrites them in the canonical layout.

Usage:

    python -m onboarder.repair_shifted_rows
        [--run-id RUN_ID]        # filter to one run (optional)
        [--course-idx N]         # filter to one course within the run
        [--dry-run]              # print plan, do not write
        [--verbose]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from . import _secrets, sheets

log = logging.getLogger("repair_shifted_rows")


def _looks_shifted_v1(raw_row: list[str]) -> bool:
    """Pattern 1 (Pain Academy / Course 3):

    Two empty placeholders (course_admin_url, failure_reason) were skipped
    so everything from col 8 onwards is shifted left by 2.

    Signals:
      * col 8 (course_admin_url) starts with a year prefix like '2026-' —
        should be empty or http://…
      * col 10 (timestamp) is not empty AND doesn't start with a year
        prefix (it holds an 11-char video_id instead).
    """
    pad = raw_row + [""] * (16 - len(raw_row))
    col8 = pad[8].strip()
    col10 = pad[10].strip()
    col8_suspect = (
        col8
        and not col8.lower().startswith("http")
        and (col8.startswith("2026-") or col8.startswith("2025-") or col8.startswith("2027-"))
    )
    col10_suspect = (
        col10
        and not (col10.startswith("2026-") or col10.startswith("2025-") or col10.startswith("2027-"))
    )
    return bool(col8_suspect and col10_suspect)


def _looks_shifted_v2(raw_row: list[str]) -> bool:
    """Pattern 2 (Body Articulate / Course 5):

    Different writer-bug — lesson_idx got duplicated into col 8, while
    course_idx + duration_sec got hijacked into cols 24 / 25.

    Signals:
      * col 8 (course_admin_url) is a SHORT digit string (1..99) and equals
        col 4 (lesson_idx) — clear "lesson_idx misrouted into admin_url".
      * col 14 (course_idx) is empty AND col 24 (transcript_excerpt) is a
        single digit (course_idx value misplaced).
      * cols 10-13 (ts/run_id/video_id/channel_id) ARE correctly populated.
    """
    pad = raw_row + [""] * (26 - len(raw_row))
    col4 = pad[4].strip()
    col8 = pad[8].strip()
    col14 = pad[14].strip()
    col24 = pad[24].strip()
    col10 = pad[10].strip()
    col8_is_lesson_idx_dup = (
        col8.isdigit()
        and len(col8) <= 3
        and col4 and col4 == col8
    )
    col14_empty_but_col24_digit = (
        not col14
        and col24.isdigit()
        and len(col24) <= 3
    )
    col10_ok = col10.startswith("2026-") or col10.startswith("2025-") or col10.startswith("2027-")
    return bool(col8_is_lesson_idx_dup and col14_empty_but_col24_digit and col10_ok)


def _looks_shifted(raw_row: list[str]) -> bool:
    """Return True if the row matches any known shift pattern."""
    return _looks_shifted_v1(raw_row) or _looks_shifted_v2(raw_row)


def _repair_row(raw_row: list[str], target_width: int) -> dict[int, str]:
    """Compute the cell-by-cell updates that move shifted values back to canonical
    columns.

    Returns a dict {col_index: new_value} containing ONLY the cells whose value
    needs to change. Caller turns this into a batched ws.update.

    Dispatches by pattern: v1 (Pain Academy) vs v2 (Body Articulate).
    """
    pad = raw_row + [""] * (max(target_width, 26) - len(raw_row))
    if _looks_shifted_v2(raw_row):
        return _repair_row_v2(pad)
    return _repair_row_v1(pad)


def _repair_row_v1(pad: list[str]) -> dict[int, str]:
    """Pain Academy pattern — values from col 8 onwards shifted left by 2."""
    updates: dict[int, str] = {}

    # Save the shifted values BEFORE blanking anything.
    shifted_ts          = pad[8]   # → target col 10
    shifted_run_id      = pad[9]   # → target col 11
    shifted_video_id    = pad[10]  # → target col 12
    shifted_channel_id  = pad[11]  # → target col 13
    # pad[12] is the gap — confirmed empty in the inspected rows
    shifted_course_idx  = pad[13]  # → target col 14
    shifted_duration    = pad[14]  # → target col 15
    shifted_lesson_desc = pad[15]  # → target col 16
    shifted_transcript  = pad[23] if len(pad) > 23 else ""  # → target col 24
    shifted_desc_ru     = pad[24] if len(pad) > 24 else ""  # → target col 25

    # course_admin_url (col 8) and failure_reason (col 9) become empty.
    if pad[8] != "":
        updates[8] = ""
    if pad[9] != "":
        updates[9] = ""

    # Move shifted values back to their canonical positions.
    if shifted_ts:
        updates[10] = shifted_ts
    if shifted_run_id:
        updates[11] = shifted_run_id
    if shifted_video_id:
        updates[12] = shifted_video_id
    if shifted_channel_id:
        updates[13] = shifted_channel_id
    if shifted_course_idx:
        updates[14] = shifted_course_idx
    if shifted_duration:
        updates[15] = shifted_duration
    if shifted_lesson_desc:
        updates[16] = shifted_lesson_desc
    if shifted_transcript:
        updates[23] = ""
        updates[24] = shifted_transcript
    if shifted_desc_ru:
        updates[25] = shifted_desc_ru

    return updates


def _repair_row_v2(pad: list[str]) -> dict[int, str]:
    """Body Articulate pattern — col 8 has lesson_idx duplicate; course_idx and
    duration_sec got misrouted to cols 24/25 (transcript_excerpt /
    lesson_description_ru). Cols 10-13 are already correct.

    Recovery:
      * col 8 → empty (clear the lesson_idx duplicate)
      * col 14 (course_idx) ← pad[24] (where it actually lived)
      * col 15 (duration_sec) ← pad[25]
      * cols 24, 25 → empty (the hijacked positions; transcript text and RU
        translation were never written, nothing to preserve)
    """
    updates: dict[int, str] = {}

    if pad[8] != "":
        updates[8] = ""

    hijacked_course_idx = pad[24].strip() if len(pad) > 24 else ""
    hijacked_duration   = pad[25].strip() if len(pad) > 25 else ""

    if hijacked_course_idx:
        updates[14] = hijacked_course_idx
    if hijacked_duration:
        updates[15] = hijacked_duration

    if hijacked_course_idx:
        updates[24] = ""
    if hijacked_duration:
        updates[25] = ""

    return updates


def _col_letter(idx: int) -> str:
    """0-based column index → A, B, …, Z, AA, AB, …"""
    if idx < 26:
        return chr(ord("A") + idx)
    return chr(ord("A") + idx // 26 - 1) + chr(ord("A") + idx % 26)


def _load_config() -> dict[str, Any]:
    path = Path("config.json")
    if not path.exists():
        path = Path(__file__).resolve().parent.parent / "config.json"
    with open(path) as f:
        return json.load(f)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Repair Sheet rows whose columns got shifted left by 2 from col I.",
    )
    p.add_argument("--run-id", default=None,
                   help="Limit to rows of this Phase 1 run_id (RAW value, not from "
                        "current col 11 — we look at the SHIFTED location col 9 too)")
    p.add_argument("--course-idx", type=int, default=None,
                   help="Limit to this course within the run "
                        "(checks both canonical col 14 and shifted col 13)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan, don't write")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = _load_config()
    onb = cfg["agents"]["operations"]["onboarder"]
    sa_path = _secrets.resolve_path(onb, "google_service_account")
    sheet_id = onb.get("google_sheet_id") or ""
    if not sheet_id:
        log.error("config: onboarder.google_sheet_id not set")
        return 2

    client = sheets.open_client(sa_path)
    ws = sheets.ensure_lessons_tab(client, sheet_id)
    all_rows = ws.get_all_values()
    header = all_rows[0]
    target_width = len(header)
    log.info(f"Sheet has {len(all_rows) - 1} data rows; header width = {target_width}")

    candidates: list[tuple[int, list[str]]] = []
    for row_idx, raw in enumerate(all_rows[1:], start=2):
        if not _looks_shifted(raw):
            continue
        pad = raw + [""] * (target_width - len(raw))

        # Determine which pattern this row matches so we know which column
        # currently holds run_id / course_idx values.
        is_v2 = _looks_shifted_v2(raw)

        # Where does the run_id live in this row's current (broken) layout?
        # v1 (Pain Academy): in col 9 (the failure_reason slot)
        # v2 (Body Articulate): in col 11 (already correct position)
        run_id_col = 11 if is_v2 else 9
        # Where does the course_idx value live?
        # v1: in col 13 (the channel_id slot)
        # v2: in col 24 (the transcript_excerpt slot, hijacked)
        course_idx_col = 24 if is_v2 else 13

        if args.run_id and pad[run_id_col].strip() != args.run_id:
            continue
        if args.course_idx is not None:
            try:
                ci_raw = int(pad[course_idx_col].strip())
            except ValueError:
                continue
            if ci_raw != int(args.course_idx):
                continue
        candidates.append((row_idx, raw))

    if not candidates:
        log.info("No shifted rows match the filters — nothing to repair.")
        return 0

    log.info(f"Found {len(candidates)} shifted rows. Planned moves:")
    for row_num, raw in candidates:
        pad = raw + [""] * (target_width - len(raw))
        pattern = "v2 (course_idx in col24)" if _looks_shifted_v2(raw) else "v1 (shift -2 from col8)"
        log.info(f"  row {row_num} [{pattern}]: status={pad[0]!r} "
                 f"course={pad[2][:30]!r} channel={pad[3][:25]!r} "
                 f"lesson_idx={pad[4]!r}")
        if _looks_shifted_v2(raw):
            log.info(f"    will move: col24→14 (course_idx={pad[24]!r}), "
                     f"col25→15 (duration_sec={pad[25]!r}), "
                     f"col8→clear (was lesson_idx dup={pad[8]!r})")
        else:
            log.info(f"    will move: col9→11 (run_id={pad[9]!r}), "
                     f"col13→14 (course_idx={pad[13]!r}), "
                     f"col10→12 (video_id={pad[10]!r})")

    if args.dry_run:
        log.info("--dry-run: stopping here. Re-run without --dry-run to apply.")
        return 0

    # Build batched updates for every shifted row.
    batch: list[dict[str, Any]] = []
    for row_num, raw in candidates:
        updates = _repair_row(raw, target_width)
        for col_idx, new_value in updates.items():
            col_letter = _col_letter(col_idx)
            batch.append({
                "range": f"{col_letter}{row_num}",
                "values": [[new_value]],
            })

    log.info(f"Applying {len(batch)} cell updates under sheet_lock…")
    with sheets.sheet_lock():
        ws.batch_update(batch, value_input_option="USER_ENTERED")
    log.info(f"✓ Repaired {len(candidates)} rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
