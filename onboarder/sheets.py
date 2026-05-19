"""Google Sheets I/O for the onboarder pipeline.

Layout (single source of truth for the operator):
    Tab `Criteria` -- Param/Value rows; bot reads at start of every run.
    Tab `Lessons`  -- ALL lesson candidates across all runs, with per-row
                      status (pending|processing|done|failed|skipped_duplicate).
                      Phase 1 appends new candidates (skipping duplicates by
                      video_id). Phase 2 reads approved+pending rows, marks
                      them processing → done.

All functions take a `client` (authorized gspread.Client) and `sheet_id` (str)
so they can be unit-tested with a fake client without touching the real Sheet.
"""

from __future__ import annotations

import logging
import threading
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

log = logging.getLogger("gateway")

# Module-level mutex for the Lessons tab. Multiple parallel wizards (one per
# Telegram forum topic) can finish Phase 1 around the same time or grab
# approved rows in Phase 2 simultaneously — without this, the sheet would
# see interleaved appends or two workers fighting over the same row.
#
# RLock so the same thread can re-acquire (e.g. when an outer "read → write"
# critical section calls a helper that also takes the lock internally).
_SHEET_LOCK = threading.RLock()


@contextmanager
def sheet_lock():
    """Context manager around the Lessons-tab mutex.

    Wrap any read-then-write critical section in this:
        with sheets.sheet_lock():
            seen = sheets.get_active_video_ids(...)
            new = [r for r in rows if r["video_id"] not in seen]
            sheets.append_lesson_rows(..., rows=new)

    Phase 1 uses it for the final dedup re-check + append; Phase 2 uses it
    for the read-approved-rows + mark-processing step. Per-video status
    updates inside Phase 2 (marking done/failed for one row) are always
    bounded to that worker's own rows and don't need the lock.
    """
    with _SHEET_LOCK:
        yield

# Header for the unified Lessons tab. Columns ordered by user-facing
# importance (status/approved/course/channel left, technical IDs right).
# IMPORTANT: append new columns at the end only — existing data depends on positions.
LESSONS_HEADER = [
    "status",                   # A: pending | processing | done | failed | skipped_duplicate
    "approved",                 # B: TRUE / FALSE checkbox (user-edited)
    "course",                   # C: full "Course N: Channel — Title"
    "channel",                  # D: channel name
    "lesson_idx",               # E: 1..M within course
    "lesson_title",             # F: video title
    "url",                      # G: youtube link
    "duration",                 # H: human-readable "20 мин"
    "course_admin_url",         # I: filled when status=done
    "failure_reason",           # J: filled when status=failed
    "timestamp",                # K: ISO datetime when added
    "run_id",                   # L: timestamp prefix for grouping
    "video_id",                 # M: youtube video id (used for dedup)
    "channel_id",               # N: youtube channel id
    "course_idx",               # O: 1..N within run
    "duration_sec",             # P: numeric
    # ── Original-language fields (Phase 1 fills them in source language) ──
    # These are the SOURCE OF TRUTH for the landing. If the operator edits
    # any of these in the Sheet, Phase 2 sends the edit to the admin API
    # (overriding the cached compose payload in pipeline.db).
    "lesson_description",       # Q: 3-5 sentences in source lang
    "course_description",       # R: course excerpt (lesson_idx=1)
    "course_tagline",           # S: short slogan (lesson_idx=1)
    "course_what_you_learn",    # T: bullet points (lesson_idx=1)
    "course_target_audience",   # U: who it's for (lesson_idx=1)
    "author_name",              # V: full author name (lesson_idx=1)
    "author_bio",               # W: 3-5 sentence bio (lesson_idx=1)
    "author_expertise",         # X: areas of expertise (lesson_idx=1)
    "transcript_excerpt",       # Y: first ~500 chars of transcript (review aid)
    # ── Russian translations (review-only, do NOT go to landing) ──
    # When the source language is non-Russian, Phase 1 also generates a
    # Russian rendering of every translatable field so the operator can
    # sanity-check what's about to ship without leaving the Sheet. These
    # cells are reference-only — the operator's edits to them are ignored
    # by Phase 2 (the *_orig column is what gets pushed to the admin).
    "lesson_description_ru",    # Z
    "course_description_ru",    # AA
    "course_tagline_ru",        # AB
    "course_what_you_learn_ru", # AC
    "course_target_audience_ru",# AD
    "author_bio_ru",            # AE
    "author_expertise_ru",      # AF
    # ── Override fields for the full landing payload (lesson_idx=1 only) ──
    # Filled by Phase 1 from the cached compose; operator can edit any of
    # these cells before clicking "Запустить обработку" and Phase 2 will
    # use the edit instead of the cached value.
    "course_title",             # AG: full course title (overrides composed)
    "course_about",             # AH: large markdown about-content
    "course_plan",              # AI: plan in delimiter format (---section: ...---)
    "course_science",           # AJ: science block in delimiter format
]

# Status enum values
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_SKIPPED_DUPLICATE = "skipped_duplicate"

# Statuses that mark a video as "owned" by another run/course (dedup gate)
ACTIVE_STATUSES = {STATUS_PROCESSING, STATUS_DONE}

CRITERIA_DEFAULTS: dict[str, Any] = {
    "min_subscribers": 5000,
    "max_subscribers": 500000,
    "min_videos_on_channel": 20,
    "max_videos_on_channel": 1000,
    "max_video_age_months": 24,
    "preferred_languages": ["ru", "en"],
    "preferred_video_length_min": 10,
    "preferred_video_length_max": 35,
}

LESSONS_TAB_NAME = "Lessons"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def open_client(service_account_file: str) -> Any:
    """Authorize gspread with a service-account JSON key. Lazy import gspread."""
    import gspread
    from google.oauth2.service_account import Credentials

    sa_path = Path(service_account_file).expanduser()
    if not sa_path.exists():
        raise FileNotFoundError(f"service account JSON not found: {sa_path}")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
    ]
    creds = Credentials.from_service_account_file(str(sa_path), scopes=scopes)
    return gspread.authorize(creds)


def open_spreadsheet(client: Any, sheet_id: str) -> Any:
    """Open the spreadsheet by ID. Caller handles SpreadsheetNotFound."""
    return client.open_by_key(sheet_id)


# ---------------------------------------------------------------------------
# Criteria tab (unchanged from before)
# ---------------------------------------------------------------------------

def read_criteria(client: Any, sheet_id: str) -> dict[str, Any]:
    """Read tab `Criteria` and merge with defaults. Missing keys fall back to defaults."""
    ss = open_spreadsheet(client, sheet_id)
    try:
        ws = ss.worksheet("Criteria")
    except Exception as e:
        log.warning(f"sheets: Criteria tab not found, using defaults ({e})")
        return dict(CRITERIA_DEFAULTS)

    rows = ws.get_all_values()
    parsed: dict[str, Any] = dict(CRITERIA_DEFAULTS)
    for row in rows[1:]:  # skip header
        if len(row) < 2:
            continue
        key = (row[0] or "").strip()
        val = (row[1] or "").strip()
        if not key:
            continue
        parsed[key] = _coerce(key, val)
    return parsed


def _coerce(key: str, val: str) -> Any:
    if key.endswith("_languages") or "," in val:
        return [v.strip() for v in val.split(",") if v.strip()]
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


# ---------------------------------------------------------------------------
# Unified Lessons tab
# ---------------------------------------------------------------------------

def ensure_lessons_tab(client: Any, sheet_id: str) -> Any:
    """Return the Lessons worksheet, creating it with header if missing.

    Migrates existing tabs by appending any missing trailing columns (the only
    safe shape change — we never reorder or remove columns, so old A..N data
    keeps working when we add Q..Y descriptions/author fields later).
    """
    ss = open_spreadsheet(client, sheet_id)
    try:
        ws = ss.worksheet(LESSONS_TAB_NAME)
    except Exception:
        ws = ss.add_worksheet(title=LESSONS_TAB_NAME, rows=200, cols=len(LESSONS_HEADER))
        ws.append_row(LESSONS_HEADER, value_input_option="USER_ENTERED")
        return ws

    _migrate_header_if_needed(ws)
    return ws


def _migrate_header_if_needed(ws: Any) -> None:
    """If existing header is shorter than LESSONS_HEADER, append the missing names.

    Refuses to migrate if any existing column name doesn't match the canonical
    header at the same position — that would mean someone reordered columns
    manually and we don't want to silently misalign data.
    """
    try:
        existing = ws.row_values(1)
    except Exception as e:
        log.warning(f"sheets: cannot read header for migration: {e}")
        return

    if existing == LESSONS_HEADER:
        return  # already up-to-date

    # Verify the prefix matches — otherwise abort migration loudly
    for i, name in enumerate(existing):
        if i >= len(LESSONS_HEADER):
            log.warning(f"sheets: Lessons tab has unexpected column at pos {i}: '{name}' "
                        f"(beyond canonical header). Skipping migration.")
            return
        if name and name != LESSONS_HEADER[i]:
            log.warning(f"sheets: Lessons tab header mismatch at pos {i}: "
                        f"got '{name}', expected '{LESSONS_HEADER[i]}'. Skipping migration.")
            return

    missing = LESSONS_HEADER[len(existing):]
    if not missing:
        return

    # Make sure the worksheet has enough columns to hold the new header
    try:
        if int(getattr(ws, "col_count", 0) or 0) < len(LESSONS_HEADER):
            ws.resize(cols=len(LESSONS_HEADER))
    except Exception as e:
        log.warning(f"sheets: cannot resize Lessons tab cols: {e}")

    # Write missing column names into row 1 starting at col len(existing)+1
    start_col = len(existing) + 1
    end_col_letter = _col_letter_idx(len(LESSONS_HEADER) - 1)
    start_col_letter = _col_letter_idx(start_col - 1)
    rng = f"{start_col_letter}1:{end_col_letter}1"
    try:
        ws.update(rng, [missing], value_input_option="USER_ENTERED")
        log.info(f"sheets: Lessons header migrated, added columns: {missing}")
    except Exception as e:
        log.warning(f"sheets: header migration write failed: {e}")


def get_active_video_ids(client: Any, sheet_id: str) -> set[str]:
    """Return EVERY video_id that has appeared in the Lessons tab, regardless of status.

    Originally this only excluded `processing` / `done` videos so the operator
    could re-surface rejected ones in a later run. In practice that meant videos
    the operator had explicitly rejected (status=failed or skipped, approved=FALSE)
    kept popping back up. We now treat any prior appearance as "seen" — if the
    operator wants to retry a previously-skipped video, they can clear its row
    in the sheet by hand.

    The function name stays as `get_active_video_ids` for back-compat with
    callers, but the semantics are now "all seen video_ids".
    """
    ws = ensure_lessons_tab(client, sheet_id)
    rows = ws.get_all_values()
    if len(rows) < 2:
        return set()

    header = rows[0]
    try:
        idx_video_id = header.index("video_id")
    except ValueError:
        log.warning("sheets: Lessons tab missing video_id column")
        return set()

    out: set[str] = set()
    for row in rows[1:]:
        if len(row) <= idx_video_id:
            continue
        vid = (row[idx_video_id] or "").strip()
        if vid:
            out.add(vid)
    return out


def get_seen_channel_ids(client: Any, sheet_id: str) -> set[str]:
    """Return every channel_id that has appeared in the Lessons tab.

    Used by Phase 1 to skip channels whose author already has a course on the
    platform. The legacy import (import_existing_courses.py) seeds these rows
    with run_id=legacy_import, so any channel that's already on truelifeflow
    won't surface again in a new run's candidate list.
    """
    ws = ensure_lessons_tab(client, sheet_id)
    rows = ws.get_all_values()
    if len(rows) < 2:
        return set()

    header = rows[0]
    try:
        idx_channel_id = header.index("channel_id")
    except ValueError:
        log.warning("sheets: Lessons tab missing channel_id column")
        return set()

    out: set[str] = set()
    for row in rows[1:]:
        if len(row) <= idx_channel_id:
            continue
        cid = (row[idx_channel_id] or "").strip()
        if cid:
            out.add(cid)
    return out


def append_lesson_rows(client: Any, sheet_id: str, *, run_id: str,
                       rows: list[dict[str, Any]]) -> list[int]:
    """Append new lesson candidates to the Lessons tab as `pending`.

    Each row dict (input) keys (all optional unless noted):
        Required-ish (basic Phase 1): course, lesson_idx, channel, channel_id,
            lesson_title, url, video_id, duration_sec, course_idx
        Extended (Phase 1 after download+transcribe+compose):
            lesson_description, course_description, course_tagline,
            course_what_you_learn, course_target_audience, author_name,
            author_bio, author_expertise, transcript_excerpt
    Unknown/missing extended fields default to empty string.

    Returns: list of 1-based absolute sheet row numbers, one per appended
    row in the same order as `rows`. Empty list if rows was empty.
    Streaming updates (update_lesson_row_partial) use these row numbers
    as anchors.
    """
    if not rows:
        return []
    ws = ensure_lessons_tab(client, sheet_id)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    values: list[list[Any]] = [_lesson_row_to_values(r, run_id=run_id, ts=ts) for r in rows]
    resp = ws.append_rows(values, value_input_option="USER_ENTERED",
                          insert_data_option="INSERT_ROWS",
                          include_values_in_response=False)
    # Parse "Lessons!A11:AJ15" → starting row 11
    updated_range = (resp or {}).get("updates", {}).get("updatedRange", "")
    m = re.search(r"![A-Z]+(\d+):[A-Z]+(\d+)", updated_range)
    if not m:
        log.warning(f"append_lesson_rows: could not parse updatedRange '{updated_range}'")
        return []
    start_row = int(m.group(1))
    return list(range(start_row, start_row + len(rows)))


def _lesson_row_to_values(r: dict[str, Any], *, run_id: str, ts: str) -> list[Any]:
    """Build a row in canonical LESSONS_HEADER order from a dict."""
    dur_sec = int(r.get("duration_sec") or 0)
    return [
        STATUS_PENDING,                                     # A status
        "FALSE",                                            # B approved
        r.get("course", ""),                                # C course
        r.get("channel", ""),                               # D channel
        r.get("lesson_idx", ""),                            # E lesson_idx
        r.get("lesson_title", r.get("title", "")),          # F lesson_title
        r.get("url", ""),                                   # G url
        _format_duration(dur_sec),                          # H duration
        "",                                                 # I course_admin_url
        "",                                                 # J failure_reason
        ts,                                                 # K timestamp
        run_id,                                             # L run_id
        r.get("video_id", ""),                              # M video_id
        r.get("channel_id", ""),                            # N channel_id
        r.get("course_idx", ""),                            # O course_idx
        dur_sec,                                            # P duration_sec
        # Original-language source-of-truth fields (Q-Y)
        r.get("lesson_description", ""),                    # Q lesson_description
        r.get("course_description", ""),                    # R course_description
        r.get("course_tagline", ""),                        # S course_tagline
        r.get("course_what_you_learn", ""),                 # T course_what_you_learn
        r.get("course_target_audience", ""),                # U course_target_audience
        r.get("author_name", ""),                           # V author_name
        r.get("author_bio", ""),                            # W author_bio
        r.get("author_expertise", ""),                      # X author_expertise
        r.get("transcript_excerpt", ""),                    # Y transcript_excerpt
        # Russian translations (Z-AF) — review aid only, not used by Phase 2
        r.get("lesson_description_ru", ""),                 # Z
        r.get("course_description_ru", ""),                 # AA
        r.get("course_tagline_ru", ""),                     # AB
        r.get("course_what_you_learn_ru", ""),              # AC
        r.get("course_target_audience_ru", ""),             # AD
        r.get("author_bio_ru", ""),                         # AE
        r.get("author_expertise_ru", ""),                   # AF
        # Extended overrides (AG-AJ) — lesson_idx=1 only; Phase 2 prefers
        # non-empty Sheet values over the cached compose payload.
        r.get("course_title", ""),                          # AG
        r.get("course_about", ""),                          # AH
        r.get("course_plan", ""),                           # AI
        r.get("course_science", ""),                        # AJ
    ]


def _row_to_dict(row: list[str], header: list[str]) -> dict[str, str]:
    """Pad row to header length and zip into a dict keyed by header names."""
    padded = row + [""] * (len(header) - len(row))
    return {h: padded[i] for i, h in enumerate(header)}


def read_pending_approved_rows(client: Any, sheet_id: str,
                               run_id: str | None = None,
                               course_idx: int | None = None,
                               ) -> list[dict[str, Any]]:
    """Return rows where Approved=TRUE AND status=pending.

    Each item carries enough metadata for Phase 2 + a `_sheet_row` index for
    later status updates. Filter by run_id if provided (otherwise all runs);
    additionally filter by `course_idx` to scope Phase 2 to a single course
    within a run.
    """
    ws = ensure_lessons_tab(client, sheet_id)
    rows = ws.get_all_values()
    if len(rows) < 2:
        return []
    header = rows[0]
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(rows[1:], start=2):  # 1-based, +1 for header
        d = _row_to_dict(raw, header)
        if (d.get("status", "").strip().lower() != STATUS_PENDING):
            continue
        if (d.get("approved", "").strip().lower() not in ("true", "1", "x", "yes", "да")):
            continue
        if run_id and d.get("run_id", "").strip() != run_id:
            continue
        if course_idx is not None:
            try:
                row_course_idx = int(d.get("course_idx") or 0)
            except ValueError:
                row_course_idx = 0
            if row_course_idx != int(course_idx):
                continue
        try:
            duration_sec = int(d.get("duration_sec") or 0)
        except ValueError:
            duration_sec = 0
        out.append({
            "course": d.get("course", ""),
            "channel": d.get("channel", ""),
            "channel_id": d.get("channel_id", ""),
            "title": d.get("lesson_title", ""),
            "url": d.get("url", ""),
            "video_id": d.get("video_id", ""),
            "duration_sec": duration_sec,
            "course_idx": _safe_int(d.get("course_idx", "")),
            "lesson_idx": _safe_int(d.get("lesson_idx", "")),
            "run_id": d.get("run_id", ""),
            # Source-of-truth original-language fields. Phase 2 prefers these
            # over pipeline.db when non-empty so operator edits in the Sheet
            # flow straight to the landing.
            "lesson_description": d.get("lesson_description", ""),
            "course_description": d.get("course_description", ""),
            "course_tagline": d.get("course_tagline", ""),
            "course_what_you_learn": d.get("course_what_you_learn", ""),
            "course_target_audience": d.get("course_target_audience", ""),
            "author_name": d.get("author_name", ""),
            "author_bio": d.get("author_bio", ""),
            "author_expertise": d.get("author_expertise", ""),
            # Russian translations (review-only). Returned for completeness so
            # callers can show them in operator-facing diagnostics, but Phase 2
            # never sends them to the admin API — *_orig is the canon.
            "lesson_description_ru": d.get("lesson_description_ru", ""),
            "course_description_ru": d.get("course_description_ru", ""),
            "course_tagline_ru": d.get("course_tagline_ru", ""),
            "course_what_you_learn_ru": d.get("course_what_you_learn_ru", ""),
            "course_target_audience_ru": d.get("course_target_audience_ru", ""),
            "author_bio_ru": d.get("author_bio_ru", ""),
            "author_expertise_ru": d.get("author_expertise_ru", ""),
            # Extended override fields (lesson_idx=1 row carries them)
            "course_title": d.get("course_title", ""),
            "course_about": d.get("course_about", ""),
            "course_plan": d.get("course_plan", ""),
            "course_science": d.get("course_science", ""),
            "_sheet_row": i,  # 1-based row index for batch_update
        })
    return out


def update_status(client: Any, sheet_id: str, *, sheet_rows: list[int],
                  new_status: str,
                  course_admin_url: str | None = None,
                  failure_reason: str | None = None) -> None:
    """Bulk-update status (and optionally admin URL / failure reason) for rows by index.

    `sheet_rows` are 1-based row numbers as returned by read_pending_approved_rows.
    Acquires the module sheet lock briefly so concurrent writers don't step
    on each other's HTTP calls.
    """
    if not sheet_rows:
        return
    with _SHEET_LOCK:
        ws = ensure_lessons_tab(client, sheet_id)

        # Map column letters from the canonical header
        col_status = _col_letter("status")           # A
        col_admin = _col_letter("course_admin_url")  # I
        col_reason = _col_letter("failure_reason")   # J

        updates: list[dict[str, Any]] = []
        for r in sheet_rows:
            updates.append({"range": f"{col_status}{r}", "values": [[new_status]]})
            if course_admin_url is not None:
                updates.append({"range": f"{col_admin}{r}", "values": [[course_admin_url]]})
            if failure_reason is not None:
                updates.append({"range": f"{col_reason}{r}", "values": [[failure_reason[:300]]]})
        ws.batch_update(updates, value_input_option="USER_ENTERED")



def update_lesson_row_partial(client: Any, sheet_id: str, *, row_number: int,
                              fields: dict[str, Any],
                              status: str | None = None,
                              failure_reason: str | None = None) -> None:
    """Update only specified columns of a single Lessons row. Preserves all
    other cells (including user-set `approved` checkbox).

    `row_number` is 1-based absolute sheet row (as returned by append_lesson_rows).
    `fields` maps LESSONS_HEADER column names to new values. Unknown keys ignored.
    `status` and `failure_reason` are shortcuts for the same-named columns.
    Empty/None values in fields are skipped — they won't blank existing cells.
    """
    with _SHEET_LOCK:
        ws = ensure_lessons_tab(client, sheet_id)
        updates: list[dict[str, Any]] = []
        # Merge convenience args into fields dict
        merged = dict(fields or {})
        if status is not None:
            merged["status"] = status
        if failure_reason is not None:
            merged["failure_reason"] = failure_reason[:300]
        for header_name, value in merged.items():
            if header_name not in LESSONS_HEADER:
                continue
            if value is None or value == "":
                continue
            col = _col_letter(header_name)
            updates.append({"range": f"{col}{row_number}", "values": [[value]]})
        if not updates:
            return
        ws.batch_update(updates, value_input_option="USER_ENTERED")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col_letter(header_name: str) -> str:
    """Header → A/B/C... column letter."""
    return _col_letter_idx(LESSONS_HEADER.index(header_name))


def _col_letter_idx(idx: int) -> str:
    """0-based column index → A/B/.../Z/AA/AB letter."""
    if idx < 26:
        return chr(ord("A") + idx)
    first = idx // 26 - 1
    second = idx % 26
    return chr(ord("A") + first) + chr(ord("A") + second)


def _format_duration(sec: int) -> str:
    if sec <= 0:
        return ""
    m, s = divmod(int(sec), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}ч {m}м"
    return f"{m} мин"


def _safe_int(v: str | int) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


def make_run_id() -> str:
    """e.g. 2026-05-05T12-34-56"""
    return time.strftime("%Y-%m-%dT%H-%M-%S")


def sheet_url(sheet_id: str, gid: int | None = None) -> str:
    base = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"
    return f"{base}#gid={gid}" if gid is not None else base
