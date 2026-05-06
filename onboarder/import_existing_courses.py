"""One-off import of already-on-platform courses into the Lessons tab.

Reads the operator's external "Курсы" sheet (a separate spreadsheet listing
courses already deployed on truelifeflow), parses lesson YouTube links per
course, and writes them into our unified Lessons tab as `status=done,
approved=TRUE, run_id=legacy_import_<timestamp>`. Phase 1 then dedupes
against these rows so we never propose:

  - the same video again (video_id dedup, already in place), and
  - the same channel/author again (channel_id dedup, added in this commit).

Per course, only the FIRST video is looked up via yt-dlp to resolve
channel_id + channel_name. All other lessons of that course inherit the
same channel — by definition (one course = one channel in this pipeline).

Usage on the bot VPS:
    cd ~/projects/jarvis-telegram-gateway
    venv/bin/python -m onboarder.import_existing_courses \\
        --external-sheet 1a2qnO2gBKgr6KtmiACF5xeEm7Wg0ACZz0h4nqF4OSdo \\
        --external-tab "новое"

The script is idempotent: video_ids that already exist in our Lessons tab
are silently skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

from . import _secrets, sheets, youtube_dl as ytdl

log = logging.getLogger("import_existing_courses")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


_VIDEO_ID_PATTERNS = [
    re.compile(r"youtu\.be/([\w-]{11})"),
    re.compile(r"[?&]v=([\w-]{11})"),
    re.compile(r"/embed/([\w-]{11})"),
    re.compile(r"/shorts/([\w-]{11})"),
]


def parse_video_id(url: str) -> str | None:
    """Extract a YouTube video_id from any of the common URL forms."""
    if not url:
        return None
    url = url.strip()
    for p in _VIDEO_ID_PATTERNS:
        m = p.search(url)
        if m:
            return m.group(1)
    return None


def read_external_sheet(client: Any, sheet_id: str, tab_name: str) -> list[dict[str, Any]]:
    """Pull rows from the external 'Курсы' sheet and parse out (course, author, video_ids[])."""
    ss = client.open_by_key(sheet_id)
    ws = ss.worksheet(tab_name)
    rows = ws.get_all_values()
    if len(rows) < 2:
        return []

    out: list[dict[str, Any]] = []
    # Layout per inspection: col 0 = course name, col 1 = instructor,
    # col 2 = cover, col 3 = comment, col 4..16 = lesson YouTube URLs.
    for r in rows[1:]:
        # Pad to expected width
        r = (r + [""] * 17)[:17]
        course_name = (r[0] or "").strip()
        author = (r[1] or "").strip()
        if not course_name and not author:
            continue
        urls = [(r[i] or "").strip() for i in range(4, 17)]
        video_ids: list[tuple[str, str]] = []  # (video_id, original_url)
        for u in urls:
            vid = parse_video_id(u)
            if vid:
                video_ids.append((vid, u))
        if not video_ids:
            continue
        out.append({
            "course_name": course_name or author,
            "author": author,
            "video_ids": video_ids,
        })
    return out


def resolve_channel_for_course(course: dict[str, Any]) -> dict[str, str]:
    """yt-dlp lookup on the FIRST video of a course → channel_id + channel_name.

    Returns {} if the lookup fails. Caller can still write rows with
    channel_id="" — they just won't help with channel-level dedup.
    """
    if not course["video_ids"]:
        return {}
    first_vid, _ = course["video_ids"][0]
    try:
        meta = ytdl.get_video_metadata(first_vid)
    except Exception as e:
        log.warning(f"yt-dlp failed on {first_vid}: {e}")
        return {}
    return {
        "channel_id": meta.get("channel_id", "") or "",
        "channel_name": meta.get("channel_name", "") or "",
    }


def build_legacy_rows(courses: list[dict[str, Any]],
                      already_seen: set[str],
                      run_id: str) -> tuple[list[list[Any]], dict[str, int]]:
    """Build raw 25-column rows ready for batched append. Skip videos already in Lessons."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_rows: list[list[Any]] = []
    stats = {"courses_imported": 0, "videos_imported": 0, "videos_skipped_dup": 0,
             "courses_skipped_empty": 0, "channels_resolved": 0, "channels_unknown": 0}

    for course_idx, course in enumerate(courses, start=1):
        # Resolve channel from first video (one course = one channel)
        log.info(f"[{course_idx}/{len(courses)}] resolving channel for course "
                 f"{course['course_name']!r} (first video {course['video_ids'][0][0]})")
        ch_info = resolve_channel_for_course(course)
        channel_id = ch_info.get("channel_id", "")
        channel_name = ch_info.get("channel_name", "") or course["author"] or ""
        if channel_id:
            stats["channels_resolved"] += 1
        else:
            stats["channels_unknown"] += 1

        added_for_course = 0
        for lesson_idx, (vid, url) in enumerate(course["video_ids"], start=1):
            if vid in already_seen:
                stats["videos_skipped_dup"] += 1
                continue
            already_seen.add(vid)  # avoid double-adding within this run
            full_course_title = (f"Курс {course_idx}: {channel_name} — {course['course_name']}"
                                 if lesson_idx == 1
                                 else f"Курс {course_idx}")
            row_dict = {
                "course": full_course_title,
                "channel": channel_name,
                "lesson_idx": lesson_idx,
                "lesson_title": "",
                "url": url,
                "duration_sec": 0,
                "video_id": vid,
                "channel_id": channel_id,
                "course_idx": course_idx,
            }
            row_values = sheets._lesson_row_to_values(row_dict, run_id=run_id, ts=ts)
            # Override status + approved: imported rows are "done" + auto-approved.
            row_values[0] = sheets.STATUS_DONE   # A status
            row_values[1] = "TRUE"                # B approved
            out_rows.append(row_values)
            added_for_course += 1

        if added_for_course:
            stats["courses_imported"] += 1
            stats["videos_imported"] += added_for_course
        else:
            stats["courses_skipped_empty"] += 1

    return out_rows, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.json",
                    help="Path to gateway config.json (for SA path + our sheet_id)")
    ap.add_argument("--external-sheet", required=True,
                    help="Spreadsheet ID of the external 'Курсы' sheet")
    ap.add_argument("--external-tab", default="новое",
                    help="Tab name within the external sheet")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse + resolve channels but do not write to Lessons")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    # Reach into the operations agent (single-agent setup).
    onb = (next(iter(cfg["agents"].values()))).get("onboarder") or {}

    sa_path = _secrets.resolve_path(onb, "google_service_account")
    our_sheet_id = onb.get("google_sheet_id") or ""
    if not our_sheet_id:
        log.error("config: onboarder.google_sheet_id not set")
        return 2

    client = sheets.open_client(sa_path)
    courses = read_external_sheet(client, args.external_sheet, args.external_tab)
    log.info(f"Parsed {len(courses)} courses from external sheet")
    if not courses:
        log.warning("No parseable courses found")
        return 1

    sheets.ensure_lessons_tab(client, our_sheet_id)
    already_seen = sheets.get_active_video_ids(client, our_sheet_id)
    log.info(f"Lessons tab already has {len(already_seen)} known video_ids — "
             f"will dedupe against these")

    run_id = "legacy_import_" + time.strftime("%Y-%m-%dT%H-%M-%S")
    rows, stats = build_legacy_rows(courses, already_seen, run_id)
    log.info(f"Stats: {stats}")
    log.info(f"Will append {len(rows)} rows under run_id={run_id}")

    if args.dry_run:
        log.info("DRY RUN — not writing to Lessons. First 3 rows preview:")
        for r in rows[:3]:
            log.info("  " + " | ".join(str(c)[:40] for c in r))
        return 0

    if not rows:
        log.info("Nothing to write (all videos already in Lessons).")
        return 0

    ws = sheets.ensure_lessons_tab(client, our_sheet_id)
    ws.append_rows(rows, value_input_option="USER_ENTERED")
    log.info(f"Imported {stats['videos_imported']} videos across "
             f"{stats['courses_imported']} courses (run_id={run_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
