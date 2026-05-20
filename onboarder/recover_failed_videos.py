"""Recovery: re-process Phase 2 failed videos and APPEND them to the
existing partial course in NewMindStart admin.

Use case: Phase 2 finished a course with some videos failed (e.g. Whisper
mis-detected language → Google Translate 400 Invalid Value). The
successful videos already created a DRAFT course; failed rows in the
Sheet have `status=failed` and the course_admin_url filled in. Instead
of deleting and re-creating the whole course (and re-uploading the
successful lessons), this script:

  1. Reads the Sheet for `status=failed` rows (filterable by run_id /
     course_idx), keeping only those whose `course_admin_url` is set
     (we need to know which existing course to append to).
  2. Re-runs each failed video through the Phase 2 per-video pipeline
     (cut → blur watermark → dub-or-skip → bunny upload) using the
     EXACT same code phase2_production uses — so any bugfix that
     landed (e.g. the welsh-misdetect skip) is in effect here too.
  3. Groups results by course_id (parsed from course_admin_url),
     POSTs each group to `/api/agent/courses/:id/append-lessons`.
  4. Updates the Sheet rows to `status=done` with the same admin URL
     so the row disappears from the "to retry" set.

CLI:

    python -m onboarder.recover_failed_videos \\
        --run-id 2026-05-20T07-09-40          (required)
        [--course-idx 4]                       (optional)
        [--voice-gender MALE|FEMALE]           (default MALE)
        [--dry-run]                            (preview only)

Without --course-idx, ALL failed rows of the run are recovered, grouped
by course_idx.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import _secrets, cache, nms_client, proxy_pool, sheets

log = logging.getLogger("recover_failed")


# ---------------------------------------------------------------------------
# Sheet scan
# ---------------------------------------------------------------------------

ADMIN_URL_RE = re.compile(r"/admin/courses/([A-Za-z0-9_-]+)(?:/|$)")


def _parse_course_id_from_admin_url(url: str) -> str | None:
    """Pull the Prisma cuid out of …/admin/courses/<id>/edit."""
    if not url:
        return None
    m = ADMIN_URL_RE.search(url)
    return m.group(1) if m else None


def _read_failed_rows(client: Any, sheet_id: str, *, run_id: str,
                     course_idx: int | None) -> list[dict[str, Any]]:
    """Return Sheet rows with status=failed and a usable course_admin_url.

    Each item carries enough data to feed phase2_production._process_one_video
    (video_id, title, url, course, course_idx, lesson_idx, …).
    """
    from .sheets import (
        ensure_lessons_tab, _row_to_dict, _safe_int,
        STATUS_FAILED,
    )

    ws = ensure_lessons_tab(client, sheet_id)
    rows = ws.get_all_values()
    if len(rows) < 2:
        return []
    header = rows[0]

    out: list[dict[str, Any]] = []
    for i, raw in enumerate(rows[1:], start=2):
        d = _row_to_dict(raw, header)
        if d.get("run_id", "").strip() != run_id:
            continue
        if d.get("status", "").strip().lower() != STATUS_FAILED:
            continue
        admin_url = d.get("course_admin_url", "").strip()
        course_id = _parse_course_id_from_admin_url(admin_url)
        if not course_id:
            log.warning(f"row {i}: status=failed but no course_admin_url — skip")
            continue
        if course_idx is not None and _safe_int(d.get("course_idx", "")) != int(course_idx):
            continue
        try:
            duration_sec = int(d.get("duration_sec") or 0)
        except ValueError:
            duration_sec = 0
        out.append({
            "_sheet_row": i,
            "_course_id": course_id,
            "_admin_url": admin_url,
            "course": d.get("course", ""),
            "course_idx": _safe_int(d.get("course_idx", "")),
            "lesson_idx": _safe_int(d.get("lesson_idx", "")),
            "video_id": d.get("video_id", "").strip(),
            "title": d.get("lesson_title", "").strip(),
            "url": d.get("url", "").strip(),
            "channel": d.get("channel", "").strip(),
            "channel_id": d.get("channel_id", "").strip(),
            "duration_sec": duration_sec,
            "lesson_description": d.get("lesson_description", "").strip(),
            "transcript_excerpt": d.get("transcript_excerpt", "").strip(),
        })
    return out


# ---------------------------------------------------------------------------
# Per-video reprocessing
# ---------------------------------------------------------------------------

def _strip_course_prefix(course_full: str, course_idx: int) -> str:
    """'Курс 4: Pain Care Clinic — Title' → 'Title'."""
    prefix = f"Курс {course_idx}:"
    s = course_full or ""
    if s.startswith(prefix):
        rest = s[len(prefix):].strip()
        if " — " in rest:
            return rest.split(" — ", 1)[1].strip()
        return rest
    return s


def _reprocess_video(row: dict[str, Any], *, onb: dict[str, Any],
                    scratch_dir: Path, voice_gender: str,
                    rotator: Any) -> dict[str, Any]:
    """Run a single failed row through phase2's per-video pipeline.

    Returns the same shape as _process_one_video_impl returns:
        {title, videoKey, videoLibraryId, duration, transcriptEn,
         originalLang, wasDubbed, lessonDescription}.
    Raises on hard failure.
    """
    from . import phase2_production

    openai_key = _secrets.resolve(onb, "openai_api_key", env="OPENAI_API_KEY")
    bunny_lib = str(onb.get("bunny_stream_library_id") or "").strip()
    if not bunny_lib:
        raise RuntimeError("config: onboarder.bunny_stream_library_id not set")
    bunny_key = _secrets.resolve(onb, "bunny_stream_api_key",
                                 env="BUNNY_STREAM_API_KEY")
    cookies_file = onb.get("youtube_cookies_file") or None

    def _g_translate_key() -> str:
        return _secrets.resolve(onb, "google_translate_api_key",
                                env="GOOGLE_TRANSLATE_API_KEY")

    def _g_tts_key() -> str:
        return _secrets.resolve(onb, "google_tts_api_key",
                                env="GOOGLE_TTS_API_KEY")

    clean_title = _strip_course_prefix(row["course"], row["course_idx"])

    lesson_in = {
        "video_id": row["video_id"],
        "title": row["title"],
        "url": row["url"],
        "course_idx": row["course_idx"],
        "lesson_idx": row["lesson_idx"],
        "lesson_description": row.get("lesson_description", ""),
    }

    # Token + chat_id are required by _process_one_video_impl for status
    # _send() calls, but we want CLI-quiet operation. _send is a no-op
    # when token is empty — phase2_production checks this explicitly.
    result = phase2_production._process_one_video(
        token="", chat_id=0,
        prefix=f"recover Курс {row['course_idx']} видео {row['lesson_idx']}",
        lesson=lesson_in,
        scratch_dir=scratch_dir,
        openai_key=openai_key,
        get_google_translate_key=_g_translate_key,
        get_google_tts_key=_g_tts_key,
        voice_gender=voice_gender,
        bunny_lib=bunny_lib,
        bunny_key=bunny_key,
        course_topic=clean_title,
        youtube_cookies_file=cookies_file,
        rotator=rotator,
        mark_cuts_word_level=bool(onb.get("mark_cuts_word_level", False)),
        blur_watermarks=bool(onb.get("blur_watermarks", False)),
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_config() -> dict[str, Any]:
    path = Path("config.json")
    if not path.exists():
        path = Path(__file__).resolve().parent.parent / "config.json"
    with open(path) as f:
        return json.load(f)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Re-process Phase 2 failed videos and append to existing course.",
    )
    p.add_argument("--run-id", required=True,
                   help="Phase 1 run_id to scan (e.g. 2026-05-20T07-09-40)")
    p.add_argument("--course-idx", type=int, default=None,
                   help="Optional: only this course within the run")
    p.add_argument("--voice-gender", default="MALE", choices=["MALE", "FEMALE"],
                   help="Voice gender for any dub that DOES happen (default MALE)")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be done, don't process or append")
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

    nms_endpoint = (onb.get("nms_endpoint") or "").rstrip("/")
    nms_token = _secrets.resolve(onb, "nms_api_token", env="AGENT_API_TOKEN")
    if not nms_endpoint or not nms_token:
        log.error("config: nms_endpoint or nms_api_token missing")
        return 2

    client = sheets.open_client(sa_path)
    rows = _read_failed_rows(client, sheet_id,
                             run_id=args.run_id, course_idx=args.course_idx)
    if not rows:
        log.info(f"No failed rows for run_id={args.run_id}"
                 + (f" course_idx={args.course_idx}" if args.course_idx else "")
                 + " — nothing to recover.")
        return 0

    by_course_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_course_id[r["_course_id"]].append(r)

    log.info(f"Recovery plan for run_id={args.run_id}:")
    for cid, rs in by_course_id.items():
        course_idx = rs[0]["course_idx"]
        titles = ", ".join(r["title"][:40] for r in rs[:3])
        more = f" (+{len(rs) - 3} more)" if len(rs) > 3 else ""
        log.info(f"  • Курс {course_idx} → admin {cid}: {len(rs)} видео ({titles}{more})")

    if args.dry_run:
        log.info("--dry-run: stopping here.")
        return 0

    # Init proxy rotator only if at least one row has no cached MP4
    needs_youtube = any(not cache.is_cached(r["video_id"]) for r in rows)
    proxy_pool_list = proxy_pool.normalise_pool(
        onb.get("youtube_proxies") or onb.get("youtube_proxy")
    )
    cookies_file = onb.get("youtube_cookies_file") or None
    rotator = proxy_pool.ProxyRotator(proxy_pool_list, cookies_file=cookies_file)
    if needs_youtube and proxy_pool_list:
        log.info(f"Initializing proxy rotator over {len(proxy_pool_list)} entries…")
        rotator.init()
        log.info(f"Using proxy: {proxy_pool._proxy_label(rotator.current)}")
    elif not needs_youtube:
        log.info("All videos cached — skipping proxy probe.")

    scratch_root = Path(onb.get("scratch_dir") or "/tmp/onboarder-recover")
    scratch_root.mkdir(parents=True, exist_ok=True)

    total_appended = 0
    total_failed = 0
    for course_id, course_rows in by_course_id.items():
        course_idx = course_rows[0]["course_idx"]
        course_scratch = scratch_root / f"recover-{args.run_id}" / f"course-{course_idx}"
        course_scratch.mkdir(parents=True, exist_ok=True)

        processed_lessons: list[dict[str, Any]] = []
        ok_rows: list[dict[str, Any]] = []
        for row in course_rows:
            log.info(f"→ Курс {course_idx} «{row['title'][:60]}» reprocessing…")
            try:
                result = _reprocess_video(
                    row, onb=onb, scratch_dir=course_scratch,
                    voice_gender=args.voice_gender, rotator=rotator,
                )
            except Exception as e:
                log.error(f"reprocess failed for {row['video_id']}: {e}", exc_info=args.verbose)
                total_failed += 1
                continue
            processed_lessons.append({
                "title": result.get("title") or row["title"],
                "description": (row.get("lesson_description")
                                or result.get("lessonDescription")
                                or result.get("transcriptEn", "")[:500]),
                "videoKey": result["videoKey"],
                "videoLibraryId": result["videoLibraryId"],
                "duration": result.get("duration"),
                "transcriptEn": result.get("transcriptEn", ""),
            })
            ok_rows.append(row)

        try:
            shutil.rmtree(course_scratch, ignore_errors=True)
        except Exception:
            pass

        if not processed_lessons:
            log.warning(f"Курс {course_idx}: no lessons survived recovery — skipping NMS append")
            continue

        log.info(f"POST /api/agent/courses/{course_id}/append-lessons "
                 f"({len(processed_lessons)} lessons)")
        try:
            resp = nms_client.append_lessons_to_course(
                endpoint=nms_endpoint, token=nms_token,
                course_id=course_id, lessons=processed_lessons,
            )
            log.info(f"  → appended {resp['appendedCount']} lessons "
                     f"(admin: {resp['adminUrl']})")
        except Exception as e:
            log.error(f"NMS append failed for course {course_id}: {e}", exc_info=args.verbose)
            total_failed += len(ok_rows)
            continue

        admin_url = resp.get("adminUrl") or course_rows[0]["_admin_url"]
        try:
            sheets.update_status(
                client, sheet_id,
                sheet_rows=[r["_sheet_row"] for r in ok_rows],
                new_status=sheets.STATUS_DONE,
                course_admin_url=admin_url,
            )
        except Exception as e:
            log.warning(f"Sheet status update failed: {e}")
        total_appended += len(ok_rows)

    log.info(f"Done. Appended {total_appended} lessons across "
             f"{len(by_course_id)} course(s). Failures: {total_failed}.")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
