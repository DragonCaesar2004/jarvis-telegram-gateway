"""Recovery: turn already-uploaded Bunny videos into a finished course.

Use case: Phase 2 uploaded videos to Bunny but the final NMS push failed
(token mismatch, network blip, etc.). The videos are sitting on Bunny as
orphans. This script:

1. Reads pending+approved rows from the Lessons sheet (filterable by run_id)
2. Lists the Bunny library, matches Lessons rows to Bunny videos by title
3. Downloads each video's MP4 from Bunny (no YouTube hit)
4. Re-Whispers them for the final EN transcript
5. Calls compose_full_course (Claude builds the course copy)
6. POSTs to /api/agent/onboard-course
7. Updates Lessons rows to status=done with admin URL

Run via:
    python -m onboarder.recover_from_bunny [run_id]

Without a run_id, picks up ALL pending+approved rows.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from . import _secrets, llm, nms_client, sheets, whisper

log = logging.getLogger("recover")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def list_bunny_videos(library_id: str, api_key: str,
                      page_size: int = 200) -> list[dict[str, Any]]:
    """List all videos in the Bunny Stream library. Returns list of {guid, title, length}."""
    import requests
    url = f"https://video.bunnycdn.com/library/{library_id}/videos"
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        r = requests.get(url, headers={"AccessKey": api_key, "Accept": "application/json"},
                         params={"page": page, "itemsPerPage": page_size, "orderBy": "date"},
                         timeout=30)
        r.raise_for_status()
        data = r.json()
        items = data.get("items") or []
        if not items:
            break
        for it in items:
            out.append({
                "guid": it.get("guid"),
                "title": (it.get("title") or "").strip(),
                "length": it.get("length", 0),
                "status": it.get("status", 0),
            })
        if len(items) < page_size:
            break
        page += 1
    return out


def fetch_bunny_video_meta(library_id: str, api_key: str, guid: str) -> dict[str, Any]:
    """GET /library/{lib}/videos/{guid} — full video metadata incl. captions/transcribing."""
    import requests
    url = f"https://video.bunnycdn.com/library/{library_id}/videos/{guid}"
    r = requests.get(url, headers={"AccessKey": api_key, "Accept": "application/json"}, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_bunny_caption_text(library_id: str, api_key: str, guid: str,
                             lang: str = "en") -> str:
    """Read Bunny-generated VTT captions and return plain text.

    Tries `lang` first, then falls back to whatever languages the video has.
    Returns empty string if no captions are available yet.
    """
    import re
    import requests

    meta = fetch_bunny_video_meta(library_id, api_key, guid)
    captions = meta.get("captions") or []
    available = [c.get("srclang") for c in captions if c.get("srclang")]

    # Pick best language: requested → en → en-US → first available
    pick = None
    for candidate in [lang, "en", "en-US", "en-GB"]:
        if candidate in available:
            pick = candidate
            break
    if not pick and available:
        pick = available[0]
    if not pick:
        log.warning(f"no captions yet for {guid} (transcribing status: {meta.get('transcribingStatus')})")
        return ""

    url = f"https://video.bunnycdn.com/library/{library_id}/videos/{guid}/captions/{pick}"
    r = requests.get(url, headers={"AccessKey": api_key}, timeout=30)
    if r.status_code != 200:
        log.warning(f"caption fetch failed {r.status_code}: {r.text[:200]}")
        return ""

    # VTT → plain text. Strip header, timestamps, cue numbers.
    text_lines: list[str] = []
    for line in r.text.splitlines():
        s = line.strip()
        if not s or s.startswith("WEBVTT") or s.startswith("NOTE") or "-->" in s:
            continue
        if re.fullmatch(r"\d+", s):  # cue number
            continue
        text_lines.append(s)
    return " ".join(text_lines).strip()


def normalize_title(s: str) -> str:
    """Strip and lowercase for fuzzy title matching."""
    return "".join(c.lower() for c in s.strip() if c.isalnum())


def main(run_id: str | None = None) -> None:
    cfg = json.loads(Path("config.json").read_text())
    onb = cfg["agents"]["operations"]["onboarder"]

    # Resolve all secrets up front
    sa_path = _secrets.resolve_path(onb, "google_service_account")
    sheet_id = onb["google_sheet_id"]
    bunny_lib = str(onb["bunny_stream_library_id"])
    bunny_key = _secrets.resolve(onb, "bunny_stream_api_key")
    openai_key = _secrets.resolve(onb, "openai_api_key", env="OPENAI_API_KEY")
    nms_endpoint = onb["nms_endpoint"].rstrip("/")
    nms_token = _secrets.resolve(onb, "nms_api_token", env="AGENT_API_TOKEN")
    scratch_root = Path(onb.get("scratch_dir") or "/tmp/onboarder")
    scratch_root.mkdir(parents=True, exist_ok=True)

    client = sheets.open_client(sa_path)

    log.info("loading pending+approved rows from Lessons…")
    rows = sheets.read_pending_approved_rows(client, sheet_id, run_id=run_id)
    if not rows:
        log.error("no pending+approved rows found")
        return
    log.info(f"got {len(rows)} rows")

    log.info(f"listing Bunny library {bunny_lib}…")
    bunny_videos = list_bunny_videos(bunny_lib, bunny_key)
    log.info(f"got {len(bunny_videos)} videos in library")
    by_norm_title: dict[str, list[dict[str, Any]]] = {}
    for bv in bunny_videos:
        by_norm_title.setdefault(normalize_title(bv["title"]), []).append(bv)

    # Group rows by course_idx
    courses: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        idx = int(r.get("course_idx") or 0)
        courses.setdefault(idx, []).append(r)
    for idx in courses:
        courses[idx].sort(key=lambda r: int(r.get("lesson_idx") or 0))

    # Mark all selected rows as processing
    sheets.update_status(client, sheet_id,
                        sheet_rows=[r["_sheet_row"] for r in rows],
                        new_status=sheets.STATUS_PROCESSING)

    for course_idx, lessons in sorted(courses.items()):
        course_full = lessons[0].get("course") or f"Course {course_idx}"
        clean_title = course_full.split(" — ", 1)[1] if " — " in course_full else course_full
        ch_name = lessons[0].get("channel") or ""
        log.info(f"=== Course {course_idx}: {clean_title} ({ch_name}) — {len(lessons)} lessons ===")

        course_scratch = scratch_root / f"recover-{int(time.time())}" / f"course-{course_idx}"
        course_scratch.mkdir(parents=True, exist_ok=True)
        course_sheet_rows = [r["_sheet_row"] for r in lessons]

        try:
            processed: list[dict[str, Any]] = []
            for lesson in lessons:
                lesson_title = lesson["title"]
                norm = normalize_title(lesson_title)
                bunny_match = (by_norm_title.get(norm) or [None])[0]
                if not bunny_match:
                    # Fuzzier match: prefix of normalized title
                    for bv_norm, bv_list in by_norm_title.items():
                        if bv_norm.startswith(norm[:30]) or norm.startswith(bv_norm[:30]):
                            bunny_match = bv_list[0]
                            log.info(f"fuzzy match: {lesson_title!r} → {bv_list[0]['title']!r}")
                            break
                if not bunny_match:
                    log.warning(f"no Bunny match for: {lesson_title!r}")
                    continue

                guid = bunny_match["guid"]
                duration = bunny_match.get("length", 0)
                log.info(f"  → {lesson_title[:50]} → bunny:{guid}")

                # Use Bunny's auto-generated captions instead of downloading + Whisper
                transcript = fetch_bunny_caption_text(bunny_lib, bunny_key, guid, lang="en")
                if not transcript:
                    log.warning(f"no captions ready for {guid}, using empty transcript")

                processed.append({
                    "title": lesson_title,
                    "videoKey": guid,
                    "videoLibraryId": bunny_lib,
                    "duration": int(duration) or None,
                    "transcriptEn": transcript,
                    "originalLang": "en",
                    "wasDubbed": False,
                })

            if not processed:
                log.error(f"course {course_idx}: no videos matched on Bunny")
                sheets.update_status(client, sheet_id, sheet_rows=course_sheet_rows,
                                    new_status=sheets.STATUS_FAILED,
                                    failure_reason="no Bunny match for any lesson")
                continue

            log.info(f"composing course copy via Claude (Opus)…")
            try:
                composed = llm.compose_full_course(
                    course_topic=clean_title, course_title=clean_title,
                    channel_name=ch_name, channel_description="",
                    lesson_transcripts=[p["transcriptEn"] for p in processed],
                )
            except Exception as e:
                log.warning(f"compose failed, using fallbacks: {e}")
                composed = None

            # Build curriculum payload
            if composed and composed.get("curriculum"):
                llm_lessons = []
                for sec in composed["curriculum"]:
                    for ll in sec.get("lessons", []):
                        llm_lessons.append({"title": ll["title"], "description": ll.get("description", "")})
                if len(llm_lessons) != len(processed):
                    log.warning(f"LLM gave {len(llm_lessons)} lessons but we have {len(processed)} videos — using flat layout")
                    curriculum_payload = [{
                        "title": "Lessons", "isBonus": False,
                        "lessons": [_lesson_payload(p, i,
                            llm_lessons[i]["title"] if i < len(llm_lessons) else p["title"],
                            llm_lessons[i].get("description", "") if i < len(llm_lessons) else "")
                            for i, p in enumerate(processed)],
                    }]
                else:
                    curriculum_payload = []
                    pi = iter(enumerate(processed))
                    for sec in composed["curriculum"]:
                        sec_p = {"title": sec.get("title", "Lessons"), "isBonus": bool(sec.get("isBonus", False)), "lessons": []}
                        for ll in sec.get("lessons", []):
                            i, p = next(pi)
                            sec_p["lessons"].append(_lesson_payload(p, i, ll["title"], ll.get("description", "")))
                        curriculum_payload.append(sec_p)
            else:
                curriculum_payload = [{
                    "title": "Lessons", "isBonus": False,
                    "lessons": [_lesson_payload(p, i, p["title"], "") for i, p in enumerate(processed)],
                }]

            if composed:
                author_payload = {"name": composed["author"]["name"] or ch_name,
                                  "bio": composed["author"]["bio"]}
                course_payload = {
                    "title": composed["course"]["title"] or clean_title,
                    "excerpt": composed["course"]["excerpt"],
                    "aboutContent": composed["course"]["aboutContent"],
                    "isAdult": composed["course"]["isAdult"],
                }
                plan_sections = composed.get("planSections") or []
                science_plan = composed.get("sciencePlan")
                testimonials = composed.get("testimonials") or []
                collection_name = composed.get("collectionName") or None
            else:
                author_payload = {"name": ch_name, "bio": f"{ch_name} — educator on YouTube."}
                course_payload = {
                    "title": clean_title,
                    "excerpt": f"A practical course on {clean_title}.",
                    "aboutContent": f"Curated lessons from {ch_name} on {clean_title}.",
                    "isAdult": False,
                }
                plan_sections, science_plan, testimonials, collection_name = [], None, [], None

            log.info(f"posting to NMS {nms_endpoint}…")
            resp = nms_client.create_draft_course(
                endpoint=nms_endpoint, token=nms_token,
                author=author_payload, course=course_payload,
                curriculum=curriculum_payload,
                plan_sections=plan_sections, science_plan=science_plan,
                testimonials=testimonials, collection_name=collection_name,
            )
            log.info(f"course created: {resp['adminUrl']}")

            sheets.update_status(client, sheet_id,
                                sheet_rows=course_sheet_rows,
                                new_status=sheets.STATUS_DONE,
                                course_admin_url=resp["adminUrl"])
            print(f"\n✅ Course {course_idx} done: {resp['adminUrl']}\n")
        except Exception as e:
            log.error(f"course {course_idx} failed: {e}", exc_info=True)
            sheets.update_status(client, sheet_id, sheet_rows=course_sheet_rows,
                                new_status=sheets.STATUS_FAILED,
                                failure_reason=str(e)[:280])
        finally:
            shutil.rmtree(course_scratch, ignore_errors=True)


def _lesson_payload(p: dict[str, Any], order: int, title: str, description: str) -> dict[str, Any]:
    return {
        "title": title or p["title"],
        "order": order,
        "description": description or (p["transcriptEn"] or "")[:500],
        "videoKey": p["videoKey"],
        "videoLibraryId": p["videoLibraryId"],
        "duration": p.get("duration"),
        "transcriptEn": p["transcriptEn"],
        "originalLang": p.get("originalLang", "en"),
        "wasDubbed": bool(p.get("wasDubbed", False)),
    }


if __name__ == "__main__":
    run_id_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(run_id_arg)
