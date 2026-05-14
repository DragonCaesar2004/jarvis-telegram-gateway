"""End-to-end recovery of a partially-uploaded course.

Takes a run_id whose course got partially into admin (some videos done,
some stuck/failed). Builds ONE consolidated new DRAFT that contains all
the original approved videos:

  1. Reads every approved row of run_id from the Lessons sheet (any status).
  2. Lists the Bunny library, matches done videos by title → reuses their
     existing videoKey (no re-upload).
  3. For rows NOT yet in Bunny, runs the full Phase 2 pipeline on them
     (download → cut → dub → upload).
  4. Pulls the cached compose payload from pipeline.db; falls back to a
     fresh compose_full_course call if missing.
  5. Builds a curriculum with ALL videos in original lesson_idx order.
  6. POSTs to /api/agent/onboard-course → new DRAFT.
  7. Updates all approved rows to status=done with the new admin URL.
  8. Optionally notifies the operator via Telegram.

Usage:
    python -m onboarder.complete_course <run_id> [--telegram-chat-id N]
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from . import (_secrets, bunny, cache, ffmpeg_cut, google_dub, llm,
               nms_client, phase2_production, pipeline_db, proxy_pool,
               sheets, whisper)
from .recover_from_bunny import (_read_all_approved_rows, list_bunny_videos,
                                 fetch_bunny_caption_text, normalize_title)

log = logging.getLogger("complete_course")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def main(run_id: str, telegram_chat_id: int | None = None) -> None:
    cfg = json.loads(Path("config.json").read_text())
    onb = cfg["agents"]["operations"]["onboarder"]

    # ── Resolve secrets ──────────────────────────────────────────────────
    sa_path = _secrets.resolve_path(onb, "google_service_account")
    sheet_id = onb["google_sheet_id"]
    bunny_lib = str(onb["bunny_stream_library_id"])
    bunny_key = _secrets.resolve(onb, "bunny_stream_api_key")
    openai_key = _secrets.resolve(onb, "openai_api_key", env="OPENAI_API_KEY")
    nms_endpoint = onb["nms_endpoint"].rstrip("/")
    nms_token = _secrets.resolve(onb, "nms_api_token", env="AGENT_API_TOKEN")
    youtube_cookies_file = onb.get("youtube_cookies_file")
    proxy_pool_list = proxy_pool.normalise_pool(
        onb.get("youtube_proxies") or onb.get("youtube_proxy"))
    scratch_root = Path(onb.get("scratch_dir") or "/tmp/onboarder")
    scratch_root.mkdir(parents=True, exist_ok=True)

    # ── Read rows ────────────────────────────────────────────────────────
    client = sheets.open_client(sa_path)
    rows = _read_all_approved_rows(client, sheet_id, run_id=run_id)
    if not rows:
        log.error(f"no approved rows for run_id={run_id}")
        return
    rows.sort(key=lambda r: (r.get("course_idx", 0), r.get("lesson_idx", 0)))
    log.info(f"got {len(rows)} approved rows, statuses="
             f"{sorted({r.get('status', '?') for r in rows})}")

    # ── Index Bunny library by normalized title ──────────────────────────
    log.info(f"listing Bunny library {bunny_lib}…")
    bunny_videos = list_bunny_videos(bunny_lib, bunny_key)
    log.info(f"got {len(bunny_videos)} videos in library")
    by_norm: dict[str, list[dict[str, Any]]] = {}
    for bv in bunny_videos:
        by_norm.setdefault(normalize_title(bv["title"]), []).append(bv)

    # ── Init proxy rotator (only if we'll need to download) ──────────────
    rotator = proxy_pool.ProxyRotator(
        proxy_pool_list, cookies_file=youtube_cookies_file)
    needs_youtube = any(
        not _find_bunny_match(r.get("title", ""), by_norm)
        for r in rows
    )
    if needs_youtube and proxy_pool_list:
        log.info(f"probing {len(proxy_pool_list)} proxies…")
        try:
            rotator.init()
            log.info(f"proxy ready: {proxy_pool._proxy_label(rotator.current)}")
        except Exception as e:
            log.warning(f"proxy probe failed: {e} — trying direct")

    # ── Get voice_gender from wizard state of the user, default MALE ─────
    voice_gender = "MALE"  # safe default; not critical for recovery

    # ── Group rows by course_idx ─────────────────────────────────────────
    courses: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        idx = int(r.get("course_idx") or 1)
        courses.setdefault(idx, []).append(r)
    for idx in courses:
        courses[idx].sort(key=lambda r: int(r.get("lesson_idx") or 0))

    # ── Read bot token for optional Telegram notification ────────────────
    bot_token = _read_first_bot_token(cfg)

    overall_results: list[dict[str, Any]] = []

    for course_idx, lessons in sorted(courses.items()):
        course_label = lessons[0].get("course") or f"Course {course_idx}"
        clean_title = (course_label.split(" — ", 1)[1]
                       if " — " in course_label else course_label)
        ch_name = lessons[0].get("channel") or ""
        log.info(f"=== Course {course_idx}: {clean_title} ({ch_name}) "
                 f"— {len(lessons)} lessons ===")

        course_scratch = scratch_root / f"complete-{int(time.time())}-c{course_idx}"
        course_scratch.mkdir(parents=True, exist_ok=True)

        try:
            processed = _build_processed_lessons(
                lessons=lessons, by_norm=by_norm,
                bunny_lib=bunny_lib, bunny_key=bunny_key,
                openai_key=openai_key,
                scratch_dir=course_scratch,
                course_topic=clean_title,
                cookies_file=youtube_cookies_file,
                rotator=rotator,
                voice_gender=voice_gender,
                onb=onb,
            )
        finally:
            shutil.rmtree(course_scratch, ignore_errors=True)

        if not processed:
            log.error(f"course {course_idx}: no lessons survived processing")
            continue

        log.info(f"course {course_idx}: {len(processed)}/{len(lessons)} lessons ready")

        # ── Build NMS payload ────────────────────────────────────────────
        composed = pipeline_db.get_course_compose(
            run_id=run_id, course_idx=course_idx)

        # If compose is missing OR has no curriculum (yesterday Phase 1 failed
        # at this step due to Claude Max limit), call compose_full_course now
        # using the transcripts we have (from pipeline_db.video_cuts +
        # Phase 2 results + Bunny captions). Save the result back to pipeline_db.
        needs_compose = (
            composed is None
            or not (composed.get("curriculum") or [])
            or len((composed.get("course", {}).get("aboutContent") or "")) < 200
        )
        if needs_compose:
            log.info(f"course {course_idx}: composed payload missing/empty in "
                     f"pipeline.db — running compose_full_course on the fly")
            transcripts = _gather_transcripts(
                lessons=lessons, processed=processed,
                bunny_lib=str(bunny_lib), bunny_key=bunny_key,
            )
            log.info(f"  gathered {sum(1 for t in transcripts if t)} non-empty "
                     f"transcripts out of {len(transcripts)}")
            compose_model = (onb.get("models") or {}).get("compose") or llm.DEFAULT_MODEL_QUALITY
            try:
                composed = llm.compose_full_course(
                    course_topic=clean_title,
                    course_title=clean_title,
                    channel_name=ch_name,
                    channel_description="",
                    lesson_transcripts=transcripts,
                    model=compose_model,
                )
                # Persist for future runs of this run_id
                try:
                    pipeline_db.save_course_compose(
                        run_id=run_id, course_idx=course_idx,
                        composed=composed,
                    )
                except Exception as e:
                    log.warning(f"  save_course_compose failed: {e}")
                log.info(f"  ✓ compose ready: title={composed['course']['title']!r}, "
                         f"about={len(composed['course']['aboutContent'])} chars, "
                         f"plan={len(composed.get('planSections', []))} sections")
            except Exception as e:
                log.error(f"  compose_full_course failed: {e}", exc_info=True)
                # Continue with whatever we have (stubs)
                composed = composed or None

        author_payload, course_payload, plan_sections, science_plan, \
            testimonials, collection_name, curriculum_payload = \
            _build_payload(composed=composed, processed=processed,
                           ch_name=ch_name, clean_title=clean_title)

        # ── Sanitize all fields to English ───────────────────────────────
        try:
            translate_key = _secrets.resolve(
                onb, "google_translate_api_key",
                env="GOOGLE_TRANSLATE_API_KEY")
            n_translated = phase2_production._sanitize_payload_to_english(
                author_payload=author_payload,
                course_payload=course_payload,
                curriculum_payload=curriculum_payload,
                plan_sections=plan_sections,
                science_plan=science_plan,
                testimonials=testimonials,
                processed_lessons=processed,
                api_key=translate_key,
            )
            if n_translated:
                log.info(f"sanitizer translated {n_translated} Cyrillic fields → EN")
        except Exception as e:
            log.warning(f"sanitizer skipped: {e}")

        # ── POST to NMS ──────────────────────────────────────────────────
        log.info(f"POST → {nms_endpoint}/api/agent/onboard-course "
                 f"({len(curriculum_payload)} sections, "
                 f"{sum(len(s['lessons']) for s in curriculum_payload)} lessons)")
        try:
            resp = nms_client.create_draft_course(
                endpoint=nms_endpoint, token=nms_token,
                author=author_payload, course=course_payload,
                curriculum=curriculum_payload,
                plan_sections=plan_sections, science_plan=science_plan,
                testimonials=testimonials, collection_name=collection_name,
            )
        except Exception as e:
            log.error(f"NMS POST failed: {e}", exc_info=True)
            continue

        admin_url = resp.get("adminUrl", "")
        log.info(f"✅ Course {course_idx} created: {admin_url}")
        overall_results.append({
            "course_idx": course_idx, "title": clean_title,
            "admin_url": admin_url, "lesson_count": len(processed),
        })

        # ── Update Sheet rows → done with new admin URL ──────────────────
        sheet_rows_idx = [r["_sheet_row"] for r in lessons]
        sheets.update_status(
            client, sheet_id, sheet_rows=sheet_rows_idx,
            new_status=sheets.STATUS_DONE,
            course_admin_url=admin_url,
        )
        log.info(f"updated {len(sheet_rows_idx)} sheet rows → done")

    # ── Final Telegram notification ──────────────────────────────────────
    if telegram_chat_id and bot_token and overall_results:
        _notify_telegram(bot_token, telegram_chat_id, run_id, overall_results)


# ───────────────────────────────────────────────────────────────────────────
# Per-video processing: reuse existing Bunny match OR run full Phase 2
# ───────────────────────────────────────────────────────────────────────────

def _build_processed_lessons(*, lessons, by_norm, bunny_lib, bunny_key,
                             openai_key, scratch_dir, course_topic,
                             cookies_file, rotator, voice_gender,
                             onb) -> list[dict[str, Any]]:
    """Return a list of lesson payloads (videoKey, transcript, etc), one per
    input lesson row. Uses Bunny match when present; otherwise downloads +
    processes the video.
    """
    processed: list[dict[str, Any]] = []
    for lesson in lessons:
        title = lesson.get("title", "")
        video_id = lesson.get("video_id", "")

        bunny_match = _find_bunny_match(title, by_norm)
        if bunny_match:
            guid = bunny_match["guid"]
            duration = int(bunny_match.get("length", 0)) or None
            log.info(f"  ✓ {title[:60]} → in Bunny ({guid})")
            transcript = fetch_bunny_caption_text(
                bunny_lib, bunny_key, guid, lang="en")
            processed.append({
                "title": title,
                "videoKey": guid,
                "videoLibraryId": bunny_lib,
                "duration": duration,
                "transcriptEn": transcript,
                "originalLang": "en",  # we don't know; harmless placeholder
                "wasDubbed": False,
                "lessonDescription": lesson.get("lesson_description", ""),
            })
            continue

        # Missing from Bunny → run full Phase 2 pipeline.
        log.info(f"  ⏳ {title[:60]} ({video_id}) — running Phase 2…")
        try:
            # Reuse the production Phase 2 per-video function; pass empty token
            # so its Telegram _send() no-ops (it swallows the API error).
            result = phase2_production._process_one_video(
                token="", chat_id=0,
                prefix=f"complete[{video_id}]",
                lesson=lesson, scratch_dir=scratch_dir,
                openai_key=openai_key,
                get_google_translate_key=lambda: _secrets.resolve(
                    onb, "google_translate_api_key",
                    env="GOOGLE_TRANSLATE_API_KEY"),
                get_google_tts_key=lambda: _secrets.resolve(
                    onb, "google_tts_api_key",
                    env="GOOGLE_TTS_API_KEY"),
                voice_gender=voice_gender,
                bunny_lib=str(bunny_lib),
                bunny_key=bunny_key,
                course_topic=course_topic,
                youtube_cookies_file=cookies_file,
                rotator=rotator,
            )
            result["lessonDescription"] = lesson.get("lesson_description", "")
            processed.append(result)
            log.info(f"  ✅ {title[:60]} uploaded → {result['videoKey']}")
        except Exception as e:
            log.error(f"  ❌ {title[:60]} failed: {e}", exc_info=True)
            continue

    return processed


def _gather_transcripts(*, lessons: list[dict[str, Any]],
                        processed: list[dict[str, Any]],
                        bunny_lib: str, bunny_key: str) -> list[str]:
    """Collect a transcript per lesson, in original lesson_idx order.

    Source priority (first non-empty wins):
      1. `processed[i].transcriptEn` — freshly produced by Phase 2 just now
      2. `pipeline_db.video_cuts.working_transcript` — Phase 1's transcript
      3. Bunny auto-generated EN captions (if ready)
    """
    # Index processed by video_id for fast lookup
    by_vid: dict[str, dict[str, Any]] = {}
    for p in processed:
        # Phase 2 result doesn't include video_id directly; fall back to
        # title-match against the lessons list.
        pass
    # Instead match by lesson title (1:1 with lessons in same order)
    proc_by_title = {p.get("title"): p for p in processed}

    out: list[str] = []
    for lesson in lessons:
        title = lesson.get("title", "")
        video_id = lesson.get("video_id", "")
        text = ""

        # 1. Freshly processed
        p = proc_by_title.get(title)
        if p and (p.get("transcriptEn") or "").strip():
            text = p["transcriptEn"]
        # 2. pipeline.db Phase 1 transcript
        if not text and video_id:
            try:
                rec = pipeline_db.get_cuts(video_id)
                if rec and rec.get("working_transcript"):
                    text = rec["working_transcript"]
            except Exception as e:
                log.warning(f"  pipeline_db read failed for {video_id}: {e}")
        # 3. Bunny captions (slow, single REST call per video)
        if not text and p and p.get("videoKey"):
            try:
                from .recover_from_bunny import fetch_bunny_caption_text
                text = fetch_bunny_caption_text(
                    bunny_lib, bunny_key, p["videoKey"], lang="en") or ""
            except Exception as e:
                log.warning(f"  bunny caption fetch failed: {e}")

        out.append(text or "")
    return out


def _find_bunny_match(title: str,
                      by_norm: dict[str, list[dict[str, Any]]]
                      ) -> dict[str, Any] | None:
    norm = normalize_title(title)
    if not norm:
        return None
    direct = by_norm.get(norm)
    if direct:
        return direct[0]
    # Fuzzy prefix match (30 chars)
    for bv_norm, bv_list in by_norm.items():
        if not bv_norm or not norm:
            continue
        if bv_norm.startswith(norm[:30]) or norm.startswith(bv_norm[:30]):
            return bv_list[0]
    return None


# ───────────────────────────────────────────────────────────────────────────
# Payload assembly (mostly mirrors phase2_production logic)
# ───────────────────────────────────────────────────────────────────────────

def _build_payload(*, composed, processed, ch_name, clean_title):
    """Build (author, course, plan, science, testimonials, collection, curriculum)
    payload tuple from the cached compose + processed lessons.
    """
    if composed and composed.get("curriculum"):
        # Flatten LLM curriculum, map 1:1 to processed videos in order
        llm_flat = []
        for sec in composed["curriculum"]:
            for ll in sec.get("lessons", []):
                llm_flat.append((sec, ll))
        if len(llm_flat) == len(processed):
            # Reconstruct sections preserving LLM structure
            curriculum_payload = []
            pi = iter(enumerate(processed))
            for sec in composed["curriculum"]:
                sec_p = {
                    "title": sec.get("title", "Lessons"),
                    "isBonus": bool(sec.get("isBonus", False)),
                    "lessons": [],
                }
                for ll in sec.get("lessons", []):
                    i, p = next(pi)
                    sec_p["lessons"].append(_lesson_payload(
                        p, i, ll["title"], ll.get("description", "")))
                curriculum_payload.append(sec_p)
        else:
            # LLM count mismatch (e.g. we added new videos) — flat layout
            log.warning(f"LLM gave {len(llm_flat)} lessons but we have "
                        f"{len(processed)} videos — using flat 'Lessons' section")
            curriculum_payload = [{
                "title": "Lessons", "isBonus": False,
                "lessons": [_lesson_payload(
                    p, i,
                    llm_flat[i][1]["title"] if i < len(llm_flat) else p["title"],
                    llm_flat[i][1].get("description", "")
                    if i < len(llm_flat) else "")
                    for i, p in enumerate(processed)],
            }]
    else:
        # No compose available — minimal flat layout
        curriculum_payload = [{
            "title": "Lessons", "isBonus": False,
            "lessons": [_lesson_payload(p, i, p["title"], "")
                        for i, p in enumerate(processed)],
        }]

    if composed:
        author_payload = {
            "name": composed["author"].get("name") or ch_name,
            "bio": composed["author"].get("bio") or "",
        }
        course_payload = {
            "title": composed["course"]["title"] or clean_title,
            "excerpt": composed["course"].get("excerpt", ""),
            "aboutContent": composed["course"].get("aboutContent", ""),
            "isAdult": bool(composed["course"].get("isAdult", False)),
        }
        plan_sections = composed.get("planSections") or []
        science_plan = composed.get("sciencePlan")
        testimonials = composed.get("testimonials") or []
        collection_name = composed.get("collectionName") or None
    else:
        author_payload = {
            "name": ch_name,
            "bio": f"{ch_name} — educator on YouTube.",
        }
        course_payload = {
            "title": clean_title,
            "excerpt": f"A practical course on {clean_title}.",
            "aboutContent": f"Curated lessons from {ch_name} on {clean_title}.",
            "isAdult": False,
        }
        plan_sections, science_plan = [], None
        testimonials, collection_name = [], None

    return (author_payload, course_payload, plan_sections, science_plan,
            testimonials, collection_name, curriculum_payload)


def _lesson_payload(p, order, title, description):
    return {
        "title": title or p["title"],
        "order": order,
        "description": description or (p.get("transcriptEn") or "")[:500],
        "videoKey": p["videoKey"],
        "videoLibraryId": p["videoLibraryId"],
        "duration": p.get("duration"),
        "transcriptEn": p.get("transcriptEn", ""),
        "originalLang": p.get("originalLang", "en"),
        "wasDubbed": bool(p.get("wasDubbed", False)),
    }


# ───────────────────────────────────────────────────────────────────────────
# Telegram notify (optional)
# ───────────────────────────────────────────────────────────────────────────

def _read_first_bot_token(cfg: dict) -> str:
    for agent_name, agent_cfg in (cfg.get("agents") or {}).items():
        path = agent_cfg.get("telegram_bot_token_file")
        if not path:
            continue
        p = Path(path).expanduser()
        if p.exists():
            return p.read_text().strip()
    return ""


def _notify_telegram(bot_token: str, chat_id: int, run_id: str,
                     results: list[dict[str, Any]]) -> None:
    import requests
    lines = [
        f"🎉 <b>Восстановление курса завершено</b>",
        f"run_id: <code>{run_id}</code>\n",
    ]
    for r in results:
        lines.append(
            f"  • Курс {r['course_idx']}: <b>{r['title']}</b> "
            f"({r['lesson_count']} уроков) → "
            f"<a href=\"{r['admin_url']}\">админка</a>"
        )
    lines.append(
        "\n⚠️ <b>Старые DRAFT-курсы того же run_id остались в админке</b> "
        "— удали их вручную через UI."
    )
    text = "\n".join(lines)
    try:
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text,
                  "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=30,
        )
    except Exception as e:
        log.warning(f"Telegram notify failed: {e}")


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    chat_id_arg: int | None = None
    cleaned_args: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--telegram-chat-id" and i + 1 < len(args):
            try:
                chat_id_arg = int(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        cleaned_args.append(a)
        i += 1

    if not cleaned_args:
        print("Usage: python -m onboarder.complete_course <run_id> "
              "[--telegram-chat-id N]")
        sys.exit(2)

    main(cleaned_args[0], telegram_chat_id=chat_id_arg)
