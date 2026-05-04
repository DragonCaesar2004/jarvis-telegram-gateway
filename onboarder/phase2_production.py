"""Phase 2: production pipeline.

Per video:
    download → working transcribe (find cuts) → ffmpeg cut →
    dub (only non-English) → final transcribe (EN) → Bunny upload

Per course:
    Claude composes title/excerpt/about/bio → POST /api/agent/onboard-course

Triggered by `wiz:start_phase2` callback in wizard.py once the user has
ticked Approved checkboxes in the Run-* sheet tab.

This is the longest-running phase: ~1-3 hours per course depending on
video count and dubbing time. Runs in a daemon thread; progress messages
are throttled.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from . import (_secrets, bunny, elevenlabs_dub, ffmpeg_cut, llm, nms_client,
               sheets, state as _state, whisper)

log = logging.getLogger("gateway")

DEFAULT_SCRATCH_DIR = "/tmp/onboarder"
PROGRESS_INTERVAL_SEC = 60


# ---------------------------------------------------------------------------
# Public entry point (called from wizard's wiz:start_phase2 callback)
# ---------------------------------------------------------------------------

def launch(token: str, agent: str, cfg: dict, chat_id: int, user_id: int) -> None:
    """Spawn the Phase 2 worker in a background daemon thread."""
    thr = threading.Thread(
        target=_worker,
        args=(token, agent, cfg, chat_id, user_id),
        name=f"phase2-{agent}-{user_id}",
        daemon=True,
    )
    thr.start()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(token: str, agent: str, cfg: dict, chat_id: int, user_id: int) -> None:
    onb = (cfg.get("onboarder") or {})
    try:
        _run(token, agent, cfg, chat_id, user_id, onb)
    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"phase2 worker crashed: {e}\n{tb}")
        _state.update(agent, user_id, step="error", error=str(e))
        _send(token, chat_id,
              f"⚠️ <b>Phase 2 упал.</b>\n\n<code>{_html_escape(str(e))[:600]}</code>\n\n"
              "Используй /cancel для возврата в чат.")


def _run(token: str, agent: str, cfg: dict, chat_id: int, user_id: int, onb: dict) -> None:
    # ── 1. Resolve secrets and config ────────────────────────────────────
    sa_path = _secrets.resolve_path(onb, "google_service_account")
    sheet_id = onb.get("google_sheet_id") or ""
    if not sheet_id:
        raise RuntimeError("config: onboarder.google_sheet_id not set")
    openai_key = _secrets.resolve(onb, "openai_api_key", env="OPENAI_API_KEY")
    youtube_cookies_file = onb.get("youtube_cookies_file") or None
    youtube_proxy = onb.get("youtube_proxy") or None
    bunny_lib = str(onb.get("bunny_stream_library_id") or "").strip()
    if not bunny_lib:
        raise RuntimeError("config: onboarder.bunny_stream_library_id not set")
    bunny_key = _secrets.resolve(onb, "bunny_stream_api_key", env="BUNNY_STREAM_API_KEY")

    # ElevenLabs only required if there are any non-English videos.
    # We resolve lazily (per-video) to allow English-only runs without the key.
    def _elevenlabs_key() -> str:
        return _secrets.resolve(onb, "elevenlabs_api_key", env="ELEVENLABS_API_KEY")

    # NMS endpoint+token are optional for now: when missing, we skip the final
    # POST and keep the Bunny videoKeys in wizard state for later manual push.
    nms_endpoint = (onb.get("nms_endpoint") or "").rstrip("/")
    nms_token: str | None = None
    try:
        nms_token = _secrets.resolve(onb, "nms_api_token", env="AGENT_API_TOKEN")
    except FileNotFoundError:
        log.warning("phase2: no NMS token configured — courses will not be auto-pushed to admin")

    scratch_root = Path(onb.get("scratch_dir") or DEFAULT_SCRATCH_DIR)
    scratch_root.mkdir(parents=True, exist_ok=True)

    # ── 2. Read approved rows from Sheet ─────────────────────────────────
    st = _state.load(agent, user_id)
    tab_name = st.get("sheet_tab")
    run_id = st.get("run_id")
    if not tab_name or not run_id:
        raise RuntimeError("wizard state missing sheet_tab/run_id — restart from /menu")

    client = sheets.open_client(sa_path)
    approved = sheets.read_approved_rows(client, sheet_id, tab_name)
    if not approved:
        raise RuntimeError(
            f"В табе {tab_name} нет строк с Approved=TRUE. "
            f"Открой Sheet, поставь галочки и нажми кнопку ещё раз."
        )

    # Group approved lessons by course_idx
    courses: dict[int, list[dict[str, Any]]] = {}
    for row in approved:
        idx = int(row.get("course_idx") or 0)
        courses.setdefault(idx, []).append(row)
    for idx in courses:
        courses[idx].sort(key=lambda r: int(r.get("lesson_idx") or 0))

    total_videos = sum(len(v) for v in courses.values())
    _send(token, chat_id,
          f"🎬 <b>Phase 2 запущен.</b>\n\n"
          f"Курсов: <b>{len(courses)}</b>\n"
          f"Видео: <b>{total_videos}</b>\n"
          f"~1-3 часа на курс. Можешь чатиться с агентом параллельно. "
          f"Прогресс прилечу отдельными сообщениями.")
    _state.update(agent, user_id, step="phase2_running")
    sheets.update_run_status(client, sheet_id, run_id, status="phase2_running")

    # ── 3. Process every video, then assemble courses ────────────────────
    course_results: list[dict[str, Any]] = []
    for course_idx in sorted(courses.keys()):
        lessons = courses[course_idx]
        course_title_full = lessons[0].get("course") or f"Курс {course_idx}"
        # The first row carries the full "Курс N: <Channel> — <title>" text;
        # later rows just say "Курс N". Strip the prefix to get the actual title.
        clean_title = _strip_course_prefix(course_title_full, course_idx)
        ch_name = lessons[0].get("channel") or ""

        _send(token, chat_id,
              f"📚 <b>Курс {course_idx}</b> — {_html_escape(ch_name)}: "
              f"начинаю обработку {len(lessons)} видео")

        course_scratch = scratch_root / f"run-{run_id}" / f"course-{course_idx}"
        course_scratch.mkdir(parents=True, exist_ok=True)
        try:
            processed_lessons = _process_course_videos(
                token=token, chat_id=chat_id, agent=agent, user_id=user_id,
                lessons=lessons, course_idx=course_idx,
                scratch_dir=course_scratch,
                openai_key=openai_key, get_elevenlabs_key=_elevenlabs_key,
                bunny_lib=bunny_lib, bunny_key=bunny_key,
                course_topic=clean_title,
                youtube_cookies_file=youtube_cookies_file,
                youtube_proxy=youtube_proxy,
            )
        finally:
            # Free disk regardless of outcome
            try:
                shutil.rmtree(course_scratch, ignore_errors=True)
            except Exception:
                pass

        if not processed_lessons:
            _send(token, chat_id,
                  f"⚠️ Курс {course_idx} пропущен — ни одно видео не обработалось до конца.")
            continue

        # ── 4. Compose course copy via Claude ────────────────────────────
        _send(token, chat_id, f"✍️ Курс {course_idx}: пишу описание и био автора через Claude…")
        try:
            composed = llm.compose_course(
                course_topic=clean_title,
                course_title=clean_title,
                channel_name=ch_name,
                channel_description="",  # Phase 1 didn't persist; could re-fetch
                lesson_transcripts=[l["transcriptEn"] for l in processed_lessons],
            )
        except Exception as e:
            log.warning(f"phase2: compose_course failed: {e}; falling back to defaults")
            composed = {
                "excerpt": f"A practical course on {clean_title}.",
                "aboutContent": f"Curated lessons from {ch_name} on {clean_title}.",
                "author_bio": f"{ch_name} — educator on YouTube.",
            }

        # ── 5. Push DRAFT course to NewMindStart (if NMS is configured) ──
        course_payload = {
            "author": {
                "name": ch_name,
                "bio": composed["author_bio"],
            },
            "course": {
                "title": clean_title,
                "excerpt": composed["excerpt"],
                "aboutContent": composed["aboutContent"],
            },
            "lessons": [
                {
                    "title": l["title"],
                    "order": idx,
                    "description": (l["transcriptEn"] or "")[:500],
                    "videoKey": l["videoKey"],
                    "videoLibraryId": l["videoLibraryId"],
                    "duration": l["duration"],
                    "transcriptEn": l["transcriptEn"],
                    "originalLang": l["originalLang"],
                    "wasDubbed": l["wasDubbed"],
                }
                for idx, l in enumerate(processed_lessons)
            ],
            "sectionTitle": "Lessons",
        }

        if nms_endpoint and nms_token:
            try:
                resp = nms_client.create_draft_course(
                    endpoint=nms_endpoint, token=nms_token,
                    author=course_payload["author"],
                    course=course_payload["course"],
                    lessons=course_payload["lessons"],
                    section_title=course_payload["sectionTitle"],
                )
                course_results.append({
                    "course_idx": course_idx, "title": clean_title,
                    "admin_url": resp["adminUrl"], "course_id": resp["courseId"],
                })
                _send(token, chat_id,
                      f"✅ <b>Курс {course_idx} создан в админке (DRAFT):</b>\n"
                      f"<a href=\"{resp['adminUrl']}\">{_html_escape(clean_title)}</a>")
            except Exception as e:
                log.error(f"phase2: NMS push failed for course {course_idx}: {e}")
                _send(token, chat_id,
                      f"⚠️ Курс {course_idx}: видео загружены на Bunny, "
                      f"но создание DRAFT не удалось:\n<code>{_html_escape(str(e))[:300]}</code>")
        else:
            # No NMS configured — at least show where the videos went
            keys = "\n".join(f"  • {l['title'][:50]} → bunny:{l['videoKey']}"
                             for l in processed_lessons)
            _send(token, chat_id,
                  f"📦 <b>Курс {course_idx}</b>: видео залиты на Bunny library {bunny_lib}.\n"
                  f"<code>{_html_escape(keys)}</code>\n\n"
                  f"<i>NMS endpoint/token не настроен — DRAFT в админке не создан.</i>")

    # ── 6. Final summary ─────────────────────────────────────────────────
    _state.update(agent, user_id, step="done", courses=course_results)
    sheets.update_run_status(client, sheet_id, run_id, status="done",
                             courses=", ".join(c.get("course_id", "") for c in course_results))

    if course_results:
        lines = [f"  • <a href=\"{c['admin_url']}\">{_html_escape(c['title'])}</a>"
                 for c in course_results]
        _send(token, chat_id,
              f"🏁 <b>Готово.</b> Курсов в админке: {len(course_results)}\n\n"
              + "\n".join(lines)
              + "\n\nПроверь и опубликуй вручную.")
    else:
        _send(token, chat_id,
              f"🏁 Phase 2 закончен, но курсы в админку не попали "
              f"(см. выше — обычно из-за отсутствия NMS-токена). Bunny ключи остались "
              f"в логах wizard'а.")


# ---------------------------------------------------------------------------
# Per-course / per-video processing
# ---------------------------------------------------------------------------

def _process_course_videos(*, token: str, chat_id: int, agent: str, user_id: int,
                           lessons: list[dict[str, Any]], course_idx: int,
                           scratch_dir: Path, openai_key: str,
                           get_elevenlabs_key, bunny_lib: str, bunny_key: str,
                           course_topic: str,
                           youtube_cookies_file: str | None = None,
                           youtube_proxy: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    total = len(lessons)
    for i, lesson in enumerate(lessons, start=1):
        prefix = f"Курс {course_idx}, видео {i}/{total}"
        try:
            row = _process_one_video(
                token=token, chat_id=chat_id, prefix=prefix,
                lesson=lesson, scratch_dir=scratch_dir,
                openai_key=openai_key, get_elevenlabs_key=get_elevenlabs_key,
                bunny_lib=bunny_lib, bunny_key=bunny_key,
                course_topic=course_topic,
                youtube_cookies_file=youtube_cookies_file,
                youtube_proxy=youtube_proxy,
            )

            out.append(row)
        except Exception as e:
            log.error(f"phase2: video {lesson.get('video_id')} failed: {e}", exc_info=True)
            _send(token, chat_id,
                  f"⚠️ {prefix}: <i>{_html_escape(lesson.get('title', '?'))[:50]}</i> — "
                  f"<code>{_html_escape(str(e))[:200]}</code>\n"
                  f"Видео пропущено, продолжаю с остальными.")
    return out


def _process_one_video(*, token: str, chat_id: int, prefix: str,
                       lesson: dict[str, Any], scratch_dir: Path,
                       openai_key: str, get_elevenlabs_key,
                       bunny_lib: str, bunny_key: str,
                       course_topic: str,
                       youtube_cookies_file: str | None = None,
                       youtube_proxy: str | None = None) -> dict[str, Any]:
    video_id = lesson["video_id"]
    title = lesson["title"]
    url = lesson["url"]

    raw_path = scratch_dir / f"{video_id}.raw.mp4"
    cleaned_path = scratch_dir / f"{video_id}.cleaned.mp4"
    final_path = scratch_dir / f"{video_id}.final.mp4"

    # 1. Download
    _send(token, chat_id, f"⏳ {prefix}: скачивание <i>{_html_escape(title)[:50]}</i>…")
    raw_path = ffmpeg_cut.download_video(
        url=url, output_path=raw_path,
        cookies_file=youtube_cookies_file,
        proxy=youtube_proxy,
    )

    # 2. Working transcribe (find cuts)
    _send(token, chat_id, f"📝 {prefix}: транскрибация для разметки вырезок…")
    working = whisper.transcribe(api_key=openai_key, file_path=raw_path,
                                 with_word_timestamps=True)
    # Normalise full language name → ISO 639-1 ("english" → "en", "russian" → "ru")
    from .elevenlabs_dub import _iso as _lang_iso
    detected_lang = _lang_iso((working.get("language") or "").lower())

    # 3. LLM marks cuts
    cuts = llm.mark_cuts(course_topic=course_topic, transcript=working)
    cuts_summary = (f"{len(cuts)} кусков" if cuts else "нет вырезок")

    # 4. Cut
    _send(token, chat_id, f"✂️ {prefix}: вырезка ({cuts_summary})…")
    cleaned_path = ffmpeg_cut.cut_segments(input_path=raw_path, cuts=cuts,
                                           output_path=cleaned_path)

    # 5. Dub if not English
    if detected_lang == "en":
        _send(token, chat_id, f"🇬🇧 {prefix}: уже на английском, дубляж пропускаем")
        final_path = cleaned_path
        was_dubbed = False
    else:
        _send(token, chat_id,
              f"🇬🇧 {prefix}: дубляж {detected_lang} → en (ElevenLabs, ~5-15 мин)…")
        elevenlabs_key = get_elevenlabs_key()
        elevenlabs_dub.dub_video(
            api_key=elevenlabs_key, file_path=cleaned_path,
            source_lang=detected_lang, target_lang="en",
            output_path=final_path, name=title[:80],
        )
        was_dubbed = True

    # 6. Final transcribe (EN, no word timestamps needed)
    _send(token, chat_id, f"📝 {prefix}: финальная транскрибация (EN)…")
    final_transcript = whisper.transcribe(api_key=openai_key, file_path=final_path,
                                          language="en", with_word_timestamps=False)

    # 7. Upload to Bunny
    duration_sec = int(ffmpeg_cut.probe_duration(final_path))
    _send(token, chat_id, f"☁️ {prefix}: загрузка на Bunny ({duration_sec}s)…")
    bunny_meta = bunny.upload_video(
        library_id=bunny_lib, api_key=bunny_key,
        file_path=final_path, title=title,
    )

    _send(token, chat_id, f"✅ {prefix}: готово")

    return {
        "title": title,
        "videoKey": bunny_meta["videoKey"],
        "videoLibraryId": bunny_meta["videoLibraryId"],
        "duration": duration_sec,
        "transcriptEn": final_transcript.get("text", ""),
        "originalLang": detected_lang,
        "wasDubbed": was_dubbed,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_course_prefix(s: str, idx: int) -> str:
    """Drop the leading 'Курс N: <Channel> —' part if present."""
    prefix_a = f"Курс {idx}: "
    prefix_b = f"Курс {idx} "
    if s.startswith(prefix_a):
        rest = s[len(prefix_a):]
        if " — " in rest:
            return rest.split(" — ", 1)[1]
        return rest
    if s.startswith(prefix_b):
        return s[len(prefix_b):]
    return s


def _send(token: str, chat_id: int, text: str) -> None:
    from gateway import tg_api  # type: ignore
    try:
        tg_api(token, "sendMessage", chat_id=chat_id, text=text, parse_mode="HTML")
    except Exception as e:
        log.warning(f"phase2 _send failed: {e}")


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )
