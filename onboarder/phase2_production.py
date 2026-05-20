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
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import _global_throttle
from pathlib import Path
from typing import Any

from . import (_secrets, bunny, cache, google_dub, ffmpeg_cut, llm,
               nms_client, pipeline_db, proxy_pool, sheets,
               state as _state, whisper)
from .proxy_pool import CookiesNeededError

log = logging.getLogger("gateway")

DEFAULT_SCRATCH_DIR = "/tmp/onboarder"
PROGRESS_INTERVAL_SEC = 60

# Thread-local context — set at the start of each background worker so the
# module-level _send / _send_with_buttons helpers can route status messages
# to the right Telegram forum topic without each call site forwarding it.
_TLS = threading.local()


# ---------------------------------------------------------------------------
# Public entry point (called from wizard's wiz:start_phase2 callback)
# ---------------------------------------------------------------------------

def launch(token: str, agent: str, cfg: dict, chat_id: int, user_id: int,
           *, thread_id: int = 0,
           run_id_override: str | None = None,
           course_idx_filter: int | None = None,
           voice_gender_override: str | None = None) -> None:
    """Spawn the Phase 2 worker in a background daemon thread.

    `thread_id` is the Telegram forum topic the run was started in. 0 means
    DM / non-forum group, preserving the original behavior.

    `run_id_override` / `course_idx_filter` scope this launch to a single
    course inside a specific run — used by the per-course «🚀 Запустить
    Курс N» button so the operator can process one course at a time
    without claiming rows from the other 4 courses in the same Sheet.

    `voice_gender_override` lets the per-course button supply a fresh
    gender choice without mutating shared wizard state (which other
    parallel per-course launches might read).
    """
    thr = threading.Thread(
        target=_worker,
        args=(token, agent, cfg, chat_id, user_id, int(thread_id or 0)),
        kwargs={
            "run_id_override": run_id_override,
            "course_idx_filter": course_idx_filter,
            "voice_gender_override": voice_gender_override,
        },
        name=(
            f"phase2-{agent}-{user_id}-{int(thread_id or 0)}"
            + (f"-c{course_idx_filter}" if course_idx_filter else "")
        ),
        daemon=True,
    )
    thr.start()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(token: str, agent: str, cfg: dict, chat_id: int, user_id: int,
            thread_id: int = 0, *,
            run_id_override: str | None = None,
            course_idx_filter: int | None = None,
            voice_gender_override: str | None = None) -> None:
    onb = (cfg.get("onboarder") or {})
    _TLS.thread_id = int(thread_id or 0)
    try:
        _run(token, agent, cfg, chat_id, user_id, onb, thread_id=int(thread_id or 0),
             run_id_override=run_id_override,
             course_idx_filter=course_idx_filter,
             voice_gender_override=voice_gender_override)
    except CookiesNeededError as e:
        # Pool exhausted — pause Phase 2 and ask user to upload fresh cookies
        log.warning(f"phase2: cookies needed: {e}")
        _state.update(agent, user_id, thread_id=int(thread_id or 0),
                      step="awaiting_cookies_pre_phase2")
        try:
            from gateway import set_user_mode, MODE_WIZARD  # type: ignore
            set_user_mode(agent, user_id, MODE_WIZARD, int(thread_id or 0))
        except Exception:
            pass
        _send_with_buttons(
            token, chat_id,
            text=(
                "⏸ <b>Phase 2 на паузе.</b>\n\n"
                f"{_html_escape(str(e))}\n\n"
                "Загрузи свежий cookies.txt с youtube.com → бот сам "
                "перезапустит Phase 2 с того видео, на котором остановился."
            ),
            buttons=[[
                {"text": "🚀 Продолжить (cookies свежие)", "callback_data": "wiz:start_phase2"},
                {"text": "✖️ Отмена", "callback_data": "wiz:cancel"},
            ]],
        )
    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"phase2 worker crashed: {e}\n{tb}")
        _state.update(agent, user_id, thread_id=int(thread_id or 0),
                      step="error", error=str(e))
        _send(token, chat_id,
              f"⚠️ <b>Phase 2 упал.</b>\n\n<code>{_html_escape(str(e))[:600]}</code>\n\n"
              "Используй /cancel для возврата в чат.")
    finally:
        _TLS.thread_id = 0


def _run(token: str, agent: str, cfg: dict, chat_id: int, user_id: int, onb: dict,
         *, thread_id: int = 0,
         run_id_override: str | None = None,
         course_idx_filter: int | None = None,
         voice_gender_override: str | None = None) -> None:
    # ── 1. Resolve secrets and config ────────────────────────────────────
    sa_path = _secrets.resolve_path(onb, "google_service_account")
    sheet_id = onb.get("google_sheet_id") or ""
    if not sheet_id:
        raise RuntimeError("config: onboarder.google_sheet_id not set")
    openai_key = _secrets.resolve(onb, "openai_api_key", env="OPENAI_API_KEY")
    youtube_cookies_file = onb.get("youtube_cookies_file") or None
    proxy_pool_list = proxy_pool.normalise_pool(
        onb.get("youtube_proxies") or onb.get("youtube_proxy")
    )
    bunny_lib = str(onb.get("bunny_stream_library_id") or "").strip()
    if not bunny_lib:
        raise RuntimeError("config: onboarder.bunny_stream_library_id not set")
    bunny_key = _secrets.resolve(onb, "bunny_stream_api_key", env="BUNNY_STREAM_API_KEY")

    # Google API keys for dubbing (only needed for non-English videos).
    # Resolved lazily so English-only runs don't require them.
    def _google_translate_key() -> str:
        return _secrets.resolve(onb, "google_translate_api_key",
                                env="GOOGLE_TRANSLATE_API_KEY")

    def _google_tts_key() -> str:
        return _secrets.resolve(onb, "google_tts_api_key",
                                env="GOOGLE_TTS_API_KEY")

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

    # ── 2. Read approved+pending rows from unified Lessons tab ──────────
    st = _state.load(agent, user_id, int(thread_id or 0))
    # Per-course launches pass overrides; bulk launches fall back to wizard state.
    run_id = run_id_override or st.get("run_id")
    voice_gender: str = str(
        voice_gender_override or st.get("voice_gender") or "MALE"
    ).upper()
    # run_id is optional now — Phase 2 picks up any approved+pending rows
    # regardless of run, since user can mix-and-match across runs in the
    # single tab. If run_id is set, we filter to that run for safety.

    client = sheets.open_client(sa_path)
    # Atomic claim under the sheet lock: read approved+pending rows AND
    # immediately stamp them as `processing` in one critical section. A
    # second Phase 2 worker that opens the lock right after will see those
    # same rows as already-processing and skip them — no two workers ever
    # process the same lesson.
    with sheets.sheet_lock():
        approved = sheets.read_pending_approved_rows(
            client, sheet_id, run_id=run_id,
            course_idx=course_idx_filter,
        )
        if not approved:
            scope = []
            if run_id:
                scope.append(f"run_id={run_id}")
            if course_idx_filter is not None:
                scope.append(f"курс {course_idx_filter}")
            scope_str = (" для " + ", ".join(scope)) if scope else ""
            raise RuntimeError(
                f"В табе Lessons нет строк со status=pending и approved=TRUE{scope_str}."
                " Открой Sheet, отметь Approved=TRUE и нажми кнопку ещё раз."
            )
        sheets.update_status(
            client, sheet_id,
            sheet_rows=[r["_sheet_row"] for r in approved],
            new_status=sheets.STATUS_PROCESSING,
        )

    # Group approved lessons by course_idx (within their run)
    courses: dict[int, list[dict[str, Any]]] = {}
    for row in approved:
        idx = int(row.get("course_idx") or 0)
        courses.setdefault(idx, []).append(row)
    for idx in courses:
        courses[idx].sort(key=lambda r: int(r.get("lesson_idx") or 0))

    total_videos = sum(len(v) for v in courses.values())
    cached_count = sum(1 for c in courses.values() for r in c
                       if cache.is_cached(r.get("video_id", "")))
    _send(token, chat_id,
          f"🎬 <b>Phase 2 запущен.</b>\n\n"
          f"Курсов: <b>{len(courses)}</b>\n"
          f"Видео: <b>{total_videos}</b> (из них {cached_count} с кешем из Phase 1 — "
          f"скачивать не нужно)\n"
          f"Все строки помечены status=processing — параллельные запуски не "
          f"подхватят их повторно.\n"
          f"~30-90 мин на курс (большую часть времени съест дубляж, если есть не-EN видео).")
    _state.update(agent, user_id, thread_id=int(thread_id or 0),
                  step="phase2_running", chat_id=chat_id)

    # ── 2.5 Initialize proxy rotator — pick first working proxy ─────────
    # Skip the probe entirely if every approved video already has a cached
    # MP4 (Phase 1 typically downloads them all). Phase 2 in that case never
    # touches YouTube — no point burning 10 sec × 30 proxies.
    rotator = proxy_pool.ProxyRotator(proxy_pool_list, cookies_file=youtube_cookies_file)
    needs_youtube = any(
        not cache.is_cached(r.get("video_id", ""))
        for c in courses.values() for r in c
    )
    if needs_youtube and proxy_pool_list:
        _send(token, chat_id,
              f"🔍 Проверяю {len(proxy_pool_list)} прокси на YouTube (~10 сек на каждый)…")

        last_progress = [time.time()]
        results: list[str] = []

        def _on_probe(idx: int, total: int, name: str, ok: bool) -> None:
            results.append(f"{'✅' if ok else '❌'} {idx}/{total} {name}")
            now = time.time()
            if ok or now - last_progress[0] > 15 or idx == total:
                _send(token, chat_id, "\n".join(results[-12:]))
                last_progress[0] = now

        try:
            rotator.init(on_progress=_on_probe)
            _send(token, chat_id,
                  f"✅ Использую прокси: <code>{proxy_pool._proxy_label(rotator.current)}</code>")
        except proxy_pool.NoWorkingProxyError as e:
            raise RuntimeError(
                f"Ни один прокси не прошёл проверку YouTube.\n\n{str(e)[:600]}\n\n"
                f"Возможные причины: cookies протухли, все IP rate-limited, "
                f"YouTube ужесточил защиту. Попробуй: обновить cookies, "
                f"добавить новые прокси в config.json, подождать 1-2 часа."
            )
    elif not needs_youtube:
        _send(token, chat_id,
              "📦 Все видео уже в кеше Phase 1 — пропускаю проверку прокси.")
    else:
        _send(token, chat_id, "⚠️ Прокси не настроен — пробую напрямую с VPS-IP")

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
                openai_key=openai_key,
                get_google_translate_key=_google_translate_key,
                get_google_tts_key=_google_tts_key,
                voice_gender=voice_gender,
                bunny_lib=bunny_lib, bunny_key=bunny_key,
                course_topic=clean_title,
                youtube_cookies_file=youtube_cookies_file,
                rotator=rotator,
                parallel_videos=int(onb.get("phase2_parallel_videos", 1) or 1),
                mark_cuts_word_level=bool(onb.get("mark_cuts_word_level", False)),
                blur_watermarks=bool(onb.get("blur_watermarks", False)),
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

        # ── 4. Compose payload — prefer Phase 1's pre-composed record ───
        composed_full: dict[str, Any] | None = None
        cached_compose = None
        if run_id:
            try:
                cached_compose = pipeline_db.get_course_compose(
                    run_id=run_id, course_idx=course_idx)
            except Exception as e:
                log.warning(f"phase2: pipeline_db.get_course_compose failed: {e}")

        if cached_compose:
            _send(token, chat_id,
                  f"📋 Курс {course_idx}: использую описания из Phase 1 (без повторного Claude)")
            composed_full = cached_compose
        else:
            _send(token, chat_id,
                  f"✍️ Курс {course_idx}: composed payload не найден в pipeline.db, "
                  f"запускаю Claude compose как fallback…")
            compose_model = (onb.get("models") or {}).get("compose") or llm.DEFAULT_MODEL_QUALITY
            try:
                composed_full = llm.compose_full_course(
                    course_topic=clean_title,
                    course_title=clean_title,
                    channel_name=ch_name,
                    channel_description="",
                    lesson_transcripts=[l["transcriptEn"] for l in processed_lessons],
                    model=compose_model,
                )
            except Exception as e:
                log.warning(f"phase2: compose_full_course fallback failed: {e}")
                composed_full = None

        # Build the curriculum: zip LLM lesson titles with our processed video metadata.
        # LLM curriculum may have multiple sections — we map ALL its lessons in order
        # against our N processed videos.
        if composed_full and composed_full.get("curriculum"):
            llm_curriculum = composed_full["curriculum"]
            llm_flat_lessons: list[tuple[dict[str, Any], int, int]] = []
            for s_idx, sec in enumerate(llm_curriculum):
                for l_idx_in_sec, lesson in enumerate(sec.get("lessons", [])):
                    llm_flat_lessons.append((lesson, s_idx, l_idx_in_sec))

            # If LLM produced a different lesson count than we have videos, fall back
            # to a single "Lessons" section using LLM titles where possible.
            if len(llm_flat_lessons) != len(processed_lessons):
                log.warning(
                    f"phase2: LLM gave {len(llm_flat_lessons)} lessons but we have "
                    f"{len(processed_lessons)} videos — flattening to single section"
                )
                merged_lessons = []
                for v_idx, video in enumerate(processed_lessons):
                    title = (llm_flat_lessons[v_idx][0]["title"]
                             if v_idx < len(llm_flat_lessons) else video["title"])
                    description = (llm_flat_lessons[v_idx][0].get("description", "")
                                   if v_idx < len(llm_flat_lessons) else "")
                    merged_lessons.append(_lesson_payload(video, v_idx, title, description))
                curriculum_payload = [{
                    "title": "Lessons",
                    "isBonus": False,
                    "lessons": merged_lessons,
                }]
            else:
                # 1-to-1 mapping: rebuild sections preserving structure
                curriculum_payload = []
                video_iter = iter(enumerate(processed_lessons))
                for s_idx, sec in enumerate(llm_curriculum):
                    sec_payload = {
                        "title": sec.get("title", f"Section {s_idx + 1}"),
                        "isBonus": bool(sec.get("isBonus", False)),
                        "lessons": [],
                    }
                    for lesson in sec.get("lessons", []):
                        v_idx, video = next(video_iter)
                        sec_payload["lessons"].append(
                            _lesson_payload(video, v_idx, lesson["title"], lesson.get("description", ""))
                        )
                    curriculum_payload.append(sec_payload)
        else:
            # Fallback: single "Lessons" section using video titles
            curriculum_payload = [{
                "title": "Lessons",
                "isBonus": False,
                "lessons": [_lesson_payload(v, idx, v["title"], "") for idx, v in enumerate(processed_lessons)],
            }]

        # Sheet override: any non-empty cell on the lesson_idx=1 row for this
        # course is treated as an operator edit and wins over the pipeline.db
        # cached compose. Empty cell → fall back to the cached compose value,
        # then to a generic stub.
        first_lesson_row = next((r for r in lessons if r.get("lesson_idx") == 1),
                                lessons[0] if lessons else {})
        sheet_excerpt = (first_lesson_row.get("course_description") or "").strip()
        sheet_author_name = (first_lesson_row.get("author_name") or "").strip()
        sheet_author_bio = (first_lesson_row.get("author_bio") or "").strip()
        sheet_title = (first_lesson_row.get("course_title") or "").strip()
        sheet_about = (first_lesson_row.get("course_about") or "").strip()
        sheet_plan_text = (first_lesson_row.get("course_plan") or "").strip()
        sheet_science_text = (first_lesson_row.get("course_science") or "").strip()

        # Author / course fields with sheet-overrides + fallbacks
        if composed_full:
            author_payload = {
                "name": (sheet_author_name
                         or composed_full["author"].get("name")
                         or ch_name),
                "bio": (sheet_author_bio
                        or composed_full["author"].get("bio")
                        or ""),
            }
            course_payload = {
                "title": (sheet_title
                          or composed_full["course"]["title"]
                          or clean_title),
                "excerpt": (sheet_excerpt
                            or composed_full["course"].get("excerpt")
                            or ""),
                "aboutContent": (sheet_about
                                 or composed_full["course"]["aboutContent"]),
                "isAdult": composed_full["course"]["isAdult"],
            }
            # Plan / Science: prefer Sheet edits (parsed) over composed cache
            plan_sections = composed_full.get("planSections") or []
            if sheet_plan_text:
                try:
                    parsed_plan = llm.parse_plan_from_sheet(sheet_plan_text)
                    if parsed_plan:
                        plan_sections = parsed_plan
                except Exception as e:
                    log.warning(f"phase2: failed to parse sheet course_plan: {e}")
            science_plan = composed_full.get("sciencePlan")  # may be None
            if sheet_science_text:
                try:
                    parsed_sci = llm.parse_science_from_sheet(sheet_science_text)
                    if parsed_sci:
                        science_plan = parsed_sci
                except Exception as e:
                    log.warning(f"phase2: failed to parse sheet course_science: {e}")
            testimonials = composed_full.get("testimonials") or []
            collection_name = composed_full.get("collectionName") or None
        else:
            author_payload = {
                "name": sheet_author_name or ch_name,
                "bio": sheet_author_bio or f"{ch_name} — educator on YouTube.",
            }
            course_payload = {
                "title": sheet_title or clean_title,
                "excerpt": sheet_excerpt or f"A practical course on {clean_title}.",
                "aboutContent": (sheet_about
                                 or f"Curated lessons from {ch_name} on {clean_title}."),
                "isAdult": False,
            }
            # Even without composed_full, the operator can paste plan/science
            # into Sheet and we'll honor them.
            plan_sections, science_plan, testimonials, collection_name = [], None, [], None
            if sheet_plan_text:
                try:
                    plan_sections = llm.parse_plan_from_sheet(sheet_plan_text) or []
                except Exception as e:
                    log.warning(f"phase2: failed to parse sheet course_plan: {e}")
            if sheet_science_text:
                try:
                    science_plan = llm.parse_science_from_sheet(sheet_science_text)
                except Exception as e:
                    log.warning(f"phase2: failed to parse sheet course_science: {e}")

        # Sheet rows for this course (used for status updates)
        course_sheet_rows = [r["_sheet_row"] for r in lessons]

        # ── 4.5 Sanitize: force every user-facing field to Latin script ───
        # Defense in depth — even if compose/sheet/fallback let Cyrillic
        # through (stale Phase 1 cache, YouTube video titles used as fallback,
        # channel name fallback for author.name), translate any Cyrillic
        # string via Google Translate before posting to NMS. Names get
        # transliterated by Translate as a byproduct of EN→EN passthrough.
        try:
            translate_key = _google_translate_key()
        except Exception as e:
            log.warning(f"phase2: no Google Translate key for sanitization: {e}")
            translate_key = ""
        if translate_key:
            sanitize_report = _sanitize_payload_to_english(
                author_payload=author_payload,
                course_payload=course_payload,
                curriculum_payload=curriculum_payload,
                plan_sections=plan_sections,
                science_plan=science_plan,
                testimonials=testimonials,
                processed_lessons=processed_lessons,
                api_key=translate_key,
            )
            if sanitize_report:
                _send(token, chat_id,
                      f"🌐 Курс {course_idx}: автоперевод в EN {sanitize_report} "
                      f"полей с кириллицей перед отправкой в админку.")

        # ── 5. Push DRAFT course to NewMindStart ────────────────────────
        if nms_endpoint and nms_token:
            try:
                resp = nms_client.create_draft_course(
                    endpoint=nms_endpoint, token=nms_token,
                    author=author_payload,
                    course=course_payload,
                    curriculum=curriculum_payload,
                    plan_sections=plan_sections,
                    science_plan=science_plan,
                    testimonials=testimonials,
                    collection_name=collection_name,
                )
                course_results.append({
                    "course_idx": course_idx, "title": clean_title,
                    "admin_url": resp["adminUrl"], "course_id": resp["courseId"],
                })
                # Mark all videos in this course as DONE with admin URL
                sheets.update_status(
                    client, sheet_id,
                    sheet_rows=course_sheet_rows,
                    new_status=sheets.STATUS_DONE,
                    course_admin_url=resp["adminUrl"],
                )
                _send(token, chat_id,
                      f"✅ <b>Курс {course_idx} создан в админке (DRAFT):</b>\n"
                      f"<a href=\"{resp['adminUrl']}\">{_html_escape(course_payload['title'])}</a>\n\n"
                      f"План: {len(plan_sections)} | Science: {'есть' if science_plan else 'нет'} | "
                      f"Отзывы: {len(testimonials)} | Коллекция: {collection_name or '—'}")
            except Exception as e:
                log.error(f"phase2: NMS push failed for course {course_idx}: {e}", exc_info=True)
                # Mark videos failed (Bunny upload was OK but draft creation died)
                sheets.update_status(
                    client, sheet_id,
                    sheet_rows=course_sheet_rows,
                    new_status=sheets.STATUS_FAILED,
                    failure_reason=f"NMS push: {str(e)[:280]}",
                )
                _send(token, chat_id,
                      f"⚠️ Курс {course_idx}: видео загружены на Bunny, "
                      f"но создание DRAFT не удалось:\n<code>{_html_escape(str(e))[:300]}</code>")
        else:
            # No NMS configured — mark done but with no admin URL
            sheets.update_status(
                client, sheet_id,
                sheet_rows=course_sheet_rows,
                new_status=sheets.STATUS_DONE,
            )
            keys = "\n".join(f"  • {l['title'][:50]} → bunny:{l['videoKey']}"
                             for l in processed_lessons)
            _send(token, chat_id,
                  f"📦 <b>Курс {course_idx}</b>: видео залиты на Bunny library {bunny_lib}.\n"
                  f"<code>{_html_escape(keys)}</code>\n\n"
                  f"<i>NMS endpoint/token не настроен — DRAFT в админке не создан.</i>")

    # ── 6. Final summary ─────────────────────────────────────────────────
    _state.update(agent, user_id, thread_id=int(thread_id or 0),
                  step="done", courses=course_results)

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

# CookiesNeededError now lives in proxy_pool (re-exported at the top of this module).


def _process_course_videos(*, token: str, chat_id: int, agent: str, user_id: int,
                           lessons: list[dict[str, Any]], course_idx: int,
                           scratch_dir: Path, openai_key: str,
                           get_google_translate_key, get_google_tts_key,
                           voice_gender: str = "MALE",
                           bunny_lib: str, bunny_key: str,
                           course_topic: str,
                           youtube_cookies_file: str | None = None,
                           rotator=None,
                           parallel_videos: int = 1,
                           mark_cuts_word_level: bool = False,
                           blur_watermarks: bool = False) -> list[dict[str, Any]]:
    """Process every approved video in a course → list of NMS lesson payloads."""
    total = len(lessons)
    parallel = max(1, min(int(parallel_videos or 1), total))

    def _kwargs_for(i: int, lesson: dict[str, Any]) -> dict[str, Any]:
        return {
            "token": token, "chat_id": chat_id,
            "prefix": f"Курс {course_idx}, видео {i}/{total}",
            "lesson": lesson, "scratch_dir": scratch_dir,
            "openai_key": openai_key,
            "get_google_translate_key": get_google_translate_key,
            "get_google_tts_key": get_google_tts_key,
            "voice_gender": voice_gender,
            "bunny_lib": bunny_lib, "bunny_key": bunny_key,
            "course_topic": course_topic,
            "youtube_cookies_file": youtube_cookies_file,
            "rotator": rotator,
            "mark_cuts_word_level": mark_cuts_word_level,
            "blur_watermarks": blur_watermarks,
        }

    # ── Sequential path (original behavior, default). ───────────────────
    if parallel == 1:
        out: list[dict[str, Any]] = []
        for i, lesson in enumerate(lessons, start=1):
            try:
                out.append(_process_one_video(**_kwargs_for(i, lesson)))
            except CookiesNeededError:
                raise
            except Exception as e:
                log.error(f"phase2: video {lesson.get('video_id')} failed: {e}",
                          exc_info=True)
                _send(token, chat_id,
                      f"⚠️ Курс {course_idx}, видео {i}/{total}: "
                      f"<i>{_html_escape(lesson.get('title', '?'))[:50]}</i> — "
                      f"<code>{_html_escape(str(e))[:200]}</code>\n"
                      f"Видео пропущено, продолжаю с остальными.")
        return out

    # ── Parallel path. ──────────────────────────────────────────────────
    _send(token, chat_id,
          f"🔀 Курс {course_idx}: обрабатываю до {parallel} видео одновременно "
          f"(всего {total}).")

    cookies_needed: CookiesNeededError | None = None
    by_idx: dict[int, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures: dict[Any, tuple[int, dict[str, Any]]] = {}
        for i, lesson in enumerate(lessons, start=1):
            fut = ex.submit(_process_one_video, **_kwargs_for(i, lesson))
            futures[fut] = (i, lesson)

        for fut in as_completed(futures):
            i, lesson = futures[fut]
            try:
                by_idx[i] = fut.result()
            except CookiesNeededError as e:
                cookies_needed = e
                # Cancel anything that hasn't started yet — proxies are dead.
                for f in futures:
                    f.cancel()
                break
            except Exception as e:
                log.error(f"phase2: video {lesson.get('video_id')} failed: {e}",
                          exc_info=True)
                _send(token, chat_id,
                      f"⚠️ Курс {course_idx}, видео {i}/{total}: "
                      f"<i>{_html_escape(lesson.get('title', '?'))[:50]}</i> — "
                      f"<code>{_html_escape(str(e))[:200]}</code>\n"
                      f"Видео пропущено, продолжаю с остальными.")

    if cookies_needed is not None:
        raise cookies_needed

    return [by_idx[i] for i in sorted(by_idx.keys())]


def _is_bot_check_error(err: Exception) -> bool:
    s = str(err).lower()
    return "sign in to confirm" in s or "not a bot" in s


def _is_truncated_download_error(err: Exception) -> bool:
    """Match yt-dlp's mid-stream truncation errors so we can rotate proxy."""
    s = str(err).lower()
    return (
        "giving up after" in s
        or "bytes read" in s and "more expected" in s
        or "connection reset" in s
        or "incomplete read" in s
        or "remote end closed connection" in s
    )


MAX_TRUNCATED_PROXY_ROTATIONS = 5
"""How many fresh proxies to try when yt-dlp's internal retries are exhausted."""


def _download_with_rotation(*, token: str, chat_id: int, prefix: str,
                            url: str, output_path, cookies_file: str | None,
                            rotator) -> "Path":
    """Download with proxy rotation on bot-check OR truncation failures.

    - Bot-check ("Sign in to confirm"): rotate until pool exhausted, then raise
      CookiesNeededError (wizard prompts user to refresh cookies).
    - Truncation ("Giving up after N retries", "X bytes read, Y more expected"):
      rotate up to MAX_TRUNCATED_PROXY_ROTATIONS times; if all fail, give up on
      this video and let caller skip it (no cookies prompt — cookies are fine).
    - Anything else: surface immediately.
    """
    truncated_tries = 0
    while True:
        proxy = rotator.current if rotator else None
        try:
            return ffmpeg_cut.download_video(
                url=url, output_path=output_path,
                cookies_file=cookies_file, proxy=proxy,
            )
        except ffmpeg_cut.FFmpegError as e:
            if _is_bot_check_error(e) and rotator is not None:
                label = proxy_pool._proxy_label(proxy) if proxy else "no-proxy"
                _send(token, chat_id,
                      f"🔁 {prefix}: bot-check на <code>{label}</code>, ищу другой прокси…")
                new_proxy = rotator.rotate()
                if new_proxy is None:
                    raise CookiesNeededError(
                        f"Все {len(rotator.pool)} прокси из пула заблокированы "
                        f"YouTube'ом. Cookies скорее всего тоже устарели — "
                        f"нужно обновить и повторить."
                    ) from e
                _send(token, chat_id,
                      f"➡️ {prefix}: переключился на <code>{proxy_pool._proxy_label(new_proxy)}</code>, повторяю…")
                continue
            if _is_truncated_download_error(e) and rotator is not None:
                truncated_tries += 1
                if truncated_tries > MAX_TRUNCATED_PROXY_ROTATIONS:
                    _send(token, chat_id,
                          f"❌ {prefix}: видео не докачалось на "
                          f"{truncated_tries} разных прокси, пропускаю.")
                    raise
                label = proxy_pool._proxy_label(proxy) if proxy else "no-proxy"
                _send(token, chat_id,
                      f"⚠️ {prefix}: обрыв на <code>{label}</code> "
                      f"(попытка {truncated_tries}/{MAX_TRUNCATED_PROXY_ROTATIONS}), "
                      f"меняю прокси…")
                new_proxy = rotator.rotate_if_still(proxy)
                if new_proxy is None:
                    raise
                _send(token, chat_id,
                      f"➡️ {prefix}: переключился на "
                      f"<code>{proxy_pool._proxy_label(new_proxy)}</code>, повторяю…")
                continue
            raise


def _process_one_video(**kwargs) -> dict:
    """Public wrapper: enforce global slot budget then call impl."""
    with _global_throttle.acquire_video_slot():
        return _process_one_video_impl(**kwargs)


def _process_one_video_impl(*, token: str, chat_id: int, prefix: str,
                       lesson: dict[str, Any], scratch_dir: Path,
                       openai_key: str,
                       get_google_translate_key, get_google_tts_key,
                       voice_gender: str = "MALE",
                       bunny_lib: str, bunny_key: str,
                       course_topic: str,
                       youtube_cookies_file: str | None = None,
                       rotator=None,
                       mark_cuts_word_level: bool = False,
                       blur_watermarks: bool = False) -> dict[str, Any]:
    """Phase 2: download → cut → dub (Google TTS) → upload."""
    from .google_dub import _iso as _lang_iso

    video_id = lesson["video_id"]
    title = lesson["title"]
    url = lesson["url"]

    cleaned_path = scratch_dir / f"{video_id}.cleaned.mp4"
    final_path = scratch_dir / f"{video_id}.final.mp4"

    # ── 1. Source MP4: prefer cache from Phase 1, otherwise download ─────
    cached_path = cache.cached_path(video_id)
    if cache.is_cached(video_id):
        _send(token, chat_id,
              f"📦 {prefix}: использую кешированный MP4 из Phase 1 "
              f"<i>{_html_escape(title)[:50]}</i>")
        raw_path = cached_path
    else:
        _send(token, chat_id,
              f"⏳ {prefix}: кеша нет — скачиваю <i>{_html_escape(title)[:50]}</i>…")
        raw_path = _download_with_rotation(
            token=token, chat_id=chat_id, prefix=prefix,
            url=url, output_path=cached_path,
            cookies_file=youtube_cookies_file,
            rotator=rotator,
        )

    # ── 2. Cuts + detected language + segments: prefer Phase 1's pipeline_db ─
    db_record = pipeline_db.get_cuts(video_id)
    cached_segments: list[dict[str, Any]] = []
    if db_record is not None:
        cuts = db_record.get("cuts") or []
        detected_lang = db_record.get("detected_lang") or ""
        cached_segments = db_record.get("segments") or []
        seg_note = (f", {len(cached_segments)} сегментов" if cached_segments
                    else ", сегментов нет (старый run — повторим Whisper)")
        _send(token, chat_id,
              f"📋 {prefix}: cuts из Phase 1 ({len(cuts)} кусков, "
              f"lang={detected_lang or '?'}{seg_note})")
    else:
        # Fallback: video has no Phase 1 handoff — recompute on the fly.
        _send(token, chat_id,
              f"📝 {prefix}: нет данных в pipeline.db — транскрибирую и размечаю…")
        working = whisper.transcribe(api_key=openai_key, file_path=raw_path,
                                     with_word_timestamps=True)
        detected_lang = _lang_iso((working.get("language") or "").lower())
        cuts = llm.mark_cuts(course_topic=course_topic, transcript=working,
                             word_level=mark_cuts_word_level) or []
        cached_segments = working.get("segments") or []

    cuts_summary = f"{len(cuts)} кусков" if cuts else "нет вырезок"

    # ── 3. Cut ───────────────────────────────────────────────────────────
    _send(token, chat_id, f"✂️ {prefix}: вырезка ({cuts_summary})…")
    cleaned_path = ffmpeg_cut.cut_segments(input_path=raw_path, cuts=cuts,
                                           output_path=cleaned_path)

    # ── 3.5. Blur watermarks (opt-in via config.onboarder.blur_watermarks) ──
    # Detection: Claude Haiku vision on 3 downscaled frames (~3K tokens).
    # Blur: ffmpeg gblur over each detected bbox. Failure here is non-fatal —
    # we keep the un-blurred cleaned video and continue.
    if blur_watermarks:
        from . import watermark_blur
        blurred_path = scratch_dir / f"{video_id}.blurred.mp4"
        _send(token, chat_id, f"🕶️ {prefix}: ищу водяные знаки…")
        try:
            wm_report = watermark_blur.process(
                cleaned_path, blurred_path,
                n_frames=3, max_height=480, model="haiku",
            )
            found = wm_report.get("watermarks") or []
            if found:
                reasons = ", ".join(
                    (w.get("reason") or "watermark")[:60] for w in found[:3]
                )
                _send(token, chat_id,
                      f"🟦 {prefix}: замутил {len(found)} ватермарк(ов) — "
                      f"<i>{_html_escape(reasons)}</i>")
                cleaned_path = blurred_path
            else:
                _send(token, chat_id,
                      f"✨ {prefix}: водяных знаков не найдено")
                # blurred_path is just a copy of cleaned_path — discard
                try:
                    blurred_path.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception as e:
            log.warning(f"phase2: watermark_blur failed for {video_id}: {e}",
                        exc_info=True)
            _send(token, chat_id,
                  f"⚠️ {prefix}: blur ватермарок упал "
                  f"(<code>{_html_escape(str(e))[:120]}</code>) — "
                  f"продолжаю с оригинальным видео.")
            try:
                blurred_path.unlink(missing_ok=True)
            except Exception:
                pass

    # ── 4. Derive cleaned-video segments — reuse Phase 1 cache when possible ──
    # Old flow re-ran Whisper on the cleaned video (extra $0.006/min + minute(s)
    # of latency). New flow: take Phase 1's word-level segments and reproject
    # them through the cuts list — zero API calls, ~0.5s word-width precision.
    cleaned_segments: list[dict[str, Any]] = []
    if cached_segments:
        from . import segment_shift
        cleaned_segments = segment_shift.shift_segments_through_cuts(
            cached_segments, cuts,
        )
        if cleaned_segments:
            _send(token, chat_id,
                  f"⚡ {prefix}: сегменты получены из кэша Phase 1 "
                  f"({len(cleaned_segments)} шт, Whisper API пропущен)")
        else:
            log.warning(f"phase2: shift produced 0 segments for {video_id}, "
                        f"falling back to Whisper")
            cached_segments = []  # force fallback below

    # ── 5. Dub if not English (Google Translate + TTS) ───────────────────
    if detected_lang == "en":
        _send(token, chat_id, f"🇬🇧 {prefix}: уже на английском, дубляж пропускаем")
        final_path = cleaned_path
        was_dubbed = False
        # Final EN transcript: join shifted segments, or re-transcribe as fallback
        if cleaned_segments:
            from . import segment_shift
            final_transcript_text = segment_shift.join_transcript(cleaned_segments)
        else:
            _send(token, chat_id, f"📝 {prefix}: транскрибация (EN, fallback)…")
            final_transcript_result = whisper.transcribe(
                api_key=openai_key, file_path=cleaned_path,
                language="en", with_word_timestamps=False)
            final_transcript_text = final_transcript_result.get("text", "")
    else:
        voice_label = "👨 мужской" if voice_gender == "MALE" else "👩 женский"
        _send(token, chat_id,
              f"🇬🇧 {prefix}: дубляж {detected_lang or '?'} → en "
              f"(Google TTS, {voice_label})…")

        # Build the working_clean dict that google_dub expects. Prefer the
        # shifted-cache segments; fall back to a fresh Whisper call on the
        # cleaned video if there's no cached data (legacy run or shift failed).
        if cleaned_segments:
            working_clean = {
                "segments": cleaned_segments,
                "language": detected_lang,
            }
        else:
            _send(token, chat_id,
                  f"📝 {prefix}: сегменты не сохранены — повторяю Whisper (fallback)…")
            working_clean = whisper.transcribe(
                api_key=openai_key, file_path=cleaned_path,
                with_word_timestamps=True)

        translate_key = get_google_translate_key()
        tts_key = get_google_tts_key()
        _, final_transcript_text = google_dub.dub_video(
            transcript=working_clean,
            input_path=cleaned_path,
            output_path=final_path,
            source_lang=detected_lang or "ru",
            target_lang="en",
            voice_gender=voice_gender,
            translate_api_key=translate_key,
            tts_api_key=tts_key,
            on_progress=lambda msg: _send(token, chat_id, f"{prefix}: {msg}"),
        )
        was_dubbed = True

    final_transcript = {"text": final_transcript_text}

    # ── 6. Upload to Bunny ───────────────────────────────────────────────
    duration_sec = int(ffmpeg_cut.probe_duration(final_path))
    _send(token, chat_id, f"☁️ {prefix}: загрузка на Bunny ({duration_sec}s)…")
    bunny_meta = bunny.upload_video(
        library_id=bunny_lib, api_key=bunny_key,
        file_path=final_path, title=title,
    )

    # ── 7. Cleanup ───────────────────────────────────────────────────────
    # Always remove transient cleaned/final files. Cache MP4 only goes once
    # Bunny upload succeeds (which it has if we got here).
    for f in (cleaned_path, final_path):
        try:
            p = Path(f)
            if p.exists() and p != cached_path:
                p.unlink()
        except Exception as e:
            log.warning(f"cleanup: could not delete {f}: {e}")
    cache.delete_cached(video_id)

    _send(token, chat_id, f"✅ {prefix}: готово")

    return {
        "title": title,
        "videoKey": bunny_meta["videoKey"],
        "videoLibraryId": bunny_meta["videoLibraryId"],
        "duration": duration_sec,
        "transcriptEn": final_transcript.get("text", ""),
        "originalLang": detected_lang,
        "wasDubbed": was_dubbed,
        "lessonDescription": lesson.get("lesson_description", ""),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lesson_payload(video: dict[str, Any], order: int,
                    title: str, description: str) -> dict[str, Any]:
    """Build per-lesson payload dict for the NMS API from a processed video + LLM-given title/desc.

    Description preference (most-trusted first):
      1. Sheet's `lessonDescription` if non-empty — this is the operator's
         source-of-truth column. If they edit it, the edit ships.
      2. The `description` arg (from compose_full_course curriculum).
      3. First 500 chars of the final EN transcript as a last-resort stub.
    """
    transcript = video.get("transcriptEn") or ""
    sheet_desc = (video.get("lessonDescription") or "").strip()
    compose_desc = (description or "").strip()
    final_desc = sheet_desc or compose_desc or transcript[:500]
    return {
        "title": title or video.get("title", f"Lesson {order + 1}"),
        "order": order,
        "description": final_desc,
        "videoKey": video["videoKey"],
        "videoLibraryId": video["videoLibraryId"],
        "duration": video.get("duration"),
        "transcriptEn": transcript,
        "originalLang": video.get("originalLang", ""),
        "wasDubbed": bool(video.get("wasDubbed", False)),
    }


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
    """Send to Telegram, routing via _TLS.thread_id when in a forum topic.

    No-op when token is empty/falsy — used by complete_course.py and other
    CLI tools that drive Phase 2 without a Telegram channel. Skipping the
    HTTP call entirely saves ~3.7s per message (Telegram returns 404 on
    empty token with retry).
    """
    if not token:
        return
    from gateway import tg_api  # type: ignore
    thread_id = int(getattr(_TLS, "thread_id", 0) or 0)
    kwargs: dict[str, Any] = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    if thread_id:
        kwargs["message_thread_id"] = thread_id
    try:
        tg_api(token, "sendMessage", **kwargs)
    except Exception as e:
        log.warning(f"phase2 _send failed: {e}")


def _send_with_buttons(token: str, chat_id: int, text: str,
                       buttons: list[list[dict[str, str]]]) -> None:
    """Send inline-keyboard message into the active forum topic (if any)."""
    from gateway import send_message_with_buttons  # type: ignore
    thread_id = int(getattr(_TLS, "thread_id", 0) or 0)
    send_message_with_buttons(token, chat_id, text, buttons,
                              message_thread_id=thread_id)


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


# ---------------------------------------------------------------------------
# Final-pass sanitizer: guarantee NO Cyrillic in any user-facing field
# ---------------------------------------------------------------------------

def _has_cyrillic(s: str) -> bool:
    """True if string contains any Cyrillic character."""
    if not s:
        return False
    return any('Ѐ' <= ch <= 'ӿ' for ch in s)


def _sanitize_payload_to_english(*, author_payload: dict, course_payload: dict,
                                 curriculum_payload: list,
                                 plan_sections: list,
                                 science_plan: dict | None,
                                 testimonials: list,
                                 processed_lessons: list,
                                 api_key: str) -> str:
    """Walk the entire NMS payload, collect Cyrillic strings, batch-translate
    them all in one Google Translate API call, write back.

    Returns a short human-readable report (e.g. "12") for the operator message,
    or empty string if nothing needed translating.

    Mutates the dicts/lists in place.
    """
    from . import google_dub

    # Collect all (setter, original) pairs that have Cyrillic
    setters: list[tuple[callable, str]] = []

    def collect(value: str, setter: callable) -> None:
        if isinstance(value, str) and _has_cyrillic(value):
            setters.append((setter, value))

    # author
    collect(author_payload.get("name", ""),
            lambda v: author_payload.__setitem__("name", v))
    collect(author_payload.get("bio", ""),
            lambda v: author_payload.__setitem__("bio", v))

    # course
    for k in ("title", "excerpt", "aboutContent"):
        collect(course_payload.get(k, ""),
                (lambda key=k: lambda v: course_payload.__setitem__(key, v))())

    # curriculum: section titles + each lesson's title + description
    for sec in curriculum_payload or []:
        collect(sec.get("title", ""),
                (lambda s=sec: lambda v: s.__setitem__("title", v))())
        for lesson in sec.get("lessons", []) or []:
            collect(lesson.get("title", ""),
                    (lambda l=lesson: lambda v: l.__setitem__("title", v))())
            collect(lesson.get("description", ""),
                    (lambda l=lesson: lambda v: l.__setitem__("description", v))())
            # Per-lesson description used by NMS flat lessons path too
            collect(lesson.get("lessonDescription", ""),
                    (lambda l=lesson: lambda v: l.__setitem__("lessonDescription", v))())

    # plan_sections
    for sec in plan_sections or []:
        collect(sec.get("title", ""),
                (lambda s=sec: lambda v: s.__setitem__("title", v))())
        for item in sec.get("items", []) or []:
            collect(item.get("title", ""),
                    (lambda it=item: lambda v: it.__setitem__("title", v))())

    # science_plan
    if science_plan:
        for k in ("headline", "subtitle"):
            collect(science_plan.get(k, ""),
                    (lambda key=k: lambda v: science_plan.__setitem__(key, v))())
        for inst in science_plan.get("institutions", []) or []:
            collect(inst.get("name", ""),
                    (lambda i=inst: lambda v: i.__setitem__("name", v))())
        for stat in science_plan.get("stats", []) or []:
            for k in ("value", "description", "citation"):
                collect(stat.get(k, ""),
                        (lambda s=stat, key=k: lambda v: s.__setitem__(key, v))())

    # testimonials
    for t in testimonials or []:
        collect(t.get("authorName", ""),
                (lambda x=t: lambda v: x.__setitem__("authorName", v))())
        collect(t.get("text", ""),
                (lambda x=t: lambda v: x.__setitem__("text", v))())

    # processed_lessons: each lesson has lessonDescription that NMS reads
    for pl in processed_lessons or []:
        collect(pl.get("lessonDescription", ""),
                (lambda x=pl: lambda v: x.__setitem__("lessonDescription", v))())

    if not setters:
        return ""

    # Batch-translate all collected strings in ONE API call
    originals = [orig for _, orig in setters]
    try:
        translated = google_dub.translate_batch(
            originals, source_lang="ru", target_lang="en", api_key=api_key,
        )
    except Exception as e:
        log.warning(f"phase2 sanitize: translate_batch failed: {e}")
        return ""

    for (setter, _), new_val in zip(setters, translated):
        if new_val:
            setter(new_val)

    return str(len(setters))
