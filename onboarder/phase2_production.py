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
               proxy_pool, sheets, state as _state, whisper)

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
    except CookiesNeededError as e:
        # Pool exhausted — pause Phase 2 and ask user to upload fresh cookies
        log.warning(f"phase2: cookies needed: {e}")
        _state.update(agent, user_id, step="awaiting_cookies_pre_phase2")
        try:
            from gateway import set_user_mode, MODE_WIZARD  # type: ignore
            set_user_mode(agent, user_id, MODE_WIZARD)
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
    proxy_pool_list = proxy_pool.normalise_pool(
        onb.get("youtube_proxies") or onb.get("youtube_proxy")
    )
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

    # ── 2. Read approved+pending rows from unified Lessons tab ──────────
    st = _state.load(agent, user_id)
    run_id = st.get("run_id")
    # run_id is optional now — Phase 2 picks up any approved+pending rows
    # regardless of run, since user can mix-and-match across runs in the
    # single tab. If run_id is set, we filter to that run for safety.

    client = sheets.open_client(sa_path)
    approved = sheets.read_pending_approved_rows(client, sheet_id, run_id=run_id)
    if not approved:
        raise RuntimeError(
            "В табе Lessons нет строк со status=pending и approved=TRUE"
            + (f" для run_id={run_id}" if run_id else "")
            + ". Открой Sheet, отметь Approved=TRUE и нажми кнопку ещё раз."
        )

    # Mark all selected rows as processing IMMEDIATELY — so a parallel run
    # or a manual click doesn't try to grab them again.
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
    _send(token, chat_id,
          f"🎬 <b>Phase 2 запущен.</b>\n\n"
          f"Курсов: <b>{len(courses)}</b>\n"
          f"Видео: <b>{total_videos}</b>\n"
          f"Все строки помечены status=processing — параллельные запуски не "
          f"подхватят их повторно.\n"
          f"~1-3 часа на курс.")
    _state.update(agent, user_id, step="phase2_running")

    # ── 2.5 Initialize proxy rotator — pick first working proxy ─────────
    rotator = proxy_pool.ProxyRotator(proxy_pool_list, cookies_file=youtube_cookies_file)
    if proxy_pool_list:
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
                openai_key=openai_key, get_elevenlabs_key=_elevenlabs_key,
                bunny_lib=bunny_lib, bunny_key=bunny_key,
                course_topic=clean_title,
                youtube_cookies_file=youtube_cookies_file,
                rotator=rotator,
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

        # ── 4. Compose full course (template-based) via Claude ──────────
        _send(token, chat_id, f"✍️ Курс {course_idx}: пишу описание, план, science, отзывы через Claude…")
        composed_full: dict[str, Any] | None = None
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
            log.warning(f"phase2: compose_full_course failed: {e}; falling back to minimal compose")
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

        # Author / course fields with fallbacks
        if composed_full:
            author_payload = {
                "name": composed_full["author"]["name"] or ch_name,
                "bio": composed_full["author"]["bio"],
            }
            course_payload = {
                "title": composed_full["course"]["title"] or clean_title,
                "excerpt": composed_full["course"]["excerpt"],
                "aboutContent": composed_full["course"]["aboutContent"],
                "isAdult": composed_full["course"]["isAdult"],
            }
            plan_sections = composed_full.get("planSections") or []
            science_plan = composed_full.get("sciencePlan")  # may be None
            testimonials = composed_full.get("testimonials") or []
            collection_name = composed_full.get("collectionName") or None
        else:
            author_payload = {"name": ch_name, "bio": f"{ch_name} — educator on YouTube."}
            course_payload = {
                "title": clean_title,
                "excerpt": f"A practical course on {clean_title}.",
                "aboutContent": f"Curated lessons from {ch_name} on {clean_title}.",
                "isAdult": False,
            }
            plan_sections, science_plan, testimonials, collection_name = [], None, [], None

        # Sheet rows for this course (used for status updates)
        course_sheet_rows = [r["_sheet_row"] for r in lessons]

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
    _state.update(agent, user_id, step="done", courses=course_results)

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

class CookiesNeededError(RuntimeError):
    """All proxies in pool are blocked by YouTube — cookies refresh required."""


def _process_course_videos(*, token: str, chat_id: int, agent: str, user_id: int,
                           lessons: list[dict[str, Any]], course_idx: int,
                           scratch_dir: Path, openai_key: str,
                           get_elevenlabs_key, bunny_lib: str, bunny_key: str,
                           course_topic: str,
                           youtube_cookies_file: str | None = None,
                           rotator=None) -> list[dict[str, Any]]:
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
                rotator=rotator,
            )

            out.append(row)
        except CookiesNeededError:
            # Pool exhausted — rethrow so the worker pauses Phase 2
            raise
        except Exception as e:
            log.error(f"phase2: video {lesson.get('video_id')} failed: {e}", exc_info=True)
            _send(token, chat_id,
                  f"⚠️ {prefix}: <i>{_html_escape(lesson.get('title', '?'))[:50]}</i> — "
                  f"<code>{_html_escape(str(e))[:200]}</code>\n"
                  f"Видео пропущено, продолжаю с остальными.")
    return out


def _is_bot_check_error(err: Exception) -> bool:
    s = str(err).lower()
    return "sign in to confirm" in s or "not a bot" in s


def _download_with_rotation(*, token: str, chat_id: int, prefix: str,
                            url: str, output_path, cookies_file: str | None,
                            rotator) -> "Path":
    """Download with proxy rotation on bot-check failures.

    Tries current rotator.current; on bot-check, rotates and retries up to len(pool).
    Raises CookiesNeededError if pool is exhausted.
    """
    last_err: Exception | None = None
    while True:
        proxy = rotator.current if rotator else None
        try:
            return ffmpeg_cut.download_video(
                url=url, output_path=output_path,
                cookies_file=cookies_file, proxy=proxy,
            )
        except ffmpeg_cut.FFmpegError as e:
            last_err = e
            if not _is_bot_check_error(e) or rotator is None:
                raise
            label = proxy_pool._proxy_label(proxy) if proxy else "no-proxy"
            _send(token, chat_id,
                  f"🔁 {prefix}: bot-check на <code>{label}</code>, ищу другой прокси…")
            new_proxy = rotator.rotate()
            if new_proxy is None:
                raise CookiesNeededError(
                    f"Все {len(rotator.pool)} прокси из пула заблокированы YouTube'ом. "
                    f"Cookies скорее всего тоже устарели — нужно обновить и повторить."
                ) from e
            _send(token, chat_id,
                  f"➡️ {prefix}: переключился на <code>{proxy_pool._proxy_label(new_proxy)}</code>, повторяю…")


def _process_one_video(*, token: str, chat_id: int, prefix: str,
                       lesson: dict[str, Any], scratch_dir: Path,
                       openai_key: str, get_elevenlabs_key,
                       bunny_lib: str, bunny_key: str,
                       course_topic: str,
                       youtube_cookies_file: str | None = None,
                       rotator=None) -> dict[str, Any]:
    video_id = lesson["video_id"]
    title = lesson["title"]
    url = lesson["url"]

    raw_path = scratch_dir / f"{video_id}.raw.mp4"
    cleaned_path = scratch_dir / f"{video_id}.cleaned.mp4"
    final_path = scratch_dir / f"{video_id}.final.mp4"

    # 1. Download (with proxy rotation on bot-check)
    _send(token, chat_id, f"⏳ {prefix}: скачивание <i>{_html_escape(title)[:50]}</i>…")
    raw_path = _download_with_rotation(
        token=token, chat_id=chat_id, prefix=prefix,
        url=url, output_path=raw_path,
        cookies_file=youtube_cookies_file,
        rotator=rotator,
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

    # 8. Clean up local video files immediately — don't wait for course cleanup
    for f in (raw_path, cleaned_path, final_path):
        try:
            p = Path(f)
            if p.exists():
                p.unlink()
        except Exception as e:
            log.warning(f"cleanup: could not delete {f}: {e}")

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

def _lesson_payload(video: dict[str, Any], order: int,
                    title: str, description: str) -> dict[str, Any]:
    """Build per-lesson payload dict for the NMS API from a processed video + LLM-given title/desc."""
    transcript = video.get("transcriptEn") or ""
    return {
        "title": title or video.get("title", f"Lesson {order + 1}"),
        "order": order,
        "description": description or transcript[:500],
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
