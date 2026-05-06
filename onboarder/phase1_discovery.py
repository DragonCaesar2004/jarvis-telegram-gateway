"""Phase 1: discovery + enrichment pipeline.

Run in a background thread. Posts progress to Telegram, writes results to Google Sheet,
updates wizard state. On completion, sends a message with the Sheet link + a "Запустить
обработку" inline button (callback_data="wiz:start_phase2").

Pipeline:
    1. Read criteria from Sheet
    2. yt-dlp search for topic → ~30 candidate videos → unique channels
    3. Top 3*N channels: fetch metadata, apply HARD filter (subs, video_count)
    4. Claude scores remaining channels for topical fit
    5. Take top N (one channel per course)
    6. For each channel: list recent videos (filtered by age), Claude picks 6-12
    7. ENRICH (per course): parallel download → Whisper → mark cuts → describe →
       research author (deep WebSearch) → compose full course payload. Cuts +
       working transcript persisted to pipeline.db. Composed payload also saved.
    8. Append fully-enriched rows to Lessons tab (real descriptions, ready to
       review on a real landing page).
    9. Telegram message with Sheet link + "Запустить обработку" button

Errors at any step → Telegram message + wizard state set to "error". User can /cancel
or click "Запустить обработку" anyway (will fail at Phase 2 if data is missing).
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import Any

from . import (_secrets, llm, phase1_enrich, proxy_pool, sheets,
               state as _state, youtube_dl as ytdl)
from .proxy_pool import CookiesNeededError

log = logging.getLogger("gateway")

# Tuning knobs (some now overridable via Criteria)
DEFAULT_SEARCH_RESULTS = 50   # criteria.search_results overrides
TARGET_PASSING_PER_COURSE = 4  # try to get this many passing channels per course
HARD_CAP_CHANNELS_TO_CHECK = 50  # absolute ceiling on metadata fetches
MIN_LLM_SCORE = 0.5
CHANNEL_VIDEOS_TO_LIST = 50
PROGRESS_INTERVAL_SEC = 30  # don't spam Telegram


# ---------------------------------------------------------------------------
# Public entry point (called from wizard's wiz:start_phase1 callback)
# ---------------------------------------------------------------------------

def launch(token: str, agent: str, cfg: dict, chat_id: int, user_id: int,
           topic: str, count: int) -> None:
    """Spawn the Phase 1 worker in a background daemon thread."""
    thr = threading.Thread(
        target=_worker,
        args=(token, agent, cfg, chat_id, user_id, topic, count),
        name=f"phase1-{agent}-{user_id}",
        daemon=True,
    )
    thr.start()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(token: str, agent: str, cfg: dict, chat_id: int, user_id: int,
            topic: str, count: int) -> None:
    onb = (cfg.get("onboarder") or {})
    try:
        _run(token, agent, cfg, chat_id, user_id, topic, count, onb)
    except CookiesNeededError as e:
        log.warning(f"phase1: cookies needed: {e}")
        _state.update(agent, user_id, step="error", error=f"cookies_needed: {e}")
        _send_with_buttons(
            token, chat_id,
            text=(
                "⏸ <b>Phase 1 на паузе.</b>\n\n"
                f"{_html_escape(str(e))[:500]}\n\n"
                "Загрузи свежий cookies.txt с youtube.com, потом /menu → 🎓 Новый курс для перезапуска."
            ),
            buttons=[[
                {"text": "📎 Загрузить cookies", "callback_data": "menu:cookies"},
                {"text": "✖️ Отмена", "callback_data": "wiz:cancel"},
            ]],
        )
    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"phase1 worker crashed: {e}\n{tb}")
        _state.update(agent, user_id, step="error", error=str(e))
        _send(token, chat_id,
              f"⚠️ <b>Phase 1 упал.</b>\n\n<code>{_html_escape(str(e))[:500]}</code>\n\n"
              "Используй /cancel для возврата в чат или /menu → 🎓 Новый курс для повтора.")


def _run(token: str, agent: str, cfg: dict, chat_id: int, user_id: int,
         topic: str, count: int, onb: dict) -> None:
    # ── 1. Resolve secrets and open Sheet ────────────────────────────────
    # Anthropic API key not needed: llm.py uses `claude -p` CLI via Max OAuth.
    sa_path = _secrets.resolve_path(onb, "google_service_account")
    sheet_id = onb.get("google_sheet_id") or ""
    if not sheet_id:
        raise RuntimeError("config: onboarder.google_sheet_id not set")

    # Phase 1 now downloads + transcribes + composes — needs Whisper key,
    # cookies for YouTube downloads, and the proxy pool for rotating around
    # bot-checks. Resolve early so we fail fast on misconfiguration.
    openai_key = _secrets.resolve(onb, "openai_api_key", env="OPENAI_API_KEY")
    youtube_cookies_file = onb.get("youtube_cookies_file") or None
    proxy_pool_list = proxy_pool.normalise_pool(
        onb.get("youtube_proxies") or onb.get("youtube_proxy")
    )
    parallel_per_course = int(onb.get("phase1_parallel_per_course") or 4)
    compose_model = (onb.get("models") or {}).get("compose") or llm.DEFAULT_MODEL_QUALITY

    client = sheets.open_client(sa_path)
    criteria = sheets.read_criteria(client, sheet_id)
    log.info(f"phase1[{user_id}] criteria: {criteria}")

    # ── 2. Ensure unified Lessons tab + collect every previously-seen video_id
    sheets.ensure_lessons_tab(client, sheet_id)
    active_video_ids = sheets.get_active_video_ids(client, sheet_id)
    log.info(f"phase1[{user_id}] {len(active_video_ids)} videos previously seen in Sheet — "
             f"deduping these (any status, including rejected/failed)")

    run_id = sheets.make_run_id()
    _state.update(agent, user_id, run_id=run_id,
                  step="phase1_running",
                  topic=topic, count=count,
                  sheet_url=sheets.sheet_url(sheet_id))

    # ── 2.5 Init proxy rotator for downloads (same probe UX as Phase 2) ──
    rotator = proxy_pool.ProxyRotator(proxy_pool_list, cookies_file=youtube_cookies_file)
    if proxy_pool_list:
        _send(token, chat_id,
              f"🔍 Проверяю {len(proxy_pool_list)} прокси на YouTube…")
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
                  f"✅ Прокси готов: <code>{proxy_pool._proxy_label(rotator.current)}</code>")
        except proxy_pool.NoWorkingProxyError as e:
            raise RuntimeError(
                f"Ни один прокси не прошёл проверку YouTube.\n\n{str(e)[:600]}\n\n"
                "Обнови cookies (/menu → 📎 Загрузить cookies) и попробуй ещё раз."
            )
    else:
        _send(token, chat_id, "⚠️ Прокси не настроен — пробую напрямую с VPS-IP")

    # ── 3. yt-dlp search → unique channels ───────────────────────────────
    search_results = _criteria_int(criteria, "search_results", DEFAULT_SEARCH_RESULTS)
    _send(token, chat_id, f"🔎 <i>Ищу каналы по теме «{_html_escape(topic)}»…</i>")
    videos = ytdl.search_videos(topic, max_results=search_results)
    if not videos:
        raise RuntimeError(f"yt-dlp search returned 0 results for '{topic}'")
    candidates = ytdl.unique_channels_from_search(videos)
    log.info(f"phase1[{user_id}] {len(candidates)} unique channel candidates from {len(videos)} videos")

    # ── 4. Greedy filter: scan candidates until we have enough passing ───
    target_passing = max(count * TARGET_PASSING_PER_COURSE, count + 3)
    cap = min(len(candidates), HARD_CAP_CHANNELS_TO_CHECK)
    _send(token, chat_id,
          f"📊 Найдено {len(candidates)} каналов в выдаче. "
          f"Проверяю метаданные (цель: {target_passing} прошедших фильтр, лимит: {cap})…")

    enriched: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    last_progress = time.time()
    checked = 0
    for ch in candidates[:cap]:
        meta = ytdl.get_channel_metadata(ch["channel_id"])
        checked += 1
        if not meta:
            continue
        if not _passes_hard_filter(meta, criteria):
            log.info(f"phase1[{user_id}] filter out: {meta['channel_name']} "
                     f"(subs={meta['subscribers']}, videos={meta['video_count']})")
            rejected.append(meta)
            continue
        enriched.append({**meta, "votes": ch.get("votes", 0),
                         "sample_titles": ch.get("sample_titles", [])})
        if time.time() - last_progress > PROGRESS_INTERVAL_SEC:
            _send(token, chat_id,
                  f"… проверено {checked}/{cap}, прошло фильтр: {len(enriched)}")
            last_progress = time.time()
        if len(enriched) >= target_passing:
            log.info(f"phase1[{user_id}] reached target {target_passing} passing channels, stop scanning")
            break

    if not enriched:
        diag = ""
        if rejected:
            sample = sorted(rejected, key=lambda m: m.get("subscribers", 0))[:5]
            lines = [f"  • {m.get('channel_name', '?')[:40]}: "
                     f"{m.get('subscribers', 0):,} subs, "
                     f"{m.get('video_count', 0)} videos"
                     for m in sample]
            diag = (f"\n\nПроверено {checked} каналов из {len(candidates)} найденных. "
                    f"Самые мелкие из отклонённых:\n" + "\n".join(lines))
        raise RuntimeError(
            f"Ни один канал не прошёл фильтр "
            f"(subs {criteria.get('min_subscribers')}-{criteria.get('max_subscribers')}, "
            f"videos {criteria.get('min_videos_on_channel')}-{criteria.get('max_videos_on_channel')})."
            f"{diag}\n\n"
            f"Tip: бамп <code>search_results</code> в Criteria до 100-200, чтобы yt-dlp "
            f"копал глубже в выдачу — мелкие каналы там обычно дальше топа."
        )

    # ── 5. Claude scores remaining channels ──────────────────────────────
    _send(token, chat_id, f"🤖 Оцениваю {len(enriched)} каналов через Claude…")
    scored = llm.score_channels(topic=topic, criteria=criteria,
                                channels=enriched)
    # Merge score into enriched lookup
    score_by_id = {s["channel_id"]: s for s in scored}
    enriched.sort(key=lambda c: score_by_id.get(c["channel_id"], {}).get("score", 0),
                  reverse=True)
    top_channels = [c for c in enriched
                    if score_by_id.get(c["channel_id"], {}).get("score", 0) >= MIN_LLM_SCORE]
    if len(top_channels) < count:
        # Fall back to best available even if below threshold
        top_channels = enriched[:count]
    top_channels = top_channels[:count]

    # ── 6. Per-channel: list videos + Claude select + dedup ─────────────
    all_lesson_rows: list[dict[str, Any]] = []
    course_summaries: list[str] = []
    skipped_total = 0
    for course_idx, ch in enumerate(top_channels, start=1):
        ch_name = ch.get("channel_name") or ch["channel_id"]
        _send(token, chat_id,
              f"🎬 Курс {course_idx}/{count} — канал «{_html_escape(ch_name)}»: "
              f"тяну видео и отбираю…")

        all_videos = ytdl.list_channel_videos(
            ch["channel_id"],
            max_results=CHANNEL_VIDEOS_TO_LIST,
            max_age_months=criteria.get("max_video_age_months"),
        )
        videos_for_llm = [_compact_video_for_llm(v) for v in all_videos]
        if not videos_for_llm:
            log.warning(f"phase1[{user_id}] channel {ch_name} has no videos in age window")
            continue

        try:
            sel = llm.select_videos(topic=topic, criteria=criteria,
                                    channel_name=ch_name, videos=videos_for_llm)
        except Exception as e:
            log.warning(f"phase1[{user_id}] select_videos failed for {ch_name}: {e}")
            continue
        course_title = sel.get("course_title")
        lessons = sel.get("lessons") or []
        if not course_title or not lessons:
            log.info(f"phase1[{user_id}] {ch_name} skipped: {sel.get('skip_reason')}")
            continue

        # Dedup: drop any lesson whose video_id is already done/processing in any run
        dedup_lessons = []
        course_skipped = 0
        for lsn in lessons:
            vid = lsn.get("video_id")
            if vid and vid in active_video_ids:
                course_skipped += 1
                log.info(f"phase1[{user_id}] dedup skip {vid} in {ch_name}")
                continue
            dedup_lessons.append(lsn)
        skipped_total += course_skipped

        if not dedup_lessons:
            _send(token, chat_id,
                  f"⚠️ Курс {course_idx} ({_html_escape(ch_name)}) пропущен — "
                  f"все {len(lessons)} видео уже обрабатывались.")
            continue

        # ── 6a. ENRICH: download + transcribe + cuts + describe + compose ──
        videos_metadata = {v["video_id"]: v for v in all_videos}
        try:
            enriched = phase1_enrich.enrich_course(
                course_idx=course_idx, run_id=run_id,
                channel_id=ch["channel_id"],
                channel_name=ch_name,
                channel_description=ch.get("description", ""),
                course_topic_input=topic,
                course_title_from_llm=course_title,
                selected_videos=dedup_lessons,
                videos_metadata=videos_metadata,
                openai_key=openai_key,
                cookies_file=youtube_cookies_file,
                rotator=rotator,
                on_progress=lambda msg: _send(token, chat_id, _html_escape(msg)),
                max_parallel=parallel_per_course,
                compose_model=compose_model,
            )
        except CookiesNeededError:
            raise  # propagate to _worker for graceful pause
        except Exception as e:
            log.warning(f"phase1[{user_id}] enrich failed for course {course_idx}: {e}",
                        exc_info=True)
            _send(token, chat_id,
                  f"⚠️ Курс {course_idx} ({_html_escape(ch_name)}): обогащение упало "
                  f"(<code>{_html_escape(str(e))[:160]}</code>). Пропускаю курс.")
            continue

        if not enriched.get("videos"):
            _send(token, chat_id,
                  f"⚠️ Курс {course_idx}: ни одно видео не довелось до конца. Пропускаю.")
            continue

        # ── 6b. Build enriched lesson rows for the Sheet ────────────────────
        final_course_title = enriched.get("course_title") or course_title
        full_course_title = f"Курс {course_idx}: {ch_name} — {final_course_title}"
        summary_extra = f" (пропущено {course_skipped} дублей)" if course_skipped else ""
        course_summaries.append(
            f"{course_idx}. {final_course_title} ({len(enriched['videos'])} уроков){summary_extra}"
        )

        course_desc = enriched.get("course_description", "")
        course_tagline = enriched.get("course_tagline", "")
        course_what = enriched.get("course_what_you_learn", "")
        course_target = enriched.get("course_target_audience", "")
        author_name = enriched.get("author_name", "") or ch_name
        author_bio = enriched.get("author_bio", "")
        author_expertise = enriched.get("author_expertise", "")

        for v in enriched["videos"]:
            is_first = (v["lesson_idx"] == 1)
            all_lesson_rows.append({
                "course": full_course_title if is_first else f"Курс {course_idx}",
                "lesson_idx": v["lesson_idx"],
                "channel": ch_name,
                "lesson_title": v["title"],
                "url": v["url"],
                "duration_sec": v.get("duration_sec", 0),
                "video_id": v["video_id"],
                "channel_id": ch["channel_id"],
                "course_idx": course_idx,
                "lesson_description": v.get("lesson_description", ""),
                "transcript_excerpt": v.get("transcript_excerpt", ""),
                # Course-level fields filled only on the first lesson row to
                # avoid blasting the same paragraph across N rows in the Sheet.
                "course_description": course_desc if is_first else "",
                "course_tagline": course_tagline if is_first else "",
                "course_what_you_learn": course_what if is_first else "",
                "course_target_audience": course_target if is_first else "",
                "author_name": author_name if is_first else "",
                "author_bio": author_bio if is_first else "",
                "author_expertise": author_expertise if is_first else "",
            })

    if not all_lesson_rows:
        raise RuntimeError("Ни один канал не дал валидной подборки уроков "
                           "(или все видео уже были обработаны раньше). "
                           "Попробуй другую тему или расширь критерии.")

    # ── 7. Append to unified Lessons tab ──────────────────────────────────
    sheets.append_lesson_rows(client, sheet_id, run_id=run_id, rows=all_lesson_rows)
    _state.update(agent, user_id, step="awaiting_approval",
                  courses_summary=course_summaries)

    # ── 8. Final Telegram message with action button ─────────────────────
    sheet_url = sheets.sheet_url(sheet_id)
    summary_lines = "\n".join(course_summaries) if course_summaries else "(нет курсов)"
    dedup_note = f"\n\n♻️ Пропущено как дубли: <b>{skipped_total}</b> видео." if skipped_total else ""
    _send_with_buttons(
        token, chat_id,
        text=(
            f"✅ <b>Phase 1 готов.</b>\n\n"
            f"Подборка ({len(all_lesson_rows)} видео в {len(course_summaries)} курсах):\n"
            f"{_html_escape(summary_lines)}{dedup_note}\n\n"
            f"📋 Открой таб <b>Lessons</b>, отметь <b>Approved=TRUE</b> "
            f"у нужных строк (status=pending). Все запуски в одной таблице:\n"
            f"{sheet_url}\n\n"
            f"Когда готов — нажми кнопку:"
        ),
        buttons=[[{"text": "🚀 Запустить обработку", "callback_data": "wiz:start_phase2"},
                  {"text": "✖️ Отмена", "callback_data": "wiz:cancel"}]],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _criteria_int(criteria: dict[str, Any], key: str, default: int = 0) -> int:
    """Read an integer-valued criterion, falling back to `default` if missing/blank/invalid."""
    v = criteria.get(key)
    if v in (None, ""):
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _passes_hard_filter(meta: dict[str, Any], criteria: dict[str, Any]) -> bool:
    """Apply min/max subs and video_count gates.

    Rules:
    - 0/empty in Criteria  → that gate is skipped (no limit).
    - 0 subscribers from yt-dlp → skip the subs gates (data missing, don't reject).
    - -1 video_count from yt-dlp → "unknown", skip video_count gates.
    """
    subs = int(meta.get("subscribers") or 0)
    vids = int(meta.get("video_count") or 0)
    vids_known = vids >= 0

    def _limit(key: str) -> int:
        v = criteria.get(key)
        try:
            return int(v) if v else 0
        except (TypeError, ValueError):
            return 0

    min_subs = _limit("min_subscribers")
    max_subs = _limit("max_subscribers")
    min_vids = _limit("min_videos_on_channel")
    max_vids = _limit("max_videos_on_channel")

    if subs > 0:
        if min_subs and subs < min_subs:
            return False
        if max_subs and subs > max_subs:
            return False
    if vids_known and vids > 0:
        if min_vids and vids < min_vids:
            return False
        if max_vids and vids > max_vids:
            return False
    return True


def _compact_video_for_llm(v: dict[str, Any]) -> dict[str, Any]:
    """Strip fields that don't help the LLM but cost tokens."""
    return {
        "video_id": v.get("video_id"),
        "title": v.get("title", ""),
        "duration_sec": v.get("duration_sec", 0),
        "view_count": v.get("view_count", 0),
        "upload_date": v.get("upload_date", ""),
    }


def _send(token: str, chat_id: int, text: str) -> None:
    """Use gateway's tg_api lazily so we honor its retries / chunking conventions."""
    from gateway import tg_api  # type: ignore
    try:
        tg_api(token, "sendMessage", chat_id=chat_id, text=text, parse_mode="HTML")
    except Exception as e:
        log.warning(f"phase1 _send failed: {e}")


def _send_with_buttons(token: str, chat_id: int, text: str,
                       buttons: list[list[dict[str, str]]]) -> None:
    from gateway import send_message_with_buttons  # type: ignore
    send_message_with_buttons(token, chat_id, text, buttons)


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )
