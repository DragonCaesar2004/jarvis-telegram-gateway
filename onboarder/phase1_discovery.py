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
    6. For each channel: list ALL videos within max_age_months window
       (flat metadata, no per-video fetches), Claude picks 5-30
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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import Any

from . import (_secrets, llm, phase1_enrich, proxy_pool, sheets,
               state as _state, youtube_dl as ytdl)
from .proxy_pool import CookiesNeededError

log = logging.getLogger("gateway")

# Tuning knobs (some now overridable via Criteria)
DEFAULT_SEARCH_RESULTS = 50   # criteria.search_results overrides
DEFAULT_SEARCH_QUERIES = 6    # criteria.search_queries overrides — number of query variations
TARGET_PASSING_PER_COURSE = 4  # legacy soft target; no longer used as early-exit
HARD_CAP_CHANNELS_TO_CHECK = 100  # absolute ceiling on metadata fetches
MIN_LLM_SCORE = 0.4  # lowered from 0.5: more channels pass scoring → more
                     # candidates for the per-channel loop, less chance of
                     # Phase 1 failing because the top-3 didn't have on-topic
                     # content. Threshold-failed channels still fall back via
                     # the unconditional ranked list further down anyway.
CHANNEL_VIDEOS_TO_LIST = None  # None = fetch the whole channel catalog (subject
                               # to youtube_dl.list_channel_videos safety cap),
                               # then filter by max_age_months in Criteria.
                               # Flat metadata is cheap; bigger pool = better
                               # Claude selection.
PROGRESS_INTERVAL_SEC = 30  # don't spam Telegram
METADATA_BATCH_SIZE = 8     # how many channels to probe in parallel; tune via
                            # onboarder.phase1_metadata_workers in config.json


# ---------------------------------------------------------------------------
# Public entry point (called from wizard's wiz:start_phase1 callback)
# ---------------------------------------------------------------------------

# Thread-local context for the running worker — lets _send pick up the right
# Telegram forum topic without updating every call site in this module.
_TLS = threading.local()


def launch(token: str, agent: str, cfg: dict, chat_id: int, user_id: int,
           topic: str, count: int,
           *, pain: str = "", audience: str = "",
           thread_id: int = 0) -> None:
    """Spawn the Phase 1 worker in a background daemon thread.

    `pain` and `audience` are optional pain-point + target-audience strings
    captured by the wizard. When empty, scoring/selection/composition fall
    back to topic-only behavior.

    `thread_id` (forum topic id) is preserved through to all status messages
    so the run posts back into the topic where the operator started it.
    """
    thr = threading.Thread(
        target=_worker,
        args=(token, agent, cfg, chat_id, user_id, topic, count,
              pain, audience, int(thread_id or 0)),
        name=f"phase1-{agent}-{user_id}-{int(thread_id or 0)}",
        daemon=True,
    )
    thr.start()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(token: str, agent: str, cfg: dict, chat_id: int, user_id: int,
            topic: str, count: int, pain: str = "", audience: str = "",
            thread_id: int = 0) -> None:
    onb = (cfg.get("onboarder") or {})
    # Stash thread_id so _send / _send_with_buttons (defined below) route to
    # the right forum topic without every call site needing the kwarg.
    _TLS.thread_id = int(thread_id or 0)
    try:
        _run(token, agent, cfg, chat_id, user_id, topic, count, onb,
             pain=pain, audience=audience, thread_id=int(thread_id or 0))
    except CookiesNeededError as e:
        log.warning(f"phase1: cookies needed: {e}")
        _state.update(agent, user_id, thread_id=int(thread_id or 0),
                      step="error", error=f"cookies_needed: {e}")
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
        _state.update(agent, user_id, thread_id=int(thread_id or 0),
                      step="error", error=str(e))
        _send(token, chat_id,
              f"⚠️ <b>Phase 1 упал.</b>\n\n<code>{_html_escape(str(e))[:500]}</code>\n\n"
              "Используй /cancel для возврата в чат или /menu → 🎓 Новый курс для повтора.")
    finally:
        _TLS.thread_id = 0


def _run(token: str, agent: str, cfg: dict, chat_id: int, user_id: int,
         topic: str, count: int, onb: dict,
         *, pain: str = "", audience: str = "",
         thread_id: int = 0,
         batch_run_id: str | None = None,
         batch_course_idx_offset: int = 0,
         batch_silent_finish: bool = False) -> int:
    """Run topic-mode Phase 1 for ONE topic.

    Standalone use returns 0 (caller doesn't need a value).

    Batch use (when batch_run_id is provided):
      - Reuses the given run_id instead of generating a new one
      - course_idx in Sheet rows is offset by batch_course_idx_offset
        (so e.g. topic #3 in a batch starts at course_idx=3, not 1)
      - If batch_silent_finish=True, skips the final "Phase 1 готов" button
        and the wizard state→awaiting_approval update — the batch caller
        does that once after all topics finish
      - Returns the number of courses (rows-with-lesson-idx-1) actually
        written to Sheet for this topic, so the caller can advance the
        course_idx offset for the next topic in the batch
    """
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
    streaming_enabled = bool(onb.get("streaming_sheet_writes", False))
    # Opt-in: send full Whisper word-level transcript to mark_cuts (~10× input
    # token cost). Default off — segment-level is enough for intro/outro cuts
    # since FFmpeg snaps to ~2s keyframes during re-encode anyway.
    mark_cuts_word_level = bool(onb.get("mark_cuts_word_level", False))

    client = sheets.open_client(sa_path)
    criteria = sheets.read_criteria(client, sheet_id)
    log.info(f"phase1[{user_id}] criteria: {criteria}")

    # ── 2. Ensure unified Lessons tab + collect previously-seen video_ids and channel_ids
    sheets.ensure_lessons_tab(client, sheet_id)
    active_video_ids = sheets.get_active_video_ids(client, sheet_id)
    blocked_channel_ids = sheets.get_seen_channel_ids(client, sheet_id)
    log.info(f"phase1[{user_id}] {len(active_video_ids)} videos and "
             f"{len(blocked_channel_ids)} channels previously seen in Sheet — "
             f"deduping these (any status, including rejected/failed/legacy_import)")

    # Batch mode reuses the caller's run_id so all topics in the batch
    # land under the same run grouping in the Sheet.
    run_id = batch_run_id or sheets.make_run_id()
    _state.update(agent, user_id, thread_id=int(thread_id or 0),
                  run_id=run_id,
                  step="phase1_running",
                  topic=topic, count=count,
                  pain=pain, audience=audience,
                  chat_id=chat_id,
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

    # ── 3. Generate query variations + parallel search → unique channels ──
    search_results = _criteria_int(criteria, "search_results", DEFAULT_SEARCH_RESULTS)
    num_queries = _criteria_int(criteria, "search_queries", DEFAULT_SEARCH_QUERIES)

    _send(token, chat_id, f"🔎 <i>Генерирую поисковые запросы по теме «{_html_escape(topic)}»…</i>")
    queries = llm.generate_search_queries(topic=topic, pain=pain, n=num_queries)
    _send(token, chat_id,
          f"🔍 Ищу по <b>{len(queries)}</b> запросам:\n" +
          "\n".join(f"  • <i>{_html_escape(q)}</i>" for q in queries))

    all_videos: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()
    search_ex = ThreadPoolExecutor(max_workers=min(len(queries), 4))
    try:
        search_futures = {search_ex.submit(ytdl.search_videos, q, search_results): q
                          for q in queries}
        for fut in as_completed(search_futures, timeout=120):
            q = search_futures[fut]
            try:
                vids = fut.result()
                added = 0
                for v in vids:
                    vid = v.get("video_id")
                    if vid and vid not in seen_video_ids:
                        seen_video_ids.add(vid)
                        all_videos.append(v)
                        added += 1
                log.info(f"phase1[{user_id}] query {q!r}: {len(vids)} results, {added} new")
            except Exception as e:
                log.warning(f"phase1[{user_id}] search query failed {q!r}: {e}")
    except FuturesTimeoutError:
        log.warning(f"phase1[{user_id}] search queries timed out, using partial results")
    finally:
        search_ex.shutdown(wait=False, cancel_futures=True)

    if not all_videos:
        raise RuntimeError(f"yt-dlp search returned 0 results for all queries: {queries}")

    candidates = ytdl.unique_channels_from_search(all_videos)
    log.info(f"phase1[{user_id}] {len(candidates)} unique channel candidates "
             f"from {len(all_videos)} videos across {len(queries)} queries")

    # ── 4. Probe metadata for ALL candidates (up to HARD_CAP) ────────────
    # We used to early-exit once `target_passing` channels passed the hard
    # filter. That made Phase 1 fail too easily — if those few candidates
    # had no on-topic videos, there was no fallback. Now we always scan
    # the full candidate pool (capped at HARD_CAP_CHANNELS_TO_CHECK=50),
    # then let Claude score and the per-channel loop iterate down the
    # ranked list until a usable channel is found. yt-dlp metadata is
    # free, so the only cost is wall time (~30-60s for 50 channels in
    # batches of 8).
    cap = min(len(candidates), HARD_CAP_CHANNELS_TO_CHECK)
    _send(token, chat_id,
          f"📊 Найдено {len(candidates)} каналов в выдаче. "
          f"Проверяю метаданные у всех (лимит {cap})…")

    enriched: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    last_progress = time.time()
    checked = 0
    metadata_workers = max(1, int(onb.get("phase1_metadata_workers")
                                  or METADATA_BATCH_SIZE))
    # Per-channel hard timeout. yt-dlp's socket_timeout caps individual HTTP
    # round-trips, but a channel page can issue many requests; this cap is
    # the outer bound. Anything past this is treated as "channel unreachable"
    # and skipped so the executor can move on.
    per_call_timeout_sec = 35
    # Probe candidates in batches so we still get the early-exit benefit when
    # `target_passing` channels pass the hard filter — but inside each batch
    # the yt-dlp calls run concurrently with as_completed + per-future
    # timeout (used to be ex.map, which blocked on the slowest one and
    # silently froze Phase 1 if a channel page hung).
    for batch_start in range(0, cap, metadata_workers):
        batch = candidates[batch_start:batch_start + metadata_workers]
        if not batch:
            break

        # Don't use `with ThreadPoolExecutor():` — its __exit__ calls
        # shutdown(wait=True), which blocks on any worker still running
        # yt-dlp (Python threads can't be killed; future.cancel() is a no-op
        # once a task has started). When a single channel page hung we'd
        # freeze the whole Phase 1. We use shutdown(wait=False, cancel_futures=True)
        # in finally — the stuck thread leaks for now (eventually finishes via
        # yt-dlp's socket_timeout), but the run keeps moving.
        ex = ThreadPoolExecutor(max_workers=len(batch))
        try:
            futures: dict[Any, dict[str, Any]] = {
                ex.submit(ytdl.get_channel_metadata, c["channel_id"]): c
                for c in batch
            }
            try:
                for fut in as_completed(futures, timeout=per_call_timeout_sec * 2):
                    ch = futures[fut]
                    checked += 1
                    try:
                        meta = fut.result(timeout=per_call_timeout_sec)
                    except (FuturesTimeoutError, Exception) as e:
                        log.warning(f"phase1[{user_id}] metadata probe timed out / "
                                    f"failed for {ch.get('channel_id')}: {e}")
                        continue
                    if not meta:
                        continue
                    # Channel-level dedup — author already on platform.
                    if meta.get("channel_id") in blocked_channel_ids:
                        log.info(f"phase1[{user_id}] skip already-on-platform channel: "
                                 f"{meta.get('channel_name')!r} (id={meta.get('channel_id')})")
                        continue
                    if not _passes_hard_filter(meta, criteria):
                        log.info(f"phase1[{user_id}] filter out: {meta['channel_name']} "
                                 f"(subs={meta['subscribers']}, videos={meta['video_count']})")
                        rejected.append(meta)
                        continue
                    enriched.append({**meta, "votes": ch.get("votes", 0),
                                     "sample_titles": ch.get("sample_titles", [])})
            except FuturesTimeoutError:
                # Outer timeout: at least one channel hung. Move on — the
                # stuck future will eventually clean itself up via
                # yt-dlp's socket_timeout.
                log.warning(f"phase1[{user_id}] metadata batch outer timeout, "
                            f"continuing with what completed ({checked} checked so far)")
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

        # Progress message after each batch (executor scope ended).
        if time.time() - last_progress > PROGRESS_INTERVAL_SEC:
            _send(token, chat_id,
                  f"… проверено {checked}/{cap}, прошло фильтр: {len(enriched)}")
            last_progress = time.time()

        # Note: no early-exit on `target_passing` anymore. We scan the full
        # candidate pool so Claude's scoring + the per-channel fallback loop
        # has the widest possible bench.

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
                                channels=enriched,
                                pain=pain, audience=audience)
    # Merge score into enriched lookup
    score_by_id = {s["channel_id"]: s for s in scored}
    enriched.sort(key=lambda c: score_by_id.get(c["channel_id"], {}).get("score", 0),
                  reverse=True)
    # Candidate pool for the per-channel loop: every channel that passed both
    # the hard filter (subs/videos) and the Claude scoring threshold, ordered
    # best-first. We DO NOT truncate to `count` here — the loop below builds
    # courses one at a time and stops once `count` succeed. If a top-scored
    # channel turns out to have no on-topic videos (select_videos skip), we
    # roll down to the next-best candidate instead of aborting Phase 1.
    candidate_channels = [c for c in enriched
                          if score_by_id.get(c["channel_id"], {}).get("score", 0) >= MIN_LLM_SCORE]
    if not candidate_channels:
        # Threshold too tight — fall back to whatever we have, ranked.
        candidate_channels = list(enriched)
    log.info(f"phase1[{user_id}] {len(candidate_channels)} channels above scoring "
             f"threshold; will try them in order until {count} courses succeed")

    # ── 6. Per-channel: list videos + Claude select + dedup ─────────────
    course_summaries: list[str] = []
    skipped_total = 0
    total_videos_written = 0
    # Batch mode offsets course_idx so each topic's course gets a unique
    # course_idx within the shared run_id (topic #1 → idx 1, topic #2 → idx 2…).
    course_idx = batch_course_idx_offset
    successful_courses = 0
    skip_reasons: list[str] = []  # for the final failure message if zero succeed
    for ch in candidate_channels:
        if successful_courses >= count:
            break
        course_idx += 1
        ch_name = ch.get("channel_name") or ch["channel_id"]
        _send(token, chat_id,
              f"🎬 Курс {course_idx} (нужно {count}, успешных {successful_courses}) — "
              f"канал «{_html_escape(ch_name)}»: тяну видео и отбираю…")

        all_videos = ytdl.list_channel_videos(
            ch["channel_id"],
            max_results=CHANNEL_VIDEOS_TO_LIST,
            max_age_months=criteria.get("max_video_age_months"),
        )
        videos_for_llm = [_compact_video_for_llm(v) for v in all_videos]
        if not videos_for_llm:
            log.warning(f"phase1[{user_id}] channel {ch_name} has no videos in age window")
            skip_reasons.append(f"{ch_name}: no videos in age window")
            continue

        try:
            sel = llm.select_videos(topic=topic, criteria=criteria,
                                    channel_name=ch_name, videos=videos_for_llm,
                                    pain=pain, audience=audience)
        except Exception as e:
            log.warning(f"phase1[{user_id}] select_videos failed for {ch_name}: {e}")
            skip_reasons.append(f"{ch_name}: select_videos error: {str(e)[:80]}")
            continue
        course_title = sel.get("course_title")
        lessons = sel.get("lessons") or []
        if not course_title or not lessons:
            reason = (sel.get("skip_reason") or "no course_title/lessons")[:120]
            log.info(f"phase1[{user_id}] {ch_name} skipped: {reason}")
            skip_reasons.append(f"{ch_name}: {reason}")
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
            skip_reasons.append(f"{ch_name}: all {len(lessons)} videos were duplicates")
            continue

        # Enforce the 5-video minimum AFTER dedup. Even if Claude picked 5+,
        # earlier runs may have eaten some via the seen-videos dedup. A 1-3
        # video "course" isn't a course — skip and let the operator find
        # another channel.
        MIN_LESSONS_PER_COURSE = 5
        if len(dedup_lessons) < MIN_LESSONS_PER_COURSE:
            _send(token, chat_id,
                  f"⚠️ Курс {course_idx} ({_html_escape(ch_name)}) пропущен — "
                  f"после дедупликации осталось только {len(dedup_lessons)} видео, "
                  f"минимум {MIN_LESSONS_PER_COURSE}. Канал не годится для отдельного курса.")
            skip_reasons.append(f"{ch_name}: only {len(dedup_lessons)}/5 unique videos after dedup")
            continue

        # ── 6a. ENRICH: download + transcribe + cuts + describe + compose ──
        videos_metadata = {v["video_id"]: v for v in all_videos}
        # Streaming pre-allocate (opt-in via streaming_sheet_writes flag).
        # Each selected video gets a pending row in the Sheet BEFORE enrich
        # starts, so the reviewer can audit lessons as soon as each one
        # finishes transcribe+describe (status=done).
        streaming_row_map: dict[int, int] = {}
        if streaming_enabled:
            streaming_row_map = phase1_enrich.pre_allocate_for_streaming(
                client, sheet_id, run_id=run_id, course_idx=course_idx,
                channel_id=ch["channel_id"], channel_name=ch_name,
                selected_videos=dedup_lessons,
                videos_metadata=videos_metadata,
            )
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
                pain=pain, audience=audience,
                sheets_client=(client if streaming_enabled else None),
                sheet_id=(sheet_id if streaming_enabled else None),
                lesson_row_map=(streaming_row_map if streaming_enabled else None),
                streaming_describe=streaming_enabled,
                mark_cuts_word_level=mark_cuts_word_level,
            )
        except CookiesNeededError:
            raise  # propagate to _worker for graceful pause
        except Exception as e:
            log.warning(f"phase1[{user_id}] enrich failed for course {course_idx}: {e}",
                        exc_info=True)
            _send(token, chat_id,
                  f"⚠️ Курс {course_idx} ({_html_escape(ch_name)}): обогащение упало "
                  f"(<code>{_html_escape(str(e))[:160]}</code>). Пропускаю курс.")
            skip_reasons.append(f"{ch_name}: enrich error: {str(e)[:80]}")
            continue

        if not enriched.get("videos"):
            _send(token, chat_id,
                  f"⚠️ Курс {course_idx}: ни одно видео не довелось до конца. Пропускаю.")
            skip_reasons.append(f"{ch_name}: no videos survived enrichment")
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
        # Extended overrides (full landing payload) — operator can edit any
        # of these in Sheet and Phase 2 will pick them up.
        course_about = enriched.get("course_about", "")
        course_plan = enriched.get("course_plan", "")
        course_science = enriched.get("course_science", "")
        # Russian translations (review-only; Phase 2 ignores them when
        # building the admin payload — the *_orig columns are canon).
        course_desc_ru = enriched.get("course_description_ru", "")
        course_tagline_ru = enriched.get("course_tagline_ru", "")
        course_what_ru = enriched.get("course_what_you_learn_ru", "")
        course_target_ru = enriched.get("course_target_audience_ru", "")
        author_bio_ru = enriched.get("author_bio_ru", "")
        author_expertise_ru = enriched.get("author_expertise_ru", "")

        # Streaming mode: rows were already pre-allocated in the Sheet at the
        # start of enrich. Just finalize row 1 with course-level fields and
        # send the per-course button.
        if streaming_enabled and streaming_row_map:
            phase1_enrich.finalize_streaming_row1(
                client, sheet_id,
                lesson_row_map=streaming_row_map,
                enriched=enriched,
                full_course_title=full_course_title,
                final_course_title=final_course_title,
                author_name_fallback=ch_name,
            )
            written_count = len(streaming_row_map)
            total_videos_written += written_count
            _send_per_course_phase2_button(
                token, chat_id, sheets.sheet_url(sheet_id),
                run_id=run_id, course_idx=course_idx, count=count,
                course_title=final_course_title, video_count=written_count,
                channel_name=ch_name,
            )
            successful_courses += 1
            continue

        # Non-streaming path: build all rows for this course in memory, then
        # write them to the Sheet under sheet_lock with late-dedup, then
        # send the per-course Phase 2 button. Each course becomes visible
        # to the reviewer as soon as it finishes (instead of all at once
        # at the end of Phase 1).
        course_rows: list[dict[str, Any]] = []
        for v in enriched["videos"]:
            is_first = (v["lesson_idx"] == 1)
            course_rows.append({
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
                "lesson_description_ru": v.get("lesson_description_ru", ""),
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
                # Extended override fields (lesson_idx=1 only) — operator can
                # edit any of these in Sheet → Phase 2 sends the edit to NMS.
                "course_title": (final_course_title if is_first else ""),
                "course_about": course_about if is_first else "",
                "course_plan": course_plan if is_first else "",
                "course_science": course_science if is_first else "",
                # Russian counterparts (also lesson_idx=1 only for course-level).
                "course_description_ru": course_desc_ru if is_first else "",
                "course_tagline_ru": course_tagline_ru if is_first else "",
                "course_what_you_learn_ru": course_what_ru if is_first else "",
                "course_target_audience_ru": course_target_ru if is_first else "",
                "author_bio_ru": author_bio_ru if is_first else "",
                "author_expertise_ru": author_expertise_ru if is_first else "",
            })

        # Atomic write of this course's rows + late-dedup. A second wizard
        # running in parallel might have committed the same video_ids while
        # we were enriching; re-check under the lock before append.
        with sheets.sheet_lock():
            latest_seen = sheets.get_active_video_ids(client, sheet_id)
            latest_blocked = sheets.get_seen_channel_ids(client, sheet_id)
            rows_to_write = [
                r for r in course_rows
                if r.get("video_id") not in latest_seen
                and r.get("channel_id") not in latest_blocked
            ]
            late_skipped = len(course_rows) - len(rows_to_write)
            if rows_to_write:
                sheets.append_lesson_rows(
                    client, sheet_id, run_id=run_id, rows=rows_to_write,
                )
        if late_skipped:
            skipped_total += late_skipped
            log.info(f"phase1[{user_id}] late dedup under sheet lock: "
                     f"dropped {late_skipped} rows for course {course_idx}")

        if not rows_to_write:
            _send(token, chat_id,
                  f"⚠️ Курс {course_idx} ({_html_escape(ch_name)}) пропущен — "
                  f"все {len(course_rows)} видео уже в Sheet (поздний дубликат).")
            skip_reasons.append(f"{ch_name}: all videos already in sheet (late dedup)")
            continue

        total_videos_written += len(rows_to_write)
        _send_per_course_phase2_button(
            token, chat_id, sheets.sheet_url(sheet_id),
            run_id=run_id, course_idx=course_idx, count=count,
            course_title=final_course_title, video_count=len(rows_to_write),
            channel_name=ch_name,
        )
        # Course built successfully; advance the success counter so the loop
        # exits after `count` good ones, not after `count` attempts.
        successful_courses += 1

    if successful_courses == 0:
        # Surface concrete skip reasons so the operator knows which channels
        # tried what and why nothing made it through.
        diag = ""
        if skip_reasons:
            diag = "\n\nПроверенные каналы и причины пропуска:\n" + "\n".join(
                f"  • {r}" for r in skip_reasons[:10]
            )
        raise RuntimeError(
            "Ни один из проверенных каналов не дал валидной подборки уроков "
            "(или все видео уже были обработаны раньше). "
            "Попробуй другую тему или расширь критерии." + diag
        )

    # Batch mode: caller (launch_topic_batch) handles state update and final
    # summary after ALL topics in the batch finish. Return how many courses
    # this topic produced so the caller can advance course_idx for the next.
    if batch_silent_finish:
        return successful_courses

    _state.update(agent, user_id, thread_id=int(thread_id or 0),
                  step="awaiting_approval",
                  courses_summary=course_summaries)

    # ── 7. Final summary — rows + per-course buttons were already sent
    #       progressively as each course finished. This is just the total.
    dedup_note = f"\n♻️ Пропущено как дубли: <b>{skipped_total}</b> видео." if skipped_total else ""
    _send(token, chat_id,
          f"🏁 <b>Phase 1 завершён.</b>\n\n"
          f"Всего курсов: <b>{successful_courses}</b>"
          + (f" (запрашивали до {count})" if successful_courses < count else "")
          + f"\nВидео: <b>{total_videos_written}</b>"
          + dedup_note
          + "\n\nКаждый курс выше — со своей кнопкой "
          + "<b>«🚀 Запустить Курс N»</b>. Жми когда проверил Sheet.")
    return successful_courses


# ---------------------------------------------------------------------------
# URL-based Phase 1 (no YouTube search — user provides video links directly)
# ---------------------------------------------------------------------------

def launch_from_urls(token: str, agent: str, cfg: dict, chat_id: int, user_id: int,
                     video_ids: list[str], *, thread_id: int = 0) -> None:
    """Spawn Phase 1 for user-provided YouTube video IDs.

    Skips discovery (search, scoring, selection) entirely. Downloads the given
    videos, transcribes, marks cuts, researches author, and composes the full
    course description — same enrich pipeline, no search overhead.
    """
    thr = threading.Thread(
        target=_worker_from_urls,
        args=(token, agent, cfg, chat_id, user_id, list(video_ids), int(thread_id or 0)),
        name=f"phase1-urls-{agent}-{user_id}-{int(thread_id or 0)}",
        daemon=True,
    )
    thr.start()


def _worker_from_urls(token: str, agent: str, cfg: dict, chat_id: int, user_id: int,
                      video_ids: list[str], thread_id: int = 0) -> None:
    onb = (cfg.get("onboarder") or {})
    _TLS.thread_id = int(thread_id or 0)
    try:
        _run_from_urls(token, agent, cfg, chat_id, user_id, video_ids, onb,
                       thread_id=int(thread_id or 0))
    except CookiesNeededError as e:
        log.warning(f"phase1_urls: cookies needed: {e}")
        _state.update(agent, user_id, thread_id=int(thread_id or 0),
                      step="error", error=f"cookies_needed: {e}")
        _send_with_buttons(
            token, chat_id,
            text=(
                "⏸ <b>Phase 1 на паузе (нужны cookies).</b>\n\n"
                f"{_html_escape(str(e))[:500]}\n\n"
                "Загрузи свежий cookies.txt с youtube.com, потом /menu → 🎓 Новый курс."
            ),
            buttons=[[
                {"text": "📎 Загрузить cookies", "callback_data": "menu:cookies"},
                {"text": "✖️ Отмена", "callback_data": "wiz:cancel"},
            ]],
        )
    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"phase1_urls worker crashed: {e}\n{tb}")
        _state.update(agent, user_id, thread_id=int(thread_id or 0),
                      step="error", error=str(e))
        _send(token, chat_id,
              f"⚠️ <b>Phase 1 упал.</b>\n\n<code>{_html_escape(str(e))[:500]}</code>\n\n"
              "Используй /cancel для возврата в чат или /menu → 🎓 Новый курс.")
    finally:
        _TLS.thread_id = 0


def _run_from_urls(token: str, agent: str, cfg: dict, chat_id: int, user_id: int,
                   video_ids: list[str], onb: dict,
                   *, thread_id: int = 0) -> None:
    from collections import Counter

    # ── 1. Resolve secrets and open Sheet ────────────────────────────────
    sa_path = _secrets.resolve_path(onb, "google_service_account")
    sheet_id = onb.get("google_sheet_id") or ""
    if not sheet_id:
        raise RuntimeError("config: onboarder.google_sheet_id not set")
    openai_key = _secrets.resolve(onb, "openai_api_key", env="OPENAI_API_KEY")
    youtube_cookies_file = onb.get("youtube_cookies_file") or None
    proxy_pool_list = proxy_pool.normalise_pool(
        onb.get("youtube_proxies") or onb.get("youtube_proxy")
    )
    parallel_per_course = int(onb.get("phase1_parallel_per_course") or 4)
    compose_model = (onb.get("models") or {}).get("compose") or llm.DEFAULT_MODEL_QUALITY
    streaming_enabled = bool(onb.get("streaming_sheet_writes", False))
    mark_cuts_word_level = bool(onb.get("mark_cuts_word_level", False))

    client = sheets.open_client(sa_path)
    sheets.ensure_lessons_tab(client, sheet_id)
    active_video_ids = sheets.get_active_video_ids(client, sheet_id)

    # Dedup: skip videos already in the Sheet (any status)
    new_video_ids = [v for v in video_ids if v not in active_video_ids]
    skipped = len(video_ids) - len(new_video_ids)
    if skipped:
        _send(token, chat_id,
              f"♻️ Пропущено {skipped} дублей (уже в таблице). "
              f"К обработке: {len(new_video_ids)} видео.")
    if not new_video_ids:
        raise RuntimeError("Все предоставленные видео уже есть в таблице (дубли).")

    run_id = sheets.make_run_id()
    _state.update(agent, user_id, thread_id=int(thread_id or 0),
                  run_id=run_id, step="phase1_running",
                  topic="(URL mode)", count=1,
                  chat_id=chat_id,
                  sheet_url=sheets.sheet_url(sheet_id))

    # ── 2. Init proxy rotator ─────────────────────────────────────────────
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
                f"Ни один прокси не прошёл проверку YouTube.\n\n{str(e)[:600]}"
            )
    else:
        _send(token, chat_id, "⚠️ Прокси не настроен — пробую напрямую с VPS-IP")

    # ── 3. Fetch video metadata for each URL ──────────────────────────────
    _send(token, chat_id,
          f"📋 Получаю метаданные {len(new_video_ids)} видео…")
    proxy_current = rotator.current if rotator else None
    videos_metadata: dict[str, Any] = {}
    for vid in new_video_ids:
        meta = ytdl.get_video_metadata(
            vid, cookies_file=youtube_cookies_file, proxy=proxy_current)
        if meta:
            videos_metadata[vid] = meta
        else:
            log.warning(f"phase1_urls: no metadata for {vid}")

    if not videos_metadata:
        raise RuntimeError(
            "Не удалось получить метаданные ни одного видео. "
            "Проверь ссылки и/или обнови cookies.")

    # Determine nominal channel (most frequent among provided videos)
    ch_counts: Counter[str] = Counter(
        m.get("channel_id") for m in videos_metadata.values() if m.get("channel_id")
    )
    dominant_ch_id = ch_counts.most_common(1)[0][0] if ch_counts else "user-provided"
    dominant_ch_meta = next(
        (m for m in videos_metadata.values() if m.get("channel_id") == dominant_ch_id),
        {},
    )
    channel_name = dominant_ch_meta.get("channel_name") or "User-provided"
    channel_description = dominant_ch_meta.get("description") or ""
    channel_id = dominant_ch_id

    if len(ch_counts) > 1:
        names = ", ".join(
            m.get("channel_name", "?")
            for m in list(videos_metadata.values())[:3]
        )
        channel_name = f"Mixed ({names})"
        channel_id = "mixed"

    # Build ordered list of videos that resolved
    selected_videos = [
        {
            "video_id": vid,
            "title": videos_metadata[vid].get("title") or f"Video {i + 1}",
            "order": i,
            "reason": "user-provided URL",
        }
        for i, vid in enumerate(new_video_ids)
        if vid in videos_metadata
    ]
    if not selected_videos:
        raise RuntimeError("Нет пригодных видео для обработки после получения метаданных.")

    # Topic hint for mark_cuts (Claude will generate the real course title in compose)
    titles_hint = ", ".join(v["title"][:40] for v in selected_videos[:4])
    if len(selected_videos) > 4:
        titles_hint += "…"
    course_topic_input = f"Course from: {titles_hint}"

    _send(token, chat_id,
          f"✅ Метаданные: <b>{_html_escape(channel_name)}</b>, "
          f"{len(selected_videos)} видео")

    # ── 4. Enrich ─────────────────────────────────────────────────────────
    _send(token, chat_id,
          f"🎬 Обрабатываю {len(selected_videos)} видео "
          f"(скачивание + транскрибация + Claude)…")
    streaming_row_map: dict[int, int] = {}
    if streaming_enabled:
        streaming_row_map = phase1_enrich.pre_allocate_for_streaming(
            client, sheet_id, run_id=run_id, course_idx=1,
            channel_id=channel_id, channel_name=channel_name,
            selected_videos=selected_videos,
            videos_metadata=videos_metadata,
        )
    try:
        enriched = phase1_enrich.enrich_course(
            course_idx=1, run_id=run_id,
            channel_id=channel_id,
            channel_name=channel_name,
            channel_description=channel_description,
            course_topic_input=course_topic_input,
            # Empty title → compose_full_course generates one from transcripts
            # (compose system prompt: "if no title provided, create an engaging,
            # clear title 3-10 words"). Channel name is the fallback if compose
            # fails — wired into enrich_course's final_course_title.
            course_title_from_llm="",
            selected_videos=selected_videos,
            videos_metadata=videos_metadata,
            openai_key=openai_key,
            cookies_file=youtube_cookies_file,
            rotator=rotator,
            on_progress=lambda msg: _send(token, chat_id, _html_escape(msg)),
            max_parallel=parallel_per_course,
            compose_model=compose_model,
            sheets_client=(client if streaming_enabled else None),
            sheet_id=(sheet_id if streaming_enabled else None),
            lesson_row_map=(streaming_row_map if streaming_enabled else None),
            streaming_describe=streaming_enabled,
            mark_cuts_word_level=bool(onb.get("mark_cuts_word_level", False)),
        )
    except CookiesNeededError:
        raise

    if not enriched.get("videos"):
        raise RuntimeError("Ни одно видео не прошло обогащение. Проверь логи.")

    # ── 5. Build Sheet rows and write ─────────────────────────────────────
    final_title = enriched.get("course_title") or channel_name
    full_title = f"Курс 1: {channel_name} — {final_title}"

    if streaming_enabled and streaming_row_map:
        # Rows already in Sheet (pre-allocated + per-video streamed). Finalize
        # row 1 with course-level fields and short-circuit the batch path.
        phase1_enrich.finalize_streaming_row1(
            client, sheet_id,
            lesson_row_map=streaming_row_map,
            enriched=enriched,
            full_course_title=full_title,
            final_course_title=final_title,
            author_name_fallback=channel_name,
        )
        _state.update(agent, user_id, thread_id=int(thread_id or 0),
                      step="awaiting_approval",
                      courses_summary=[f"1. {final_title} ({len(enriched['videos'])} уроков)"])
        _send_per_course_phase2_button(
            token, chat_id, sheets.sheet_url(sheet_id),
            run_id=run_id, course_idx=1, count=1,
            course_title=final_title, video_count=len(enriched["videos"]),
            channel_name=channel_name,
        )
        return

    all_lesson_rows: list[dict[str, Any]] = []
    for v in enriched["videos"]:
        is_first = (v["lesson_idx"] == 1)
        all_lesson_rows.append({
            "course": full_title if is_first else "Курс 1",
            "lesson_idx": v["lesson_idx"],
            "channel": channel_name,
            "lesson_title": v["title"],
            "url": v["url"],
            "duration_sec": v.get("duration_sec", 0),
            "video_id": v["video_id"],
            "channel_id": channel_id,
            "course_idx": 1,
            "lesson_description": v.get("lesson_description", ""),
            "lesson_description_ru": v.get("lesson_description_ru", ""),
            "transcript_excerpt": v.get("transcript_excerpt", ""),
            "course_description": enriched.get("course_description", "") if is_first else "",
            "course_tagline": enriched.get("course_tagline", "") if is_first else "",
            "course_what_you_learn": enriched.get("course_what_you_learn", "") if is_first else "",
            "course_target_audience": enriched.get("course_target_audience", "") if is_first else "",
            "author_name": enriched.get("author_name", "") if is_first else "",
            "author_bio": enriched.get("author_bio", "") if is_first else "",
            "author_expertise": enriched.get("author_expertise", "") if is_first else "",
            # Extended override fields (lesson_idx=1 only)
            "course_title": (final_title if is_first else ""),
            "course_about": enriched.get("course_about", "") if is_first else "",
            "course_plan": enriched.get("course_plan", "") if is_first else "",
            "course_science": enriched.get("course_science", "") if is_first else "",
            "course_description_ru": enriched.get("course_description_ru", "") if is_first else "",
            "course_tagline_ru": enriched.get("course_tagline_ru", "") if is_first else "",
            "course_what_you_learn_ru": enriched.get("course_what_you_learn_ru", "") if is_first else "",
            "course_target_audience_ru": enriched.get("course_target_audience_ru", "") if is_first else "",
            "author_bio_ru": enriched.get("author_bio_ru", "") if is_first else "",
            "author_expertise_ru": enriched.get("author_expertise_ru", "") if is_first else "",
        })

    with sheets.sheet_lock():
        latest_seen = sheets.get_active_video_ids(client, sheet_id)
        rows_to_write = [r for r in all_lesson_rows
                         if r.get("video_id") not in latest_seen]
        if rows_to_write:
            sheets.append_lesson_rows(client, sheet_id, run_id=run_id,
                                      rows=rows_to_write)

    _state.update(agent, user_id, thread_id=int(thread_id or 0),
                  step="awaiting_approval",
                  courses_summary=[f"1. {final_title} ({len(enriched['videos'])} уроков)"])

    # ── 6. Per-course Telegram button (rows already written above) ────────
    _send_per_course_phase2_button(
        token, chat_id, sheets.sheet_url(sheet_id),
        run_id=run_id, course_idx=1, count=1,
        course_title=final_title, video_count=len(enriched["videos"]),
        channel_name=channel_name,
    )


# ---------------------------------------------------------------------------
# URL multi-batch: process N courses sequentially within one run_id
# ---------------------------------------------------------------------------

def launch_from_url_groups(token: str, agent: str, cfg: dict, chat_id: int,
                           user_id: int, url_groups: list[list[str]],
                           *, thread_id: int = 0) -> None:
    """Spawn Phase 1 for a batch of URL groups (one course per group).

    Each group's videos become one course_idx within a single shared run_id.
    Sheet rows for all courses are written at the end so the operator sees a
    single, complete review payload.
    """
    thr = threading.Thread(
        target=_worker_from_url_groups,
        args=(token, agent, cfg, chat_id, user_id,
              [list(g) for g in url_groups], int(thread_id or 0)),
        name=f"phase1-url-groups-{agent}-{user_id}-{int(thread_id or 0)}",
        daemon=True,
    )
    thr.start()


def _worker_from_url_groups(token: str, agent: str, cfg: dict, chat_id: int,
                            user_id: int, url_groups: list[list[str]],
                            thread_id: int = 0) -> None:
    onb = (cfg.get("onboarder") or {})
    _TLS.thread_id = int(thread_id or 0)
    try:
        _run_from_url_groups(token, agent, cfg, chat_id, user_id, url_groups, onb,
                             thread_id=int(thread_id or 0))
    except CookiesNeededError as e:
        log.warning(f"phase1_url_groups: cookies needed: {e}")
        _state.update(agent, user_id, thread_id=int(thread_id or 0),
                      step="error", error=f"cookies_needed: {e}")
        _send_with_buttons(
            token, chat_id,
            text=(
                "⏸ <b>Phase 1 на паузе (нужны cookies).</b>\n\n"
                f"{_html_escape(str(e))[:500]}\n\n"
                "Загрузи свежий cookies.txt с youtube.com, потом /menu → 🎓 Новый курс."
            ),
            buttons=[[
                {"text": "📎 Загрузить cookies", "callback_data": "menu:cookies"},
                {"text": "✖️ Отмена", "callback_data": "wiz:cancel"},
            ]],
        )
    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"phase1_url_groups worker crashed: {e}\n{tb}")
        _state.update(agent, user_id, thread_id=int(thread_id or 0),
                      step="error", error=str(e))
        _send(token, chat_id,
              f"⚠️ <b>Phase 1 упал.</b>\n\n<code>{_html_escape(str(e))[:500]}</code>\n\n"
              "Используй /cancel для возврата в чат или /menu → 🎓 Новый курс.")
    finally:
        _TLS.thread_id = 0


def _run_from_url_groups(token: str, agent: str, cfg: dict, chat_id: int,
                        user_id: int, url_groups: list[list[str]], onb: dict,
                        *, thread_id: int = 0) -> None:
    """Multi-course URL batch. Reuses per-course logic via _process_one_url_course."""
    from collections import Counter

    # ── 1. Resolve secrets and open Sheet (once for entire batch) ─────────
    sa_path = _secrets.resolve_path(onb, "google_service_account")
    sheet_id = onb.get("google_sheet_id") or ""
    if not sheet_id:
        raise RuntimeError("config: onboarder.google_sheet_id not set")
    openai_key = _secrets.resolve(onb, "openai_api_key", env="OPENAI_API_KEY")
    youtube_cookies_file = onb.get("youtube_cookies_file") or None
    proxy_pool_list = proxy_pool.normalise_pool(
        onb.get("youtube_proxies") or onb.get("youtube_proxy")
    )
    parallel_per_course = int(onb.get("phase1_parallel_per_course") or 4)
    compose_model = (onb.get("models") or {}).get("compose") or llm.DEFAULT_MODEL_QUALITY
    streaming_enabled = bool(onb.get("streaming_sheet_writes", False))
    mark_cuts_word_level = bool(onb.get("mark_cuts_word_level", False))

    client = sheets.open_client(sa_path)
    sheets.ensure_lessons_tab(client, sheet_id)

    run_id = sheets.make_run_id()
    _state.update(agent, user_id, thread_id=int(thread_id or 0),
                  run_id=run_id, step="phase1_running",
                  topic=f"(URL batch: {len(url_groups)} courses)",
                  count=len(url_groups), chat_id=chat_id,
                  sheet_url=sheets.sheet_url(sheet_id))

    # ── 2. Init proxy rotator (once) ──────────────────────────────────────
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
                f"Ни один прокси не прошёл проверку YouTube.\n\n{str(e)[:600]}"
            )
    else:
        _send(token, chat_id, "⚠️ Прокси не настроен — пробую напрямую с VPS-IP")

    # ── 3. Loop over course groups, enrich each, accumulate rows ──────────
    all_lesson_rows: list[dict[str, Any]] = []
    course_summaries: list[str] = []
    courses_done = 0
    courses_failed: list[str] = []

    for course_idx, video_ids in enumerate(url_groups, start=1):
        _send(token, chat_id,
              f"\n📚 <b>Курс {course_idx}/{len(url_groups)}</b> — {len(video_ids)} видео")
        # Re-read active video_ids from Sheet on EACH course iteration —
        # earlier courses in this batch just wrote rows that need to be
        # respected by later courses' dedup check.
        active_video_ids = sheets.get_active_video_ids(client, sheet_id)
        try:
            course_rows, summary = _process_one_url_course(
                course_idx=course_idx, run_id=run_id, video_ids=video_ids,
                active_video_ids=active_video_ids,
                openai_key=openai_key, cookies_file=youtube_cookies_file,
                rotator=rotator, parallel_per_course=parallel_per_course,
                compose_model=compose_model, token=token, chat_id=chat_id,
                streaming_enabled=streaming_enabled,
                sheets_client=client,
                sheet_id=sheet_id,
                mark_cuts_word_level=mark_cuts_word_level,
            )
        except CookiesNeededError:
            raise  # propagate to worker handler
        except Exception as e:
            log.error(f"course {course_idx} crashed: {e}", exc_info=True)
            _send(token, chat_id,
                  f"⚠️ Курс {course_idx} упал: <code>{_html_escape(str(e))[:200]}</code>. "
                  f"Продолжаю со следующими.")
            courses_failed.append(f"Курс {course_idx}: {str(e)[:80]}")
            continue

        if not course_rows and not streaming_enabled:
            # In streaming mode, an empty list means "rows already in Sheet"
            # — it is success, not failure.
            courses_failed.append(f"Курс {course_idx}: нет уроков после обработки")
            continue

        all_lesson_rows.extend(course_rows)
        course_summaries.append(summary)
        courses_done += 1

        # Write rows incrementally per-course so progress is visible in Sheet
        # even if a later course in the batch crashes.
        with sheets.sheet_lock():
            latest_seen = sheets.get_active_video_ids(client, sheet_id)
            new_rows = [r for r in course_rows
                        if r.get("video_id") not in latest_seen]
            if new_rows:
                sheets.append_lesson_rows(client, sheet_id, run_id=run_id,
                                          rows=new_rows)

    # In streaming mode, all_lesson_rows stays empty (rows live only in Sheet),
    # so a non-zero courses_done with empty all_lesson_rows is still success.
    if courses_done == 0:
        raise RuntimeError(
            "Ни один курс из пакета не дал валидных уроков. "
            + ("\n".join(courses_failed) if courses_failed else "")
        )

    _state.update(agent, user_id, thread_id=int(thread_id or 0),
                  step="awaiting_approval",
                  courses_summary=course_summaries)

    # ── 4. Final summary — per-course buttons were already sent in
    #       _process_one_url_course as each course finished.
    summary_text = "\n".join(f"  {s}" for s in course_summaries)
    fail_text = ""
    if courses_failed:
        fail_text = "\n\n⚠️ <b>Упавшие курсы:</b>\n" + "\n".join(
            f"  • {_html_escape(f)}" for f in courses_failed)
    _send(token, chat_id,
          f"🏁 <b>Phase 1 завершён (пакет URL).</b>\n\n"
          f"Успешных курсов: <b>{courses_done}/{len(url_groups)}</b>\n"
          f"Всего уроков: <b>{len(all_lesson_rows)}</b>\n\n"
          f"{summary_text}{fail_text}\n\n"
          "Каждый курс выше — со своей кнопкой "
          "<b>«🚀 Запустить Курс N»</b>.")


def _process_one_url_course(*, course_idx: int, run_id: str,
                            video_ids: list[str],
                            active_video_ids: set[str],
                            openai_key: str, cookies_file: str | None,
                            rotator: proxy_pool.ProxyRotator,
                            parallel_per_course: int, compose_model: str,
                            token: str, chat_id: int,
                            # Streaming sheet writes (opt-in; default OFF). When
                            # all three are provided, the helper pre-allocates +
                            # streams writes + finalizes row 1 with course-level
                            # fields, then returns [] so the orchestrator's batch
                            # append is a no-op.
                            streaming_enabled: bool = False,
                            sheets_client: Any = None,
                            sheet_id: str | None = None,
                            mark_cuts_word_level: bool = False,
                            ) -> tuple[list[dict[str, Any]], str]:
    """Process ONE course of a URL batch. Returns (sheet_rows, summary_str).

    Mirrors the per-course steps from _run_from_urls but with the course_idx
    threaded through everywhere so multi-batch Sheet rows are properly tagged.
    """
    from collections import Counter

    # Dedup vs Sheet
    new_video_ids = [v for v in video_ids if v not in active_video_ids]
    skipped = len(video_ids) - len(new_video_ids)
    if skipped:
        _send(token, chat_id,
              f"  ♻️ Курс {course_idx}: пропущено {skipped} дублей из Sheet")
    if not new_video_ids:
        raise RuntimeError(
            f"Курс {course_idx}: все {len(video_ids)} видео уже в Sheet")

    # Fetch metadata
    _send(token, chat_id,
          f"  📋 Курс {course_idx}: метаданные {len(new_video_ids)} видео…")
    proxy_current = rotator.current if rotator else None
    videos_metadata: dict[str, Any] = {}
    for vid in new_video_ids:
        meta = ytdl.get_video_metadata(
            vid, cookies_file=cookies_file, proxy=proxy_current)
        if meta:
            videos_metadata[vid] = meta
    if not videos_metadata:
        raise RuntimeError(
            f"Курс {course_idx}: не получил метаданные ни одного видео")

    # Determine nominal channel
    ch_counts: Counter[str] = Counter(
        m.get("channel_id") for m in videos_metadata.values() if m.get("channel_id")
    )
    dominant_ch_id = ch_counts.most_common(1)[0][0] if ch_counts else "user-provided"
    dominant_ch_meta = next(
        (m for m in videos_metadata.values() if m.get("channel_id") == dominant_ch_id),
        {},
    )
    channel_name = dominant_ch_meta.get("channel_name") or "User-provided"
    channel_description = dominant_ch_meta.get("description") or ""
    channel_id = dominant_ch_id
    if len(ch_counts) > 1:
        names = ", ".join(m.get("channel_name", "?")
                          for m in list(videos_metadata.values())[:3])
        channel_name = f"Mixed ({names})"
        channel_id = "mixed"

    selected_videos = [
        {
            "video_id": vid,
            "title": videos_metadata[vid].get("title") or f"Video {i + 1}",
            "order": i,
            "reason": "user-provided URL",
        }
        for i, vid in enumerate(new_video_ids)
        if vid in videos_metadata
    ]
    if not selected_videos:
        raise RuntimeError(f"Курс {course_idx}: нет пригодных видео")

    titles_hint = ", ".join(v["title"][:40] for v in selected_videos[:4])
    if len(selected_videos) > 4:
        titles_hint += "…"
    course_topic_input = f"Course from: {titles_hint}"

    _send(token, chat_id,
          f"  🎬 Курс {course_idx} — <b>{_html_escape(channel_name)}</b>: "
          f"скачиваю + транскрибирую + compose ({len(selected_videos)} видео)…")

    streaming_row_map: dict[int, int] = {}
    if streaming_enabled and sheets_client is not None and sheet_id:
        streaming_row_map = phase1_enrich.pre_allocate_for_streaming(
            sheets_client, sheet_id, run_id=run_id, course_idx=course_idx,
            channel_id=channel_id, channel_name=channel_name,
            selected_videos=selected_videos,
            videos_metadata=videos_metadata,
        )

    enriched = phase1_enrich.enrich_course(
        course_idx=course_idx, run_id=run_id,
        channel_id=channel_id, channel_name=channel_name,
        channel_description=channel_description,
        course_topic_input=course_topic_input,
        course_title_from_llm="",  # compose will generate
        selected_videos=selected_videos,
        videos_metadata=videos_metadata,
        openai_key=openai_key, cookies_file=cookies_file, rotator=rotator,
        on_progress=lambda msg: _send(token, chat_id,
                                      f"  [Курс {course_idx}] {_html_escape(msg)}"),
        max_parallel=parallel_per_course,
        compose_model=compose_model,
        sheets_client=(sheets_client if streaming_enabled else None),
        sheet_id=(sheet_id if streaming_enabled else None),
        lesson_row_map=(streaming_row_map if streaming_enabled else None),
        streaming_describe=streaming_enabled,
        mark_cuts_word_level=mark_cuts_word_level,
    )

    if not enriched.get("videos"):
        raise RuntimeError(f"Курс {course_idx}: ни одно видео не прошло обогащение")

    final_title = enriched.get("course_title") or channel_name
    full_title = f"Курс {course_idx}: {channel_name} — {final_title}"
    course_label = f"Курс {course_idx}"

    rows: list[dict[str, Any]] = []
    for v in enriched["videos"]:
        is_first = (v["lesson_idx"] == 1)
        rows.append({
            "course": full_title if is_first else course_label,
            "lesson_idx": v["lesson_idx"],
            "channel": channel_name,
            "lesson_title": v["title"],
            "url": v["url"],
            "duration_sec": v.get("duration_sec", 0),
            "video_id": v["video_id"],
            "channel_id": channel_id,
            "course_idx": course_idx,
            "lesson_description": v.get("lesson_description", ""),
            "lesson_description_ru": v.get("lesson_description_ru", ""),
            "transcript_excerpt": v.get("transcript_excerpt", ""),
            "course_description": enriched.get("course_description", "") if is_first else "",
            "course_tagline": enriched.get("course_tagline", "") if is_first else "",
            "course_what_you_learn": enriched.get("course_what_you_learn", "") if is_first else "",
            "course_target_audience": enriched.get("course_target_audience", "") if is_first else "",
            "author_name": enriched.get("author_name", "") if is_first else "",
            "author_bio": enriched.get("author_bio", "") if is_first else "",
            "author_expertise": enriched.get("author_expertise", "") if is_first else "",
            "course_title": (final_title if is_first else ""),
            "course_about": enriched.get("course_about", "") if is_first else "",
            "course_plan": enriched.get("course_plan", "") if is_first else "",
            "course_science": enriched.get("course_science", "") if is_first else "",
            "course_description_ru": enriched.get("course_description_ru", "") if is_first else "",
            "course_tagline_ru": enriched.get("course_tagline_ru", "") if is_first else "",
            "course_what_you_learn_ru": enriched.get("course_what_you_learn_ru", "") if is_first else "",
            "course_target_audience_ru": enriched.get("course_target_audience_ru", "") if is_first else "",
            "author_bio_ru": enriched.get("author_bio_ru", "") if is_first else "",
            "author_expertise_ru": enriched.get("author_expertise_ru", "") if is_first else "",
        })

    if streaming_enabled and streaming_row_map and sheets_client is not None and sheet_id:
        phase1_enrich.finalize_streaming_row1(
            sheets_client, sheet_id,
            lesson_row_map=streaming_row_map,
            enriched=enriched,
            full_course_title=full_title,
            final_course_title=final_title,
            author_name_fallback=channel_name,
        )
        # Rows already streamed — short-circuit so orchestrator's batch append
        # is a no-op for this course.
        rows = []

    summary = (f"{course_idx}. {final_title} "
               f"({len(enriched['videos'])} уроков, {channel_name})")
    sheet_url_safe = sheets.sheet_url(sheet_id) if sheet_id else ""
    if sheet_url_safe:
        _send_per_course_phase2_button(
            token, chat_id, sheet_url_safe,
            run_id=run_id, course_idx=course_idx, count=0,
            course_title=final_title, video_count=len(enriched["videos"]),
            channel_name=channel_name,
        )
    else:
        _send(token, chat_id,
              f"  ✅ Курс {course_idx} готов: <b>{_html_escape(final_title)}</b> "
              f"({len(enriched['videos'])} уроков)")
    return rows, summary


# ---------------------------------------------------------------------------
# Topic multi-batch: run topic-mode Phase 1 sequentially for N topics
# ---------------------------------------------------------------------------

def launch_topic_batch(token: str, agent: str, cfg: dict, chat_id: int,
                       user_id: int, topic_groups: list[dict[str, str]],
                       *, thread_id: int = 0) -> None:
    """Spawn Phase 1 worker that processes N topic+description pairs sequentially.

    Each topic becomes one course_idx within a shared run_id. After all topics
    are processed, one consolidated "Запустить обработку" button is sent.
    """
    thr = threading.Thread(
        target=_worker_topic_batch,
        args=(token, agent, cfg, chat_id, user_id,
              [dict(tg) for tg in topic_groups], int(thread_id or 0)),
        name=f"phase1-topic-batch-{agent}-{user_id}-{int(thread_id or 0)}",
        daemon=True,
    )
    thr.start()


def _worker_topic_batch(token: str, agent: str, cfg: dict, chat_id: int,
                        user_id: int, topic_groups: list[dict[str, str]],
                        thread_id: int = 0) -> None:
    onb = (cfg.get("onboarder") or {})
    _TLS.thread_id = int(thread_id or 0)
    try:
        _run_topic_batch(token, agent, cfg, chat_id, user_id, topic_groups, onb,
                         thread_id=int(thread_id or 0))
    except CookiesNeededError as e:
        log.warning(f"phase1_topic_batch: cookies needed: {e}")
        _state.update(agent, user_id, thread_id=int(thread_id or 0),
                      step="error", error=f"cookies_needed: {e}")
        _send_with_buttons(
            token, chat_id,
            text=(
                "⏸ <b>Phase 1 на паузе (нужны cookies).</b>\n\n"
                f"{_html_escape(str(e))[:500]}\n\n"
                "Загрузи свежий cookies.txt с youtube.com, потом /menu → 🎓 Новый курс."
            ),
            buttons=[[
                {"text": "📎 Загрузить cookies", "callback_data": "menu:cookies"},
                {"text": "✖️ Отмена", "callback_data": "wiz:cancel"},
            ]],
        )
    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"phase1_topic_batch worker crashed: {e}\n{tb}")
        _state.update(agent, user_id, thread_id=int(thread_id or 0),
                      step="error", error=str(e))
        _send(token, chat_id,
              f"⚠️ <b>Phase 1 упал.</b>\n\n<code>{_html_escape(str(e))[:500]}</code>\n\n"
              "Используй /cancel для возврата в чат или /menu → 🎓 Новый курс.")
    finally:
        _TLS.thread_id = 0


def _run_topic_batch(token: str, agent: str, cfg: dict, chat_id: int,
                     user_id: int, topic_groups: list[dict[str, str]], onb: dict,
                     *, thread_id: int = 0) -> None:
    """Sequentially run topic-mode Phase 1 for each topic in the batch.

    Reuses one shared run_id. Each topic's course gets a unique course_idx
    via the batch_course_idx_offset param on _run. Sheet rows are written
    incrementally by each _run call (no extra accumulation needed here)
    so partial progress survives mid-batch crashes.
    """
    sheet_id = onb.get("google_sheet_id") or ""
    if not sheet_id:
        raise RuntimeError("config: onboarder.google_sheet_id not set")

    # Generate ONE run_id for the whole batch. Each topic's call to _run
    # reuses it (via batch_run_id) so all rows land under the same group.
    run_id = sheets.make_run_id()

    _state.update(agent, user_id, thread_id=int(thread_id or 0),
                  run_id=run_id,
                  step="phase1_running",
                  topic=f"(topic batch: {len(topic_groups)} courses)",
                  count=len(topic_groups),
                  chat_id=chat_id,
                  sheet_url=sheets.sheet_url(sheet_id))

    completed_topics: list[str] = []
    failed_topics: list[str] = []
    course_idx_offset = 0  # advances by number of courses produced per topic

    for topic_idx, tg in enumerate(topic_groups, start=1):
        topic = (tg.get("topic") or "").strip()
        pain = (tg.get("pain") or "").strip()
        if not topic:
            continue

        _send(token, chat_id,
              f"\n📚 <b>Тема {topic_idx}/{len(topic_groups)}: "
              f"{_html_escape(topic)}</b>")

        try:
            # Each topic in the batch ALSO tries to produce up to 5 courses
            # (same as single-topic mode). Phase 1's per-channel loop stops
            # at the first 5 that succeed, or at whatever it found if fewer.
            courses_made = _run(
                token, agent, cfg, chat_id, user_id,
                topic=topic, count=5, onb=onb,
                pain=pain, audience="",
                thread_id=thread_id,
                batch_run_id=run_id,
                batch_course_idx_offset=course_idx_offset,
                batch_silent_finish=True,
            )
            course_idx_offset += int(courses_made or 0)
            if courses_made:
                completed_topics.append(
                    f"  ✓ Тема {topic_idx}: «{topic}» — {courses_made} курс(ов)"
                )
            else:
                failed_topics.append(f"  ✗ Тема {topic_idx}: «{topic}» — 0 курсов")
        except CookiesNeededError:
            raise  # propagate to outer worker
        except Exception as e:
            log.error(f"topic batch: topic {topic_idx} «{topic}» failed: {e}",
                      exc_info=True)
            _send(token, chat_id,
                  f"⚠️ Тема {topic_idx} «{_html_escape(topic)}» упала: "
                  f"<code>{_html_escape(str(e))[:200]}</code>. Продолжаю.")
            failed_topics.append(f"  ✗ Тема {topic_idx}: «{topic}» — {str(e)[:80]}")
            continue

    if not completed_topics:
        raise RuntimeError(
            "Ни одна тема из пакета не дала курсов.\n"
            + "\n".join(failed_topics)
        )

    # Final state + button (only here, not per topic)
    _state.update(agent, user_id, thread_id=int(thread_id or 0),
                  step="awaiting_approval",
                  courses_summary=completed_topics)

    success_block = "\n".join(completed_topics)
    fail_block = ("\n\n⚠️ <b>Не сработали:</b>\n" + "\n".join(failed_topics)
                  if failed_topics else "")
    _send(token, chat_id,
          f"🏁 <b>Phase 1 завершён (пакет тем).</b>\n\n"
          f"Успешных тем: <b>{len(completed_topics)}/{len(topic_groups)}</b>\n"
          f"Всего курсов: <b>{course_idx_offset}</b>\n\n"
          f"{success_block}{fail_block}\n\n"
          "Каждый курс выше — со своей кнопкой "
          "<b>«🚀 Запустить Курс N»</b>. Жми по одной.")


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
    """Use gateway's tg_api lazily so we honor its retries / chunking conventions.

    Reads the active forum topic from `_TLS.thread_id` (set by `_worker` at
    start) so callers don't have to pass thread_id explicitly.
    """
    from gateway import tg_api  # type: ignore
    thread_id = int(getattr(_TLS, "thread_id", 0) or 0)
    kwargs: dict[str, Any] = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    if thread_id:
        kwargs["message_thread_id"] = thread_id
    try:
        tg_api(token, "sendMessage", **kwargs)
    except Exception as e:
        log.warning(f"phase1 _send failed: {e}")


def _send_with_buttons(token: str, chat_id: int, text: str,
                       buttons: list[list[dict[str, str]]]) -> None:
    from gateway import send_message_with_buttons  # type: ignore
    thread_id = int(getattr(_TLS, "thread_id", 0) or 0)
    send_message_with_buttons(token, chat_id, text, buttons,
                              message_thread_id=thread_id)


def _send_per_course_phase2_button(token: str, chat_id: int, sheet_url: str, *,
                                   run_id: str, course_idx: int, count: int,
                                   course_title: str, video_count: int,
                                   channel_name: str) -> None:
    """Sent right after a course's rows hit the Sheet — gives the operator
    a course-specific «🚀 Запустить Курс N» button so they can launch Phase 2
    for that one course without waiting for the rest of Phase 1.
    """
    text = (
        f"✅ <b>Курс {course_idx} готов</b> "
        f"<i>({_html_escape(channel_name)})</i>\n"
        f"<b>{_html_escape(course_title)}</b>\n\n"
        f"📋 {video_count} видео в Sheet. Проверь, поставь "
        f"<b>Approved=TRUE</b> у нужных строк и жми кнопку:\n"
        f"{sheet_url}"
    )
    _send_with_buttons(
        token, chat_id, text,
        buttons=[[{"text": f"🚀 Запустить Курс {course_idx}",
                   "callback_data": f"wiz:p2c:{run_id}:{course_idx}"}]],
    )


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )
