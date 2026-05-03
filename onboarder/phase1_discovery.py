"""Phase 1: discovery pipeline.

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
    7. Append to Run-* tab + Runs index, mark "awaiting_approval"
    8. Telegram message with Sheet link + "Запустить обработку" button

Errors at any step → Telegram message + wizard state set to "error". User can /cancel
or click "Запустить обработку" anyway (will fail at Phase 2 if data is missing).
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import Any

from . import _secrets, llm, sheets, state as _state, youtube_dl as ytdl

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

    client = sheets.open_client(sa_path)
    criteria = sheets.read_criteria(client, sheet_id)
    log.info(f"phase1[{user_id}] criteria: {criteria}")

    # ── 2. Create Run row + Run-* tab ────────────────────────────────────
    run_id = sheets.make_run_id()
    tab_name = sheets.run_tab_name(run_id)
    sheets.append_run(client, sheet_id, run_id=run_id, topic=topic,
                      count=count, status="phase1_searching", sheet_tab=tab_name)
    sheets.create_run_tab(client, sheet_id, tab_name)
    _state.update(agent, user_id, run_id=run_id, sheet_tab=tab_name,
                  step="phase1_running",
                  sheet_url=sheets.sheet_tab_url(sheet_id))

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
    sheets.update_run_status(client, sheet_id, run_id, status="phase1_scoring")
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

    # ── 6. Per-channel: list videos + Claude select ──────────────────────
    sheets.update_run_status(client, sheet_id, run_id, status="phase1_selecting")
    all_lesson_rows: list[dict[str, Any]] = []
    course_summaries: list[str] = []
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

        full_course_title = f"Курс {course_idx}: {ch_name} — {course_title}"
        course_summaries.append(f"{course_idx}. {course_title} ({len(lessons)} уроков)")
        videos_by_id = {v["video_id"]: v for v in all_videos}
        for lesson_idx, lsn in enumerate(lessons, start=1):
            vid = lsn.get("video_id")
            v = videos_by_id.get(vid) or {}
            all_lesson_rows.append({
                "course": full_course_title if lesson_idx == 1 else f"Курс {course_idx}",
                "lesson_idx": lesson_idx,
                "channel": ch_name,
                "title": lsn.get("title") or v.get("title", ""),
                "url": v.get("url") or f"https://youtu.be/{vid}",
                "duration_sec": v.get("duration_sec", 0),
                "video_id": vid,
                "channel_id": ch["channel_id"],
                "course_idx": course_idx,
            })

    if not all_lesson_rows:
        raise RuntimeError("Ни один канал не дал валидной подборки уроков. "
                           "Попробуй другую тему или расширь критерии.")

    # ── 7. Write to Sheet ────────────────────────────────────────────────
    sheets.append_lesson_rows(client, sheet_id, tab_name, all_lesson_rows)
    sheets.update_run_status(client, sheet_id, run_id, status="awaiting_approval")
    _state.update(agent, user_id, step="awaiting_approval",
                  courses_summary=course_summaries)

    # ── 8. Final Telegram message with action button ─────────────────────
    sheet_url = sheets.sheet_tab_url(sheet_id)
    summary_lines = "\n".join(course_summaries) if course_summaries else "(нет курсов)"
    _send_with_buttons(
        token, chat_id,
        text=(
            f"✅ <b>Phase 1 готов.</b>\n\n"
            f"Подборка ({len(all_lesson_rows)} видео в {len(course_summaries)} курсах):\n"
            f"{_html_escape(summary_lines)}\n\n"
            f"📋 Открой Sheet, проверь подборку, поставь галочки в колонке "
            f"<b>Approved</b> у нужных строк, отредактируй названия если нужно:\n"
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
