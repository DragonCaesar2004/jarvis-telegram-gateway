"""Wizard state machine: form → Phase 1 → approval → Phase 2.

Thread-aware: all state and outgoing messages carry an optional
`thread_id` (Telegram forum topic). thread_id=0 means DM / non-forum group,
preserving the original behavior. With non-zero thread_id one user can run
several wizards in parallel — one per forum topic — and all status messages
land back in the right topic.

Public API consumed by gateway.py:
    handle_wizard_message(token, agent, cfg, chat_id, user_id, text, msg, thread_id=0)
        Called by gateway's mode-router when user is in MODE_WIZARD and sends a message.

    start_wizard(token, agent, cfg, chat_id, user_id, *, thread_id=0)
        Called from menu callback ("menu:onboard") to begin a fresh wizard run.

    register_callbacks(register_func)
        Called once at gateway startup to register inline-button handlers
        (prefix "wiz:") with the gateway's callback dispatcher.

    clear_wizard_state(agent, user_id, thread_id=0)
        Called by /cancel and "menu:chat" to wipe in-progress form/run state.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import state as _state
from . import clarify as _clarify

# Matches youtube.com/watch?v=ID and youtu.be/ID (with or without https://).
# Handles URLs separated by spaces, newlines, or multiple spaces.
_YOUTUBE_URL_RE = re.compile(
    r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?[^\s]*?v=|youtu\.be/)([A-Za-z0-9_-]{11})'
)


def _parse_youtube_urls(text: str) -> list[str]:
    """Extract unique YouTube video IDs from text (space/newline-separated URLs)."""
    return list(dict.fromkeys(_YOUTUBE_URL_RE.findall(text)))


def _parse_topic_groups(text: str) -> list[dict[str, str]]:
    """Split topic+description text by separator lines into a list of courses.

    Each block (between separator lines of =/-/_/*/#) becomes one course.
    Within a block:
      - first non-empty line  → `topic`
      - remaining lines       → `pain` (description)

    Returns list of {"topic": str, "pain": str}. Empty blocks dropped.
    Example input:
        Kegel yoga для мужчин
        Курс должен включать упражнения тазового дна
        ===
        Постпартум восстановление
        Для мам после родов
    → [{"topic": "Kegel yoga для мужчин", "pain": "Курс должен включать..."},
        {"topic": "Постпартум восстановление", "pain": "Для мам после родов"}]
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        is_sep = bool(s) and len(set(s)) == 1 and s[0] in "=-_*#"
        if is_sep:
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)

    out: list[dict[str, str]] = []
    for block in blocks:
        non_empty = [line for line in block if line.strip()]
        if not non_empty:
            continue
        topic = non_empty[0].strip()
        pain = "\n".join(non_empty[1:]).strip()
        if topic:
            out.append({"topic": topic, "pain": pain})
    return out


def _parse_youtube_url_groups(text: str) -> list[list[str]]:
    """Split text into course groups by separator lines, return list-of-lists of video_ids.

    A separator line is one whose stripped content is one or more of the
    same character from {=, -, _, *, #} — e.g.:
        url1
        url2
        ===
        url3
        url4
        ---
        url5

    Returns 3 groups: [url1,url2], [url3,url4], [url5].

    If there are no separator lines, returns a single group containing all
    URLs from the text (backward-compatible with single-course paste).

    Empty groups (no valid URLs in a block) are dropped silently. Within each
    group, video_ids are deduplicated; cross-group duplicates are NOT removed
    (operator may legitimately want the same video in two courses).
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        is_sep = bool(s) and len(set(s)) == 1 and s[0] in "=-_*#"
        if is_sep:
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)

    groups: list[list[str]] = []
    for block in blocks:
        vids = _parse_youtube_urls("\n".join(block))
        if vids:
            groups.append(vids)
    return groups

log = logging.getLogger("gateway")

# Form step identifiers
STEP_ASK_TOPIC = "ask_topic"
STEP_ASK_PAIN = "ask_pain"          # specific customer pain the course solves
STEP_ASK_AUDIENCE = "ask_audience"  # who the course is for
STEP_ASK_COUNT = "ask_count"
STEP_CLARIFY = "clarify"            # awaiting operator's free-form answer to clarify_questions
STEP_CONFIRM = "confirm"
STEP_PHASE1_RUNNING = "phase1_running"
STEP_AWAITING_APPROVAL = "awaiting_approval"
STEP_AWAITING_COOKIES_PRE_PHASE1 = "awaiting_cookies_pre_phase1"  # F1: gate before Phase 1 launch
STEP_AWAITING_COOKIES_PRE_PHASE2 = "awaiting_cookies_pre_phase2"  # forced refresh before phase2
STEP_PHASE2_RUNNING = "phase2_running"
STEP_DONE = "done"
STEP_AWAITING_COOKIES = "awaiting_cookies"  # standalone cookies upload (from /menu)
STEP_ASK_VOICE = "ask_voice"              # LEGACY: gender selection before Phase 1 launch
STEP_ASK_VOICE_PHASE2 = "ask_voice_phase2"  # gender selection right before Phase 2 launch

# Inputs that mean "no answer" for optional pain/audience steps.
_SKIP_TOKENS = {"-", "—", "skip", "/skip", "пропустить", "нет", "no"}

# F1: how fresh YouTube cookies must be for Phase 1 to launch without
# asking the operator to re-upload. Six hours covers the common case of
# "I uploaded cookies this morning, want to launch this evening" while
# still catching truly stale cookies that would crash mid-Phase 1.
_COOKIES_MAX_AGE_SEC = 6 * 3600


def _cookies_fresh(cfg: dict[str, Any], max_age_sec: int = _COOKIES_MAX_AGE_SEC) -> bool:
    """True iff YouTube cookies file exists AND was modified within
    `max_age_sec` seconds. Used by the pre-Phase-1 gate so the operator
    refreshes cookies BEFORE Phase 1 launches rather than discovering
    staleness mid-batch through a proxy-pool-exhausted error.
    """
    onb = (cfg.get("onboarder") or {})
    path_str = onb.get("youtube_cookies_file") or "~/.secrets/youtube-cookies.txt"
    try:
        p = Path(path_str).expanduser()
    except (TypeError, ValueError):
        return False
    if not p.exists():
        return False
    try:
        return (time.time() - p.stat().st_mtime) <= max_age_sec
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Lifecycle entry points
# ---------------------------------------------------------------------------

def start_wizard(token: str, agent: str, cfg: dict, chat_id: int, user_id: int,
                 *, thread_id: int = 0) -> None:
    """Begin a new wizard run: clear previous state, ask first question."""
    _state.clear(agent, user_id, thread_id)
    _state.save(agent, user_id, {
        "step": STEP_ASK_TOPIC,
        "chat_id": chat_id,
        "thread_id": int(thread_id or 0),
    }, thread_id)
    _send(token, chat_id,
          "🎓 <b>Новый курс</b>\n\n"
          "Два режима:\n\n"
          "• <b>Тема</b> — напиши тему, бот сам найдёт YouTube-каналы и видео\n"
          "  Пример: <i>«AI для маркетологов»</i>\n\n"
          "• <b>Прямые ссылки</b> — вставь YouTube-ссылки через пробел или с новой строки, "
          "бот возьмёт именно эти видео и сам сгенерирует тему/описание\n"
          "  Пример: <i>https://youtu.be/abc123 https://youtu.be/def456</i>\n\n"
          "<i>/cancel — выход в чат с агентом.</i>",
          thread_id=thread_id)


def start_cookies_upload(token: str, agent: str, cfg: dict, chat_id: int, user_id: int,
                         *, thread_id: int = 0) -> None:
    """Enter the wizard 'awaiting cookies file' state.

    Preserves existing run state (sheet_tab, run_id, step) so that uploading
    cookies during an active Phase 1/Phase 2 run doesn't wipe the run context.
    """
    existing = _state.load(agent, user_id, thread_id)
    # Stash the previous step so we can restore it after upload
    _state.save(agent, user_id, {
        **existing,
        "_prev_step": existing.get("step"),
        "step": STEP_AWAITING_COOKIES,
        "chat_id": chat_id,
        "thread_id": int(thread_id or 0),
    }, thread_id)
    _send(token, chat_id,
          "📎 <b>Загрузка YouTube cookies</b>\n\n"
          "Отправь следующим сообщением файл <code>cookies.txt</code> "
          "(Netscape-format) — прикрепи его как документ.\n\n"
          "<b>Как получить файл:</b>\n"
          "1. Поставь расширение <i>Get cookies.txt LOCALLY</i> в Chrome/Edge\n"
          "2. Открой youtube.com (залогинен)\n"
          "3. Иконка расширения → Export As → cookies.txt\n"
          "4. Перетащи скачанный файл сюда\n\n"
          "<i>/cancel — отмена.</i>",
          thread_id=thread_id)


def clear_wizard_state(agent: str, user_id: int, thread_id: int = 0) -> None:
    _state.clear(agent, user_id, thread_id)


def handle_wizard_message(token: str, agent: str, cfg: dict, chat_id: int,
                          user_id: int, text: str, msg: dict,
                          *, thread_id: int = 0) -> None:
    """Route a non-command text message based on wizard step."""
    st = _state.load(agent, user_id, thread_id)
    if not st:
        # Mode is wizard but no state — recover gracefully.
        start_wizard(token, agent, cfg, chat_id, user_id, thread_id=thread_id)
        return

    step = st.get("step", STEP_ASK_TOPIC)

    if step in (STEP_AWAITING_COOKIES, STEP_AWAITING_COOKIES_PRE_PHASE2,
                STEP_AWAITING_COOKIES_PRE_PHASE1):
        _handle_cookies_upload(token, agent, cfg, chat_id, user_id, msg,
                               thread_id=thread_id)
        return

    if step == STEP_ASK_TOPIC:
        topic = text.strip()
        if not topic:
            _send(token, chat_id, "Тема не может быть пустой. Попробуй ещё раз.",
                  thread_id=thread_id)
            return

        # URL mode: user pasted YouTube links.
        # If they used separator lines (===, ---, ***, ###) → multi-course batch.
        url_groups = _parse_youtube_url_groups(topic)
        if url_groups:
            total = sum(len(g) for g in url_groups)
            if len(url_groups) == 1:
                # Single course — preserve the existing wizard state shape
                # (url_mode + video_ids) so legacy callbacks still work.
                video_ids = url_groups[0]
                _state.update(agent, user_id, thread_id=thread_id,
                              url_mode=True, video_ids=video_ids,
                              topic="", count=1, step=STEP_CONFIRM)
                url_list = "\n".join(f"  youtu.be/{v}" for v in video_ids[:8])
                more = f"\n  …и ещё {len(video_ids) - 8}" if len(video_ids) > 8 else ""
                _send_with_buttons(
                    token, chat_id,
                    f"🔗 <b>Режим прямых ссылок</b>\n\n"
                    f"Распознано видео: <b>{len(video_ids)}</b>\n"
                    f"<code>{url_list}{more}</code>\n\n"
                    f"Тема и описания курса будут сгенерированы автоматически "
                    f"по транскрибации видео. Запустить Phase 1?",
                    [[{"text": "🚀 Поехали", "callback_data": "wiz:start_phase1"},
                      {"text": "✖️ Отмена", "callback_data": "wiz:cancel"}]],
                    thread_id=thread_id,
                )
            else:
                # Multi-course batch
                _state.update(agent, user_id, thread_id=thread_id,
                              url_mode=True, url_groups=url_groups,
                              video_ids=[], topic="",
                              count=len(url_groups), step=STEP_CONFIRM)
                summary = "\n".join(
                    f"  Курс {i+1}: <b>{len(g)}</b> видео ({', '.join('youtu.be/' + v for v in g[:3])}{', …' if len(g) > 3 else ''})"
                    for i, g in enumerate(url_groups)
                )
                _send_with_buttons(
                    token, chat_id,
                    f"🔗 <b>Пакетный режим — {len(url_groups)} курс(ов)</b>\n\n"
                    f"Всего видео: <b>{total}</b>\n\n"
                    f"{summary}\n\n"
                    f"Каждый курс обрабатывается отдельно, темы и описания "
                    f"генерируются автоматически по транскриптам. "
                    f"Sheet наполнится последовательно (~10-20 мин на курс). "
                    f"Запустить Phase 1 для всех {len(url_groups)} курсов?",
                    [[{"text": f"🚀 Поехали ({len(url_groups)} курсов)",
                       "callback_data": "wiz:start_phase1"},
                      {"text": "✖️ Отмена", "callback_data": "wiz:cancel"}]],
                    thread_id=thread_id,
                )
            return

        # Multi-topic batch: separator lines present in plain text → each block
        # is a topic with optional description. Skip the per-topic STEP_ASK_PAIN
        # since the description is already in each block.
        topic_groups = _parse_topic_groups(topic)
        if len(topic_groups) >= 2:
            # Persist topic_groups + reset any prior clarify state. Step is
            # NOT set to CONFIRM yet — we spawn a clarify round for topic 0,
            # which sets step=STEP_CLARIFY (or auto-advances if LLM returns
            # no questions). The final confirm screen is rendered by
            # _send_confirm_after_clarify() once all topics are done.
            _state.update(agent, user_id, thread_id=thread_id,
                          topic_groups=topic_groups,
                          topic="", pain="",
                          count=len(topic_groups),
                          clarify_questions_per_topic=[],
                          clarify_answers=[])
            summary = "\n".join(
                f"  Курс {i+1}: <b>{_html_escape(tg['topic'])}</b>"
                + (f"\n    <i>{_html_escape(tg['pain'][:120])}{'…' if len(tg['pain']) > 120 else ''}</i>"
                   if tg.get('pain') else "")
                for i, tg in enumerate(topic_groups)
            )
            _send(token, chat_id,
                  f"📦 <b>Пакетный режим — {len(topic_groups)} тем(ы)</b>\n\n"
                  f"{summary}\n\n"
                  f"Сейчас я задам по каждой теме 2-4 уточняющих вопроса — "
                  f"они помогут точнее отобрать каналы.\n\n"
                  f"🤔 Тема 1/{len(topic_groups)}: подбираю вопросы…",
                  thread_id=thread_id)
            _spawn_clarify_round(
                token, agent, cfg, chat_id, user_id,
                thread_id=thread_id,
                topic=topic_groups[0].get("topic", ""),
                description=topic_groups[0].get("pain", ""),
                topic_index=0, total_topics=len(topic_groups),
            )
            return

        # Normal topic flow: ask for description
        _state.update(agent, user_id, thread_id=thread_id,
                      topic=topic, step=STEP_ASK_PAIN)
        _send(token, chat_id,
              f"Тема: <b>{_html_escape(topic)}</b>\n\n"
              "Опиши <b>что должно быть в этом курсе</b>: какие темы покрыть, "
              "для кого, какую боль/задачу решает, какой результат у студента в конце.\n\n"
              "Чем подробнее — тем точнее Claude отберёт каналы и видео.\n\n"
              "Например: «Восстановление коленного сустава после артроскопии: "
              "анатомия, упражнения по неделям 1-12, ошибки в реабилитации, "
              "когда возвращаться к спорту. Для людей 35-55 после операции.»\n\n"
              "<i>Пропустить — отправь <code>-</code> или <code>/skip</code>.</i>",
              thread_id=thread_id)
        return

    if step == STEP_ASK_PAIN:
        desc_raw = text.strip()
        description = "" if desc_raw.lower() in _SKIP_TOKENS else desc_raw
        # Persist as `pain` (legacy field name; LLM helpers expect this kwarg).
        # Single-topic mode: try to find UP TO 5 channels (and thus 5 courses)
        # matching the topic. If criteria match fewer, Phase 1 stops at what
        # it found (won't fail unless 0 succeed). To request exactly one
        # course per topic, use the multi-topic batch (===) format.
        # Step is NOT set to CONFIRM here — first we run a clarify round.
        # _spawn_clarify_round will either set step=STEP_CLARIFY (with
        # questions) or auto-advance to STEP_CONFIRM (when LLM had nothing
        # worth asking).
        st = _state.update(agent, user_id, thread_id=thread_id,
                           pain=description, count=5,
                           clarify_questions_per_topic=[],
                           clarify_answers=[])
        topic = st.get("topic", "")
        _send(token, chat_id,
              "🤔 Подбираю уточняющие вопросы (5-15 сек)…",
              thread_id=thread_id)
        _spawn_clarify_round(
            token, agent, cfg, chat_id, user_id,
            thread_id=thread_id,
            topic=topic, description=description,
            topic_index=0, total_topics=1,
        )
        return

    if step == STEP_CLARIFY:
        answer = text.strip()
        if not answer:
            _send(token, chat_id,
                  "Пустой ответ. Напиши что-то одним сообщением, "
                  "или нажми «⏭ Без уточнений» чтобы пропустить.",
                  thread_id=thread_id)
            return

        st = _state.load(agent, user_id, thread_id)
        answers = list(st.get("clarify_answers") or [])
        topic_groups = list(st.get("topic_groups") or [])
        total = len(topic_groups) if topic_groups else 1

        # The current round's index is whichever slot we're about to fill.
        # `_spawn_clarify_round` only appends an "" answer for AUTO-SKIPPED
        # rounds (LLM returned no questions). For rounds that asked the user,
        # `answers` is still at the pre-round length.
        current_index = len(answers)
        # Defensive: if somehow we lost track, clamp.
        if current_index >= total:
            current_index = total - 1
        answers.append(answer)

        _state.update(agent, user_id, thread_id=thread_id,
                      clarify_answers=answers)

        _advance_after_clarify(
            token, agent, cfg, chat_id, user_id,
            thread_id=thread_id,
            just_finished_index=current_index,
            total_topics=total,
        )
        return

    if step == STEP_ASK_AUDIENCE:
        # Legacy fallback: mid-flight wizards may still sit on this removed
        # step. Skip it silently and route to confirm so the user isn't stuck.
        log.info(f"[{agent}] wizard advancing legacy {step!r} → confirm")
        _state.update(agent, user_id, thread_id=thread_id,
                      count=1, step=STEP_CONFIRM)
        topic = _state.load(agent, user_id, thread_id).get("topic", "")
        _send_with_buttons(
            token, chat_id,
            f"<b>Подтверждение:</b>\n\nТема: <b>{_html_escape(topic)}</b>\n\n"
            f"Запустить Phase 1?",
            [[{"text": "🚀 Поехали", "callback_data": "wiz:start_phase1"},
              {"text": "✖️ Отмена", "callback_data": "wiz:cancel"}]],
            thread_id=thread_id,
        )
        return

    if step == STEP_ASK_COUNT:
        # Legacy fallback: ignore the count entry, force 1, route to confirm.
        log.info(f"[{agent}] wizard collapsing legacy STEP_ASK_COUNT → confirm")
        _state.update(agent, user_id, thread_id=thread_id,
                      count=1, step=STEP_CONFIRM)
        topic = _state.load(agent, user_id, thread_id).get("topic", "")
        _send_with_buttons(
            token, chat_id,
            f"<b>Подтверждение:</b>\n\nТема: <b>{_html_escape(topic)}</b>\n\n"
            f"Запустить Phase 1?",
            [[{"text": "🚀 Поехали", "callback_data": "wiz:start_phase1"},
              {"text": "✖️ Отмена", "callback_data": "wiz:cancel"}]],
            thread_id=thread_id,
        )
        return

    if step in (STEP_PHASE1_RUNNING, STEP_PHASE2_RUNNING):
        _send(token, chat_id,
              f"⏳ Сейчас идёт фоновая обработка ({step}). "
              f"Сообщения в чате с агентом доступны параллельно через /menu → 💬 Чат с агентом.",
              thread_id=thread_id)
        return

    if step == STEP_AWAITING_APPROVAL:
        _send(token, chat_id,
              "Жду подтверждения подборки в Sheet → нажми кнопку «Запустить обработку» под последним сообщением, "
              "или /cancel для выхода.",
              thread_id=thread_id)
        return

    # Unknown step: reset
    log.warning(f"[{agent}] wizard unknown step={step!r}, restarting")
    start_wizard(token, agent, cfg, chat_id, user_id, thread_id=thread_id)


# ---------------------------------------------------------------------------
# Callback handlers (inline buttons)
# ---------------------------------------------------------------------------

def register_callbacks(register_func: Callable[[str, Any], None]) -> None:
    """Hook into gateway's callback dispatcher. Called once at startup."""
    register_func("wiz:", _wizard_callback_handler)


def _wizard_callback_handler(token: str, agent: str, cfg: dict, cq: dict) -> None:
    """Dispatch wiz:* callbacks. Imported lazily by gateway.dispatch_callback_query."""
    from gateway import answer_callback_query, set_user_mode, MODE_CHAT, tg_api  # type: ignore

    cq_id = cq.get("id", "")
    data = cq.get("data", "")
    msg = cq.get("message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    user_id = (cq.get("from") or {}).get("id")
    # Topic the button was clicked in (forum support). 0 = main thread / DM.
    thread_id = int(msg.get("message_thread_id") or 0)
    if chat_id is None or user_id is None:
        answer_callback_query(token, cq_id)
        return

    action = data.split(":", 1)[1] if ":" in data else ""

    if action == "cancel":
        clear_wizard_state(agent, user_id, thread_id)
        set_user_mode(agent, user_id, MODE_CHAT, thread_id)
        answer_callback_query(token, cq_id, "Отменено")
        try:
            tg_api(token, "sendMessage", chat_id=chat_id,
                   message_thread_id=thread_id or None,
                   text="<b>Отменено.</b> Обратно в чат с агентом.",
                   parse_mode="HTML")
        except Exception:
            pass
        return

    if action == "skip_clarify":
        # Operator chose to skip all (remaining) clarifications and jump
        # straight to confirm. Pad the clarify_answers list with empty
        # strings for every topic that didn't get an answer yet, so the
        # confirm screen and start_phase1 handler see a consistent shape.
        st = _state.load(agent, user_id, thread_id)
        topic_groups = list(st.get("topic_groups") or [])
        answers = list(st.get("clarify_answers") or [])
        qpt = list(st.get("clarify_questions_per_topic") or [])
        total = len(topic_groups) if topic_groups else 1
        while len(answers) < total:
            answers.append("")
        while len(qpt) < total:
            qpt.append([])
        _state.update(agent, user_id, thread_id=thread_id,
                      clarify_answers=answers,
                      clarify_questions_per_topic=qpt)
        answer_callback_query(token, cq_id, "Без уточнений")
        _send_confirm_after_clarify(token, agent, chat_id, user_id, thread_id)
        return

    if action == "start_phase1":
        st = _state.load(agent, user_id, thread_id)
        url_mode = bool(st.get("url_mode"))
        url_groups = list(st.get("url_groups") or [])
        topic_groups = list(st.get("topic_groups") or [])
        video_ids = list(st.get("video_ids") or [])
        topic = st.get("topic") or ""
        description = st.get("pain", "")  # stored under `pain` for back-compat
        # Clarification arrays (filled during the clarify rounds, parallel
        # to topic_groups for multi or length-1 for single topic). Used
        # below to enrich `pain` before launching Phase 1.
        clarify_answers = list(st.get("clarify_answers") or [])
        clarify_questions_per_topic = list(
            st.get("clarify_questions_per_topic") or [])
        if url_groups:
            count = len(url_groups)
        elif topic_groups:
            count = len(topic_groups)
        else:
            # Single-topic / single-URL path: respect what STEP_ASK_PAIN
            # (count=5) or the URL handler (count=1) saved into state.
            # Falling back to a hardcoded 1 here used to silently clobber
            # the "up to 5 courses per topic" intent.
            count = int(st.get("count") or 1)

        has_input = (topic
                     or (url_mode and (video_ids or url_groups))
                     or topic_groups)
        if not has_input:
            answer_callback_query(token, cq_id, "Состояние формы потеряно", show_alert=True)
            clear_wizard_state(agent, user_id, thread_id)
            return

        # Build the canonical (kind, args) for Phase 1 launch. Done BEFORE
        # the cookies-freshness gate so we can stash these into state and
        # resume on cookies upload without recomputing clarification merges.
        kind, args = _build_phase1_launch_args(
            url_mode=url_mode, url_groups=url_groups, video_ids=video_ids,
            topic_groups=topic_groups, topic=topic, description=description,
            count=count,
            clarify_answers=clarify_answers,
            clarify_questions_per_topic=clarify_questions_per_topic,
        )

        # F1: cookies-freshness gate. If mtime > 6h or missing → ask the
        # operator to re-upload before Phase 1 launches. Avoids the
        # mid-run "all 20 proxies blocked, please upload cookies" cascade
        # we saw before this feature. After upload, Case D in
        # _handle_cookies_upload resumes with the same (kind, args).
        if not _cookies_fresh(cfg):
            _state.update(agent, user_id, thread_id=thread_id,
                          count=count,
                          step=STEP_AWAITING_COOKIES_PRE_PHASE1,
                          _pending_launch_kind=kind,
                          _pending_launch_args=args)
            answer_callback_query(token, cq_id, "Сначала свежие cookies")
            _send_with_buttons(
                token, chat_id,
                text=(
                    "📎 <b>Cookies устарели или отсутствуют.</b>\n\n"
                    "Свежие YouTube cookies нужны ПЕРЕД Phase 1 (последний раз "
                    "сохранил &gt; 6 часов назад). Загрузи файл "
                    "<code>cookies.txt</code> следующим сообщением — Phase 1 "
                    "стартует сам, как только пришлёшь.\n\n"
                    "<b>Как получить файл:</b>\n"
                    "1. Поставь расширение <i>Get cookies.txt LOCALLY</i> в Chrome/Edge\n"
                    "2. Открой youtube.com (залогинен)\n"
                    "3. Иконка расширения → Export As → cookies.txt\n"
                    "4. Перетащи файл сюда"
                ),
                buttons=[[{"text": "✖️ Отмена", "callback_data": "wiz:cancel"}]],
                thread_id=thread_id,
            )
            return

        # Phase 1 doesn't need voice_gender. The dubbing choice happens at
        # Phase 2 launch — after the operator has seen the actual video set
        # in the Sheet and can judge whether dubbing is even needed.
        _state.update(agent, user_id, thread_id=thread_id,
                      count=count, step=STEP_PHASE1_RUNNING)
        answer_callback_query(token, cq_id, "Phase 1 запущен")

        _dispatch_phase1(token, agent, cfg, chat_id, user_id,
                         thread_id=thread_id, kind=kind, args=args)
        return

    if action in ("voice_male", "voice_female"):
        # Voice gender selection: now happens right before Phase 2 launch.
        # See skip_cookies (and the cookies upload handler) for how the user
        # gets here — both routes set step=STEP_ASK_VOICE_PHASE2 and prompt
        # for the voice.
        voice_gender = "MALE" if action == "voice_male" else "FEMALE"
        _state.update(agent, user_id, thread_id=thread_id,
                      voice_gender=voice_gender, step=STEP_PHASE2_RUNNING)
        answer_callback_query(token, cq_id, "Запускаю Phase 2")
        voice_label = "👨 мужской" if voice_gender == "MALE" else "👩 женский"
        try:
            tg_api(token, "sendMessage", chat_id=chat_id,
                   message_thread_id=thread_id or None,
                   text=(
                       f"🎬 <b>Phase 2 запущен.</b>\n\n"
                       f"Голос дубляжа: {voice_label} "
                       f"(используется только для не-английских видео)"
                   ),
                   parse_mode="HTML")
        except Exception:
            pass
        try:
            from . import phase2_production
            phase2_production.launch(token, agent, cfg, chat_id, user_id,
                                     thread_id=thread_id)
        except Exception as e:
            log.exception(f"[{agent}] failed to launch phase2: {e}")
            try:
                tg_api(token, "sendMessage", chat_id=chat_id,
                       message_thread_id=thread_id or None,
                       text=f"⚠️ Не удалось запустить Phase 2: {e}")
            except Exception:
                pass
            _state.update(agent, user_id, thread_id=thread_id, step="error", error=str(e))
        return

    # Per-course Phase 2 launch button. Triggered from Phase 1's per-course
    # «🚀 Запустить Курс N» message. Format: wiz:p2c:{run_id}:{course_idx}
    if action.startswith("p2c:"):
        parts = action.split(":")
        if len(parts) != 3:
            answer_callback_query(token, cq_id, "Битый callback", show_alert=True)
            return
        run_id = parts[1]
        try:
            course_idx = int(parts[2])
        except ValueError:
            answer_callback_query(token, cq_id, "Битый course_idx", show_alert=True)
            return
        answer_callback_query(token, cq_id, f"Курс {course_idx}: выбор голоса")
        # Strip the button off the Phase 1 message so a second click doesn't
        # spawn another worker for the same course.
        message_id = msg.get("message_id")
        if message_id:
            try:
                tg_api(token, "editMessageReplyMarkup",
                       chat_id=chat_id, message_id=message_id,
                       reply_markup={"inline_keyboard": []})
            except Exception as e:
                log.warning(f"p2c: could not strip button: {e}")
        # Ask voice (per course, per user preference).
        try:
            tg_api(token, "sendMessage", chat_id=chat_id,
                   message_thread_id=thread_id or None,
                   text=(
                       f"🎙 <b>Курс {course_idx}: голос дубляжа?</b>\n\n"
                       f"<i>(применяется только к не-английским видео)</i>"
                   ),
                   parse_mode="HTML",
                   reply_markup={"inline_keyboard": [[
                       {"text": "👨 Мужской",
                        "callback_data": f"wiz:p2cv:{run_id}:{course_idx}:MALE"},
                       {"text": "👩 Женский",
                        "callback_data": f"wiz:p2cv:{run_id}:{course_idx}:FEMALE"},
                   ]]})
        except Exception as e:
            log.exception(f"[{agent}] p2c: failed to ask voice: {e}")
        return

    # Voice picked for a per-course Phase 2 launch.
    # Format: wiz:p2cv:{run_id}:{course_idx}:{MALE|FEMALE}
    if action.startswith("p2cv:"):
        parts = action.split(":")
        if len(parts) != 4:
            answer_callback_query(token, cq_id, "Битый callback", show_alert=True)
            return
        run_id = parts[1]
        try:
            course_idx = int(parts[2])
        except ValueError:
            answer_callback_query(token, cq_id, "Битый course_idx", show_alert=True)
            return
        voice_gender = parts[3].upper()
        if voice_gender not in ("MALE", "FEMALE"):
            voice_gender = "MALE"
        voice_label = "👨 мужской" if voice_gender == "MALE" else "👩 женский"
        answer_callback_query(token, cq_id, f"Курс {course_idx} запущен")
        # Strip the voice picker buttons.
        message_id = msg.get("message_id")
        if message_id:
            try:
                tg_api(token, "editMessageText",
                       chat_id=chat_id, message_id=message_id,
                       text=(
                           f"🎬 <b>Курс {course_idx} запущен.</b>\n"
                           f"Голос: {voice_label}"
                       ),
                       parse_mode="HTML")
            except Exception as e:
                log.warning(f"p2cv: could not update prompt: {e}")
        try:
            from . import phase2_production
            phase2_production.launch(
                token, agent, cfg, chat_id, user_id,
                thread_id=thread_id,
                run_id_override=run_id,
                course_idx_filter=course_idx,
                voice_gender_override=voice_gender,
            )
        except Exception as e:
            log.exception(f"[{agent}] p2cv: failed to launch phase2 "
                          f"(run={run_id} course={course_idx}): {e}")
            try:
                tg_api(token, "sendMessage", chat_id=chat_id,
                       message_thread_id=thread_id or None,
                       text=f"⚠️ Не удалось запустить Курс {course_idx}: {e}")
            except Exception:
                pass
        return

    if action == "start_phase2":
        # Force a cookies refresh before Phase 2 — common failure mode is
        # forgetting to update YouTube cookies between runs.
        _state.update(agent, user_id, thread_id=thread_id,
                      step=STEP_AWAITING_COOKIES_PRE_PHASE2)
        answer_callback_query(token, cq_id, "Сначала обнови cookies")
        _send_with_buttons(
            token, chat_id,
            text=(
                "🍪 <b>Обнови YouTube cookies перед запуском.</b>\n\n"
                "Cookies живут 2-4 недели — лучше залить свежие, чтобы Phase 2 "
                "не упал в середине.\n\n"
                "Открой <b>youtube.com</b> в Chrome (залогинен) → расширение "
                "<i>Get cookies.txt LOCALLY</i> → <b>Export All</b> → прикрепи файл сюда.\n\n"
                "Если уверен что cookies свежие — жми «Пропустить»."
            ),
            buttons=[[
                {"text": "⏭ Пропустить (cookies свежие)", "callback_data": "wiz:skip_cookies"},
                {"text": "✖️ Отмена", "callback_data": "wiz:cancel"},
            ]],
            thread_id=thread_id,
        )
        return

    if action == "skip_cookies":
        # User asserts cookies are fresh — proceed to voice question (instead
        # of launching Phase 2 directly). Voice is only used for non-English
        # videos; if none of the approved videos are non-EN we still ask once
        # for simplicity (the answer is just ignored downstream).
        _state.update(agent, user_id, thread_id=thread_id,
                      step=STEP_ASK_VOICE_PHASE2)
        answer_callback_query(token, cq_id)
        _send_with_buttons(
            token, chat_id,
            text=(
                "🎙 <b>Какой голос использовать для дубляжа на английский?</b>\n\n"
                "(применяется только к видео, которые не на английском — "
                "английские пройдут без озвучки)"
            ),
            buttons=[[
                {"text": "👨 Мужской", "callback_data": "wiz:voice_male"},
                {"text": "👩 Женский", "callback_data": "wiz:voice_female"},
            ]],
            thread_id=thread_id,
        )
        return

    answer_callback_query(token, cq_id)


# ---------------------------------------------------------------------------
# Helpers (use gateway's send functions to honor HTML chunking, retries, etc.)
# ---------------------------------------------------------------------------

def _send(token: str, chat_id: int, text: str, *, thread_id: int = 0) -> None:
    from gateway import tg_api  # type: ignore
    kwargs: dict[str, Any] = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    if thread_id:
        kwargs["message_thread_id"] = int(thread_id)
    try:
        tg_api(token, "sendMessage", **kwargs)
    except Exception as e:
        log.warning(f"wizard _send failed: {e}")


def _send_with_buttons(token: str, chat_id: int, text: str,
                       buttons: list[list[dict[str, str]]],
                       *, thread_id: int = 0) -> None:
    from gateway import send_message_with_buttons  # type: ignore
    send_message_with_buttons(token, chat_id, text, buttons,
                              message_thread_id=int(thread_id or 0))


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


# ---------------------------------------------------------------------------
# Clarification round (between description and confirm)
# ---------------------------------------------------------------------------
#
# State shape (set when we enter the clarify step):
#
#   step                            = STEP_CLARIFY
#   clarify_questions_per_topic     = list[list[str]] — parallel to
#                                     topic_groups (or length 1 for single
#                                     topic). [] for topics that the LLM
#                                     decided needed no clarification.
#   clarify_answers                 = list[str] — parallel to above. "" for
#                                     skipped / no-questions topics.
#
# Both lists grow incrementally during multi-topic batches: we generate
# questions for one topic, collect the answer, then generate for the next.

def _spawn_clarify_round(token: str, agent: str, cfg: dict,
                         chat_id: int, user_id: int, *,
                         thread_id: int, topic: str, description: str,
                         topic_index: int, total_topics: int) -> None:
    """Generate clarify questions for `topic` in a daemon thread, then either:
      * if LLM produced questions → set step=STEP_CLARIFY and message the
        operator with the questions + a "Без уточнений" button, OR
      * if LLM returned [] (description already exhaustive) → record an empty
        answer for this topic and either auto-advance to the next topic in a
        batch, or jump to STEP_CONFIRM for single-topic flows.

    Runs in a background thread so the Telegram polling loop isn't blocked
    by the 5-15s Claude CLI call. Multiple concurrent clarify rounds for the
    same (user, thread) are NOT expected (one wizard per topic-id) so no
    locking is needed beyond the state file's own atomic-replace.
    """
    def _bg() -> None:
        try:
            questions = _clarify.generate_questions(
                topic=topic, description=description)
        except Exception as e:
            log.warning(f"wizard: clarify generation crashed: {e}")
            questions = []

        st = _state.load(agent, user_id, thread_id)
        qpt = list(st.get("clarify_questions_per_topic") or [])
        answers = list(st.get("clarify_answers") or [])
        # Pad if we somehow skipped indexes (shouldn't happen, but be safe)
        while len(qpt) < topic_index:
            qpt.append([])
        while len(answers) < topic_index:
            answers.append("")

        # Append THIS topic's questions (may be empty list).
        if len(qpt) == topic_index:
            qpt.append(questions)
        else:
            qpt[topic_index] = questions

        # If LLM had no questions for this topic — auto-fill empty answer
        # and advance. The operator never sees this round.
        if not questions:
            if len(answers) == topic_index:
                answers.append("")
            else:
                answers[topic_index] = ""
            _state.update(agent, user_id, thread_id=thread_id,
                          clarify_questions_per_topic=qpt,
                          clarify_answers=answers)
            _advance_after_clarify(token, agent, cfg, chat_id, user_id,
                                   thread_id=thread_id,
                                   just_finished_index=topic_index,
                                   total_topics=total_topics)
            return

        # We have questions — show them and wait for operator's text reply.
        _state.update(agent, user_id, thread_id=thread_id,
                      step=STEP_CLARIFY,
                      clarify_questions_per_topic=qpt,
                      clarify_answers=answers)

        if total_topics > 1:
            header = (f"💡 <b>Тема {topic_index + 1}/{total_topics}:</b> "
                      f"<b>{_html_escape(topic)}</b>\n\n"
                      f"Уточняющие вопросы:\n")
        else:
            header = "💡 <b>Уточняющие вопросы перед запуском</b>\n\n"

        qs_text = "\n".join(
            f"{i + 1}. {_html_escape(q)}" for i, q in enumerate(questions))

        msg = (
            header + qs_text + "\n\n"
            "<i>Ответь одним сообщением — свободным текстом по всем вопросам "
            "сразу. Или нажми «Без уточнений» чтобы пропустить.</i>"
        )

        _send_with_buttons(
            token, chat_id, msg,
            [[{"text": "⏭ Без уточнений", "callback_data": "wiz:skip_clarify"},
              {"text": "✖️ Отмена", "callback_data": "wiz:cancel"}]],
            thread_id=thread_id,
        )

    threading.Thread(
        target=_bg, daemon=True,
        name=f"clarify-{agent}-{user_id}-{topic_index}",
    ).start()


def _advance_after_clarify(token: str, agent: str, cfg: dict,
                            chat_id: int, user_id: int, *,
                            thread_id: int, just_finished_index: int,
                            total_topics: int) -> None:
    """Decide what to do after a clarify round ended (either answered or
    auto-skipped because LLM produced no questions).

    If there are more topics in a multi-batch → kick off the next round.
    Otherwise → show the final confirmation screen.
    """
    next_index = just_finished_index + 1
    if next_index < total_topics:
        # More topics to clarify
        st = _state.load(agent, user_id, thread_id)
        topic_groups = list(st.get("topic_groups") or [])
        next_tg = topic_groups[next_index] if next_index < len(topic_groups) else {}
        _send(token, chat_id,
              f"🤔 Тема {next_index + 1}/{total_topics}: подбираю вопросы…",
              thread_id=thread_id)
        _spawn_clarify_round(
            token, agent, cfg, chat_id, user_id,
            thread_id=thread_id,
            topic=next_tg.get("topic", ""),
            description=next_tg.get("pain", ""),
            topic_index=next_index, total_topics=total_topics,
        )
        return

    # All clarify rounds done → show confirm screen
    _send_confirm_after_clarify(token, agent, chat_id, user_id, thread_id)


def _send_confirm_after_clarify(token: str, agent: str,
                                 chat_id: int, user_id: int,
                                 thread_id: int) -> None:
    """Final confirmation screen after all clarify rounds completed.

    Sets step=STEP_CONFIRM, shows topic(s), description(s), and any
    clarifications the operator gave. Buttons: 🚀 Поехали / ✖️ Отмена.
    """
    _state.update(agent, user_id, thread_id=thread_id, step=STEP_CONFIRM)
    st = _state.load(agent, user_id, thread_id)
    topic_groups = list(st.get("topic_groups") or [])
    answers = list(st.get("clarify_answers") or [])

    if topic_groups:
        # Multi-topic batch
        lines = []
        for i, tg in enumerate(topic_groups):
            t = _html_escape(tg.get("topic", "") or "(без темы)")
            ans = (answers[i] if i < len(answers) else "").strip()
            line = f"  Курс {i + 1}: <b>{t}</b>"
            if ans:
                snip = _html_escape(ans[:160])
                if len(ans) > 160:
                    snip += "…"
                line += f"\n    <i>уточнения: {snip}</i>"
            lines.append(line)
        msg = (
            f"📦 <b>Пакетный режим — {len(topic_groups)} тем(ы)</b>\n\n"
            + "\n".join(lines) + "\n\n"
            f"Запустить Phase 1 для всех {len(topic_groups)} курсов?"
        )
        _send_with_buttons(
            token, chat_id, msg,
            [[{"text": f"🚀 Поехали ({len(topic_groups)} курсов)",
               "callback_data": "wiz:start_phase1"},
              {"text": "✖️ Отмена", "callback_data": "wiz:cancel"}]],
            thread_id=thread_id,
        )
        return

    # Single topic
    topic = st.get("topic", "")
    description = st.get("pain", "")
    desc_line = (_html_escape(description) if description
                 else "<i>(не указано)</i>")
    ans = (answers[0] if answers else "").strip()
    clar_block = ""
    if ans:
        snip = _html_escape(ans[:300])
        if len(ans) > 300:
            snip += "…"
        clar_block = f"\nУточнения: <i>{snip}</i>"

    _send_with_buttons(
        token, chat_id,
        f"<b>Подтверждение:</b>\n\n"
        f"Тема: <b>{_html_escape(topic)}</b>\n"
        f"Описание курса: {desc_line}{clar_block}\n\n"
        f"Запустить Phase 1? Соберу <b>до 5 курсов</b> на эту тему "
        f"(каждый — отдельный канал, прошедший критерии Sheet).",
        [[{"text": "🚀 Поехали", "callback_data": "wiz:start_phase1"},
          {"text": "✖️ Отмена", "callback_data": "wiz:cancel"}]],
        thread_id=thread_id,
    )


def _combine_pain_with_clarification(pain: str, questions: list[str],
                                      answer: str) -> str:
    """Concatenate the operator's free-form clarification answer onto the
    end of `pain` so downstream prompts (discovery, scoring, compose) pick
    it up automatically via the existing OPERATOR-SPECIFIED COURSE DIRECTION
    block in llm._pain_audience_block().
    """
    extra = _clarify.format_for_pain(questions, answer)
    if not extra:
        return pain
    return (pain or "") + extra


# ---------------------------------------------------------------------------
# Phase 1 launch helpers (F1: cookies-gate + resume after upload)
# ---------------------------------------------------------------------------
#
# Used by both `wiz:start_phase1` (direct launch with fresh cookies) and
# `_handle_cookies_upload` Case D (resume after operator uploaded cookies).
# Splitting the args-building from the preamble/dispatch lets us stash the
# canonical (kind, args) tuple into wizard state between the two events.

def _build_phase1_launch_args(*, url_mode: bool, url_groups: list,
                               video_ids: list, topic_groups: list,
                               topic: str, description: str, count: int,
                               clarify_answers: list,
                               clarify_questions_per_topic: list,
                              ) -> tuple[str, dict[str, Any]]:
    """Decide which Phase 1 launch kind applies + bundle its args.

    Returns (kind, args) where kind is one of:
      "topic_batch" — multi-topic batch (===-separated). args has
                      `topic_groups` (with each topic's pain enriched).
      "url_groups"  — multi-course URL batch. args has `url_groups`.
      "url_single"  — single-URL list. args has `video_ids`.
      "topic_single" — single topic. args has `topic`, `pain` (enriched), `count`.
    """
    if topic_groups and not url_mode:
        # Multi-topic batch: enrich each topic's pain with its clarification
        enriched: list[dict[str, str]] = []
        for i, tg in enumerate(topic_groups):
            base_pain = tg.get("pain", "") or ""
            ans = clarify_answers[i] if i < len(clarify_answers) else ""
            qs = (clarify_questions_per_topic[i]
                  if i < len(clarify_questions_per_topic) else [])
            new_pain = _combine_pain_with_clarification(base_pain, qs, ans)
            enriched.append({**tg, "pain": new_pain})
        return ("topic_batch", {"topic_groups": enriched})

    if url_mode and url_groups:
        return ("url_groups", {"url_groups": url_groups})

    if url_mode and video_ids:
        return ("url_single", {"video_ids": video_ids})

    # Single-topic: enrich the single-pain with the single clarification
    single_ans = clarify_answers[0] if clarify_answers else ""
    single_qs = (clarify_questions_per_topic[0]
                 if clarify_questions_per_topic else [])
    enriched_pain = _combine_pain_with_clarification(
        description, single_qs, single_ans)
    return ("topic_single",
            {"topic": topic, "pain": enriched_pain, "count": int(count)})


def _dispatch_phase1(token: str, agent: str, cfg: dict, chat_id: int,
                     user_id: int, *, thread_id: int,
                     kind: str, args: dict[str, Any]) -> None:
    """Send Phase 1 preamble message + spawn the discovery worker.

    Mirrors the 4 launch kinds from the wizard's start_phase1 handler.
    Centralised here so the cookies-gate resume path (Case D in
    _handle_cookies_upload) can reuse the same preamble + dispatch.

    On launch exception: sets state.step=error, sends fail message.
    """
    from gateway import tg_api  # type: ignore
    from . import phase1_discovery

    def _send_text(text: str) -> None:
        try:
            tg_api(token, "sendMessage", chat_id=chat_id,
                   message_thread_id=thread_id or None,
                   text=text, parse_mode="HTML")
        except Exception:
            pass

    if kind == "topic_batch":
        topic_groups = args.get("topic_groups") or []
        _send_text(
            f"🔍 <b>Phase 1 запущен — пакет из {len(topic_groups)} тем.</b>\n\n"
            "Каждая тема пройдёт полную Phase 1: поиск YouTube-каналов, "
            "скоринг через Claude, отбор видео, скачивание, "
            "транскрибация, compose. Курсы пишутся в Sheet по ходу.\n\n"
            f"Примерно <b>~{20 * len(topic_groups)}-{40 * len(topic_groups)} мин</b> "
            "на весь пакет. Можешь вернуться в чат "
            "(<code>/menu</code> → 💬 Чат), пришлю одно итоговое "
            "сообщение когда всё будет готово."
        )
        try:
            phase1_discovery.launch_topic_batch(
                token, agent, cfg, chat_id, user_id,
                topic_groups=topic_groups, thread_id=thread_id,
            )
        except Exception as e:
            log.exception(f"[{agent}] failed to launch phase1_topic_batch: {e}")
            _send_text(f"⚠️ Не удалось запустить Phase 1: {e}")
            _state.update(agent, user_id, thread_id=thread_id, step="error", error=str(e))
        return

    if kind == "url_groups":
        url_groups = args.get("url_groups") or []
        total = sum(len(g) for g in url_groups)
        _send_text(
            f"🔗 <b>Phase 1 запущен — пакет из {len(url_groups)} курсов.</b>\n\n"
            f"Всего видео: <b>{total}</b>\n\n"
            "Курсы обрабатываются последовательно. Каждый: скачивание, "
            "Whisper, разметка вырезок, исследование автора, compose. "
            "Тему и названия генерирую по транскриптам.\n\n"
            f"Примерно <b>~{15 * len(url_groups)}-{30 * len(url_groups)} мин</b> на весь пакет. "
            "Можешь вернуться в чат (<code>/menu</code> → 💬 Чат), "
            "пришлю одно итоговое сообщение когда всё будет готово."
        )
        try:
            phase1_discovery.launch_from_url_groups(
                token, agent, cfg, chat_id, user_id,
                url_groups=url_groups, thread_id=thread_id,
            )
        except Exception as e:
            log.exception(f"[{agent}] failed to launch phase1_from_url_groups: {e}")
            _send_text(f"⚠️ Не удалось запустить Phase 1: {e}")
            _state.update(agent, user_id, thread_id=thread_id, step="error", error=str(e))
        return

    if kind == "url_single":
        video_ids = args.get("video_ids") or []
        _send_text(
            "🔗 <b>Phase 1 запущен (прямые ссылки).</b>\n\n"
            f"Видео в обработке: <b>{len(video_ids)}</b>\n\n"
            "Скачиваю видео, транскрибирую (Whisper), размечаю вырезки, "
            "исследую автора и составляю описание курса. "
            "Тему и названия генерирую автоматически по транскрипту.\n\n"
            "~15-30 минут. Можешь вернуться в чат "
            "(<code>/menu</code> → 💬 Чат), пришлю результат как будет готово."
        )
        try:
            phase1_discovery.launch_from_urls(
                token, agent, cfg, chat_id, user_id,
                video_ids=video_ids, thread_id=thread_id,
            )
        except Exception as e:
            log.exception(f"[{agent}] failed to launch phase1_from_urls: {e}")
            _send_text(f"⚠️ Не удалось запустить Phase 1: {e}")
            _state.update(agent, user_id, thread_id=thread_id, step="error", error=str(e))
        return

    # topic_single (default fallback)
    topic = args.get("topic") or ""
    pain = args.get("pain") or ""
    count = int(args.get("count") or 5)
    desc_block = (f"\nОписание: <b>{_html_escape(pain[:300])}</b>" if pain else "")
    _send_text(
        "🔍 <b>Phase 1 запущен.</b>\n\n"
        f"Тема: <b>{_html_escape(topic)}</b>"
        f"{desc_block}\n\n"
        "Phase 1 ищет каналы (до 5), скачивает видео, "
        "транскрибирует (Whisper), размечает вырезки, пишет "
        "описания уроков и курса, ищет инфу об авторе через "
        "WebSearch. На выходе в Sheet будут реальные описания, "
        "готовые для лендинга.\n\n"
        "~20-45 минут на каждый курс (5 курсов = ~2-4 часа). "
        "Можешь вернуться в чат с агентом (<code>/menu</code> → "
        "💬 Чат), пришлю результат как будет готово."
    )
    try:
        phase1_discovery.launch(token, agent, cfg, chat_id, user_id,
                                topic, count, pain=pain, thread_id=thread_id)
    except Exception as e:
        log.exception(f"[{agent}] failed to launch phase1: {e}")
        _send_text(f"⚠️ Не удалось запустить Phase 1: {e}")
        _state.update(agent, user_id, thread_id=thread_id, step="error", error=str(e))


# ---------------------------------------------------------------------------
# Cookie upload handler
# ---------------------------------------------------------------------------

def _handle_cookies_upload(token: str, agent: str, cfg: dict,
                           chat_id: int, user_id: int, msg: dict,
                           *, thread_id: int = 0) -> None:
    """Save attached document as YouTube cookies.txt at the configured path."""
    from gateway import download_telegram_file, set_user_mode, MODE_CHAT  # type: ignore
    import os
    import shutil
    from pathlib import Path

    doc = msg.get("document")
    if not doc:
        _send(token, chat_id,
              "Жду файл прикреплением (как документ). Просто текст не подходит. "
              "Если передумал — /cancel.",
              thread_id=thread_id)
        return

    file_id = doc.get("file_id")
    file_name = doc.get("file_name") or "cookies.txt"
    if not file_id:
        _send(token, chat_id, "⚠️ Не удалось прочитать file_id. Попробуй ещё раз.",
              thread_id=thread_id)
        return

    onb = (cfg.get("onboarder") or {})
    target = onb.get("youtube_cookies_file") or "~/.secrets/youtube-cookies.txt"
    target_path = Path(target).expanduser()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    local = download_telegram_file(token, file_id, "document", file_name)
    if not local:
        _send(token, chat_id, "⚠️ Не смог скачать файл из Telegram (>20MB или ошибка сети).",
              thread_id=thread_id)
        return

    # Validate: Netscape cookies.txt starts with `# Netscape HTTP Cookie File`
    # or at least has tab-separated YouTube domain entries.
    try:
        head = local.read_text(errors="replace")[:4096]
    except Exception as e:
        _send(token, chat_id, f"⚠️ Не смог прочитать файл: {e}", thread_id=thread_id)
        return

    if "youtube.com" not in head and "Netscape" not in head:
        _send(token, chat_id,
              "⚠️ Файл не похож на Netscape cookies.txt от youtube.com "
              "(не нашёл ни 'Netscape', ни 'youtube.com' в первых 4KB). "
              "Перепроверь, что экспортировал именно с youtube.com.",
              thread_id=thread_id)
        return

    try:
        shutil.move(str(local), str(target_path))
        os.chmod(target_path, 0o600)
    except Exception as e:
        _send(token, chat_id, f"⚠️ Не смог сохранить файл в {target_path}: {e}",
              thread_id=thread_id)
        return

    size_kb = target_path.stat().st_size / 1024
    line_count = sum(1 for _ in target_path.open("r", errors="replace"))

    st = _state.load(agent, user_id, thread_id)
    current_step = st.get("step")
    prev_step = st.pop("_prev_step", None)

    # Case A: cookies were requested as part of the pre-Phase 2 gate.
    # Move to voice-selection step (voice is asked right before Phase 2 now
    # that we know the actual approved video set in the Sheet).
    if current_step == STEP_AWAITING_COOKIES_PRE_PHASE2:
        st["step"] = STEP_ASK_VOICE_PHASE2
        _state.save(agent, user_id, st, thread_id)
        _send_with_buttons(
            token, chat_id,
            text=(
                f"✅ <b>Cookies обновлены</b> ({size_kb:.1f} KB, {line_count} строк).\n\n"
                f"🎙 <b>Какой голос использовать для дубляжа?</b>\n"
                f"(применяется только к не-английским видео)"
            ),
            buttons=[[
                {"text": "👨 Мужской", "callback_data": "wiz:voice_male"},
                {"text": "👩 Женский", "callback_data": "wiz:voice_female"},
            ]],
            thread_id=thread_id,
        )
        return

    # Case D (F1): cookies were requested by the pre-Phase 1 gate. Auto-
    # resume the Phase 1 launch with the stashed (kind, args) — no return
    # to chat, no manual re-trigger needed. This is the explicit UX fix
    # for the 2026-05-26 incident where Phase 1 died after cookies error
    # and the operator didn't realise they had to re-run the wizard.
    if current_step == STEP_AWAITING_COOKIES_PRE_PHASE1:
        kind = st.pop("_pending_launch_kind", None)
        args = st.pop("_pending_launch_args", None) or {}
        if not kind:
            # Defensive: no stashed args (shouldn't happen if state went
            # through start_phase1). Fall back to clean-state behaviour.
            log.warning(f"PRE_PHASE1 cookies upload but no _pending_launch_kind in state")
            st["step"] = "ask_topic"
            _state.save(agent, user_id, st, thread_id)
            _send(token, chat_id,
                  f"✅ Cookies обновлены ({size_kb:.1f} KB).\n\n"
                  "Состояние wizard потеряно — открой <code>/menu</code> → 🎓 Новый курс заново.",
                  thread_id=thread_id)
            return
        st["step"] = STEP_PHASE1_RUNNING
        _state.save(agent, user_id, st, thread_id)
        _send(token, chat_id,
              f"✅ <b>Cookies обновлены</b> ({size_kb:.1f} KB, {line_count} строк).\n"
              "🚀 Запускаю Phase 1…",
              thread_id=thread_id)
        _dispatch_phase1(token, agent, cfg, chat_id, user_id,
                         thread_id=thread_id, kind=kind, args=args)
        return

    # Case B: standalone cookies upload during an active run (e.g. user is on
    # awaiting_approval and uploaded cookies via /menu) → restore prev step
    if prev_step and prev_step not in (STEP_AWAITING_COOKIES, STEP_DONE, "error", ""):
        st["step"] = prev_step
        _state.save(agent, user_id, st, thread_id)
        confirm_text = f"✅ <b>Cookies обновлены</b> ({size_kb:.1f} KB, {line_count} строк).\n\n"
        if prev_step == STEP_AWAITING_APPROVAL:
            confirm_text += "Подборка в Sheet всё ещё ждёт. Нажми кнопку:"
            _send_with_buttons(token, chat_id, confirm_text,
                               [[{"text": "🚀 Запустить обработку", "callback_data": "wiz:start_phase2"},
                                 {"text": "✖️ Отмена", "callback_data": "wiz:cancel"}]],
                               thread_id=thread_id)
        else:
            _send(token, chat_id, confirm_text + f"Продолжаю с шага: <code>{prev_step}</code>.",
                  thread_id=thread_id)
        return

    # Case C: standalone upload from /menu without active run → return to chat
    clear_wizard_state(agent, user_id, thread_id)
    set_user_mode(agent, user_id, MODE_CHAT, thread_id)
    _send(token, chat_id,
          f"✅ <b>Cookies сохранены</b> ({size_kb:.1f} KB, {line_count} строк).\n\n"
          f"Следующий запуск Phase 2 будет использовать этот файл.\n"
          f"Возвращаюсь в чат с агентом.",
          thread_id=thread_id)
