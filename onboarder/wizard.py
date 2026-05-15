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
from typing import Any, Callable

from . import state as _state

# Matches youtube.com/watch?v=ID and youtu.be/ID (with or without https://).
# Handles URLs separated by spaces, newlines, or multiple spaces.
_YOUTUBE_URL_RE = re.compile(
    r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?[^\s]*?v=|youtu\.be/)([A-Za-z0-9_-]{11})'
)


def _parse_youtube_urls(text: str) -> list[str]:
    """Extract unique YouTube video IDs from text (space/newline-separated URLs)."""
    return list(dict.fromkeys(_YOUTUBE_URL_RE.findall(text)))

log = logging.getLogger("gateway")

# Form step identifiers
STEP_ASK_TOPIC = "ask_topic"
STEP_ASK_PAIN = "ask_pain"          # specific customer pain the course solves
STEP_ASK_AUDIENCE = "ask_audience"  # who the course is for
STEP_ASK_COUNT = "ask_count"
STEP_CONFIRM = "confirm"
STEP_PHASE1_RUNNING = "phase1_running"
STEP_AWAITING_APPROVAL = "awaiting_approval"
STEP_AWAITING_COOKIES_PRE_PHASE2 = "awaiting_cookies_pre_phase2"  # forced refresh before phase2
STEP_PHASE2_RUNNING = "phase2_running"
STEP_DONE = "done"
STEP_AWAITING_COOKIES = "awaiting_cookies"  # standalone cookies upload (from /menu)
STEP_ASK_VOICE = "ask_voice"              # LEGACY: gender selection before Phase 1 launch
STEP_ASK_VOICE_PHASE2 = "ask_voice_phase2"  # gender selection right before Phase 2 launch

# Inputs that mean "no answer" for optional pain/audience steps.
_SKIP_TOKENS = {"-", "—", "skip", "/skip", "пропустить", "нет", "no"}


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

    if step in (STEP_AWAITING_COOKIES, STEP_AWAITING_COOKIES_PRE_PHASE2):
        _handle_cookies_upload(token, agent, cfg, chat_id, user_id, msg,
                               thread_id=thread_id)
        return

    if step == STEP_ASK_TOPIC:
        topic = text.strip()
        if not topic:
            _send(token, chat_id, "Тема не может быть пустой. Попробуй ещё раз.",
                  thread_id=thread_id)
            return

        # URL mode: user pasted YouTube links (space/newline/multi-space separated)
        video_ids = _parse_youtube_urls(topic)
        if video_ids:
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
        # Always 1 course per wizard run — count is fixed, no separate step.
        st = _state.update(agent, user_id, thread_id=thread_id,
                           pain=description, count=1, step=STEP_CONFIRM)
        topic = st.get("topic", "")
        desc_line = (_html_escape(description)
                     if description else "<i>(не указано)</i>")
        _send_with_buttons(
            token, chat_id,
            f"<b>Подтверждение:</b>\n\n"
            f"Тема: <b>{_html_escape(topic)}</b>\n"
            f"Описание курса: {desc_line}\n\n"
            f"Запустить Phase 1 (поиск каналов и видео для одного курса)?",
            [[{"text": "🚀 Поехали", "callback_data": "wiz:start_phase1"},
              {"text": "✖️ Отмена", "callback_data": "wiz:cancel"}]],
            thread_id=thread_id,
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

    if action == "start_phase1":
        st = _state.load(agent, user_id, thread_id)
        url_mode = bool(st.get("url_mode"))
        video_ids = list(st.get("video_ids") or [])
        topic = st.get("topic") or ""
        description = st.get("pain", "")  # stored under `pain` for back-compat
        count = 1  # one course per wizard run, always

        if not topic and not (url_mode and video_ids):
            answer_callback_query(token, cq_id, "Состояние формы потеряно", show_alert=True)
            clear_wizard_state(agent, user_id, thread_id)
            return

        # Phase 1 doesn't need voice_gender. The dubbing choice happens at
        # Phase 2 launch — after the operator has seen the actual video set
        # in the Sheet and can judge whether dubbing is even needed.
        _state.update(agent, user_id, thread_id=thread_id,
                      count=count, step=STEP_PHASE1_RUNNING)
        answer_callback_query(token, cq_id, "Phase 1 запущен")

        if url_mode and video_ids:
            try:
                tg_api(token, "sendMessage", chat_id=chat_id,
                       message_thread_id=thread_id or None,
                       text=(
                           "🔗 <b>Phase 1 запущен (прямые ссылки).</b>\n\n"
                           f"Видео в обработке: <b>{len(video_ids)}</b>\n\n"
                           "Скачиваю видео, транскрибирую (Whisper), размечаю вырезки, "
                           "исследую автора и составляю описание курса. "
                           "Тему и названия генерирую автоматически по транскрипту.\n\n"
                           "~15-30 минут. Можешь вернуться в чат "
                           "(<code>/menu</code> → 💬 Чат), пришлю результат как будет готово."
                       ),
                       parse_mode="HTML")
            except Exception:
                pass
            try:
                from . import phase1_discovery
                phase1_discovery.launch_from_urls(
                    token, agent, cfg, chat_id, user_id,
                    video_ids=video_ids, thread_id=thread_id,
                )
            except Exception as e:
                log.exception(f"[{agent}] failed to launch phase1_from_urls: {e}")
                try:
                    tg_api(token, "sendMessage", chat_id=chat_id,
                           message_thread_id=thread_id or None,
                           text=f"⚠️ Не удалось запустить Phase 1: {e}")
                except Exception:
                    pass
                _state.update(agent, user_id, thread_id=thread_id, step="error", error=str(e))
            return

        # Normal topic-based flow
        desc_block = (f"\nОписание: <b>{_html_escape(description)[:300]}</b>"
                      if description else "")
        try:
            tg_api(token, "sendMessage", chat_id=chat_id,
                   message_thread_id=thread_id or None,
                   text=(
                       "🔍 <b>Phase 1 запущен.</b>\n\n"
                       f"Тема: <b>{_html_escape(topic)}</b>"
                       f"{desc_block}\n\n"
                       "Phase 1 ищет каналы, скачивает видео, транскрибирует "
                       "(Whisper), размечает вырезки, пишет описания уроков и "
                       "курса, ищет инфу об авторе через WebSearch. На выходе "
                       "в Sheet будут реальные описания, готовые для лендинга.\n\n"
                       "~20-45 минут на курс. Можешь вернуться в чат с агентом "
                       "(<code>/menu</code> → 💬 Чат), пришлю результат как "
                       "будет готово."
                   ),
                   parse_mode="HTML")
        except Exception:
            pass
        try:
            from . import phase1_discovery
            phase1_discovery.launch(token, agent, cfg, chat_id, user_id,
                                    topic, count,
                                    pain=description,
                                    thread_id=thread_id)
        except Exception as e:
            log.exception(f"[{agent}] failed to launch phase1: {e}")
            try:
                tg_api(token, "sendMessage", chat_id=chat_id,
                       message_thread_id=thread_id or None,
                       text=f"⚠️ Не удалось запустить Phase 1: {e}")
            except Exception:
                pass
            _state.update(agent, user_id, thread_id=thread_id, step="error", error=str(e))
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
