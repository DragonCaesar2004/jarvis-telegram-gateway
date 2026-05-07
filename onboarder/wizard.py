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
from typing import Any, Callable

from . import state as _state

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
          "Какая тема курсов? (например: «AI для маркетологов»)\n\n"
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
        # Pain step intentionally skipped — operator decided not to ask for it.
        # The infra (state field, llm helpers' `pain=` kwargs) is kept so we can
        # reinstate the step later by flipping `STEP_ASK_AUDIENCE` back to
        # `STEP_ASK_PAIN` here. Until then `pain` stays an empty string.
        _state.update(agent, user_id, thread_id=thread_id,
                      topic=topic, step=STEP_ASK_AUDIENCE)
        _send(token, chat_id,
              f"Тема: <b>{_html_escape(topic)}</b>\n\n"
              "Кто <b>целевая аудитория</b>? "
              "Например: «мужчины 45+ с малоподвижным образом жизни» или "
              "«предприниматели B2B, которые хотят масштабироваться через AI».\n\n"
              "<i>Пропустить — отправь <code>-</code> или <code>/skip</code>.</i>",
              thread_id=thread_id)
        return

    if step == STEP_ASK_PAIN:
        # Legacy: a wizard from before we removed the pain step might still
        # have its state file pointing here. Treat the input as audience to
        # keep the user moving forward instead of force-restarting them.
        log.info(f"[{agent}] wizard advancing legacy STEP_ASK_PAIN state to audience")
        aud_raw = text.strip()
        audience = "" if aud_raw.lower() in _SKIP_TOKENS else aud_raw
        _state.update(agent, user_id, thread_id=thread_id,
                      audience=audience, step=STEP_ASK_COUNT)
        shown_aud = _html_escape(audience) if audience else "<i>(пропущено)</i>"
        _send(token, chat_id,
              f"Аудитория: <b>{shown_aud}</b>\n\n"
              "Сколько курсов сделать? (число от 1 до 5)",
              thread_id=thread_id)
        return

    if step == STEP_ASK_AUDIENCE:
        aud_raw = text.strip()
        audience = "" if aud_raw.lower() in _SKIP_TOKENS else aud_raw
        _state.update(agent, user_id, thread_id=thread_id,
                      audience=audience, step=STEP_ASK_COUNT)
        shown_aud = _html_escape(audience) if audience else "<i>(пропущено)</i>"
        _send(token, chat_id,
              f"Аудитория: <b>{shown_aud}</b>\n\n"
              "Сколько курсов сделать? (число от 1 до 5)",
              thread_id=thread_id)
        return

    if step == STEP_ASK_COUNT:
        try:
            count = int(text.strip())
            if not (1 <= count <= 5):
                raise ValueError("range")
        except ValueError:
            _send(token, chat_id, "Нужно число от 1 до 5. Попробуй ещё раз.",
                  thread_id=thread_id)
            return
        st = _state.update(agent, user_id, thread_id=thread_id,
                           count=count, step=STEP_CONFIRM)
        topic = st.get("topic", "")
        audience = st.get("audience", "")
        aud_line = _html_escape(audience) if audience else "<i>(не указана)</i>"
        _send_with_buttons(
            token, chat_id,
            f"<b>Подтверждение:</b>\n\n"
            f"Тема: <b>{_html_escape(topic)}</b>\n"
            f"Аудитория: {aud_line}\n"
            f"Кол-во курсов: <b>{count}</b>\n\n"
            f"Запустить Phase 1 (поиск каналов и видео)?",
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
        topic = st.get("topic")
        count = st.get("count")
        audience = st.get("audience", "")
        if not topic or not count:
            answer_callback_query(token, cq_id, "Состояние формы потеряно", show_alert=True)
            clear_wizard_state(agent, user_id, thread_id)
            return
        _state.update(agent, user_id, thread_id=thread_id, step=STEP_PHASE1_RUNNING)
        answer_callback_query(token, cq_id, "Phase 1 запущен")
        aud_line = f"\nАудитория: <b>{_html_escape(audience)}</b>" if audience else ""
        try:
            tg_api(token, "sendMessage", chat_id=chat_id,
                   message_thread_id=thread_id or None,
                   text=(
                       "🔍 <b>Phase 1 запущен.</b>\n\n"
                       f"Тема: <b>{_html_escape(topic)}</b>"
                       f"{aud_line}\n"
                       f"Курсов: <b>{count}</b>\n\n"
                       "Phase 1 теперь делает всю тяжёлую работу: ищет каналы, "
                       "скачивает видео, транскрибирует, размечает вырезки, "
                       "пишет описания уроков и курса, ищет инфу об авторе через "
                       "WebSearch. На выходе в Sheet будут реальные описания, "
                       "готовые для лендинга.\n\n"
                       "Займёт ~40-90 минут. Можешь вернуться в чат с агентом "
                       "(<code>/menu</code> → 💬 Чат), пришлю результат как будет готово."
                   ),
                   parse_mode="HTML")
        except Exception:
            pass
        try:
            from . import phase1_discovery
            phase1_discovery.launch(token, agent, cfg, chat_id, user_id,
                                    topic, count,
                                    audience=audience,
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
        # User asserts cookies are fresh — proceed straight to Phase 2
        _state.update(agent, user_id, thread_id=thread_id, step=STEP_PHASE2_RUNNING)
        answer_callback_query(token, cq_id, "Запускаю Phase 2")
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

    # Case A: cookies were requested as part of the pre-Phase 2 gate → autostart Phase 2
    if current_step == STEP_AWAITING_COOKIES_PRE_PHASE2:
        st["step"] = STEP_PHASE2_RUNNING
        _state.save(agent, user_id, st, thread_id)
        _send(token, chat_id,
              f"✅ <b>Cookies обновлены</b> ({size_kb:.1f} KB, {line_count} строк).\n"
              f"Запускаю Phase 2…",
              thread_id=thread_id)
        try:
            from . import phase2_production
            phase2_production.launch(token, agent, cfg, chat_id, user_id,
                                     thread_id=thread_id)
        except Exception as e:
            log.exception(f"phase2 autostart after cookies upload failed: {e}")
            _send(token, chat_id, f"⚠️ Не удалось запустить Phase 2: {e}",
                  thread_id=thread_id)
            _state.update(agent, user_id, thread_id=thread_id, step="error", error=str(e))
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
