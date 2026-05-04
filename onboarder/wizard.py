"""Wizard state machine: form → Phase 1 → approval → Phase 2.

Public API consumed by gateway.py:
    handle_wizard_message(token, agent, cfg, chat_id, user_id, text, msg)
        Called by gateway's mode-router when user is in MODE_WIZARD and sends a message.

    start_wizard(token, agent, cfg, chat_id, user_id)
        Called from menu callback ("menu:onboard") to begin a fresh wizard run.

    register_callbacks(register_func)
        Called once at gateway startup to register inline-button handlers
        (prefix "wiz:") with the gateway's callback dispatcher.

    clear_wizard_state(agent, user_id)
        Called by /cancel and "menu:chat" to wipe in-progress form/run state.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from . import state as _state

log = logging.getLogger("gateway")

# Form step identifiers
STEP_ASK_TOPIC = "ask_topic"
STEP_ASK_COUNT = "ask_count"
STEP_CONFIRM = "confirm"
STEP_PHASE1_RUNNING = "phase1_running"
STEP_AWAITING_APPROVAL = "awaiting_approval"
STEP_PHASE2_RUNNING = "phase2_running"
STEP_DONE = "done"
STEP_AWAITING_COOKIES = "awaiting_cookies"  # bot is waiting for a cookies.txt upload


# ---------------------------------------------------------------------------
# Lifecycle entry points
# ---------------------------------------------------------------------------

def start_wizard(token: str, agent: str, cfg: dict, chat_id: int, user_id: int) -> None:
    """Begin a new wizard run: clear previous state, ask first question."""
    _state.clear(agent, user_id)
    _state.save(agent, user_id, {"step": STEP_ASK_TOPIC, "chat_id": chat_id})
    _send(token, chat_id,
          "🎓 <b>Новый курс</b>\n\n"
          "Какая тема курсов? (например: «AI для маркетологов»)\n\n"
          "<i>/cancel — выход в чат с агентом.</i>")


def start_cookies_upload(token: str, agent: str, cfg: dict, chat_id: int, user_id: int) -> None:
    """Enter the wizard 'awaiting cookies file' state.

    Preserves existing run state (sheet_tab, run_id, step) so that uploading
    cookies during an active Phase 1/Phase 2 run doesn't wipe the run context.
    """
    existing = _state.load(agent, user_id)
    # Stash the previous step so we can restore it after upload
    _state.save(agent, user_id, {
        **existing,
        "_prev_step": existing.get("step"),
        "step": STEP_AWAITING_COOKIES,
        "chat_id": chat_id,
    })
    _send(token, chat_id,
          "📎 <b>Загрузка YouTube cookies</b>\n\n"
          "Отправь следующим сообщением файл <code>cookies.txt</code> "
          "(Netscape-format) — прикрепи его как документ.\n\n"
          "<b>Как получить файл:</b>\n"
          "1. Поставь расширение <i>Get cookies.txt LOCALLY</i> в Chrome/Edge\n"
          "2. Открой youtube.com (залогинен)\n"
          "3. Иконка расширения → Export As → cookies.txt\n"
          "4. Перетащи скачанный файл сюда\n\n"
          "<i>/cancel — отмена.</i>")


def clear_wizard_state(agent: str, user_id: int) -> None:
    _state.clear(agent, user_id)


def handle_wizard_message(token: str, agent: str, cfg: dict, chat_id: int,
                          user_id: int, text: str, msg: dict) -> None:
    """Route a non-command text message based on wizard step."""
    st = _state.load(agent, user_id)
    if not st:
        # Mode is wizard but no state — recover gracefully.
        start_wizard(token, agent, cfg, chat_id, user_id)
        return

    step = st.get("step", STEP_ASK_TOPIC)

    if step == STEP_AWAITING_COOKIES:
        _handle_cookies_upload(token, agent, cfg, chat_id, user_id, msg)
        return

    if step == STEP_ASK_TOPIC:
        topic = text.strip()
        if not topic:
            _send(token, chat_id, "Тема не может быть пустой. Попробуй ещё раз.")
            return
        _state.update(agent, user_id, topic=topic, step=STEP_ASK_COUNT)
        _send(token, chat_id,
              f"Тема: <b>{_html_escape(topic)}</b>\n\n"
              "Сколько курсов сделать? (число от 1 до 5)")
        return

    if step == STEP_ASK_COUNT:
        try:
            count = int(text.strip())
            if not (1 <= count <= 5):
                raise ValueError("range")
        except ValueError:
            _send(token, chat_id, "Нужно число от 1 до 5. Попробуй ещё раз.")
            return
        st = _state.update(agent, user_id, count=count, step=STEP_CONFIRM)
        topic = st.get("topic", "")
        _send_with_buttons(
            token, chat_id,
            f"<b>Подтверждение:</b>\n\n"
            f"Тема: <b>{_html_escape(topic)}</b>\n"
            f"Кол-во курсов: <b>{count}</b>\n\n"
            f"Запустить Phase 1 (поиск каналов и видео)?",
            [[{"text": "🚀 Поехали", "callback_data": "wiz:start_phase1"},
              {"text": "✖️ Отмена", "callback_data": "wiz:cancel"}]],
        )
        return

    if step in (STEP_PHASE1_RUNNING, STEP_PHASE2_RUNNING):
        _send(token, chat_id,
              f"⏳ Сейчас идёт фоновая обработка ({step}). "
              f"Сообщения в чате с агентом доступны параллельно через /menu → 💬 Чат с агентом.")
        return

    if step == STEP_AWAITING_APPROVAL:
        _send(token, chat_id,
              "Жду подтверждения подборки в Sheet → нажми кнопку «Запустить обработку» под последним сообщением, "
              "или /cancel для выхода.")
        return

    # Unknown step: reset
    log.warning(f"[{agent}] wizard unknown step={step!r}, restarting")
    start_wizard(token, agent, cfg, chat_id, user_id)


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
    if chat_id is None or user_id is None:
        answer_callback_query(token, cq_id)
        return

    action = data.split(":", 1)[1] if ":" in data else ""

    if action == "cancel":
        clear_wizard_state(agent, user_id)
        set_user_mode(agent, user_id, MODE_CHAT)
        answer_callback_query(token, cq_id, "Отменено")
        try:
            tg_api(token, "sendMessage", chat_id=chat_id,
                   text="<b>Отменено.</b> Обратно в чат с агентом.",
                   parse_mode="HTML")
        except Exception:
            pass
        return

    if action == "start_phase1":
        st = _state.load(agent, user_id)
        topic = st.get("topic")
        count = st.get("count")
        if not topic or not count:
            answer_callback_query(token, cq_id, "Состояние формы потеряно", show_alert=True)
            clear_wizard_state(agent, user_id)
            return
        _state.update(agent, user_id, step=STEP_PHASE1_RUNNING)
        answer_callback_query(token, cq_id, "Phase 1 запущен")
        try:
            tg_api(token, "sendMessage", chat_id=chat_id,
                   text=(
                       "🔍 <b>Phase 1 запущен.</b>\n\n"
                       f"Тема: <b>{_html_escape(topic)}</b>\n"
                       f"Курсов: <b>{count}</b>\n\n"
                       "Это займёт ~10-20 минут. Можешь пока вернуться в чат с агентом "
                       "(<code>/menu</code> → 💬 Чат), пришлю результат как будет готово."
                   ),
                   parse_mode="HTML")
        except Exception:
            pass
        try:
            from . import phase1_discovery
            phase1_discovery.launch(token, agent, cfg, chat_id, user_id, topic, count)
        except Exception as e:
            log.exception(f"[{agent}] failed to launch phase1: {e}")
            try:
                tg_api(token, "sendMessage", chat_id=chat_id,
                       text=f"⚠️ Не удалось запустить Phase 1: {e}")
            except Exception:
                pass
            _state.update(agent, user_id, step="error", error=str(e))
        return

    if action == "start_phase2":
        # Triggered when user has approved the Sheet and clicks "Запустить обработку"
        _state.update(agent, user_id, step=STEP_PHASE2_RUNNING)
        answer_callback_query(token, cq_id, "Phase 2 запущен")
        try:
            from . import phase2_production
            phase2_production.launch(token, agent, cfg, chat_id, user_id)
        except Exception as e:
            log.exception(f"[{agent}] failed to launch phase2: {e}")
            try:
                tg_api(token, "sendMessage", chat_id=chat_id,
                       text=f"⚠️ Не удалось запустить Phase 2: {e}")
            except Exception:
                pass
            _state.update(agent, user_id, step="error", error=str(e))
        return

    answer_callback_query(token, cq_id)


# ---------------------------------------------------------------------------
# Helpers (use gateway's send functions to honor HTML chunking, retries, etc.)
# ---------------------------------------------------------------------------

def _send(token: str, chat_id: int, text: str) -> None:
    from gateway import tg_api  # type: ignore
    try:
        tg_api(token, "sendMessage", chat_id=chat_id, text=text, parse_mode="HTML")
    except Exception as e:
        log.warning(f"wizard _send failed: {e}")


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


# ---------------------------------------------------------------------------
# Cookie upload handler
# ---------------------------------------------------------------------------

def _handle_cookies_upload(token: str, agent: str, cfg: dict,
                           chat_id: int, user_id: int, msg: dict) -> None:
    """Save attached document as YouTube cookies.txt at the configured path."""
    from gateway import download_telegram_file, set_user_mode, MODE_CHAT  # type: ignore
    import os
    import shutil
    from pathlib import Path

    doc = msg.get("document")
    if not doc:
        _send(token, chat_id,
              "Жду файл прикреплением (как документ). Просто текст не подходит. "
              "Если передумал — /cancel.")
        return

    file_id = doc.get("file_id")
    file_name = doc.get("file_name") or "cookies.txt"
    if not file_id:
        _send(token, chat_id, "⚠️ Не удалось прочитать file_id. Попробуй ещё раз.")
        return

    onb = (cfg.get("onboarder") or {})
    target = onb.get("youtube_cookies_file") or "~/.secrets/youtube-cookies.txt"
    target_path = Path(target).expanduser()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    local = download_telegram_file(token, file_id, "document", file_name)
    if not local:
        _send(token, chat_id, "⚠️ Не смог скачать файл из Telegram (>20MB или ошибка сети).")
        return

    # Validate: Netscape cookies.txt starts with `# Netscape HTTP Cookie File`
    # or at least has tab-separated YouTube domain entries.
    try:
        head = local.read_text(errors="replace")[:4096]
    except Exception as e:
        _send(token, chat_id, f"⚠️ Не смог прочитать файл: {e}")
        return

    if "youtube.com" not in head and "Netscape" not in head:
        _send(token, chat_id,
              "⚠️ Файл не похож на Netscape cookies.txt от youtube.com "
              "(не нашёл ни 'Netscape', ни 'youtube.com' в первых 4KB). "
              "Перепроверь, что экспортировал именно с youtube.com.")
        return

    try:
        shutil.move(str(local), str(target_path))
        os.chmod(target_path, 0o600)
    except Exception as e:
        _send(token, chat_id, f"⚠️ Не смог сохранить файл в {target_path}: {e}")
        return

    size_kb = target_path.stat().st_size / 1024
    line_count = sum(1 for _ in target_path.open("r", errors="replace"))

    # Restore previous run state — don't wipe sheet_tab / run_id
    st = _state.load(agent, user_id)
    prev_step = st.pop("_prev_step", None)
    if prev_step and prev_step not in (STEP_AWAITING_COOKIES, STEP_DONE, "error", ""):
        # There was an active run — restore it
        st["step"] = prev_step
        _state.save(agent, user_id, st)
        # Stay in wizard mode so user can continue
        confirm_text = (
            f"✅ <b>Cookies обновлены.</b>\n\n"
            f"Размер: {size_kb:.1f} KB, строк: {line_count}\n\n"
        )
        if prev_step == STEP_AWAITING_APPROVAL:
            confirm_text += "Подборка в Sheet всё ещё ждёт тебя. Нажми кнопку чтобы запустить обработку:"
            _send_with_buttons(token, chat_id, confirm_text,
                               [[{"text": "🚀 Запустить обработку", "callback_data": "wiz:start_phase2"},
                                 {"text": "✖️ Отмена", "callback_data": "wiz:cancel"}]])
        else:
            _send(token, chat_id, confirm_text + f"Продолжаю с шага: <code>{prev_step}</code>.")
    else:
        # No active run — clear state and return to chat
        clear_wizard_state(agent, user_id)
        set_user_mode(agent, user_id, MODE_CHAT)
        _send(token, chat_id,
              f"✅ <b>Cookies сохранены.</b>\n\n"
              f"Размер: {size_kb:.1f} KB, строк: {line_count}\n\n"
              f"Следующий запуск Phase 2 будет использовать этот файл.\n"
              f"Возвращаюсь в чат с агентом.")
