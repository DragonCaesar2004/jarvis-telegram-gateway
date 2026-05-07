"""Wizard per-(user, thread) state: which step in the form, which run is active.

File: state/wizard-{agent}-{user_id}.json              (DM / non-forum chat — thread_id=0)
File: state/wizard-{agent}-{user_id}-{thread_id}.json  (Telegram forum topic)

Threading the state by message_thread_id lets one operator run several wizards
in parallel — one per forum topic. DM behavior is unchanged because we keep the
old filename when thread_id is 0 / falsy.

Schema (unchanged):
    {
      "step": "ask_topic" | "ask_count" | "confirm" | "phase1_running" | "awaiting_approval" | "phase2_running",
      "topic": "AI for marketers",
      "count": 3,
      "run_id": "2026-05-03T20-50-12",
      "thread_id": 12345,             # NEW: forum topic id (0 if DM/group main thread)
      "chat_id": -1001234567890,      # NEW: where to post background updates
      "phase1_started_at": "...",
      "sheet_url": "...",
      "phase2_started_at": "...",
      "courses": [{"course_id": "...", "admin_url": "..."}, ...]
    }
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Resolved at runtime to gateway's STATE_DIR. Lazy import to avoid circular dep.
def _state_dir() -> Path:
    from gateway import STATE_DIR  # type: ignore
    return STATE_DIR


def _file(agent: str, user_id: int, thread_id: int = 0) -> Path:
    """Path to the per-user, per-thread wizard state file.

    Falsy thread_id (0 / None) → original DM filename so existing state files
    keep working without migration.
    """
    if thread_id:
        return _state_dir() / f"wizard-{agent}-{user_id}-{int(thread_id)}.json"
    return _state_dir() / f"wizard-{agent}-{user_id}.json"


def load(agent: str, user_id: int, thread_id: int = 0) -> dict[str, Any]:
    f = _file(agent, user_id, thread_id)
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except Exception:
        return {}


def save(agent: str, user_id: int, state: dict[str, Any], thread_id: int = 0) -> None:
    f = _file(agent, user_id, thread_id)
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    os.replace(tmp, f)


def clear(agent: str, user_id: int, thread_id: int = 0) -> None:
    _file(agent, user_id, thread_id).unlink(missing_ok=True)


def update(agent: str, user_id: int, *, thread_id: int = 0, **fields: Any) -> dict[str, Any]:
    state = load(agent, user_id, thread_id)
    state.update(fields)
    save(agent, user_id, state, thread_id)
    return state
