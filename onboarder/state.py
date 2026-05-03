"""Wizard per-user state: which step in the form, which run is active.

File: state/wizard-{agent}-{user_id}.json
Schema:
    {
      "step": "ask_topic" | "ask_count" | "confirm" | "phase1_running" | "awaiting_approval" | "phase2_running",
      "topic": "AI for marketers",
      "count": 3,
      "run_id": "2026-05-03T20-50-12",
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


def _file(agent: str, user_id: int) -> Path:
    return _state_dir() / f"wizard-{agent}-{user_id}.json"


def load(agent: str, user_id: int) -> dict[str, Any]:
    f = _file(agent, user_id)
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except Exception:
        return {}


def save(agent: str, user_id: int, state: dict[str, Any]) -> None:
    f = _file(agent, user_id)
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    os.replace(tmp, f)


def clear(agent: str, user_id: int) -> None:
    _file(agent, user_id).unlink(missing_ok=True)


def update(agent: str, user_id: int, **fields: Any) -> dict[str, Any]:
    state = load(agent, user_id)
    state.update(fields)
    save(agent, user_id, state)
    return state
