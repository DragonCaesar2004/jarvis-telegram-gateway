"""Persistent log of operator-rejected courses.

Stored as append-only JSONL at `state/rejections.jsonl`. Each line is one
rejection record:

    {"ts": "2026-05-27T12:34:56Z",
     "run_id": "2026-05-26T19-31-20",
     "course_idx": 4,
     "channel": "Some Channel Name",
     "course_title": "Calm Body, Clear Mind: …",
     "reason": "low quality, mostly theory",
     "user_id": 5228613243,
     "thread_id": 200}

Consumed by:
  - wizard._finalize_rejection (writer)  — appends a record when the operator
    presses «❌ Отклонить» on a per-course Phase 2 button.
  - phase1_discovery._run (reader)       — pulls recent rejections at the
    start of every Phase 1 to feed llm.score_channels / llm.select_videos
    so the discovery LLM avoids producing similar courses again.

Failure-safe: any error reading the file returns []; the pipeline continues
without rejection feedback. Any error writing logs at WARNING level — the
caller (the reject UX) still completes Sheet update + cache delete so
operator's intent is honoured even if the JSONL append fails.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("gateway")

REJECTIONS_FILE_NAME = "rejections.jsonl"

# Cross-thread serialisation for the append-line. Append-mode writes of short
# JSON lines are POSIX-atomic on local disk, but the lock keeps the code self-
# documenting and avoids interleaved partial flushes in any edge case.
_LOCK = threading.Lock()


def _path() -> Path:
    """Resolve the JSONL path lazily (gateway.STATE_DIR may not exist at
    import time in some test scenarios)."""
    from gateway import STATE_DIR  # type: ignore  # lazy import
    return STATE_DIR / REJECTIONS_FILE_NAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_rejection(*, run_id: str, course_idx: int,
                     channel: str, course_title: str,
                     reason: str,
                     user_id: int, thread_id: int = 0) -> bool:
    """Append one rejection record to the JSONL. Returns True on success.

    Any write failure logs WARNING and returns False — caller should still
    update the Sheet and clear the cache (operator's intent is honoured).
    """
    rec: dict[str, Any] = {
        "ts": _now_iso(),
        "run_id": str(run_id or ""),
        "course_idx": int(course_idx or 0),
        "channel": str(channel or ""),
        "course_title": str(course_title or ""),
        "reason": str(reason or ""),
        "user_id": int(user_id or 0),
        "thread_id": int(thread_id or 0),
    }
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    p = _path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            with p.open("a", encoding="utf-8") as f:
                f.write(line)
        return True
    except OSError as e:
        log.warning(f"rejections.append failed: {e} — record dropped: "
                    f"run={run_id} course={course_idx}")
        return False


def load_recent_rejections(n: int = 20) -> list[dict[str, Any]]:
    """Return up to the last N rejection records. Returns [] if file
    missing or unreadable. Bad-JSON lines are skipped silently."""
    p = _path()
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # One bad line shouldn't kill the whole feed.
                    continue
    except OSError as e:
        log.warning(f"rejections.load failed: {e} — returning empty list")
        return []
    if n and len(out) > n:
        out = out[-n:]
    return out
