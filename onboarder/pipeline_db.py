"""SQLite-backed handoff store between Phase 1 and Phase 2.

Phase 1 computes cut timecodes (intro/promo segments to remove) on the working
transcript and persists them here, keyed by `video_id`. Phase 2 reads the
cuts back without re-running Whisper on the original.

Schema:
    video_cuts(
        video_id TEXT PRIMARY KEY,
        cuts_json TEXT NOT NULL,           -- JSON array of {start, end, reason}
        working_transcript TEXT,           -- full working transcript (text)
        detected_lang TEXT,                -- ISO code from Whisper ("en", "ru", ...)
        duration_sec INTEGER,              -- raw video duration before cuts
        computed_at TEXT NOT NULL          -- ISO8601 timestamp
    )

The DB file lives at <gateway state_dir>/pipeline.db. Lazy import of gateway
keeps this module standalone-testable.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("gateway")


def _db_path() -> Path:
    from gateway import STATE_DIR  # type: ignore
    return STATE_DIR / "pipeline.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS video_cuts (
            video_id TEXT PRIMARY KEY,
            cuts_json TEXT NOT NULL,
            working_transcript TEXT,
            detected_lang TEXT,
            duration_sec INTEGER,
            computed_at TEXT NOT NULL
        )
    """)
    # Migration: add segments_json column (full Whisper segments with word-level
    # timestamps) so Phase 2 can derive cleaned-timeline segments without
    # re-running Whisper on the cut video. Safe to run multiple times.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(video_cuts)").fetchall()}
    if "segments_json" not in cols:
        conn.execute("ALTER TABLE video_cuts ADD COLUMN segments_json TEXT")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS course_compose (
            run_id TEXT NOT NULL,
            course_idx INTEGER NOT NULL,
            composed_json TEXT NOT NULL,
            computed_at TEXT NOT NULL,
            PRIMARY KEY (run_id, course_idx)
        )
    """)
    return conn


def save_cuts(video_id: str, *, cuts: list[dict[str, Any]],
              working_transcript: str = "",
              detected_lang: str = "",
              duration_sec: int = 0,
              segments: list[dict[str, Any]] | None = None) -> None:
    """Upsert cut data for one video. Safe to call multiple times.

    `segments`: full Whisper segments with word-level timestamps. Phase 2
    reuses them to derive cleaned-timeline segments without a second Whisper
    call. Schema is whatever Whisper returns:
        [{"start": float, "end": float, "text": str,
          "words": [{"start": float, "end": float, "word": str}, ...]}, ...]
    """
    if not video_id:
        log.warning("pipeline_db.save_cuts: empty video_id, skipping")
        return
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = json.dumps(cuts or [], ensure_ascii=False)
    segments_payload = json.dumps(segments or [], ensure_ascii=False) if segments else None
    with _connect() as conn:
        conn.execute("""
            INSERT INTO video_cuts(video_id, cuts_json, working_transcript,
                                   detected_lang, duration_sec, computed_at,
                                   segments_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                cuts_json = excluded.cuts_json,
                working_transcript = excluded.working_transcript,
                detected_lang = excluded.detected_lang,
                duration_sec = excluded.duration_sec,
                computed_at = excluded.computed_at,
                segments_json = COALESCE(excluded.segments_json, video_cuts.segments_json)
        """, (video_id, payload, working_transcript, detected_lang,
              int(duration_sec or 0), ts, segments_payload))


def get_cuts(video_id: str) -> dict[str, Any] | None:
    """Return {cuts, working_transcript, detected_lang, duration_sec,
    computed_at, segments} or None if no row exists.

    `segments` is the Whisper segments array (with word-level timestamps) when
    Phase 1 captured it; empty list for legacy rows that predate the column.
    """
    if not video_id:
        return None
    with _connect() as conn:
        row = conn.execute("""
            SELECT cuts_json, working_transcript, detected_lang,
                   duration_sec, computed_at, segments_json
              FROM video_cuts WHERE video_id = ?
        """, (video_id,)).fetchone()
    if not row:
        return None
    cuts_json, transcript, lang, dur, computed_at, segments_json = row
    try:
        cuts = json.loads(cuts_json or "[]")
    except json.JSONDecodeError:
        cuts = []
    try:
        segments = json.loads(segments_json or "[]") if segments_json else []
    except json.JSONDecodeError:
        segments = []
    return {
        "cuts": cuts,
        "working_transcript": transcript or "",
        "detected_lang": lang or "",
        "duration_sec": int(dur or 0),
        "computed_at": computed_at or "",
        "segments": segments,
    }


def delete_cuts(video_id: str) -> bool:
    """Remove cut data for one video. Returns True if a row was deleted."""
    if not video_id:
        return False
    with _connect() as conn:
        cur = conn.execute("DELETE FROM video_cuts WHERE video_id = ?", (video_id,))
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# course_compose: Phase 1 stores the FULL compose_full_course() output here
# (curriculum, plan, science, testimonials, author bio, etc). Phase 2 reads
# back and POSTs to NewMindStart admin without re-running Claude.
# ---------------------------------------------------------------------------

def save_course_compose(*, run_id: str, course_idx: int,
                        composed: dict[str, Any]) -> None:
    """Upsert composed course payload for (run_id, course_idx)."""
    if not run_id or not course_idx:
        log.warning("pipeline_db.save_course_compose: bad keys, skipping")
        return
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = json.dumps(composed or {}, ensure_ascii=False)
    with _connect() as conn:
        conn.execute("""
            INSERT INTO course_compose(run_id, course_idx, composed_json, computed_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(run_id, course_idx) DO UPDATE SET
                composed_json = excluded.composed_json,
                computed_at = excluded.computed_at
        """, (run_id, int(course_idx), payload, ts))


def get_course_compose(*, run_id: str, course_idx: int) -> dict[str, Any] | None:
    """Return composed payload or None."""
    if not run_id or not course_idx:
        return None
    with _connect() as conn:
        row = conn.execute("""
            SELECT composed_json FROM course_compose
             WHERE run_id = ? AND course_idx = ?
        """, (run_id, int(course_idx))).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0] or "{}")
    except json.JSONDecodeError:
        return None
