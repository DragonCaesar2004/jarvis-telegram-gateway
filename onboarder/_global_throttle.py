"""Global throttle for per-video work across ALL parallel courses.

When multiple courses run concurrently (different Telegram forum topics each
trigger their own Phase 1 / Phase 2), each course's ThreadPoolExecutor limits
parallelism *within* a course (phase1_parallel_per_course / phase2_parallel_videos).
But there is no built-in cap on total simultaneous video pipelines, so 10 courses
× 8 slots could spawn 80 concurrent downloads + ffmpeg + transcriptions — enough
to OOM an 8 GB box.

This module provides a single BoundedSemaphore that wraps every per-video unit
of work in Phase 1 enrich and Phase 2 production. Slot budget defaults to 16 and
can be overridden via the ONBOARDER_MAX_GLOBAL_SLOTS env var or by calling
configure() once at startup.
"""

from __future__ import annotations

import contextlib
import os
import threading

_lock = threading.Lock()
_sem: threading.BoundedSemaphore | None = None
_max_slots: int = 0


def configure(max_slots: int) -> None:
    """Initialise the global semaphore. First caller wins; subsequent calls are no-ops."""
    global _sem, _max_slots
    with _lock:
        if _sem is None:
            _max_slots = max(1, int(max_slots))
            _sem = threading.BoundedSemaphore(_max_slots)


def _ensure() -> threading.BoundedSemaphore:
    global _sem
    if _sem is None:
        default = int(os.environ.get("ONBOARDER_MAX_GLOBAL_SLOTS", "16"))
        configure(default)
    assert _sem is not None
    return _sem


@contextlib.contextmanager
def acquire_video_slot():
    """Block until a global video-processing slot is free, then release on exit."""
    sem = _ensure()
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


def current_max() -> int:
    """Return the configured cap (after first acquire/configure)."""
    _ensure()
    return _max_slots
