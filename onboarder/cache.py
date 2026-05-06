"""MP4 cache between Phase 1 (download+transcribe) and Phase 2 (cut+upload).

Phase 1 downloads each video once and keeps the MP4 here. Phase 2 reuses the
cached file (no re-download → no second YouTube hit, no second proxy/cookies
ride, ~30% faster). After Bunny upload succeeds in Phase 2, the cached file
is deleted.

Layout: ~/onboarder-cache/{video_id}.mp4
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("gateway")

_DEFAULT_DIR = Path.home() / "onboarder-cache"


def cache_dir() -> Path:
    """Return the cache directory, honoring ONBOARDER_CACHE_DIR env override."""
    override = os.environ.get("ONBOARDER_CACHE_DIR")
    d = Path(override).expanduser() if override else _DEFAULT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached_path(video_id: str) -> Path:
    """Path where the MP4 for `video_id` lives (whether or not it exists)."""
    return cache_dir() / f"{video_id}.mp4"


def is_cached(video_id: str) -> bool:
    """True iff a non-empty MP4 file is on disk for `video_id`."""
    p = cached_path(video_id)
    try:
        return p.exists() and p.stat().st_size > 0
    except OSError:
        return False


def delete_cached(video_id: str) -> bool:
    """Remove the cached MP4. Returns True if a file was actually deleted."""
    p = cached_path(video_id)
    try:
        if p.exists():
            p.unlink()
            log.info(f"cache: deleted {p.name}")
            return True
    except OSError as e:
        log.warning(f"cache: delete {p.name} failed: {e}")
    return False


def cache_size_bytes() -> int:
    """Total bytes used by the cache (for diagnostics / Telegram /status)."""
    total = 0
    try:
        for p in cache_dir().iterdir():
            if p.is_file():
                total += p.stat().st_size
    except OSError:
        pass
    return total
