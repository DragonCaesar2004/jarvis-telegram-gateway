"""Bunny Stream upload (mirrors NewMindStart's src/lib/bunny.ts uploadVideo).

Two-step upload:
    1. POST https://video.bunnycdn.com/library/{lib}/videos  → { guid }
    2. PUT  https://video.bunnycdn.com/library/{lib}/videos/{guid}  (raw bytes)

Then we wait briefly for Bunny to register the upload (the GUID is usable
immediately for storing in Lesson.videoKey, but transcoding takes minutes —
that's Bunny's problem, not ours).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger("gateway")

API_BASE = "https://video.bunnycdn.com"
DEFAULT_TIMEOUT = 30
UPLOAD_TIMEOUT = 1800   # 30 min for big files


class BunnyError(RuntimeError):
    pass


def upload_video(*, library_id: str, api_key: str, file_path: str | Path,
                 title: str) -> dict[str, Any]:
    """Create video entry then upload bytes. Returns metadata for NMS payload.

    Returns:
        {
          "videoKey": str,         # the GUID (Bunny's internal id)
          "videoLibraryId": str,   # the library id (string)
          "duration": int | None,  # seconds — populated from filesystem if ffprobe ran;
                                   # otherwise None and Phase 2 will pull it elsewhere
          "size_bytes": int,
        }
    """
    if not library_id or not api_key:
        raise BunnyError("bunny: library_id and api_key are required")
    p = Path(file_path)
    if not p.exists():
        raise BunnyError(f"bunny: file not found: {p}")

    # ── Step 1: create video entry ─────────────────────────────────────
    create_url = f"{API_BASE}/library/{library_id}/videos"
    try:
        r1 = requests.post(
            create_url,
            headers={
                "AccessKey": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={"title": title},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as e:
        raise BunnyError(f"bunny: network error on create: {e}") from e
    if r1.status_code >= 400:
        raise BunnyError(f"bunny create failed: {r1.status_code} {r1.text[:300]}")
    try:
        guid = r1.json().get("guid")
    except ValueError:
        raise BunnyError(f"bunny create: non-JSON response: {r1.text[:300]}")
    if not guid:
        raise BunnyError(f"bunny create: missing 'guid' in response: {r1.text[:300]}")

    # ── Step 2: upload bytes ───────────────────────────────────────────
    upload_url = f"{API_BASE}/library/{library_id}/videos/{guid}"
    size = p.stat().st_size
    log.info(f"bunny: uploading {p.name} ({size/1024/1024:.1f} MB) to library {library_id}")
    try:
        with p.open("rb") as f:
            r2 = requests.put(
                upload_url,
                headers={
                    "AccessKey": api_key,
                    "Content-Type": "application/octet-stream",
                },
                data=f,
                timeout=UPLOAD_TIMEOUT,
            )
    except requests.RequestException as e:
        # Best-effort cleanup so we don't leave orphan video entries
        _try_delete(library_id, api_key, guid)
        raise BunnyError(f"bunny: network error during upload: {e}") from e

    if r2.status_code >= 400:
        _try_delete(library_id, api_key, guid)
        raise BunnyError(f"bunny upload failed: {r2.status_code} {r2.text[:300]}")

    return {
        "videoKey": guid,
        "videoLibraryId": str(library_id),
        "duration": None,
        "size_bytes": size,
    }


def delete_video(*, library_id: str, api_key: str, video_guid: str) -> None:
    """Best-effort delete (ignore 404)."""
    url = f"{API_BASE}/library/{library_id}/videos/{video_guid}"
    try:
        r = requests.delete(url, headers={"AccessKey": api_key}, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        log.warning(f"bunny delete network error: {e}")
        return
    if r.status_code not in (200, 204, 404):
        log.warning(f"bunny delete returned {r.status_code}: {r.text[:200]}")


def _try_delete(library_id: str, api_key: str, guid: str) -> None:
    try:
        delete_video(library_id=library_id, api_key=api_key, video_guid=guid)
    except Exception as e:
        log.warning(f"bunny: orphan cleanup failed for {guid}: {e}")
