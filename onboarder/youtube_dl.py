"""yt-dlp wrappers for the discovery phase.

All metadata-only operations (no actual video download). Phase 2 will use
a separate `download_video()` for the real .mp4 fetch.

Functions:
    search_videos(query, max_results)
        ytsearch{N}:query → list of video dicts with channel info attached.

    unique_channels_from_search(videos)
        Group search results by channel_id, return list with vote count
        (more videos in search = stronger topical match).

    get_channel_metadata(channel_id_or_url)
        Fetch subscribers, video_count, description, language hints.

    list_channel_videos(channel_id_or_url, max_results, max_age_months)
        Recent videos (flat listing — no per-video full metadata yet).

    get_video_metadata(video_id, with_full_metadata=False)
        Single video info; full metadata costs an extra HTTP round-trip.

All functions return plain dicts (JSON-serializable) so they can be passed
straight to llm.score_channels() / llm.select_videos().
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger("gateway")


# ---------------------------------------------------------------------------
# Internal: yt-dlp instance with sane defaults
# ---------------------------------------------------------------------------

def _ydl(extra_opts: dict | None = None) -> Any:
    """Build a YoutubeDL instance. Lazy import so gateway core doesn't need yt_dlp."""
    from yt_dlp import YoutubeDL
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
        "nocheckcertificate": True,
    }
    if extra_opts:
        opts.update(extra_opts)
    return YoutubeDL(opts)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_videos(query: str, max_results: int = 30) -> list[dict[str, Any]]:
    """Run `ytsearch{N}:query` and return list of {video_id, title, channel_id,
    channel_name, duration_sec, upload_date, view_count, description}.
    """
    search_url = f"ytsearch{int(max_results)}:{query}"
    with _ydl({"extract_flat": "in_playlist"}) as ydl:
        try:
            info = ydl.extract_info(search_url, download=False)
        except Exception as e:
            log.warning(f"youtube_dl: search failed for {query!r}: {e}")
            return []

    entries = info.get("entries") or []
    out: list[dict[str, Any]] = []
    for e in entries:
        if not e:
            continue
        out.append({
            "video_id": e.get("id"),
            "title": e.get("title") or "",
            "channel_id": e.get("channel_id") or e.get("uploader_id") or "",
            "channel_name": e.get("channel") or e.get("uploader") or "",
            "channel_url": e.get("channel_url") or e.get("uploader_url") or "",
            "duration_sec": int(e.get("duration") or 0),
            "view_count": int(e.get("view_count") or 0),
            "url": e.get("url") or e.get("webpage_url") or f"https://youtu.be/{e.get('id')}",
        })
    return out


def unique_channels_from_search(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group search results by channel_id. Returns list ordered by vote count desc.

    Each item: {channel_id, channel_name, channel_url, votes, sample_titles[]}
    where votes = number of videos from that channel that appeared in the search.
    """
    by_channel: dict[str, dict[str, Any]] = {}
    for v in videos:
        cid = v.get("channel_id") or ""
        if not cid:
            continue
        bucket = by_channel.setdefault(cid, {
            "channel_id": cid,
            "channel_name": v.get("channel_name", ""),
            "channel_url": v.get("channel_url", ""),
            "votes": 0,
            "sample_titles": [],
        })
        bucket["votes"] += 1
        if len(bucket["sample_titles"]) < 5:
            bucket["sample_titles"].append(v.get("title", ""))
    return sorted(by_channel.values(), key=lambda x: x["votes"], reverse=True)


# ---------------------------------------------------------------------------
# Channel metadata
# ---------------------------------------------------------------------------

def get_channel_metadata(channel_id_or_url: str) -> dict[str, Any]:
    """Fetch subscriber count, video count, description, etc.

    Accepts either a bare channel_id ("UC...") or full channel URL.
    Returns {} on failure (caller should skip the channel).
    """
    url = _channel_url(channel_id_or_url)
    with _ydl({"extract_flat": True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as e:
            log.warning(f"youtube_dl: channel metadata failed for {url}: {e}")
            return {}

    return {
        "channel_id": info.get("channel_id") or info.get("id") or "",
        "channel_name": info.get("channel") or info.get("uploader") or info.get("title") or "",
        "channel_url": info.get("channel_url") or info.get("webpage_url") or url,
        "subscribers": int(info.get("channel_follower_count") or 0),
        "video_count": _extract_video_count(info),
        "description": (info.get("description") or "")[:2000],
        "language": _extract_language(info),
        "thumbnail": _best_thumbnail(info),
    }


def list_channel_videos(channel_id_or_url: str, max_results: int = 50,
                        max_age_months: int | None = None) -> list[dict[str, Any]]:
    """List recent videos from a channel. Flat listing — no per-video full metadata.

    Returns up to `max_results` videos sorted newest-first. If `max_age_months`
    is set, filter out videos older than that.
    """
    url = _channel_videos_url(channel_id_or_url)
    cutoff_ts: float | None = None
    if max_age_months:
        cutoff_ts = time.time() - (max_age_months * 30.4 * 24 * 3600)

    with _ydl({"extract_flat": "in_playlist", "playlistend": max_results}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as e:
            log.warning(f"youtube_dl: channel videos failed for {url}: {e}")
            return []

    entries = info.get("entries") or []
    out: list[dict[str, Any]] = []
    for e in entries:
        if not e:
            continue
        upload_ts = _parse_upload_date(e.get("upload_date") or e.get("timestamp"))
        if cutoff_ts and upload_ts and upload_ts < cutoff_ts:
            continue
        out.append({
            "video_id": e.get("id"),
            "title": e.get("title") or "",
            "duration_sec": int(e.get("duration") or 0),
            "view_count": int(e.get("view_count") or 0),
            "upload_date": e.get("upload_date") or "",
            "upload_ts": upload_ts,
            "url": e.get("url") or f"https://youtu.be/{e.get('id')}",
            "description": "",  # flat listing doesn't include description
        })
    return out


def get_video_metadata(video_id: str) -> dict[str, Any]:
    """Full metadata for one video (description, language, etc.)."""
    url = f"https://youtu.be/{video_id}"
    with _ydl() as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as e:
            log.warning(f"youtube_dl: video metadata failed for {video_id}: {e}")
            return {}
    return {
        "video_id": info.get("id") or video_id,
        "title": info.get("title") or "",
        "description": info.get("description") or "",
        "duration_sec": int(info.get("duration") or 0),
        "view_count": int(info.get("view_count") or 0),
        "upload_date": info.get("upload_date") or "",
        "channel_id": info.get("channel_id") or info.get("uploader_id") or "",
        "channel_name": info.get("channel") or info.get("uploader") or "",
        "language": info.get("language") or "",
        "url": info.get("webpage_url") or url,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _channel_url(channel_id_or_url: str) -> str:
    if channel_id_or_url.startswith("http"):
        return channel_id_or_url
    if channel_id_or_url.startswith("@"):
        return f"https://www.youtube.com/{channel_id_or_url}"
    return f"https://www.youtube.com/channel/{channel_id_or_url}"


def _channel_videos_url(channel_id_or_url: str) -> str:
    base = _channel_url(channel_id_or_url).rstrip("/")
    if base.endswith("/videos"):
        return base
    return base + "/videos"


def _extract_video_count(info: dict[str, Any]) -> int:
    # yt-dlp puts this in different keys depending on extractor version
    for k in ("playlist_count", "n_entries", "channel_video_count"):
        v = info.get(k)
        if isinstance(v, int) and v > 0:
            return v
    entries = info.get("entries")
    if isinstance(entries, list):
        return len(entries)
    return 0


def _extract_language(info: dict[str, Any]) -> str | None:
    # yt-dlp doesn't reliably expose channel language; try a few hints
    for k in ("language", "original_language"):
        v = info.get(k)
        if v:
            return str(v)
    return None


def _best_thumbnail(info: dict[str, Any]) -> str | None:
    thumbs = info.get("thumbnails") or []
    if not thumbs:
        return info.get("thumbnail")
    # Largest by area
    def area(t: dict) -> int:
        return int(t.get("width") or 0) * int(t.get("height") or 0)
    return max(thumbs, key=area).get("url")


def _parse_upload_date(value: Any) -> float | None:
    """yt-dlp gives upload_date as 'YYYYMMDD' string OR timestamp int. Return Unix ts."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    if len(s) == 8 and s.isdigit():
        try:
            dt = datetime(int(s[:4]), int(s[4:6]), int(s[6:8]), tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None
    return None
