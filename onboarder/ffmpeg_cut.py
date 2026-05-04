"""FFmpeg helpers for the onboarder pipeline.

Two operations:

1. cut_segments(input_path, cuts, output_path)
   Given a list of [(start, end)] ranges to REMOVE, build the keep-list
   and concat the remaining segments into one MP4.

2. download_video(url, output_path)
   Use yt-dlp to fetch best video+audio merged as MP4.
   (Lives here for convenience — both use FFmpeg under the hood.)

3. probe_duration(path) -> float
   ffprobe wrapper to get exact seconds.

All functions raise FFmpegError on failure with the relevant stderr tail.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

log = logging.getLogger("gateway")


class FFmpegError(RuntimeError):
    pass


def _check_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise FFmpegError(f"{name!r} not found in PATH — apt install {name}")


def _ytdlp_cmd() -> list[str]:
    """Resolve yt-dlp invocation: CLI if on PATH, else `python -m yt_dlp`.

    Systemd-run services often don't have the venv's bin/ on PATH, so the
    CLI 'yt-dlp' lookup fails. The Python module is always available because
    it's installed in the same venv that runs gateway.
    """
    cli = shutil.which("yt-dlp")
    if cli:
        return [cli]
    return [sys.executable, "-m", "yt_dlp"]


def probe_duration(path: str | Path) -> float:
    """Return media duration in seconds. Raises if file is not media."""
    _check_tool("ffprobe")
    p = Path(path)
    if not p.exists():
        raise FFmpegError(f"probe_duration: {p} not found")
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(p)],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        raise FFmpegError(f"ffprobe failed: {r.stderr[:300]}")
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        raise FFmpegError(f"ffprobe parse error: {e}") from e


def download_video(*, url: str, output_path: str | Path,
                   max_height: int = 1080, timeout: int = 1800,
                   cookies_file: str | None = None,
                   proxy: str | None = None) -> Path:
    """Download MP4 via yt-dlp. Returns the actual output path.

    Format selector: best video up to max_height + best audio, merge to MP4.

    `cookies_file`: optional Netscape-format cookies.txt. Required when YouTube
    starts demanding "Sign in to confirm you're not a bot" (datacenter IPs hit
    this fast). Export via a browser extension on a logged-in machine.

    `proxy`: optional URL like 'http://user:pass@host:port' or 'socks5://host:port'.
    Required for production downloads from VPS — YouTube blocks datacenter IPs
    even with valid cookies+PO Token.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best"
    log.info(f"yt-dlp: download {url} → {out} (proxy={'yes' if proxy else 'no'})")
    args = _ytdlp_cmd() + [
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--no-warnings",
        "--remote-components", "ejs:github",  # EJS challenge solver (needed for 2025+ YouTube)
        "-o", str(out),
    ]
    if cookies_file:
        cookies_path = Path(cookies_file).expanduser()
        if cookies_path.exists():
            args.extend(["--cookies", str(cookies_path)])
        else:
            log.warning(f"yt-dlp: cookies file {cookies_path} missing, downloading without")
    if proxy:
        args.extend(["--proxy", proxy])
    args.append(url)
    # Ensure deno is on PATH for ejs challenge solving
    env = os.environ.copy()
    home = str(Path.home())
    deno_bin = f"{home}/.deno/bin"
    if deno_bin not in env.get("PATH", ""):
        env["PATH"] = f"{deno_bin}:{env.get('PATH', '')}"
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
    if r.returncode != 0:
        raise FFmpegError(f"yt-dlp failed: {r.stderr[-500:]}")
    if not out.exists():
        # yt-dlp sometimes appends an extension; try to find it
        candidates = sorted(out.parent.glob(out.name + "*"))
        if candidates:
            return candidates[0]
        raise FFmpegError(f"yt-dlp produced no output at {out}")
    return out


def cut_segments(*, input_path: str | Path, cuts: list[dict],
                 output_path: str | Path) -> Path:
    """Remove the given cut ranges and concat the rest into one MP4.

    Args:
        input_path:  source video file
        cuts:        list of {"start": float, "end": float, ...} (extras ignored)
        output_path: destination .mp4

    Returns the output_path Path.
    Empty `cuts` → just copy the file.
    """
    _check_tool("ffmpeg")
    src = Path(input_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FFmpegError(f"cut_segments: input {src} not found")

    # Normalize and validate cuts
    normalized = _normalize_cuts(cuts)
    duration = probe_duration(src)
    keep = _invert_cuts(normalized, duration)

    if not keep:
        raise FFmpegError("cut_segments: keep list is empty (cuts cover entire video)")

    # If we're keeping everything (or nearly), just copy
    if len(keep) == 1 and keep[0][0] <= 0.05 and keep[0][1] >= duration - 0.05:
        shutil.copyfile(src, dst)
        return dst

    # Re-encode each keep segment to its own file, then concat with the demuxer.
    # We re-encode (not copy) to guarantee keyframe alignment at cut points;
    # otherwise concat-copy can produce broken playback.
    with tempfile.TemporaryDirectory(prefix="ffmpeg-cut-") as tmpdir:
        tmp = Path(tmpdir)
        seg_paths: list[Path] = []
        for i, (s, e) in enumerate(keep):
            seg = tmp / f"seg{i:03d}.mp4"
            r = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(src),
                    "-ss", f"{s:.3f}",
                    "-to", f"{e:.3f}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-c:a", "aac", "-b:a", "160k",
                    "-movflags", "+faststart",
                    str(seg),
                ],
                capture_output=True, text=True,
                timeout=max(300, int((e - s) * 4)),
            )
            if r.returncode != 0 or not seg.exists():
                raise FFmpegError(
                    f"ffmpeg segment {i} ({s:.1f}-{e:.1f}) failed: {r.stderr[-400:]}"
                )
            seg_paths.append(seg)

        # Build concat list file
        list_file = tmp / "concat.txt"
        list_file.write_text("\n".join(f"file '{p}'" for p in seg_paths))

        r = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                "-movflags", "+faststart",
                str(dst),
            ],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0 or not dst.exists():
            raise FFmpegError(f"ffmpeg concat failed: {r.stderr[-400:]}")

    return dst


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_cuts(cuts: list[dict]) -> list[tuple[float, float]]:
    """Coerce, sort, merge overlapping ranges. Drops zero-length items."""
    pairs: list[tuple[float, float]] = []
    for c in cuts or []:
        try:
            s = float(c.get("start", 0))
            e = float(c.get("end", 0))
        except (TypeError, ValueError):
            continue
        if e > s:
            pairs.append((s, e))
    pairs.sort()
    merged: list[tuple[float, float]] = []
    for s, e in pairs:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _invert_cuts(cuts: list[tuple[float, float]], total: float) -> list[tuple[float, float]]:
    """Given cut ranges in [0, total], return the keep-list."""
    keep: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in cuts:
        if s > cursor + 0.05:
            keep.append((cursor, min(s, total)))
        cursor = max(cursor, e)
    if cursor < total - 0.05:
        keep.append((cursor, total))
    return keep
