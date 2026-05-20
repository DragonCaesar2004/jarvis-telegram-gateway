"""Auto-detect and blur watermarks/channel logos in videos via Claude vision.

Phase 2 standalone helper. Sample N frames from the cleaned video,
downscale to a small JPEG, ask Claude Haiku where the watermark sits
(normalized bbox in 0-1 coords), then apply an ffmpeg boxblur over each
detected region.

Standalone CLI for testing:

    python -m onboarder.watermark_blur INPUT.mp4 OUTPUT.mp4 \\
        [--frames 3] [--max-height 480] [--quality 5] \\
        [--model haiku] [--blur 20] [--keep-frames] [--verbose]

When the user is happy with the visual result we'll wire it into
phase2_production between the `cut` and `dub` steps.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("watermark_blur")


# ---------------------------------------------------------------------------
# Vision detection
# ---------------------------------------------------------------------------

DETECTION_SYSTEM = """You analyze video frames to find LOGOS and WATERMARKS that the editor needs to blur out before publishing.

## What COUNTS as a watermark (REPORT these — they are exactly what we are looking for)

- **Social media handles** prefixed with `@` (e.g. `@username`, `@nancybadillo13`)
- **Platform icons** burned into a corner: Instagram camera, TikTok note, YouTube play, Facebook `f`, X / Twitter bird
- **Channel names or brand text** placed in a corner or along an edge
- **Subscribe bugs** ("Subscribe", "Follow", "Like and subscribe", bell-icon overlays)
- **URLs / domain names** burned into the frame (`example.com`, `www.…`)
- **Show / channel logos** (a designed mark or wordmark in a corner)
- **Hashtags** in a static corner position (`#brandname`)

If you see ANY of those in a fixed position across frames, you MUST report it. Do not second-guess these — they are unambiguous brand marks.

## What does NOT count (do NOT report)

- The presenter's face, hands, body, hair, clothing
- On-screen instructional text directly tied to the lesson (exercise names, anatomical labels, timestamps, captions/subtitles, set counts)
- Furniture, plants, decor in the background
- Lighting flares, reflections
- The actual lesson content

## How to be sure

I give you N frames sampled from the SAME video. A real watermark sits in the **same pixel position** in every one of them. If a candidate is in different spots between frames, it is content, not a watermark — skip it.

## Output format

Return ONLY a JSON object, no commentary, no code fences:

```
{
  "watermarks": [
    {
      "x": 0.00,
      "y": 0.00,
      "w": 0.22,
      "h": 0.08,
      "reason": "Instagram handle @nancybadillo13 in top-left corner"
    }
  ]
}
```

- `x`, `y` = top-left corner of the bbox, as a fraction (0.0 - 1.0) of frame width / height.
- `w`, `h` = width / height of the bbox, same fraction units.
- Pad the tight bbox by ~2% of frame width/height on each side so the blur fully covers the mark.
- If no watermarks: `{"watermarks": []}`.
- Multiple watermarks: list each one separately.
- Reject any candidate whose center lies in the **central 60% × 60%** of the frame — that area is almost always content.

Default toward REPORTING when you see a handle, an icon, a URL, or branded text in an edge position. Misses are worse than blurring over a corner of background.
"""


def _claude_token() -> str | None:
    """Return the OAuth token if available — gateway already has it in env."""
    return os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")


def _run_claude_vision(frame_paths: list[Path], model: str, timeout: int) -> dict[str, Any]:
    """Send frames to claude -p via the Read tool; parse JSON reply."""
    frames_block = "\n".join(
        f"- Frame {i + 1}: {p.absolute()}" for i, p in enumerate(frame_paths)
    )
    user = (
        f"Here are {len(frame_paths)} frames sampled from the same video. "
        f"Use the Read tool to load each file, then return the watermark JSON.\n\n"
        f"{frames_block}\n\n"
        "Return ONLY the JSON described in the system prompt, no commentary, "
        "no code fences."
    )
    full_prompt = DETECTION_SYSTEM + "\n\n---\n\n" + user

    add_dirs: list[str] = []
    seen: set[str] = set()
    for p in frame_paths:
        d = str(p.absolute().parent)
        if d not in seen:
            seen.add(d)
            add_dirs.append(d)

    cmd = ["claude", "-p", "--model", model,
           "--output-format", "text",
           "--permission-mode", "bypassPermissions"]
    for d in add_dirs:
        cmd.extend(["--add-dir", d])

    env = os.environ.copy()
    env.setdefault("PATH", f"{Path.home()}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin")

    log.debug(f"claude cmd: {cmd}")
    r = subprocess.run(cmd, input=full_prompt, capture_output=True,
                       text=True, timeout=timeout, env=env)
    if r.returncode != 0:
        raise RuntimeError(
            f"claude vision exited {r.returncode}: stderr={r.stderr[:300]} "
            f"stdout={r.stdout[:300]}"
        )
    text = (r.stdout or "").strip()
    if not text:
        raise RuntimeError(f"claude vision returned empty stdout. stderr={r.stderr[:300]!r}")

    # Strip code fences if model wrapped JSON in ```json ... ```
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl >= 0:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[: text.rfind("```")].rstrip()

    # Tolerate trailing prose after the JSON object.
    try:
        obj, end_idx = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"claude vision returned non-JSON: {e}\nRaw: {text[:500]}")
    if end_idx < len(text):
        tail = text[end_idx:].strip()
        if tail:
            log.debug(f"ignored {len(tail)} trailing chars after JSON")
    return obj


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def _probe_duration(path: Path) -> float:
    """Get duration in seconds via ffprobe."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         str(path)],
        text=True,
    ).strip()
    return float(out)


def _probe_resolution(path: Path) -> tuple[int, int]:
    """Get (width, height) in pixels via ffprobe."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error",
         "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=s=x:p=0",
         str(path)],
        text=True,
    ).strip()
    w, h = out.split("x")
    return int(w), int(h)


def _sample_frames(input_path: Path, n_frames: int, max_height: int,
                   quality: int, scratch_dir: Path, trim_seconds: float) -> list[Path]:
    """Sample n_frames JPEGs from the video, evenly spaced, downscaled.

    Skips the first/last `trim_seconds` so we don't accidentally sample an
    intro/outro card (they'd skew the "same place across frames" signal).
    """
    duration = _probe_duration(input_path)
    usable_start = trim_seconds if duration > 2 * trim_seconds + 4 else 0.5
    usable_end = max(usable_start + 1, duration - trim_seconds)

    timestamps = [
        usable_start + (usable_end - usable_start) * (i + 0.5) / n_frames
        for i in range(n_frames)
    ]

    out_paths: list[Path] = []
    for i, ts in enumerate(timestamps):
        out = scratch_dir / f"frame_{i:02d}_t{int(ts)}.jpg"
        # -ss before -i = fast seek (less accurate but enough for our purposes)
        # scale=-2:H keeps aspect ratio; -2 forces even width (required by yuv420)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", str(ts), "-i", str(input_path),
             "-vframes", "1",
             "-vf", f"scale=-2:{max_height}",
             "-q:v", str(quality),
             str(out)],
            check=True,
        )
        out_paths.append(out)
    return out_paths


# ---------------------------------------------------------------------------
# Filter graph construction
# ---------------------------------------------------------------------------

def _build_blur_filter(bboxes: list[dict[str, Any]], video_w: int, video_h: int,
                      blur_strength: int) -> str:
    """Build an ffmpeg filter_complex graph that blurs every bbox.

    For N bboxes:
        [0:v]split=N+1[base][b0_in][b1_in]...[bN-1_in];
        [bi_in]crop=W:H:X:Y,boxblur=R[bi];
        [base][b0]overlay=X0:Y0[t0];[t0][b1]overlay=X1:Y1[t1];... → final

    Returns the filter_complex string ready for `-filter_complex`.
    """
    if not bboxes:
        return ""

    pixel_boxes = []
    for bb in bboxes:
        x = int(float(bb["x"]) * video_w)
        y = int(float(bb["y"]) * video_h)
        w = int(float(bb["w"]) * video_w)
        h = int(float(bb["h"]) * video_h)
        # Clamp into frame
        x = max(0, min(video_w - 1, x))
        y = max(0, min(video_h - 1, y))
        w = max(2, min(video_w - x, w))
        h = max(2, min(video_h - y, h))
        # ffmpeg crop expects even dims for yuv420
        w = w - (w % 2)
        h = h - (h % 2)
        pixel_boxes.append((x, y, w, h))

    n = len(pixel_boxes)
    split_labels = ["base"] + [f"b{i}_in" for i in range(n)]
    parts = [f"[0:v]split={n + 1}" + "".join(f"[{lbl}]" for lbl in split_labels)]

    for i, (x, y, w, h) in enumerate(pixel_boxes):
        # gblur (Gaussian blur) instead of boxblur — `sigma` has no upper
        # cap, unlike boxblur where chroma radius is hard-capped at 14.
        # Visual quality is also smoother. `sigma=N` ≈ a moderately stronger
        # blur than `boxblur=N/2`.
        parts.append(
            f"[b{i}_in]crop={w}:{h}:{x}:{y},gblur=sigma={blur_strength}[b{i}]"
        )

    prev = "base"
    for i, (x, y, _, _) in enumerate(pixel_boxes):
        out_label = "out" if i == n - 1 else f"t{i}"
        parts.append(f"[{prev}][b{i}]overlay={x}:{y}[{out_label}]")
        prev = out_label

    return ";".join(parts)


def _apply_blur(input_path: Path, output_path: Path,
                bboxes: list[dict[str, Any]],
                video_w: int, video_h: int,
                blur_strength: int) -> None:
    """Re-encode video with the boxblur filtergraph for each bbox."""
    filtergraph = _build_blur_filter(bboxes, video_w, video_h, blur_strength)
    if not filtergraph:
        # No watermarks — just copy through
        shutil.copy(str(input_path), str(output_path))
        return

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(input_path),
        "-filter_complex", filtergraph,
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]
    log.debug(f"ffmpeg blur cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process(input_path: str | Path, output_path: str | Path, *,
            n_frames: int = 3,
            max_height: int = 480,
            jpeg_quality: int = 5,
            model: str = "haiku",
            blur_strength: int = 20,
            trim_seconds: float = 3.0,
            timeout: int = 120,
            keep_frames: bool = False) -> dict[str, Any]:
    """Detect + blur watermarks. Returns a report dict.

    Returns:
        {
          "input": ..., "output": ...,
          "duration_sec": float, "width": int, "height": int,
          "frames_sampled": int,
          "watermarks": [{x, y, w, h, reason}, ...],
          "filtergraph": str,
          "blurred": bool,   # False if no watermarks were detected
        }
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input video not found: {input_path}")

    duration = _probe_duration(input_path)
    video_w, video_h = _probe_resolution(input_path)
    log.info(f"video: {video_w}x{video_h}, {duration:.1f}s")

    scratch = Path(tempfile.mkdtemp(prefix="watermark-blur-"))
    log.debug(f"scratch dir: {scratch}")

    try:
        frames = _sample_frames(input_path, n_frames=n_frames,
                                max_height=max_height, quality=jpeg_quality,
                                scratch_dir=scratch, trim_seconds=trim_seconds)
        log.info(f"sampled {len(frames)} frames at max_height={max_height}, "
                 f"quality={jpeg_quality}")

        detection = _run_claude_vision(frames, model=model, timeout=timeout)
        wms = detection.get("watermarks") or []
        log.info(f"claude found {len(wms)} watermark region(s)")
        for w in wms:
            log.info(f"  • {w.get('reason', '?')[:80]}  "
                     f"x={w.get('x'):.3f} y={w.get('y'):.3f} "
                     f"w={w.get('w'):.3f} h={w.get('h'):.3f}")

        filtergraph = _build_blur_filter(wms, video_w, video_h, blur_strength)
        _apply_blur(input_path, output_path, wms, video_w, video_h, blur_strength)

        return {
            "input": str(input_path),
            "output": str(output_path),
            "duration_sec": duration,
            "width": video_w,
            "height": video_h,
            "frames_sampled": len(frames),
            "watermarks": wms,
            "filtergraph": filtergraph,
            "blurred": bool(wms),
            "scratch_dir": str(scratch) if keep_frames else None,
        }
    finally:
        if not keep_frames:
            try:
                shutil.rmtree(scratch, ignore_errors=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> int:
    p = argparse.ArgumentParser(
        description="Detect + blur watermarks in a video via Claude vision.",
    )
    p.add_argument("input", help="input video path")
    p.add_argument("output", help="output video path")
    p.add_argument("--frames", type=int, default=3,
                   help="frames to sample (default: 3)")
    p.add_argument("--max-height", type=int, default=480,
                   help="downscale sampled frames to this height (default: 480)")
    p.add_argument("--quality", type=int, default=5,
                   help="JPEG quality 2-31, lower=better (default: 5)")
    p.add_argument("--model", default="haiku",
                   help="claude model (default: haiku)")
    p.add_argument("--blur", type=int, default=20,
                   help="boxblur strength (default: 20)")
    p.add_argument("--trim", type=float, default=3.0,
                   help="skip first/last N seconds when sampling (default: 3)")
    p.add_argument("--timeout", type=int, default=120,
                   help="claude vision timeout in seconds (default: 120)")
    p.add_argument("--keep-frames", action="store_true",
                   help="keep extracted JPEG frames on disk for inspection")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="debug logging")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        report = process(
            args.input, args.output,
            n_frames=args.frames,
            max_height=args.max_height,
            jpeg_quality=args.quality,
            model=args.model,
            blur_strength=args.blur,
            trim_seconds=args.trim,
            timeout=args.timeout,
            keep_frames=args.keep_frames,
        )
    except Exception as e:
        log.error(f"failed: {e}")
        return 1

    print("\n=== detection report ===")
    print(json.dumps({k: v for k, v in report.items() if k != "filtergraph"},
                     indent=2, ensure_ascii=False))
    if report["filtergraph"]:
        print(f"\nfilter_complex:\n  {report['filtergraph']}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
