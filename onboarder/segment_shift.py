"""Reproject Whisper segments from original timeline to cleaned timeline.

Phase 1 stores Whisper segments (with word-level timestamps) in pipeline.db,
along with the list of cuts FFmpeg will remove. Phase 2 doesn't re-run Whisper
on the cleaned video — instead, it uses this module to:

  - drop words/segments whose timestamps fall inside a cut
  - shift remaining timestamps backwards by the total duration of cuts that
    occurred before each timestamp

The output mimics what Whisper would have returned if it transcribed the
cleaned video directly. Downstream (google_dub.dub_video, final EN transcript
joining) uses (start, end, text) per segment.

Word-level granularity is critical: Whisper segments span 5-30 seconds, but
cuts often happen mid-sentence (intro/outro/promo). Without word timestamps
we'd either drop entire segments at cut boundaries (large desync) or include
words that aren't in the cleaned audio. Word-level shifting bounds the error
to ~one word width (~0.3-0.5 sec).
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("gateway")


def shift_segments_through_cuts(
    segments: list[dict[str, Any]],
    cuts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return new segments aligned to the cleaned video timeline.

    Args:
        segments: Whisper output. Each item:
            {"start": float, "end": float, "text": str,
             "words": [{"start": float, "end": float, "word": str}, ...]}
            (words optional; falls back to segment-level granularity if missing)
        cuts: list of cut ranges in original-video time:
            [{"start": float, "end": float, "reason": str}, ...]

    Returns:
        list of {"start": float, "end": float, "text": str} in cleaned-video time.
        Words inside cuts are dropped. Empty segments (no surviving words) are
        excluded. Order preserved.
    """
    if not segments:
        return []

    # Normalize cuts: list of (start, end) tuples sorted by start
    sorted_cuts = sorted(
        [(float(c.get("start", 0)), float(c.get("end", 0)))
         for c in (cuts or [])
         if float(c.get("end", 0)) > float(c.get("start", 0))],
        key=lambda x: x[0],
    )

    if not sorted_cuts:
        # No cuts — just strip word-level info and pass through
        return [
            {"start": float(s.get("start", 0)),
             "end": float(s.get("end", 0)),
             "text": (s.get("text") or "").strip()}
            for s in segments
            if (s.get("text") or "").strip()
        ]

    def is_in_cut(t: float) -> bool:
        for c_start, c_end in sorted_cuts:
            if c_start <= t < c_end:
                return True
            if c_start > t:
                break  # cuts sorted, no later cut can contain t
        return False

    def shift(t: float) -> float:
        """Map original timestamp → cleaned timestamp.

        If t falls inside a cut, snap to the cut's start (post-shift).
        Otherwise subtract total cut duration that occurred before t.
        """
        delta = 0.0
        for c_start, c_end in sorted_cuts:
            if c_end <= t:
                delta += (c_end - c_start)
            elif c_start <= t < c_end:
                return max(0.0, c_start - delta)
            else:
                break
        return max(0.0, t - delta)

    out: list[dict[str, Any]] = []
    for seg in segments:
        words = seg.get("words") or []

        if not words:
            # No word-level info — treat segment as atomic.
            s_start = float(seg.get("start", 0))
            s_end = float(seg.get("end", 0))
            # If both endpoints fall inside any cut, drop. Otherwise shift.
            if is_in_cut(s_start) and is_in_cut(s_end):
                continue
            new_start = shift(s_start)
            new_end = shift(s_end)
            if new_end <= new_start:
                continue
            out.append({
                "start": new_start,
                "end": new_end,
                "text": (seg.get("text") or "").strip(),
            })
            continue

        # Word-level: keep only words whose midpoint is outside every cut.
        kept = []
        for w in words:
            try:
                w_start = float(w.get("start", 0))
                w_end = float(w.get("end", 0))
            except (TypeError, ValueError):
                continue
            w_mid = (w_start + w_end) / 2.0
            if is_in_cut(w_mid):
                continue
            kept.append({
                "start": w_start,
                "end": w_end,
                "word": (w.get("word") or "").strip(),
            })

        if not kept:
            continue  # entire segment fell inside cuts

        new_start = shift(kept[0]["start"])
        new_end = shift(kept[-1]["end"])
        if new_end <= new_start:
            continue
        new_text = " ".join(w["word"] for w in kept if w["word"]).strip()
        if not new_text:
            continue
        out.append({
            "start": new_start,
            "end": new_end,
            "text": new_text,
        })

    return out


def join_transcript(segments: list[dict[str, Any]]) -> str:
    """Concatenate shifted segments into a plain final transcript text."""
    return " ".join((s.get("text") or "").strip() for s in segments).strip()
