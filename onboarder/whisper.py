"""OpenAI Whisper API wrapper for the onboarder pipeline.

Two operations:
    transcribe(file_path, *, language=None) -> {text, language, words, duration}
        word-level timestamps for cut detection.
    detect_language(file_path) -> str (ISO 639-1)
        cheap pass that uses verbose_json with no word timestamps.

Whisper pricing (~$0.006/min) — pretty cheap. For phase 2 we transcribe
twice per video (once on original to find cuts, once on final post-edit
output to store in DB), so budget ~$0.012/min × ~25 min × 7 lessons ≈ $2/course.

File size limit: OpenAI enforces 25 MB. Before sending we compress audio to
16kHz mono 64kbps mp3. 90-min video ≈ 43 MB uncompressed → 43 MB, so for
very long files we also split and merge transcriptions.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("gateway")

WHISPER_MODEL = "whisper-1"
WHISPER_MAX_BYTES = 24 * 1024 * 1024   # 24 MB hard cap (API limit is 25 MB)


class WhisperError(RuntimeError):
    pass


def _client(api_key: str) -> Any:
    """Lazy-import OpenAI client."""
    from openai import OpenAI
    if not api_key:
        raise WhisperError("openai api key is empty")
    return OpenAI(api_key=api_key)


def _compress_audio(src: Path, dst: Path) -> Path:
    """Extract & compress audio to 16kHz mono 64kbps mp3 for Whisper.

    30 min video → ~14 MB. 90 min → ~43 MB (still needs splitting).
    """
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-vn", "-ar", "16000", "-ac", "1", "-ab", "64k",
         "-f", "mp3", str(dst)],
        capture_output=True, timeout=300,
    )
    if r.returncode != 0:
        raise WhisperError(f"ffmpeg audio compress failed: {r.stderr[-400:]}")
    return dst


def _split_audio(src: Path, tmpdir: Path, chunk_bytes: int = WHISPER_MAX_BYTES) -> list[Path]:
    """Split large mp3 into ~chunk_bytes pieces. Returns list of chunk paths."""
    size = src.stat().st_size
    if size <= chunk_bytes:
        return [src]

    # Estimate duration from file size (64kbps = 8000 bytes/sec)
    bytes_per_sec = 8000
    chunk_sec = max(60, chunk_bytes // bytes_per_sec - 10)

    chunks: list[Path] = []
    offset = 0
    idx = 0
    while True:
        chunk = tmpdir / f"chunk_{idx:03d}.mp3"
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-ss", str(offset), "-t", str(chunk_sec),
             "-c", "copy", str(chunk)],
            capture_output=True, timeout=120,
        )
        if r.returncode != 0 or not chunk.exists() or chunk.stat().st_size < 1000:
            break
        chunks.append(chunk)
        offset += chunk_sec
        idx += 1

    return chunks or [src]


def _transcribe_one(client: Any, path: Path, language: str | None,
                    with_word_timestamps: bool) -> dict[str, Any]:
    """Transcribe a single file (already ≤25 MB)."""
    granularities = ["word", "segment"] if with_word_timestamps else ["segment"]
    with path.open("rb") as f:
        try:
            resp = client.audio.transcriptions.create(
                model=WHISPER_MODEL, file=f,
                response_format="verbose_json",
                timestamp_granularities=granularities,
                language=language,
            )
        except Exception as e:
            raise WhisperError(f"whisper API error: {e}") from e
    data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    return {
        "text": data.get("text", ""),
        "language": data.get("language", "") or "",
        "duration": float(data.get("duration") or 0.0),
        "words": data.get("words") or [],
        "segments": data.get("segments") or [],
    }


def transcribe(*, api_key: str, file_path: str | Path,
               language: str | None = None,
               with_word_timestamps: bool = True) -> dict[str, Any]:
    """Transcribe an audio/video file. Returns:
        {
          "text": str,                # plain transcript
          "language": str,            # ISO 639-1 (e.g. "en", "ru")
          "duration": float,          # seconds
          "words": [{"word": str, "start": float, "end": float}, ...],
          "segments": [...]           # raw whisper segments (caller may ignore)
        }
    """
    p = Path(file_path)
    if not p.exists():
        raise WhisperError(f"whisper: file not found: {p}")

    client = _client(api_key)
    log.info(f"whisper: transcribing {p.name} ({p.stat().st_size/1024/1024:.1f} MB), "
             f"lang={language or 'auto'}")

    with tempfile.TemporaryDirectory(prefix="whisper-") as tmpdir:
        tmp = Path(tmpdir)
        # Compress to 16kHz mono 64kbps — keeps quality good for speech, cuts size ~5x
        compressed = _compress_audio(p, tmp / "audio.mp3")
        log.info(f"whisper: compressed {p.stat().st_size/1024/1024:.1f} MB → "
                 f"{compressed.stat().st_size/1024/1024:.1f} MB")

        chunks = _split_audio(compressed, tmp)
        log.info(f"whisper: {len(chunks)} chunk(s)")

        if len(chunks) == 1:
            return _transcribe_one(client, chunks[0], language, with_word_timestamps)

        # Multi-chunk: transcribe each, merge text + shift timestamps
        merged_text = ""
        merged_words: list[Any] = []
        merged_segs: list[Any] = []
        detected_lang = language or ""
        total_dur = 0.0
        time_offset = 0.0

        for chunk in chunks:
            res = _transcribe_one(client, chunk, language or detected_lang or None,
                                  with_word_timestamps)
            if not detected_lang:
                detected_lang = res.get("language", "")
            merged_text += (" " if merged_text else "") + res["text"]
            chunk_dur = res.get("duration") or 0.0
            for w in res.get("words") or []:
                merged_words.append({**w,
                    "start": w.get("start", 0) + time_offset,
                    "end": w.get("end", 0) + time_offset})
            for s in res.get("segments") or []:
                merged_segs.append({**s,
                    "start": s.get("start", 0) + time_offset,
                    "end": s.get("end", 0) + time_offset})
            time_offset += chunk_dur
            total_dur += chunk_dur

        return {
            "text": merged_text.strip(),
            "language": detected_lang,
            "duration": total_dur,
            "words": merged_words,
            "segments": merged_segs,
        }


def detect_language(*, api_key: str, file_path: str | Path) -> str:
    """Quick language detection. Same cost as full transcription, but caller
    may call this before deciding whether to do the full word-timestamp pass.
    """
    result = transcribe(api_key=api_key, file_path=file_path,
                        with_word_timestamps=False)
    return result["language"]
