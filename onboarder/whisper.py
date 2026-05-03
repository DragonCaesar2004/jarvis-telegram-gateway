"""OpenAI Whisper API wrapper for the onboarder pipeline.

Two operations:
    transcribe(file_path, *, language=None) -> {text, language, words, duration}
        word-level timestamps for cut detection.
    detect_language(file_path) -> str (ISO 639-1)
        cheap pass that uses verbose_json with no word timestamps.

Whisper pricing (~$0.006/min) — pretty cheap. For phase 2 we transcribe
twice per video (once on original to find cuts, once on final post-edit
output to store in DB), so budget ~$0.012/min × ~25 min × 7 lessons ≈ $2/course.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("gateway")

WHISPER_MODEL = "whisper-1"  # OpenAI's Whisper API model


class WhisperError(RuntimeError):
    pass


def _client(api_key: str) -> Any:
    """Lazy-import OpenAI client."""
    from openai import OpenAI
    if not api_key:
        raise WhisperError("openai api key is empty")
    return OpenAI(api_key=api_key)


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
    timestamp_granularities = ["word", "segment"] if with_word_timestamps else ["segment"]

    log.info(f"whisper: transcribing {p.name} ({p.stat().st_size/1024/1024:.1f} MB), "
             f"lang={language or 'auto'}")

    with p.open("rb") as f:
        try:
            resp = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=f,
                response_format="verbose_json",
                timestamp_granularities=timestamp_granularities,
                language=language,  # None = auto-detect
            )
        except Exception as e:
            raise WhisperError(f"whisper API error: {e}") from e

    # OpenAI's TranscriptionVerbose object → plain dict
    data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    return {
        "text": data.get("text", ""),
        "language": data.get("language", "") or "",
        "duration": float(data.get("duration") or 0.0),
        "words": data.get("words") or [],
        "segments": data.get("segments") or [],
    }


def detect_language(*, api_key: str, file_path: str | Path) -> str:
    """Quick language detection. Same cost as full transcription, but caller
    may call this before deciding whether to do the full word-timestamp pass.
    """
    result = transcribe(api_key=api_key, file_path=file_path,
                        with_word_timestamps=False)
    return result["language"]
