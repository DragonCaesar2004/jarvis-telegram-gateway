"""Whisper API wrapper for the onboarder pipeline.

Supports TWO providers via the same OpenAI-compatible SDK surface:

  * "openai" (default) — whisper-1 model on api.openai.com.
    Pricing $0.006/min ($0.36/h). Battle-tested, slower (~1-3× realtime).

  * "groq" — whisper-large-v3 model on api.groq.com/openai/v1.
    Pricing $0.00185/min ($0.111/h) — 3.2× cheaper. Faster (~10-20× realtime).
    Slightly better language detection (v3 > v2 of the model).
    Newer service — slightly less proven uptime than OpenAI.

Two operations:
    transcribe(file_path, *, language=None, provider=None) ->
        {text, language, words, duration, segments}
        word-level timestamps for cut detection.
    detect_language(file_path, provider=None) -> str
        cheap pass that uses verbose_json with no word timestamps.

The `provider` kwarg overrides the module-level default set via
`set_default_provider()` from gateway startup. Per-call override lets us
add a future fallback path (try groq, on error fall back to openai).

File size limit: 25 MB for both providers. Before sending we compress audio
to 16kHz mono 64kbps mp3. 90-min video ≈ 43 MB uncompressed → 43 MB, so for
very long files we also split and merge transcriptions. Groq also caps
audio duration at ~25 min per request — handled by a smaller chunk_bytes.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("gateway")

# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------

# Per-provider settings. Both expose OpenAI-compatible client surface.
WHISPER_PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "model": "whisper-1",
        "base_url": None,  # OpenAI SDK default
        # OpenAI accepts up to 25 MB, no explicit per-request duration cap
        "max_chunk_bytes": 24 * 1024 * 1024,
    },
    "groq": {
        "model": "whisper-large-v3",
        "base_url": "https://api.groq.com/openai/v1",
        # Groq enforces BOTH 25 MB file size AND a hard 25-min audio duration
        # cap per request when response_format=verbose_json with timestamps.
        # At 64 kbps that's ~11 MB worst case → use 10 MB to stay under both.
        "max_chunk_bytes": 10 * 1024 * 1024,
    },
}

_DEFAULT_PROVIDER: str = "openai"  # mutated by set_default_provider()
WHISPER_MAX_BYTES = 24 * 1024 * 1024   # legacy back-compat constant


# ---------------------------------------------------------------------------
# Language normalisation
# ---------------------------------------------------------------------------
#
# Whisper's verbose_json `language` field returns the FULL English name of the
# detected language ("english", "russian", "spanish", …), not an ISO 639-1
# code. If we feed that back as the `language=` parameter on a subsequent
# call, OpenAI returns 400:
#     "Invalid language 'english'. Language parameter must be specified in
#      ISO-639-1 format."
# This triggered specifically on multi-chunk audio (>~50 min on OpenAI,
# >~20 min on Groq) where chunks 2+ inherit the detected language from
# chunk 1 to skip re-detection. Single-chunk callers were unaffected
# because phase1_enrich/phase2 already pass the return value through
# elevenlabs_dub._iso themselves.

_WHISPER_LANG_TO_ISO: dict[str, str] = {
    "english": "en", "russian": "ru", "spanish": "es", "french": "fr",
    "german": "de", "portuguese": "pt", "italian": "it", "chinese": "zh",
    "japanese": "ja", "korean": "ko", "arabic": "ar", "hindi": "hi",
    "turkish": "tr", "polish": "pl", "dutch": "nl", "ukrainian": "uk",
    "vietnamese": "vi", "indonesian": "id", "thai": "th", "swedish": "sv",
    "norwegian": "no", "finnish": "fi", "danish": "da", "czech": "cs",
    "greek": "el", "hebrew": "he", "romanian": "ro", "hungarian": "hu",
    "bulgarian": "bg", "catalan": "ca", "croatian": "hr", "slovak": "sk",
    "slovenian": "sl", "estonian": "et", "latvian": "lv", "lithuanian": "lt",
    "malay": "ms", "tamil": "ta", "telugu": "te", "bengali": "bn",
    "urdu": "ur", "persian": "fa", "filipino": "tl", "tagalog": "tl",
}


def _normalize_lang_to_iso(lang: str) -> str | None:
    """Whisper full-word lang → ISO 639-1. None if we can't map it.

    None means "treat as auto-detect" — safer than passing back an unrecognised
    string that may or may not be a valid ISO. Multi-chunk callers chain this
    through `language or detected_lang or None`, so None correctly falls
    through to fresh auto-detection on the next chunk.
    """
    if not lang:
        return None
    s = lang.strip().lower()
    if not s:
        return None
    # Already-ISO (two-letter code) passes through unchanged.
    if len(s) == 2 and s.isalpha():
        return s
    return _WHISPER_LANG_TO_ISO.get(s)


def set_default_provider(provider: str) -> None:
    """Set the module-level default provider. Idempotent. Validates input."""
    global _DEFAULT_PROVIDER
    p = (provider or "openai").lower()
    if p not in WHISPER_PROVIDERS:
        raise WhisperError(
            f"unknown whisper provider {provider!r}; "
            f"known: {sorted(WHISPER_PROVIDERS)}"
        )
    if p != _DEFAULT_PROVIDER:
        log.info(f"whisper: default provider switched {_DEFAULT_PROVIDER!r} → {p!r}")
    _DEFAULT_PROVIDER = p


def _resolve_provider(provider: str | None) -> str:
    """Return validated provider name (per-call override falls back to default)."""
    p = (provider or _DEFAULT_PROVIDER).lower()
    if p not in WHISPER_PROVIDERS:
        raise WhisperError(
            f"unknown whisper provider {provider!r}; "
            f"known: {sorted(WHISPER_PROVIDERS)}"
        )
    return p


class WhisperError(RuntimeError):
    pass


def _client(api_key: str, provider: str | None = None) -> Any:
    """Lazy-import OpenAI-compatible client. Points at Groq endpoint when
    provider='groq' (Groq exposes a subset of the OpenAI API surface, so
    the same SDK works as drop-in)."""
    from openai import OpenAI
    if not api_key:
        raise WhisperError(f"{provider or _DEFAULT_PROVIDER} api key is empty")
    cfg = WHISPER_PROVIDERS[_resolve_provider(provider)]
    if cfg["base_url"]:
        return OpenAI(api_key=api_key, base_url=cfg["base_url"])
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
                    with_word_timestamps: bool, *,
                    provider: str | None = None) -> dict[str, Any]:
    """Transcribe a single file (already ≤ provider's per-request limit)."""
    model = WHISPER_PROVIDERS[_resolve_provider(provider)]["model"]
    granularities = ["word", "segment"] if with_word_timestamps else ["segment"]
    with path.open("rb") as f:
        try:
            resp = client.audio.transcriptions.create(
                model=model, file=f,
                response_format="verbose_json",
                timestamp_granularities=granularities,
                language=language,
            )
        except Exception as e:
            raise WhisperError(f"whisper API error ({provider or _DEFAULT_PROVIDER}): {e}") from e
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
               with_word_timestamps: bool = True,
               provider: str | None = None) -> dict[str, Any]:
    """Transcribe an audio/video file. Returns:
        {
          "text": str,                # plain transcript
          "language": str,            # ISO 639-1 (e.g. "en", "ru")
          "duration": float,          # seconds
          "words": [{"word": str, "start": float, "end": float}, ...],
          "segments": [...]           # raw whisper segments (caller may ignore)
        }

    `provider` overrides module-level default (set via set_default_provider).
    Use this for fallback flows ("try groq, on error retry with openai").
    """
    p = Path(file_path)
    if not p.exists():
        raise WhisperError(f"whisper: file not found: {p}")

    resolved = _resolve_provider(provider)
    client = _client(api_key, provider=resolved)
    chunk_cap = WHISPER_PROVIDERS[resolved]["max_chunk_bytes"]
    log.info(f"whisper[{resolved}]: transcribing {p.name} "
             f"({p.stat().st_size/1024/1024:.1f} MB), lang={language or 'auto'}")

    with tempfile.TemporaryDirectory(prefix="whisper-") as tmpdir:
        tmp = Path(tmpdir)
        # Compress to 16kHz mono 64kbps — keeps quality good for speech, cuts size ~5x
        compressed = _compress_audio(p, tmp / "audio.mp3")
        log.info(f"whisper[{resolved}]: compressed "
                 f"{p.stat().st_size/1024/1024:.1f} MB → "
                 f"{compressed.stat().st_size/1024/1024:.1f} MB")

        chunks = _split_audio(compressed, tmp, chunk_bytes=chunk_cap)
        log.info(f"whisper[{resolved}]: {len(chunks)} chunk(s)")

        if len(chunks) == 1:
            return _transcribe_one(client, chunks[0], language,
                                   with_word_timestamps, provider=resolved)

        # Multi-chunk: transcribe each, merge text + shift timestamps.
        # `detected_lang` is the language hint passed to chunks 2+ to skip
        # re-detection. It MUST be in ISO 639-1 form ("en", not "english") —
        # OpenAI rejects full-name input with HTTP 400. We normalise the
        # first-chunk auto-detected response through _normalize_lang_to_iso
        # before reusing it; if normalisation fails (unknown name), we leave
        # detected_lang as None and chunks 2+ also auto-detect.
        merged_text = ""
        merged_words: list[Any] = []
        merged_segs: list[Any] = []
        detected_lang: str | None = _normalize_lang_to_iso(language or "")
        total_dur = 0.0
        time_offset = 0.0

        for chunk in chunks:
            res = _transcribe_one(client, chunk, language or detected_lang or None,
                                  with_word_timestamps, provider=resolved)
            if not detected_lang:
                detected_lang = _normalize_lang_to_iso(res.get("language", ""))
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


def detect_language(*, api_key: str, file_path: str | Path,
                    provider: str | None = None) -> str:
    """Quick language detection. Same cost as full transcription, but caller
    may call this before deciding whether to do the full word-timestamp pass.
    """
    result = transcribe(api_key=api_key, file_path=file_path,
                        with_word_timestamps=False, provider=provider)
    return result["language"]
