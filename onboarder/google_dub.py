"""Google Cloud dubbing: Translate → TTS → FFmpeg mix.

Pipeline (per non-English video):
    1. Whisper already ran on cleaned video → segments with word timestamps
    2. Batch-translate all segment texts via Google Cloud Translation API v2
    3. Synthesize each translated segment via Google Cloud TTS v1 (parallel)
    4. Build a new audio track with FFmpeg:
         - silent base (full video duration)
         - each TTS clip placed at its segment's start timestamp
    5. Merge new audio track with video (replacing original audio)

Result: English TTS voice plays only when the speaker was talking;
silence during pauses / music / non-speech.

The translated segments are joined and returned as the final English
transcript — no separate Whisper call needed for dubbed videos.

Auth: both APIs accept a simple API key (no OAuth needed).
Set keys via config.json onboarder.google_translate_api_key / google_tts_api_key
or env vars GOOGLE_TRANSLATE_API_KEY / GOOGLE_TTS_API_KEY.
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger("gateway")

TRANSLATE_API = "https://translation.googleapis.com/language/translate/v2"
TTS_API = "https://texttospeech.googleapis.com/v1/text:synthesize"

# WaveNet voices (highest quality free tier; switch to Chirp3-HD for studio quality)
_VOICES: dict[str, dict[str, str]] = {
    "MALE": {"languageCode": "en-US", "name": "en-US-Wavenet-D", "ssmlGender": "MALE"},
    "FEMALE": {"languageCode": "en-US", "name": "en-US-Wavenet-F", "ssmlGender": "FEMALE"},
}

# Max parallel TTS workers. Cloud TTS rate limit is 300 RPM on Basic tier,
# so 8 parallel is safe for typical 20-30 min lecture (~150 segments).
TTS_WORKERS = 8


class DubError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Google Translate (batch)
# ---------------------------------------------------------------------------

# Google Translate v2 limits per request:
#   q array length: max 128 strings
#   total chars in q: max 30_000
#   single string: max 5_000 chars
# We chunk well under these so a long lecture (200+ Whisper segments) doesn't
# silently fail with HTTP 400. Picking 100 and 25k leaves headroom for the
# request envelope and source/target/format fields.
TRANSLATE_MAX_STRINGS_PER_REQ = 100
TRANSLATE_MAX_CHARS_PER_REQ = 25_000


def _chunk_for_translate(items: list[tuple[int, str]]
                         ) -> list[list[tuple[int, str]]]:
    """Split (idx, text) pairs into chunks under Google's per-request limits."""
    chunks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_chars = 0
    for idx, t in items:
        t_chars = len(t)
        # If adding this string would push us over limits, start a new chunk.
        # First-item check protects against a single huge string blocking forever.
        if current and (
            len(current) >= TRANSLATE_MAX_STRINGS_PER_REQ
            or current_chars + t_chars > TRANSLATE_MAX_CHARS_PER_REQ
        ):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append((idx, t))
        current_chars += t_chars
    if current:
        chunks.append(current)
    return chunks


def translate_batch(texts: list[str], source_lang: str, target_lang: str,
                    api_key: str, *, raise_on_failure: bool = False) -> list[str]:
    """Translate many texts. Chunks under Google's 128-string / 30K-char limits.

    `raise_on_failure=True` (used by dub_video): propagate the error so the
    caller doesn't accidentally feed source-language text to downstream TTS.
    `raise_on_failure=False` (default, used by sanitizer): return originals so
    a translate outage doesn't kill the whole Phase 2 — sanitization is a
    best-effort layer over already-English content.
    """
    non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
    if not non_empty:
        return texts

    result = list(texts)
    chunks = _chunk_for_translate(non_empty)

    for chunk_idx, chunk in enumerate(chunks):
        payload = {
            "q": [t for _, t in chunk],
            "source": _iso(source_lang),
            "target": _iso(target_lang),
            "format": "text",
        }
        r = None
        try:
            r = requests.post(TRANSLATE_API, params={"key": api_key},
                              json=payload, timeout=60)
            r.raise_for_status()
            translations = r.json()["data"]["translations"]
        except Exception as e:
            body = ""
            try:
                if r is not None:
                    body = r.text[:400]
            except Exception:
                pass
            msg = (f"google_dub: translate_batch failed at chunk "
                   f"{chunk_idx + 1}/{len(chunks)} "
                   f"({len(chunk)} strings, "
                   f"{sum(len(t) for _, t in chunk)} chars): {e}")
            if body:
                msg = f"{msg} body={body}"
            if raise_on_failure:
                raise DubError(msg) from e
            log.warning(f"{msg} — using originals for this chunk")
            continue  # leave originals in `result` for this chunk

        for (orig_idx, _), tr in zip(chunk, translations):
            result[orig_idx] = tr.get("translatedText", texts[orig_idx])

    return result


# ---------------------------------------------------------------------------
# Google Cloud TTS (per segment, parallelised)
# ---------------------------------------------------------------------------

def synthesize_speech(text: str, voice_gender: str, api_key: str) -> bytes:
    """Synthesize one text → MP3 bytes. Returns b'' on empty input."""
    if not text.strip():
        return b""
    voice = _VOICES.get((voice_gender or "MALE").upper(), _VOICES["MALE"])
    payload = {
        "input": {"text": text},
        "voice": voice,
        "audioConfig": {"audioEncoding": "MP3", "speakingRate": 1.05},
    }
    r = requests.post(TTS_API, params={"key": api_key}, json=payload, timeout=60)
    r.raise_for_status()
    b64 = r.json().get("audioContent", "")
    if not b64:
        raise DubError(f"TTS returned empty audioContent for text: {text[:60]!r}")
    return base64.b64decode(b64)


def _synthesize_parallel(texts: list[str], voice_gender: str, api_key: str,
                         max_workers: int = TTS_WORKERS) -> list[bytes | None]:
    """Synthesize all texts in parallel. Returns list aligned to input (None on error)."""
    results: list[bytes | None] = [None] * len(texts)

    def _job(idx: int, text: str) -> tuple[int, bytes | None]:
        if not text.strip():
            return idx, b""
        try:
            return idx, synthesize_speech(text, voice_gender, api_key)
        except Exception as e:
            log.warning(f"google_dub: TTS failed for segment {idx}: {e}")
            return idx, None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_job, i, t): i for i, t in enumerate(texts)}
        for fut in as_completed(futures):
            idx, data = fut.result()
            results[idx] = data

    return results


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def _probe_duration(path: Path) -> float:
    """Return video/audio duration in seconds via ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(r.stdout).get("streams", [])
    return max((float(s.get("duration", 0)) for s in streams), default=0.0)


def _build_audio_track(clips: list[tuple[float, Path]],
                       total_duration: float,
                       output: Path) -> None:
    """Mix a silent base with TTS clips placed at their start timestamps."""
    # Base: generate silence
    base_cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"anullsrc=r=44100:cl=mono:d={total_duration:.3f}",
        "-c:a", "pcm_s16le", str(output),
    ]
    subprocess.run(base_cmd, check=True, capture_output=True)

    if not clips:
        return

    # Overlay each TTS clip at its timestamp using adelay + amix
    inputs: list[str] = ["-i", str(output)]
    for _, mp3 in clips:
        inputs += ["-i", str(mp3)]

    filter_parts: list[str] = ["[0:a]acopy[base]"]
    labels = ["[base]"]
    for i, (start_sec, _) in enumerate(clips):
        delay_ms = int(start_sec * 1000)
        label = f"[d{i}]"
        filter_parts.append(f"[{i+1}:a]adelay={delay_ms}|{delay_ms}[d{i}]")
        labels.append(label)

    mix_in = "".join(labels)
    filter_parts.append(
        f"{mix_in}amix=inputs={len(labels)}:duration=first:normalize=0[out]"
    )

    mixed = output.with_suffix(".mixed.wav")
    cmd = (
        ["ffmpeg", "-y"] + inputs
        + ["-filter_complex", ";".join(filter_parts)]
        + ["-map", "[out]", "-ar", "44100", "-ac", "1", str(mixed)]
    )
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise DubError(f"ffmpeg amix failed: {r.stderr[-600:]}")

    # Replace silent base with the mixed result
    mixed.replace(output)


def _merge_audio_video(video: Path, audio: Path, output: Path) -> None:
    """Replace video's audio track with new audio."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-i", str(audio),
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise DubError(f"ffmpeg merge failed: {r.stderr[-600:]}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def dub_video(*,
              transcript: dict[str, Any],
              input_path: str | Path,
              output_path: str | Path,
              source_lang: str,
              target_lang: str = "en",
              voice_gender: str = "MALE",
              translate_api_key: str,
              tts_api_key: str,
              on_progress: Any = None) -> tuple[Path, str]:
    """Translate + synthesize + merge. Returns (dubbed_video_path, en_transcript_text).

    The returned transcript is the joined translated segments — callers can use
    it directly as the final English transcript without a separate Whisper call.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    segments = transcript.get("segments") or []
    if not segments:
        raise DubError("transcript has no segments — run Whisper with word_timestamps=True")

    src_iso = _iso(source_lang)
    tgt_iso = _iso(target_lang)

    # ── 1. Batch-translate all segment texts ─────────────────────────────
    seg_texts = [(s.get("text") or "").strip() for s in segments]

    if src_iso != tgt_iso:
        _emit(on_progress, f"🌐 Перевожу {len(seg_texts)} сегментов (Google Translate)…")
        # raise_on_failure=True: never let source-language text fall through to
        # TTS — that produces English-accented Russian, the worst of both
        # worlds. If translate fails (API disabled, network), fail the video.
        translated_texts = translate_batch(
            seg_texts, src_iso, tgt_iso, translate_api_key,
            raise_on_failure=True,
        )
    else:
        translated_texts = seg_texts

    en_transcript = " ".join(t for t in translated_texts if t)

    # ── 2. Synthesize all segments (parallel) ────────────────────────────
    _emit(on_progress, f"🔊 Синтезирую голос для {len(translated_texts)} сегментов "
                       f"(Google TTS, {voice_gender.lower()})…")
    audio_bytes_list = _synthesize_parallel(translated_texts, voice_gender, tts_api_key)

    with tempfile.TemporaryDirectory(prefix="google_dub_") as tmpdir:
        tmp = Path(tmpdir)

        # Save each TTS clip to a temp MP3 file
        clips: list[tuple[float, Path]] = []
        for i, (seg, audio_bytes) in enumerate(zip(segments, audio_bytes_list)):
            if not audio_bytes:
                continue
            start = float(seg.get("start") or 0)
            mp3 = tmp / f"seg_{i:05d}.mp3"
            mp3.write_bytes(audio_bytes)
            clips.append((start, mp3))

        if not clips:
            raise DubError("no TTS audio produced — all segments failed synthesis")

        _emit(on_progress, f"🎛 Монтирую аудио ({len(clips)} клипов)…")

        # ── 3. Build audio track ──────────────────────────────────────────
        total_dur = _probe_duration(input_path)
        audio_track = tmp / "track.wav"
        _build_audio_track(clips, total_dur, audio_track)

        # ── 4. Merge with video ───────────────────────────────────────────
        _emit(on_progress, "🎬 Подмешиваю аудио в видео…")
        _merge_audio_video(input_path, audio_track, output_path)

    _emit(on_progress, "✅ Дубляж готов")
    return output_path, en_transcript


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LANG_MAP = {
    "english": "en", "russian": "ru", "spanish": "es", "french": "fr",
    "german": "de", "portuguese": "pt", "italian": "it", "chinese": "zh",
    "japanese": "ja", "korean": "ko", "arabic": "ar", "hindi": "hi",
    "turkish": "tr", "polish": "pl", "dutch": "nl", "ukrainian": "uk",
}


def _iso(lang: str) -> str:
    """Normalise full language name → ISO 639-1 ('russian' → 'ru')."""
    return _LANG_MAP.get((lang or "").lower(), (lang or "").lower()[:2] or "en")


def _emit(callback: Any, msg: str) -> None:
    if callback:
        try:
            callback(msg)
        except Exception:
            pass
