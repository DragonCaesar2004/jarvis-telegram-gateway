"""ElevenLabs Dubbing Studio API wrapper.

Three steps:
    1. submit_dub(file_path, source_lang, target_lang) → dubbing_id
    2. wait_for_completion(dubbing_id, timeout=2400) → polls every 30s
    3. download_dubbed(dubbing_id, target_lang, output_path) → saves dubbed MP4

`dub_video()` ties all three together as a single blocking call. Used by
phase2_production for non-English videos. English videos skip dubbing entirely.

Pricing (2026): ~2000 credits per minute of dubbed audio. A typical Pro plan
($99/mo) gets 500K credits — about 4 hours of dubbing per month.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger("gateway")

API_BASE = "https://api.elevenlabs.io/v1"
DEFAULT_POLL_INTERVAL = 30
DEFAULT_TIMEOUT = 2400  # 40 min — enough for ~30 min of source video


class DubError(RuntimeError):
    pass


_LANG_NAME_TO_ISO: dict[str, str] = {
    "english": "en", "russian": "ru", "spanish": "es", "french": "fr",
    "german": "de", "portuguese": "pt", "italian": "it", "chinese": "zh",
    "japanese": "ja", "korean": "ko", "arabic": "ar", "hindi": "hi",
    "turkish": "tr", "polish": "pl", "dutch": "nl", "ukrainian": "uk",
}


def _iso(lang: str) -> str:
    """Normalise full language name → ISO 639-1 code ('english' → 'en')."""
    return _LANG_NAME_TO_ISO.get(lang.lower(), lang.lower())


def submit_dub(*, api_key: str, file_path: str | Path,
               source_lang: str, target_lang: str = "en",
               name: str | None = None,
               watermark: bool = False,
               timeout: int = 300) -> str:
    """POST /v1/dubbing → returns dubbing_id (str)."""
    p = Path(file_path)
    if not p.exists():
        raise DubError(f"dub: file not found: {p}")

    url = f"{API_BASE}/dubbing"
    fields = {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "watermark": "true" if watermark else "false",
        "highest_resolution": "true",
    }
    if name:
        fields["name"] = name

    source_lang = _iso(source_lang)
    target_lang = _iso(target_lang)
    log.info(f"elevenlabs: submitting dub {p.name}, {source_lang} → {target_lang}")
    try:
        with p.open("rb") as f:
            r = requests.post(
                url,
                headers={"xi-api-key": api_key, "Accept": "application/json"},
                files={"file": (p.name, f, "video/mp4")},
                data=fields,
                timeout=timeout,
            )
    except requests.RequestException as e:
        raise DubError(f"dub submit network error: {e}") from e

    if r.status_code >= 400:
        raise DubError(f"dub submit {r.status_code}: {r.text[:400]}")

    try:
        data = r.json()
    except ValueError:
        raise DubError(f"dub submit non-JSON: {r.text[:300]}")

    dubbing_id = data.get("dubbing_id")
    if not dubbing_id:
        raise DubError(f"dub submit missing 'dubbing_id': {data}")
    return dubbing_id


def get_status(*, api_key: str, dubbing_id: str) -> dict[str, Any]:
    """GET /v1/dubbing/{id} → {status: queued|dubbing|dubbed|failed, ...}"""
    url = f"{API_BASE}/dubbing/{dubbing_id}"
    try:
        r = requests.get(url, headers={"xi-api-key": api_key,
                                       "Accept": "application/json"}, timeout=30)
    except requests.RequestException as e:
        raise DubError(f"dub status network error: {e}") from e
    if r.status_code >= 400:
        raise DubError(f"dub status {r.status_code}: {r.text[:300]}")
    return r.json()


def wait_for_completion(*, api_key: str, dubbing_id: str,
                        poll_interval: int = DEFAULT_POLL_INTERVAL,
                        timeout: int = DEFAULT_TIMEOUT,
                        on_progress: Any = None) -> dict[str, Any]:
    """Poll status until 'dubbed' or 'failed'. Returns final status dict.

    on_progress: optional callback(status_dict) called on each non-terminal poll.
    """
    deadline = time.time() + timeout
    while True:
        st = get_status(api_key=api_key, dubbing_id=dubbing_id)
        status = st.get("status")
        if status == "dubbed":
            return st
        if status == "failed":
            raise DubError(f"dub failed: {st}")
        if time.time() >= deadline:
            raise DubError(f"dub timed out after {timeout}s, last status: {status}")
        if on_progress:
            try:
                on_progress(st)
            except Exception:
                pass
        time.sleep(poll_interval)


def download_dubbed(*, api_key: str, dubbing_id: str, target_lang: str,
                    output_path: str | Path, timeout: int = 300) -> Path:
    """Download dubbed MP4 to output_path. Returns the Path."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Newer API: /v1/dubbing/{id}/audio/{lang} returns the dubbed video file
    url = f"{API_BASE}/dubbing/{dubbing_id}/audio/{target_lang}"
    log.info(f"elevenlabs: downloading dubbed file → {out}")
    try:
        with requests.get(url,
                          headers={"xi-api-key": api_key},
                          stream=True, timeout=timeout) as r:
            if r.status_code >= 400:
                raise DubError(f"dub download {r.status_code}: {r.text[:300]}")
            with out.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
                    if chunk:
                        f.write(chunk)
    except requests.RequestException as e:
        raise DubError(f"dub download network error: {e}") from e
    return out


def dub_video(*, api_key: str, file_path: str | Path,
              source_lang: str, target_lang: str = "en",
              output_path: str | Path,
              name: str | None = None,
              on_progress: Any = None) -> Path:
    """Submit + wait + download. Blocking. Returns path to dubbed file."""
    dubbing_id = submit_dub(api_key=api_key, file_path=file_path,
                            source_lang=source_lang, target_lang=target_lang,
                            name=name)
    log.info(f"elevenlabs: dubbing_id={dubbing_id} (will poll until done)")
    wait_for_completion(api_key=api_key, dubbing_id=dubbing_id,
                        on_progress=on_progress)
    return download_dubbed(api_key=api_key, dubbing_id=dubbing_id,
                           target_lang=target_lang, output_path=output_path)
