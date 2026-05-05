"""Proxy pool: test residential proxies, pick a working one for YouTube downloads.

YouTube rate-limits individual IPs. Even paid residential proxies get
"Sign in to confirm you're not a bot" responses if hit too often. We solve
this by:

1. Keeping a list of N proxies in config (`onboarder.youtube_proxies`).
2. Before Phase 2 starts: probe each proxy with a quick metadata fetch on
   a known video. Return the first one that succeeds.
3. (Optional) During Phase 2, if a download fails with bot-check error,
   exclude the current proxy and pick the next working one.

The probe is fast — ~5-15 seconds per proxy, so probing 20 takes a couple
minutes worst-case. Usually the first 1-3 work and we stop early.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

log = logging.getLogger("gateway")

# A short, ubiquitous video to probe with. Using one of the user's actual course
# videos would be ideal but breaks reuse. dQw4w9WgXcQ is heavily cached → false
# positives. We use a less-cached but still public video for a realistic check.
DEFAULT_PROBE_VIDEO = "iG9CE55wbtY"  # TED talk, ~20 min, public, decent traffic
PROBE_TIMEOUT_SEC = 25


class NoWorkingProxyError(RuntimeError):
    pass


def find_working_proxy(*, proxies: list[str],
                       cookies_file: str | None = None,
                       probe_video_id: str = DEFAULT_PROBE_VIDEO,
                       on_progress=None) -> str:
    """Probe proxies in order, return the first that fetches metadata successfully.

    `on_progress(idx, total, name, ok)` is called for each tested proxy so the
    caller (e.g. Phase 2) can stream status to Telegram.

    Raises NoWorkingProxyError if every proxy fails.
    """
    if not proxies:
        raise NoWorkingProxyError("proxy pool is empty (set onboarder.youtube_proxies in config)")

    failures: list[str] = []
    for idx, proxy in enumerate(proxies, start=1):
        name = _proxy_label(proxy)
        ok, err = _probe(proxy, cookies_file, probe_video_id)
        if on_progress:
            try:
                on_progress(idx, len(proxies), name, ok)
            except Exception:
                pass
        if ok:
            log.info(f"proxy_pool: working proxy found ({idx}/{len(proxies)}): {name}")
            return proxy
        failures.append(f"  {idx}. {name}: {err[:120]}")

    raise NoWorkingProxyError(
        f"all {len(proxies)} proxies failed YouTube probe (video={probe_video_id}):\n"
        + "\n".join(failures[:10])
    )


def _probe(proxy: str, cookies_file: str | None, video_id: str) -> tuple[bool, str]:
    """Run a fast yt-dlp metadata fetch through the proxy. Returns (ok, error_snippet)."""
    # Use the same Python interpreter that's running the gateway — that's the
    # venv where yt-dlp was pip-installed. Direct `yt-dlp` binary isn't in
    # systemd's PATH.
    args = [
        sys.executable, "-m", "yt_dlp",
        "--proxy", proxy,
        "--remote-components", "ejs:github",
        "--skip-download",
        "--no-warnings",
        "--print", "ok",
        f"https://youtu.be/{video_id}",
    ]
    if cookies_file:
        cookies_path = Path(cookies_file).expanduser()
        if cookies_path.exists():
            args.extend(["--cookies", str(cookies_path)])

    env = os.environ.copy()
    home = str(Path.home())
    deno_bin = f"{home}/.deno/bin"
    if deno_bin not in env.get("PATH", ""):
        env["PATH"] = f"{deno_bin}:{env.get('PATH', '')}"

    try:
        r = subprocess.run(args, capture_output=True, text=True,
                           timeout=PROBE_TIMEOUT_SEC, env=env)
    except subprocess.TimeoutExpired:
        return False, "probe timeout"
    except FileNotFoundError:
        return False, "python -m yt_dlp not found"

    out = (r.stdout or "").strip()
    err = (r.stderr or "")
    # yt-dlp returns 0 on success regardless of what `--print` actually emits.
    # Treat any zero exit as success — it means YouTube returned playable metadata.
    if r.returncode == 0:
        return True, ""
    # Common error patterns we care about
    snippet = err.split("\n")[0] if err else f"exit {r.returncode}"
    return False, snippet


def _proxy_label(proxy_url: str) -> str:
    """Human-friendly label like 'host:port' (no credentials)."""
    # http://user:pass@host:port → host:port
    try:
        if "@" in proxy_url:
            return proxy_url.split("@", 1)[1]
        # http://host:port
        return proxy_url.split("//", 1)[-1]
    except Exception:
        return proxy_url


class ProxyRotator:
    """Holds a pool of proxies + the currently-active one.

    Phase 2 creates one of these at the start. Each video uses .current. If
    a download fails with bot-check, .rotate() picks the next working proxy
    (re-probing the unused remainder). Returns None when the pool is exhausted.
    """

    def __init__(self, pool: list[str], cookies_file: str | None = None):
        self.pool = list(pool)
        self.cookies_file = cookies_file
        self.tried: set[str] = set()
        self.current: str | None = None

    def init(self, on_progress=None) -> str | None:
        """Pick the first working proxy. Returns it or raises NoWorkingProxyError."""
        if not self.pool:
            return None
        proxy = find_working_proxy(
            proxies=self.pool,
            cookies_file=self.cookies_file,
            on_progress=on_progress,
        )
        self.tried.add(proxy)
        self.current = proxy
        return proxy

    def rotate(self, on_progress=None) -> str | None:
        """Pick the next working proxy from untried pool. None if exhausted."""
        remaining = [p for p in self.pool if p not in self.tried]
        if not remaining:
            return None
        try:
            proxy = find_working_proxy(
                proxies=remaining,
                cookies_file=self.cookies_file,
                on_progress=on_progress,
            )
            self.tried.add(proxy)
            self.current = proxy
            return proxy
        except NoWorkingProxyError:
            # All remaining failed too — mark them tried so we don't loop forever
            for p in remaining:
                self.tried.add(p)
            return None


def normalise_pool(cfg_value) -> list[str]:
    """Accept both `youtube_proxy` (str, single) and `youtube_proxies` (list of str)."""
    if not cfg_value:
        return []
    if isinstance(cfg_value, str):
        return [cfg_value] if cfg_value.strip() else []
    if isinstance(cfg_value, list):
        return [p for p in cfg_value if isinstance(p, str) and p.strip()]
    return []
