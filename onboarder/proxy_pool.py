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
import random
import subprocess
import sys
import tempfile
import threading
import time
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


class CookiesNeededError(RuntimeError):
    """All proxies in the pool are blocked by YouTube — cookies refresh required."""
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

    Thread-safe: callers may share one rotator across worker threads. When a
    worker hits a bot-check, it calls rotate_locked(failing_proxy) — only the
    first thread to land on a given failing proxy actually triggers a probe;
    later threads observe the already-rotated `.current` and proceed.
    """

    def __init__(self, pool: list[str], cookies_file: str | None = None):
        self.pool = list(pool)
        self.cookies_file = cookies_file
        self.tried: set[str] = set()
        self.current: str | None = None
        self._lock = threading.Lock()
        # Round-robin index — kept for back-compat but no longer the primary
        # picker. The LRU policy in next_round_robin() supersedes it.
        self._rr_idx = 0
        # Short-term blacklist: proxy → epoch-time when it can be tried again.
        # Used by mark_bad() so 403/timeout proxies are skipped for a few
        # minutes instead of being hammered repeatedly.
        self._blacklist: dict[str, float] = {}
        # LRU tracking: proxy → epoch-time it was last handed out via
        # next_round_robin(). Picker prefers the proxy with the OLDEST entry
        # (or 0 for never-used proxies — they get picked first). This is
        # smoother than naïve round-robin: when one proxy is blacklisted and
        # comes back, it sits "cold" at the head of the queue, getting one
        # request to test the waters before being slammed again.
        self._last_used: dict[str, float] = {}

    def init(self, on_progress=None) -> str | None:
        """Pick the first working proxy. Returns it or raises NoWorkingProxyError."""
        if not self.pool:
            return None
        proxy = find_working_proxy(
            proxies=self.pool,
            cookies_file=self.cookies_file,
            on_progress=on_progress,
        )
        with self._lock:
            self.tried.add(proxy)
            self.current = proxy
        return proxy

    def rotate(self, on_progress=None) -> str | None:
        """Pick the next working proxy from untried pool. None if exhausted."""
        with self._lock:
            remaining = [p for p in self.pool if p not in self.tried]
        if not remaining:
            return None
        try:
            proxy = find_working_proxy(
                proxies=remaining,
                cookies_file=self.cookies_file,
                on_progress=on_progress,
            )
            with self._lock:
                self.tried.add(proxy)
                self.current = proxy
            return proxy
        except NoWorkingProxyError:
            # All remaining failed too — mark them tried so we don't loop forever
            with self._lock:
                for p in remaining:
                    self.tried.add(p)
            return None

    def rotate_if_still(self, failing_proxy: str | None, on_progress=None) -> str | None:
        """Rotate only if `.current` is still the proxy that failed.

        Lets multiple workers share a single rotator: the first failure triggers
        the rotation, subsequent failures (on the now-stale proxy) are no-ops.
        Returns the (possibly new) current proxy.
        """
        with self._lock:
            already_rotated = self.current != failing_proxy
            current_snapshot = self.current
        if already_rotated:
            return current_snapshot
        return self.rotate(on_progress=on_progress)

    def next_round_robin(self, *, jitter_seconds: float = 0.5) -> str | None:
        """Pick the least-recently-used non-blacklisted proxy + apply jitter.

        Algorithm:
          1. Among non-blacklisted proxies, pick the one with the OLDEST
             `_last_used` timestamp (0 for never-used → these go first).
          2. Mark it as just-used (update _last_used[picked] = now).
          3. Release the lock.
          4. Sleep a small random amount (0..jitter_seconds, default 0..0.5s)
             OUTSIDE the lock — this desynchronises parallel workers who
             would otherwise all hit their picked proxy in the same
             millisecond and look bot-like to YouTube's CDN.
          5. Return the proxy.

        Why LRU instead of strict round-robin:
          * If proxy 7 got blacklisted for 5 min, came back, and is now
            "cold" — LRU picks it FIRST among the candidates, which lets it
            ease back in with one request instead of being slammed.
          * Brand-new proxies (never used in this process) tie at
            _last_used=0 and get picked deterministically in pool order
            until each has been used once. After that, true LRU cycle.

        Fallback if EVERY proxy is currently blacklisted: returns the one
        whose ban expires soonest (no _last_used update — we're using a
        desperate fallback, not a "fresh" pick).

        Thread-safe: critical section under self._lock; jitter sleep happens
        after release so 16 parallel callers don't block each other.

        Args:
          jitter_seconds: max random delay before returning. Pass 0.0 to
            disable (useful for tests or fast smoke-checks).
        """
        if not self.pool:
            return None
        now = time.time()
        picked: str | None = None
        used_fallback = False
        with self._lock:
            candidates = [p for p in self.pool
                          if self._blacklist.get(p, 0.0) <= now]
            if candidates:
                # Pick the proxy that has been idle the longest. .get(p, 0.0)
                # makes never-used proxies "infinitely old" (tied at 0) — they
                # win against all already-used proxies on first selection.
                picked = min(candidates, key=lambda p: self._last_used.get(p, 0.0))
                self._last_used[picked] = now
            elif self._blacklist:
                # Every proxy is blacklisted. Return the one whose ban expires
                # soonest so the pipeline keeps moving. Do NOT update
                # _last_used: this isn't a fresh pick, it's a desperate fallback.
                picked = min(self._blacklist.items(), key=lambda kv: kv[1])[0]
                used_fallback = True
            else:
                # Pool exists but has no entries we know about — shouldn't
                # happen in practice, but return first for safety.
                picked = self.pool[0]

        # Jitter is applied OUTSIDE the lock so parallel callers don't
        # serialize on it. Skip jitter when we're using the fallback —
        # waiting just delays a probably-doomed retry.
        if picked is not None and jitter_seconds > 0 and not used_fallback:
            time.sleep(random.uniform(0.0, float(jitter_seconds)))
        return picked

    def mark_bad(self, proxy: str | None, ttl_seconds: int = 300) -> None:
        """Add `proxy` to the short-term blacklist so next_round_robin skips it.

        Default TTL is 5 min — long enough for YouTube to drop a CDN rate
        limit, short enough that good proxies aren't lost for the rest of the
        run. Called by phase1_enrich when a proxy returns 403 / bot-check /
        truncated download.
        """
        if not proxy:
            return
        with self._lock:
            self._blacklist[proxy] = time.time() + max(1, int(ttl_seconds))


def normalise_pool(cfg_value) -> list[str]:
    """Accept both `youtube_proxy` (str, single) and `youtube_proxies` (list of str)."""
    if not cfg_value:
        return []
    if isinstance(cfg_value, str):
        return [cfg_value] if cfg_value.strip() else []
    if isinstance(cfg_value, list):
        return [p for p in cfg_value if isinstance(p, str) and p.strip()]
    return []
