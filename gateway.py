#!/usr/bin/env python3
"""
claude-telegram-gateway -- Universal Telegram <-> Claude Code router.

Routes Telegram messages to Claude Code CLI sessions per agent.
Each agent gets its own Telegram bot, Claude workspace, and session state.

Features:
- Producer-consumer architecture (OOB commands like /stop handled instantly)
- Markdown -> Telegram HTML conversion
- Voice/audio transcription via Groq Whisper
- Hot memory (rolling journal per agent)
- OpenViking semantic memory push (optional)
- Session management with --resume
- Media handling (photo, video, voice, document, sticker)
- Group chat support with @mention / name detection
- Real-time progress tracking (tool calls, subagent dispatches, TodoWrite plans)
- Forward context: agent sees who forwarded the message
- Message reactions: eyes emoji on received messages
- Inline buttons: send_message_with_buttons + callback query dispatch
- Sticker cache: cached descriptions for repeated stickers
- Webhook API: HTTP endpoint for external message injection
- Per-topic routing: route group-chat topics to specific agents
- Streaming modes: off / partial / progress (configurable per agent)

Usage:
    1. Copy config.example.json -> config.json
    2. Fill in agent config (telegram token, workspace, etc.)
    3. python3 gateway.py

Config structure (config.json):
    {
        "allowed_user_ids": [123456789],
        "webhook_port": 9090,
        "webhook_token": "your-secret-token",
        "agents": {
            "myagent": {
                "enabled": true,
                "workspace": "/path/to/agent/.claude",
                "model": "sonnet",
                "timeout_sec": 120,
                "streaming_mode": "partial",
                "bot_token": "123456:AABBccdd...",
                "groq_api_key": "YOUR_GROQ_API_KEY",
                "groq_api_key_file": "/path/to/groq.key",
                "openviking_url": "http://127.0.0.1:1933",
                "openviking_key_file": "/path/to/ov.key",
                "openviking_account": "default",
                "agent_names": ["myagent", "agent"],
                "topic_routing": {"-1001234567890": ["42", "99"]},
                "env": {"KEY": "value"}
            }
        }
    }

Token resolution order (priority first):
- "bot_token"                (inline, EdgeLab Day 1 format)
- "telegram_bot_token"       (inline, legacy Silvana/Kaelthas format)
- "telegram_bot_token_file"  (path to file, legacy)
Allowlist key resolution: "allowed_user_ids" first, "allowlist_user_ids" fallback.
Same inline/file pattern for groq_api_key / groq_api_key_file.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fcntl as _fcntl
import random as _random
import re as _re
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor

import requests

import atexit as _atexit

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

# Bounded thread pool for L4 (OV + Cognee) pushes — prevents unbounded thread
# spawning on message bursts. Renamed from _OV_POOL in T-19 when the gateway
# started routing semantic-memory writes through silvana_cognee.dual_write.
_L4_POOL = _ThreadPoolExecutor(max_workers=2, thread_name_prefix="l4-push")

# Active claude subprocesses per (agent, chat_id) for /stop command
_ACTIVE_PROCS: dict[tuple[str, int], Any] = {}

# Per-agent message queues for producer-consumer architecture
_MSG_QUEUES: dict[str, queue.Queue] = {}

# Shutdown event to gracefully stop all threads
_SHUTDOWN_EVENT = threading.Event()

# Out-of-band commands handled instantly by producer thread
_OOB_COMMANDS = frozenset({"/stop", "/cancel", "/status", "/reset", "/new"})


@_atexit.register
def _shutdown_l4_pool() -> None:
    """Graceful drain of pending L4 pushes on gateway shutdown."""
    try:
        _L4_POOL.shutdown(wait=True, cancel_futures=False)
    except Exception:
        pass


# --- L4 bridge to silvana_cognee (subprocess to cognee venv) ---------------
# Gateway runs under system Python; silvana_cognee is installed into the
# dedicated cognee venv (has cognee==1.0.0 + its LanceDB/Kuzu wheels).
# We shell out instead of importing to keep gateway's dep footprint intact
# and to keep a single rollback surface (delete this block + rename pool).
_L4_VENV_PY = Path("/Users/jasonqwwen/cognee-silvana/venv/bin/python")
_L4_CLI_ENABLED = _L4_VENV_PY.exists()

# M3: shared-Thrall HTTP transport. When an agent opts into ``l4_transport =
# "http"`` the gateway skips the local cognee venv entirely and POSTs the
# gateway event straight to the shared Cognee FastAPI on Thrall. Agents
# running on hosts without a local venv (Kaelthas, Arthas, Illidan) rely on
# this path; agents with a venv can keep ``subprocess`` (default) during the
# M5 backfill + M6 cutover window for safe rollback.
_L4_HTTP_TIMEOUT_S = 30.0
_L4_HTTP_COGNIFY_TIMEOUT_S = 90.0
# Max per-message body size before we truncate. Matches
# ``silvana_cognee.sanitize.clamp_len`` default so HTTP and subprocess
# produce byte-identical ingests for the same input.
_L4_HTTP_CLAMP_CHARS = 3000
# Byte-for-byte parity with silvana_cognee.sanitize.TRUNCATION_MARKER. No
# leading newline, no space before [truncated]. If Python tightens this,
# tests/test_gateway_l4_http.py::test_clamp_over_limit_adds_marker will
# catch the drift before a cutover makes it user-visible.
_L4_HTTP_TRUNCATION_MARKER = "\u2026[truncated]"
# Parity with ``silvana_cognee.dual_write._PUSH_WHITELIST`` — categories we
# willingly push to the graph. ``external_media`` deliberately absent (agents
# forward third-party content we don't want extracted as user preference).
# Must stay in lockstep; drift is caught by tests/test_gateway_l4_http.py.
_L4_HTTP_PUSH_WHITELIST = frozenset({
    "own_text", "own_voice", "forwarded", "group_mention",
})
# Cached api-key reads keyed by (path, mtime_ns) so we do not re-read the
# key file on every Telegram message. Gateway runs single-threaded per chat,
# but _L4_POOL workers share this dict — protected by a Lock.
_L4_HTTP_KEY_CACHE: dict[tuple[str, int], str] = {}
_L4_HTTP_KEY_LOCK = threading.Lock()


def _l4_http_scrub(text: str) -> str:
    """Mirror ``silvana_cognee.sanitize.scrub_nulls`` — strip U+0000 only."""
    if not isinstance(text, str):
        return ""
    return text.replace("\x00", "")


def _l4_http_clamp(text: str) -> str:
    """Mirror ``silvana_cognee.sanitize.clamp_len`` with the default 3000 cap
    and the same trailing marker shape. Symmetric with subprocess output so
    operators can diff ingests across transports during the cutover window.
    """
    if len(text) <= _L4_HTTP_CLAMP_CHARS:
        return text
    return text[:_L4_HTTP_CLAMP_CHARS] + _L4_HTTP_TRUNCATION_MARKER


def _l4_http_read_key(key_path: Path) -> str | None:
    """Read the Cognee API key file with an mtime-keyed cache.

    Returns ``None`` if the file is missing, unreadable, or empty — the caller
    logs and skips. We key on ``(absolute path, mtime_ns)`` so a key rotation
    (file rewrite bumps mtime) naturally invalidates the cached value without
    a restart.
    """
    try:
        st = key_path.stat()
    except OSError as exc:
        log.warning("[l4-http] api key stat failed for %s: %s", key_path, exc)
        return None
    cache_key = (str(key_path.resolve()), st.st_mtime_ns)
    with _L4_HTTP_KEY_LOCK:
        cached = _L4_HTTP_KEY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        value = key_path.read_text().strip()
    except OSError as exc:
        log.warning("[l4-http] api key read failed for %s: %s", key_path, exc)
        return None
    if not value:
        log.warning("[l4-http] api key file is empty: %s", key_path)
        return None
    with _L4_HTTP_KEY_LOCK:
        _L4_HTTP_KEY_CACHE[cache_key] = value
    return value


def _l4_transport(cfg: dict) -> str:
    """Return ``"http"`` or ``"subprocess"`` based on agent cfg + env.

    Resolution order:
      1. ``cfg.l4_transport`` (per-agent explicit choice).
      2. ``L4_TRANSPORT`` env var (operator override for kicking the tyres).
      3. ``"subprocess"`` (safe default — matches pre-M3 behaviour).
    """
    raw = cfg.get("l4_transport")
    if not isinstance(raw, str) or not raw:
        raw = os.environ.get("L4_TRANSPORT", "subprocess")
    value = raw.strip().lower()
    return "http" if value == "http" else "subprocess"


def _l4_http_ready(cfg: dict) -> bool:
    """Cheap pre-flight check so we never attempt a partial HTTP push."""
    required = ("cognee_remote_url", "cognee_remote_api_key_file",
                "cognee_agent_slug")
    return all(isinstance(cfg.get(k), str) and cfg[k] for k in required)


def _l4_qualify_dataset(slug: str, prefix: str, name: str) -> str:
    """Mirror ``silvana_cognee.remote.qualify_dataset`` so the gateway writes
    into the same namespace the Python client does. Idempotent on re-apply.
    """
    marker = f"{slug}__"
    base = name or "default"
    while base.startswith(marker):
        base = base[len(marker):]
    if prefix and not base.startswith(prefix):
        base = f"{prefix}{base}"
    return f"{slug}__{base}"


def _l4_http_call(subcmd: str, cfg: dict, agent: str, chat_id: int,
                  payload: dict, extra: dict | None = None) -> None:
    """Direct POST to Cognee FastAPI /api/v1/add. Fire-and-forget.

    Parallel to ``_l4_call`` but without a subprocess: the gateway event is
    materialised as a text blob and pushed straight to Thrall. This removes
    the need for a cognee venv on the agent's host and is the cutover path
    (M6) for agents that run on Mac mini / VPS peers without a local install.

    Always invoked from inside ``_L4_POOL`` workers so a slow HTTP request
    can't stall the gateway consumer. On any failure we log at WARN with an
    ``[l4-http]`` tag and return — symmetry with ``_l4_call``.

    Args:
        subcmd: ``"gateway-write"`` or ``"gateway-write-group"`` (kept for log
            parity with the subprocess path).
        cfg: Agent config dict.
        agent: Agent slug (``"silvana"``, ``"claude"``, …).
        chat_id: Telegram chat / group id, used in metadata.
        payload: ``{user_text|message_text, agent_response, extra_meta}``.
        extra: Optional ``{source_category, user_handle, …}`` hints.
    """
    if not _l4_http_ready(cfg):
        log.warning("[l4-http] %s skipped: missing cognee_remote_url / key / slug", subcmd)
        return

    extra = extra or {}
    source = extra.get("source_category", "own_text")

    # H2: whitelist gating must mirror silvana_cognee.dual_write.should_push_to_l4
    # so the HTTP path can't quietly ingest categories the Python path filters
    # out (e.g. external_media — third-party media that would pollute the graph
    # as if it were the user's own preference). Drift here → silent divergence
    # during the M6 cutover, which is exactly what we spent M2 preventing.
    if source not in _L4_HTTP_PUSH_WHITELIST:
        log.debug("[l4-http] %s skipped: source_category=%r not in whitelist",
                  subcmd, source)
        return

    # Sanitise user-controlled strings BEFORE we check for emptiness: scrub_nulls
    # can turn a payload that is "just \x00s" into empty, and clamp_len enforces
    # the 3000-char cap so the gateway emits bytes identical to what the CLI
    # produces for the same input (tests/test_gateway_l4_http.py pins this).
    raw_user = payload.get("user_text", "") or ""
    raw_resp = payload.get("agent_response", "") or ""
    raw_msg = payload.get("message_text", "") or ""
    user_text = _l4_http_clamp(_l4_http_scrub(raw_user))
    agent_resp = _l4_http_clamp(_l4_http_scrub(raw_resp))
    message_text = _l4_http_clamp(_l4_http_scrub(raw_msg))

    remote_url = cfg["cognee_remote_url"].rstrip("/")
    key_path = Path(expand(cfg["cognee_remote_api_key_file"]))
    api_key = _l4_http_read_key(key_path)
    if not api_key:
        return  # _l4_http_read_key already logged the reason

    slug = cfg["cognee_agent_slug"]
    prefix = cfg.get("cognee_dataset_prefix") or ""
    qualified = _l4_qualify_dataset(slug, prefix, source)

    # Build an episode body that matches silvana_cognee.dual_write._compose_episode
    # byte-for-byte so diffs across transports stay meaningful. For group
    # writes the Python path sets agent_response="" and passes message_text as
    # user_text; we do the same so the shared _compose_episode logic produces
    # identical blobs regardless of which transport landed the event.
    if subcmd == "gateway-write-group":
        if not message_text.strip():
            log.debug("[l4-http] %s skipped: empty message_text", subcmd)
            return
        user_handle = payload.get("user_handle", "anon")
        body_user = message_text
        body_resp = ""
        channel = "group"
        filename = f"group-{chat_id}-{int(time.time())}.txt"
    else:
        if not (user_text.strip() or agent_resp.strip()):
            log.debug("[l4-http] %s skipped: empty user+agent text", subcmd)
            return
        user_handle = None
        body_user = user_text
        body_resp = agent_resp
        channel = "dm"
        filename = f"dm-{chat_id}-{int(time.time())}.txt"

    # Byte-parity with silvana_cognee.dual_write._now_iso: UTC isoformat
    # with microsecond precision. ``time.strftime`` produced naive
    # second-precision strings, which made HTTP and subprocess episode
    # headers diff-incompatible and defeated the M6 cutover audit plan.
    # Honour COGNEE_MIGRATION_FROZEN_NOW so the reference tests on the
    # Python side can pin a deterministic clock in both transports.
    now_str = (
        os.environ.get("COGNEE_MIGRATION_FROZEN_NOW")
        or datetime.now(timezone.utc).isoformat()
    )
    body_parts = [
        f"[chat:{chat_id} agent:{agent} at {now_str}]",
        "[extraction hint: decision]",
        f"[source:{source}] {body_user}",
    ]
    if body_resp:
        body_parts.append("")
        body_parts.append(f"[agent_response:] {body_resp}")
    body = "\n".join(body_parts)

    meta: dict = {
        "agent": agent,
        "chat_id": chat_id,
        "source_category": source,
        "channel": channel,
        **(payload.get("extra_meta") or {}),
    }
    if user_handle is not None and "user_handle" not in meta:
        meta["user_handle"] = user_handle

    headers = {"X-Api-Key": api_key, "Accept": "application/json"}
    files = {"data": (filename, body.encode("utf-8"), "text/plain")}
    form = {"datasetName": qualified, "metadata": json.dumps(meta, default=str)}
    try:
        r = requests.post(
            f"{remote_url}/api/v1/add",
            headers=headers, files=files, data=form,
            timeout=_L4_HTTP_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        log.warning("[l4-http] %s add transport error: %s", subcmd, exc)
        return
    if r.status_code >= 400:
        log.warning(
            "[l4-http] %s add rc=%s body=%s",
            subcmd, r.status_code, (r.text or "")[:200],
        )
        return
    log.info("[l4-http] %s add ok dataset=%s", subcmd, qualified)

    # Optional cognify — when cfg.l4_http_cognify is truthy we also trigger
    # a background graph rebuild so fresh writes become searchable without
    # waiting for a nightly job. Off by default to keep latency bounded.
    if cfg.get("l4_http_cognify") is True:
        try:
            rc = requests.post(
                f"{remote_url}/api/v1/cognify",
                headers={**headers, "Content-Type": "application/json"},
                json={"datasets": [qualified], "runInBackground": True},
                timeout=_L4_HTTP_COGNIFY_TIMEOUT_S,
            )
            if rc.status_code >= 400:
                log.warning(
                    "[l4-http] %s cognify rc=%s body=%s",
                    subcmd, rc.status_code, (rc.text or "")[:200],
                )
        except requests.RequestException as exc:
            log.warning("[l4-http] %s cognify transport error: %s", subcmd, exc)


def _l4_call(subcmd: str, cli_args: list[str], payload: dict) -> None:
    """Run ``silvana_cognee.cli`` in the cognee venv. Fire-and-forget.

    Invoked only from inside ``_L4_POOL`` worker threads, so a slow subprocess
    (up to the 90s timeout) never blocks the gateway main loop or consumer.
    On any failure we log at WARN with an ``[l4]`` tag and return — gateway
    never sees an exception and a message is never lost from its perspective.

    Args:
        subcmd: ``"gateway-write"`` or ``"gateway-write-group"``.
        cli_args: Positional CLI flags (``["--agent", "silvana", ...]``).
        payload: Body forwarded to the CLI over stdin as JSON.
    """
    if not _L4_CLI_ENABLED:
        log.debug("[l4] venv not found at %s; skipping cognee path", _L4_VENV_PY)
        return
    try:
        proc = subprocess.run(
            [str(_L4_VENV_PY), "-m", "silvana_cognee.cli", subcmd, *cli_args],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=90,
        )
        if proc.returncode != 0:
            log.warning(
                "[l4] %s rc=%s stderr=%s",
                subcmd, proc.returncode, (proc.stderr or "")[:400],
            )
            return
        # Parse the last non-empty stdout line as the status JSON.
        stdout = (proc.stdout or "").strip()
        if stdout:
            try:
                status = json.loads(stdout.splitlines()[-1])
                log.info("[l4] %s result=%s", subcmd, status)
            except json.JSONDecodeError:
                log.debug("[l4] %s stdout not json: %s", subcmd, stdout[:200])
    except subprocess.TimeoutExpired:
        log.warning("[l4] %s timed out after 90s", subcmd)
    except FileNotFoundError as e:
        log.warning("[l4] %s venv python missing: %s", subcmd, e)
    except Exception as e:
        log.warning("[l4] %s failed: %s", subcmd, e)


def _l4_backend(cfg: dict) -> str:
    """Read the configured L4 backend from agent cfg ('ov' default)."""
    backend = cfg.get("l4_backend", "ov")
    if isinstance(backend, str) and backend.lower() in ("ov", "cognee", "dual"):
        return backend.lower()
    return "ov"


# ---------------------------------------------------------------------------
# Paths -- relative to working directory (cwd where gateway.py is launched)
# ---------------------------------------------------------------------------

_BASE_DIR = Path.cwd()
LOG_PATH = _BASE_DIR / "gateway.log"
CONFIG_PATH = _BASE_DIR / "config.json"
STATE_DIR = _BASE_DIR / "state"
MEDIA_DIR = _BASE_DIR / "media-inbound"
STICKER_CACHE_PATH = STATE_DIR / "sticker-cache.json"
STATE_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# In-memory sticker description cache (loaded from disk on startup)
_sticker_cache: dict[str, str] = {}
if STICKER_CACHE_PATH.exists():
    try:
        _sticker_cache = json.loads(STICKER_CACHE_PATH.read_text())
    except Exception:
        _sticker_cache = {}


def _save_sticker_cache() -> None:
    try:
        STICKER_CACHE_PATH.write_text(json.dumps(_sticker_cache, ensure_ascii=False))
    except Exception:
        pass


def _get_sticker_description(
    uid: str, emoji: str, set_name: str, local_path: Path | None
) -> str:
    """Get or build sticker description. Cache by file_unique_id."""
    if uid and uid in _sticker_cache:
        return _sticker_cache[uid]

    parts: list[str] = []
    if emoji:
        parts.append(f"emoji {emoji}")
    if set_name:
        parts.append(f"set \"{set_name}\"")
    if local_path:
        parts.append(str(local_path))

    desc = ", ".join(parts) if parts else "sticker (no description)"

    if uid:
        _sticker_cache[uid] = desc
        _save_sticker_cache()

    return desc


# Legacy fallback for Groq key (checked after agent config)
GROQ_KEY_FILE = Path.home() / ".secrets" / "groq-api-key"

TG_MAX_FILE_MB = 20  # Telegram Bot API hard limit
MEDIA_EXTENSIONS = {
    "voice": ".ogg",
    "audio": "",  # keep original
    "video": ".mp4",
    "video_note": ".mp4",
    "photo": ".jpg",
    "document": "",  # keep original
    "sticker": ".webp",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("gateway")


# ---------------------------------------------------------------------------
# Placeholder -- generic, no per-agent pools
# ---------------------------------------------------------------------------

def _media_to_input_type(msg: dict, source_tag: str) -> str:
    """Map Telegram message to input_type for placeholder selection."""
    if source_tag == "forwarded":
        return "forwarded"
    if "voice" in msg or "video_note" in msg or "audio" in msg:
        return "voice"
    if "video" in msg:
        return "video"
    if "photo" in msg:
        return "photo"
    if "document" in msg:
        return "document"
    if "sticker" in msg:
        return "sticker"
    return "text"


def get_placeholder(agent: str, input_type: str = "text") -> str:
    """Pick a generic placeholder for any agent/input_type."""
    _GENERIC: dict[str, list[str]] = {
        "text": ["думаю", "обрабатываю", "анализирую", "работаю", "принял"],
        "voice": ["слушаю голосовое", "транскрибирую", "разбираю речь"],
        "video": ["смотрю видео", "анализирую видео"],
        "photo": ["рассматриваю изображение", "изучаю картинку"],
        "document": ["читаю документ", "изучаю файл"],
        "forwarded": ["изучаю пересланное", "разбираю материал"],
        "sticker": ["принял стикер"],
    }
    pool = _GENERIC.get(input_type) or _GENERIC["text"]
    return _random.choice(pool) + "..."


# ---------------------------------------------------------------------------
# HTML / Markdown utilities
# ---------------------------------------------------------------------------

# HTML parse error detection
_PARSE_ERR_RE = _re.compile(
    r"can't parse entities|parse entities|find end of the entity|unsupported start tag|unexpected end tag",
    _re.IGNORECASE,
)


def escape_html(text: str) -> str:
    """Escape 3 chars for Telegram HTML body. Order matters: & first."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def escape_html_attr(text: str) -> str:
    return escape_html(text).replace('"', "&quot;")


def is_html_parse_error(resp_body: Any) -> bool:
    if not isinstance(resp_body, dict):
        return False
    desc = resp_body.get("description", "") or ""
    return bool(_PARSE_ERR_RE.search(desc))


# Markdown -> Telegram HTML converter
_MD_CODEBLOCK_RE = _re.compile(r"```([a-zA-Z0-9_+\-]*)\n?(.*?)```", _re.DOTALL)
_MD_INLINECODE_RE = _re.compile(r"`([^`\n]+?)`")
_MD_BOLD_RE = _re.compile(r"\*\*([^\*\n]+?)\*\*")
_MD_ITALIC_STAR_RE = _re.compile(r"(?<![\w\*])\*([^\*\n]+?)\*(?!\w)")
# Underscore italic: require whitespace/punctuation boundary to avoid breaking
# identifiers like config_file.py, snake_case, etc.
_MD_ITALIC_UND_RE = _re.compile(r"(?:(?<=\s)|(?<=^)|(?<=[.,;:!?\(\[]))_([^_\n]+?)_(?=\s|$|[.,;:!?\)\]])")
_MD_LINK_RE = _re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MD_HEADING_RE = _re.compile(r"^(#{1,6})\s+(.+)$", _re.MULTILINE)
_MD_STRIKE_RE = _re.compile(r"~~([^~\n]+?)~~")
_MD_TABLE_RE = _re.compile(
    r"(?:^[ \t]*\|.+\|[ \t]*\n)+"
    r"(?:^[ \t]*\|[-:\s|]+\|[ \t]*\n)?"
    r"(?:^[ \t]*\|.+\|[ \t]*\n)*",
    _re.MULTILINE,
)


def _md_table_to_pre(table_text: str) -> str:
    """Convert a markdown table block into a <pre> block with aligned columns."""
    lines = [ln.strip() for ln in table_text.strip().splitlines()]
    # Filter out separator rows (|---|---|)
    data_lines = [ln for ln in lines if not _re.match(r"^\|[\s\-:|]+\|$", ln)]
    if not data_lines:
        return table_text

    # Parse cells
    rows: list[list[str]] = []
    for ln in data_lines:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return table_text

    # Calculate max width per column
    num_cols = max(len(r) for r in rows)
    col_widths = [0] * num_cols
    for row in rows:
        for i, cell in enumerate(row):
            if i < num_cols:
                col_widths[i] = max(col_widths[i], len(cell))

    # Build aligned output
    out_lines: list[str] = []
    for row in rows:
        parts: list[str] = []
        for i in range(num_cols):
            cell = row[i] if i < len(row) else ""
            parts.append(cell.ljust(col_widths[i]))
        out_lines.append("  ".join(parts).rstrip())

    return "<pre>" + escape_html("\n".join(out_lines)) + "</pre>"


def markdown_to_telegram_html(text: str) -> str:
    """Convert common markdown to Telegram-supported HTML. Safe for already-HTML content."""
    if not text:
        return text

    # Step 1: extract fenced code blocks first (preserve content)
    placeholders: dict[str, str] = {}

    def _save_codeblock(m: _re.Match) -> str:
        lang = m.group(1) or ""
        code = m.group(2).rstrip("\n")
        escaped = escape_html(code)
        if lang:
            html = f'<pre><code class="language-{escape_html_attr(lang)}">{escaped}</code></pre>'
        else:
            html = f"<pre><code>{escaped}</code></pre>"
        key = f"\x00CB{len(placeholders)}\x00"
        placeholders[key] = html
        return key

    text = _MD_CODEBLOCK_RE.sub(_save_codeblock, text)

    # Step 1.5: convert markdown tables to <pre> blocks (before HTML escaping)
    def _save_table(m: _re.Match) -> str:
        html = _md_table_to_pre(m.group(0))
        key = f"\x00TB{len(placeholders)}\x00"
        placeholders[key] = html
        return key

    text = _MD_TABLE_RE.sub(_save_table, text)

    # Step 2: inline code
    def _save_inlinecode(m: _re.Match) -> str:
        code = m.group(1)
        html = f"<code>{escape_html(code)}</code>"
        key = f"\x00IC{len(placeholders)}\x00"
        placeholders[key] = html
        return key

    text = _MD_INLINECODE_RE.sub(_save_inlinecode, text)

    # Step 3: escape HTML special chars in remaining text
    # (but preserve already-present HTML tags the agent might have used)
    # We accept agent's <b>, <i>, <code>, <pre>, <a>, <s>, <tg-spoiler>, <blockquote>
    _SAFE_TAGS = ("b", "i", "s", "code", "pre", "a", "tg-spoiler", "blockquote", "u")
    _TAG_RE = _re.compile(
        r"</?(" + "|".join(_SAFE_TAGS) + r")(?:\s[^>]*)?>", _re.IGNORECASE
    )

    tag_placeholders: dict[str, str] = {}

    def _save_tag(m: _re.Match) -> str:
        key = f"\x00TG{len(tag_placeholders)}\x00"
        tag_placeholders[key] = m.group(0)
        return key

    text = _TAG_RE.sub(_save_tag, text)
    # Now escape
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Restore known tags
    for key, val in tag_placeholders.items():
        text = text.replace(key, val)

    # Step 4: markdown transformations
    text = _MD_HEADING_RE.sub(lambda m: f"<b>{m.group(2)}</b>", text)
    text = _MD_BOLD_RE.sub(r"<b>\1</b>", text)
    text = _MD_STRIKE_RE.sub(r"<s>\1</s>", text)
    text = _MD_ITALIC_STAR_RE.sub(r"<i>\1</i>", text)
    text = _MD_ITALIC_UND_RE.sub(r"<i>\1</i>", text)
    text = _MD_LINK_RE.sub(
        lambda m: f'<a href="{escape_html_attr(m.group(2))}">{m.group(1)}</a>', text
    )

    # Step 5: restore code placeholders
    for key, val in placeholders.items():
        text = text.replace(key, val)

    return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def expand(p: str) -> str:
    return os.path.expanduser(p)


def _resolve_token(cfg: dict, direct_key: str, file_key: str) -> str | None:
    """Resolve a secret from config: direct value first, then file fallback."""
    val = cfg.get(direct_key)
    if val:
        return val.strip()
    fpath = cfg.get(file_key)
    if fpath:
        p = Path(expand(fpath))
        if p.exists():
            return p.read_text().strip()
    return None


def _resolve_telegram_token(cfg: dict) -> str:
    """Resolve Telegram bot token from agent config. Raises if not found.

    Accepted keys (in priority order):
      - bot_token              (inline, EdgeLab Day 1 format)
      - telegram_bot_token     (inline, legacy Silvana/Kaelthas format)
      - telegram_bot_token_file (path to file with token, legacy)
    """
    token = _resolve_token(cfg, "bot_token", "telegram_bot_token_file")
    if not token:
        token = _resolve_token(cfg, "telegram_bot_token", "telegram_bot_token_file")
    if not token:
        raise ValueError(
            "No telegram token: set 'bot_token' (or 'telegram_bot_token' / "
            "'telegram_bot_token_file') in agent config"
        )
    return token


def _resolve_groq_key(cfg: dict) -> str | None:
    """Resolve Groq API key: agent config first, then global fallback file."""
    key = _resolve_token(cfg, "groq_api_key", "groq_api_key_file")
    if key:
        return key
    # Global fallback
    if GROQ_KEY_FILE.exists():
        return GROQ_KEY_FILE.read_text().strip()
    return None


# ---------------------------------------------------------------------------
# Telegram API
# ---------------------------------------------------------------------------

_MAX_DOCUMENT_BYTES = 50 * 1024 * 1024  # Telegram sendDocument limit: 50 MB

def tg_api(token: str, method: str, retry: int = 2, **params: Any) -> dict:
    """Telegram API call with retry on 429/5xx/network errors."""
    url = f"https://api.telegram.org/bot{token}/{method}"
    last_exc: Any = None
    for attempt in range(retry + 1):
        try:
            r = requests.post(url, json=params, timeout=30)
            if r.status_code == 429:
                # Rate limit -- honor retry_after if present
                wait = r.json().get("parameters", {}).get("retry_after", 1)
                time.sleep(min(wait, 30))
                continue
            if r.status_code >= 500:
                last_exc = Exception(f"telegram {r.status_code}: {r.text[:200]}")
                time.sleep(1 + attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_exc = e
            if attempt < retry:
                time.sleep(1 + attempt)
                continue
            raise
    if last_exc:
        raise last_exc
    return {}


def _send_one(token: str, chat_id: int, text: str, reply_to: int | None, parse_mode: str | None) -> None:
    """Send single message with HTML parse_mode, fallback to plain on parse error."""
    params: dict[str, Any] = {"chat_id": chat_id, "text": text}
    if reply_to:
        params["reply_to_message_id"] = reply_to
        params["allow_sending_without_reply"] = True
    if parse_mode:
        params["parse_mode"] = parse_mode
    try:
        tg_api(token, "sendMessage", **params)
    except requests.HTTPError as e:
        body = None
        try:
            body = e.response.json()
        except Exception:
            pass
        if parse_mode and is_html_parse_error(body):
            # Retry without parse_mode (plain text)
            params.pop("parse_mode", None)
            tg_api(token, "sendMessage", **params)
            return
        raise


def send_message(
    token: str, chat_id: int, text: str, reply_to: int | None = None, html: bool = True
) -> None:
    """Send text to Telegram, chunking by 4000 chars at paragraph boundaries when possible."""
    # Convert markdown -> HTML (if HTML mode)
    if html:
        text = markdown_to_telegram_html(text)

    # Chunk by paragraphs (double newline), fallback to 4000-char slices
    limit = 4000
    if len(text) <= limit:
        chunks = [text]
    else:
        chunks = []
        paragraphs = text.split("\n\n")
        current = ""
        for p in paragraphs:
            if len(current) + len(p) + 2 <= limit:
                current = current + "\n\n" + p if current else p
            else:
                if current:
                    chunks.append(current)
                if len(p) > limit:
                    # Try splitting by single newline first, then hard split
                    sub_lines = p.split("\n")
                    sub_current = ""
                    for sl in sub_lines:
                        if len(sub_current) + len(sl) + 1 <= limit:
                            sub_current = sub_current + "\n" + sl if sub_current else sl
                        else:
                            if sub_current:
                                chunks.append(sub_current)
                            if len(sl) > limit:
                                for i in range(0, len(sl), limit):
                                    chunks.append(sl[i : i + limit])
                                sub_current = ""
                            else:
                                sub_current = sl
                    if sub_current:
                        chunks.append(sub_current)
                    current = ""
                else:
                    current = p
        if current:
            chunks.append(current)
    parse_mode = "HTML" if html else None
    for chunk in chunks:
        _send_one(token, chat_id, chunk, reply_to, parse_mode)
        reply_to = None  # only first chunk is a reply


def send_chat_action(token: str, chat_id: int, action: str = "typing") -> None:
    try:
        tg_api(token, "sendChatAction", chat_id=chat_id, action=action)
    except Exception as e:
        log.warning(f"sendChatAction failed: {e}")


def set_reaction(
    token: str, chat_id: int, message_id: int, emoji: str = "\U0001f440"
) -> None:
    """Set emoji reaction on a message (ack). Default: eyes emoji."""
    try:
        tg_api(
            token, "setMessageReaction",
            chat_id=chat_id,
            message_id=message_id,
            reaction=json.dumps([{"type": "emoji", "emoji": emoji}]),
        )
    except Exception as e:
        log.debug(f"setMessageReaction failed (non-critical): {e}")


def send_message_with_buttons(
    token: str,
    chat_id: int,
    text: str,
    buttons: list[list[dict[str, str]]],
    reply_to: int | None = None,
    html: bool = True,
) -> dict | None:
    """Send message with inline keyboard buttons.

    Args:
        buttons: 2D list of button rows, each button is {"text": "...", "callback_data": "..."}
                 or {"text": "...", "url": "..."} for URL buttons.
    Returns:
        Telegram API response dict or None on error.
    """
    if html:
        text = markdown_to_telegram_html(text)
    params: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text[:4000],
        "reply_markup": json.dumps({"inline_keyboard": buttons}),
    }
    if reply_to:
        params["reply_to_message_id"] = reply_to
        params["allow_sending_without_reply"] = True
    if html:
        params["parse_mode"] = "HTML"
    try:
        return tg_api(token, "sendMessage", **params)
    except requests.HTTPError as e:
        body = None
        try:
            body = e.response.json()
        except Exception:
            pass
        if html and is_html_parse_error(body):
            params.pop("parse_mode", None)
            return tg_api(token, "sendMessage", **params)
        log.warning(f"send_message_with_buttons failed: {e}")
        return None


def answer_callback_query(
    token: str, callback_query_id: str, text: str = "", show_alert: bool = False
) -> None:
    """Answer a callback query (inline button press)."""
    try:
        tg_api(
            token, "answerCallbackQuery",
            callback_query_id=callback_query_id,
            text=text,
            show_alert=show_alert,
        )
    except Exception as e:
        log.warning(f"answerCallbackQuery failed: {e}")


def send_document(
    token: str,
    chat_id: int,
    file_path: str,
    caption: str | None = None,
    reply_to: int | None = None,
) -> dict:
    """Send a local file to Telegram via sendDocument API.

    Args:
        token: Telegram bot token.
        chat_id: Target chat ID.
        file_path: Absolute path to local file.
        caption: Optional caption (HTML, max 1024 chars).
        reply_to: Optional message ID to reply to.

    Returns:
        Telegram API response dict.
    """
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    data: dict[str, Any] = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption[:1024]
        data["parse_mode"] = "HTML"
    if reply_to:
        data["reply_to_message_id"] = reply_to
        data["allow_sending_without_reply"] = True
    p = Path(file_path)
    if not p.exists():
        log.warning(f"send_document: file not found: {file_path}")
        return {}
    file_size = p.stat().st_size
    if file_size > _MAX_DOCUMENT_BYTES:
        log.warning(
            f"send_document: file too large"
            f" ({file_size} bytes): {file_path}"
        )
        return {}
    last_exc: Any = None
    for attempt in range(3):
        try:
            with open(p, "rb") as f:
                files = {"document": (p.name, f)}
                r = requests.post(
                    url, data=data, files=files, timeout=120,
                )
            if r.status_code == 429:
                wait = r.json().get(
                    "parameters", {},
                ).get("retry_after", 1)
                time.sleep(min(wait, 30))
                continue
            if r.status_code >= 500:
                last_exc = Exception(
                    f"telegram {r.status_code}: {r.text[:200]}"
                )
                time.sleep(1 + attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_exc = e
            if attempt < 2:
                time.sleep(1 + attempt)
                continue
            raise
    if last_exc:
        raise last_exc
    return {}


# Pending callback handlers: {callback_data_prefix: handler_func}
# Handler signature: handler(token, agent, cfg, callback_query) -> None
_CALLBACK_HANDLERS: dict[str, Any] = {}


def register_callback_handler(prefix: str, handler: Any) -> None:
    """Register a handler for callback_data starting with prefix."""
    _CALLBACK_HANDLERS[prefix] = handler


def dispatch_callback_query(token: str, agent: str, cfg: dict, cq: dict) -> None:
    """Route a callback_query to the appropriate handler."""
    data = cq.get("data", "")
    for prefix, handler in _CALLBACK_HANDLERS.items():
        if data.startswith(prefix):
            try:
                handler(token, agent, cfg, cq)
            except Exception as e:
                log.exception(f"callback handler error ({prefix}): {e}")
                answer_callback_query(token, cq["id"], f"Error: {e}", show_alert=True)
            return
    # No handler matched -- acknowledge silently
    answer_callback_query(token, cq["id"])


# ---------------------------------------------------------------------------
# Group chat: agent name detection
# ---------------------------------------------------------------------------

def _get_agent_names(agent: str, cfg: dict) -> list[str]:
    """Get agent name aliases from config, fallback to [agent]."""
    names = cfg.get("agent_names")
    if names and isinstance(names, list):
        return [n.lower() for n in names]
    return [agent.lower()]


def is_addressed_to_agent(agent: str, msg: dict, bot_username: str | None, cfg: dict | None = None) -> bool:
    """For group chats: check if message explicitly addresses this agent.

    Returns True if:
    - DM (private chat) -- always addressed
    - Group chat AND (@bot_username mentioned OR agent name appears in text)
    """
    chat_type = (msg.get("chat") or {}).get("type", "private")
    if chat_type == "private":
        return True  # DM -- always address

    # Group/supergroup -- need explicit mention or name
    text = (
        msg.get("text") or msg.get("caption")
        or msg.get("_voice_transcript") or ""
    ).lower()
    if not text:
        return False

    # Check @bot_username mention
    if bot_username:
        if f"@{bot_username.lower()}" in text:
            return True

    # Check agent name aliases
    names = _get_agent_names(agent, cfg or {})
    for name in names:
        if name in text:
            return True

    # Check reply-to-bot: if user replied to bot's message
    reply_to = msg.get("reply_to_message") or {}
    reply_from = (reply_to.get("from") or {})
    if reply_from.get("is_bot") and reply_from.get("username", "").lower() == (bot_username or "").lower():
        return True

    return False


# ---------------------------------------------------------------------------
# Message classification
# ---------------------------------------------------------------------------

def classify_source(msg: dict) -> tuple[str, str]:
    """Classify message source. Returns (source_tag, human_label).

    Tags:
    - own_text: user's own typed message (includes caption under photo/video/doc)
    - own_voice: user's voice/video_note (direct speech)
    - forwarded: forwarded from someone else
    - external_media: bare media (photo/video/document) without caption text
    """
    # Forward detection first -- forwarded content is always external source
    if (msg.get("forward_from") or msg.get("forward_from_chat")
            or msg.get("forward_sender_name") or msg.get("forward_origin")):
        origin = msg.get("forward_origin") or {}
        name = (
            (msg.get("forward_from") or {}).get("first_name")
            or (msg.get("forward_from_chat") or {}).get("title")
            or msg.get("forward_sender_name")
            or origin.get("sender_user_name")
            or "unknown"
        )
        return ("forwarded", f"forwarded from: {name}")
    # Voice/video_note = user speaking directly
    if "voice" in msg or "video_note" in msg:
        return ("own_voice", "user voice message")
    # Media with caption = user's own text with attachment
    has_media = any(k in msg for k in ("audio", "video", "photo", "document", "sticker"))
    has_caption = bool(msg.get("caption"))
    if has_media and has_caption:
        return ("own_text", "user text with attachment")
    if has_media:
        return ("external_media", "media attachment from user")
    return ("own_text", "user text")


# ---------------------------------------------------------------------------
# Media handling
# ---------------------------------------------------------------------------

def resolve_media_ref(msg: dict) -> dict | None:
    """Detect media in Telegram message, return {type, file_id, file_name, mime_type, file_size}."""
    if "photo" in msg and msg["photo"]:
        # photo is array of sizes; take largest
        largest = msg["photo"][-1]
        return {
            "type": "photo",
            "file_id": largest.get("file_id"),
            "file_name": None,
            "mime_type": "image/jpeg",
            "file_size": largest.get("file_size", 0),
        }
    for t in ("voice", "audio", "video", "video_note", "document", "sticker"):
        if t in msg:
            obj = msg[t]
            return {
                "type": t,
                "file_id": obj.get("file_id"),
                "file_name": obj.get("file_name"),
                "mime_type": obj.get("mime_type"),
                "file_size": obj.get("file_size", 0),
            }
    return None


def download_telegram_file(token: str, file_id: str, media_type: str, file_name: str | None) -> Path | None:
    """Download file from Telegram Bot API, save to MEDIA_DIR. Returns path or None."""
    try:
        r = tg_api(token, "getFile", file_id=file_id)
        file_info = r.get("result") or {}
        file_path = file_info.get("file_path")
        file_size = file_info.get("file_size", 0)
        if not file_path:
            log.warning(f"getFile: no file_path for {file_id}")
            return None
        if file_size > TG_MAX_FILE_MB * 1024 * 1024:
            log.warning(f"file too big: {file_size} bytes > {TG_MAX_FILE_MB}MB")
            return None
        # Derive local filename
        # Force known extensions for types where Groq/processors are strict
        if media_type == "voice":
            ext = ".ogg"  # Telegram gives .oga but Groq only accepts .ogg/.opus
        elif media_type == "video_note":
            ext = ".mp4"
        else:
            ext = Path(file_path).suffix or MEDIA_EXTENSIONS.get(media_type, "")
        if file_name:
            sanitized = _re.sub(r"[^\w\-. ]", "_", Path(file_name).stem)[:40]
            local_name = f"{sanitized}---{uuid.uuid4()}{ext}"
        else:
            local_name = f"{uuid.uuid4()}{ext}"
        local_path = MEDIA_DIR / local_name
        # Download
        url = f"https://api.telegram.org/file/bot{token}/{file_path}"
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        log.info(f"downloaded {media_type} -> {local_path.name} ({file_size} bytes)")
        return local_path
    except Exception as e:
        log.warning(f"download_telegram_file failed: {e}")
        return None


def transcribe_audio(path: Path, agent_cfg: dict | None = None, language: str = "ru") -> str | None:
    """Transcribe audio via Groq Whisper. Returns transcript text or None."""
    key = _resolve_groq_key(agent_cfg or {})
    if not key:
        log.warning("GROQ key missing, skip transcription")
        return None
    try:
        with open(path, "rb") as f:
            files = {"file": (path.name, f, "audio/ogg")}
            data = {
                "model": "whisper-large-v3-turbo",
                "response_format": "text",
                "language": language,
            }
            r = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                files=files,
                data=data,
                timeout=60,
            )
        if r.status_code == 200:
            return r.text.strip()
        log.warning(f"groq transcribe HTTP {r.status_code}: {r.text[:200]}")
        return None
    except Exception as e:
        log.warning(f"groq transcribe failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Telegram message editing / deletion
# ---------------------------------------------------------------------------

def edit_message(token: str, chat_id: int, message_id: int, text: str, html: bool = True) -> None:
    """Edit Telegram message. Silent on 'message not modified'. HTML fallback to plain.
    Retry once on transient network errors (backoff 500ms).
    """
    text = text[:4000]
    params: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id, "text": text}
    if html:
        params["parse_mode"] = "HTML"

    def _try(attempt: int = 0) -> bool:
        try:
            tg_api(token, "editMessageText", **params)
            return True
        except requests.HTTPError as e:
            body = None
            try:
                body = e.response.json()
            except Exception:
                pass
            desc = (body.get("description", "") if isinstance(body, dict) else "").lower()
            if "message is not modified" in desc:
                return True
            if html and is_html_parse_error(body):
                params.pop("parse_mode", None)
                try:
                    tg_api(token, "editMessageText", **params)
                    return True
                except Exception as e2:
                    log.warning(f"editMessageText plain fallback failed: {e2}")
                    return False
            log.warning(f"editMessageText HTTP {e}")
            return False
        except requests.RequestException as e:
            # Network/timeout error -- retry once
            if attempt < 1:
                time.sleep(0.5)
                return _try(attempt + 1)
            log.warning(f"editMessageText network error after retry: {e}")
            return False
        except Exception as e:
            log.warning(f"editMessageText failed: {e}")
            return False

    _try()


def delete_message(token: str, chat_id: int, message_id: int) -> None:
    try:
        tg_api(token, "deleteMessage", chat_id=chat_id, message_id=message_id)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Mode state (chat | wizard) -- per (agent, user_id)
# ---------------------------------------------------------------------------

MODE_CHAT = "chat"
MODE_WIZARD = "wizard"


def _mode_file(agent: str, user_id: int) -> Path:
    return STATE_DIR / f"mode-{agent}-{user_id}.txt"


def get_user_mode(agent: str, user_id: int) -> str:
    """Return current mode for a user. Default: chat (existing Claude behavior)."""
    f = _mode_file(agent, user_id)
    if not f.exists():
        return MODE_CHAT
    val = f.read_text().strip()
    return val if val in (MODE_CHAT, MODE_WIZARD) else MODE_CHAT


def set_user_mode(agent: str, user_id: int, mode: str) -> None:
    if mode not in (MODE_CHAT, MODE_WIZARD):
        raise ValueError(f"invalid mode: {mode}")
    f = _mode_file(agent, user_id)
    if mode == MODE_CHAT:
        f.unlink(missing_ok=True)
    else:
        f.write_text(mode)


# ---------------------------------------------------------------------------
# Commands (/status, /reset, /new, /compact, /stop, /menu, /help)
# ---------------------------------------------------------------------------

def _get_workspace(agent: str, cfg: dict) -> str:
    """Get workspace path from agent config."""
    return expand(cfg["workspace"])


def handle_command(token: str, chat_id: int, agent: str, cmd: str, args: str, cfg: dict | None = None) -> bool:
    """Handle /status /reset /new /help commands. Returns True if handled."""
    cfg = cfg or {}
    sid_file = STATE_DIR / f"sid-{agent}-{chat_id}.txt"
    first_file = STATE_DIR / f"sid-{agent}-{chat_id}.first"

    if cmd == "/status":
        workspace = _get_workspace(agent, cfg) if cfg.get("workspace") else None
        core_dir = Path(workspace) / "core" if workspace else None

        hot_kb = decisions_kb = memory_kb = rules_kb = 0.0
        if core_dir:
            hot_file = core_dir / "hot" / "recent.md"
            decisions_file = core_dir / "warm" / "decisions.md"
            memory_file = core_dir / "MEMORY.md"
            rules_file = core_dir / "rules.md"
            hot_kb = hot_file.stat().st_size / 1024 if hot_file.exists() else 0
            decisions_kb = decisions_file.stat().st_size / 1024 if decisions_file.exists() else 0
            memory_kb = memory_file.stat().st_size / 1024 if memory_file.exists() else 0
            rules_kb = rules_file.stat().st_size / 1024 if rules_file.exists() else 0

        if sid_file.exists():
            sid = sid_file.read_text().strip()
            age = time.time() - sid_file.stat().st_mtime
            age_str = f"{int(age/3600)}ч {int((age%3600)/60)}м" if age > 3600 else f"{int(age/60)}м"
            session_extra = ""
            jsonl_dir = Path.home() / ".claude" / "projects"
            for d in jsonl_dir.iterdir() if jsonl_dir.exists() else []:
                jf = d / f"{sid}.jsonl"
                if jf.exists():
                    size_kb = jf.stat().st_size / 1024
                    turns = sum(1 for _ in jf.open())
                    session_extra = f"\nturns: {turns} | {size_kb:.0f} KB"
                    break
            text = (
                f"<b>сессия активна</b>\n"
                f"id: <code>{sid[:8]}...</code>\n"
                f"возраст: {age_str}{session_extra}\n\n"
                f"<b>память</b>\n"
                f"rules: {rules_kb:.1f} KB\n"
                f"warm (decisions): {decisions_kb:.1f} KB\n"
                f"hot (recent): {hot_kb:.1f} KB\n"
                f"cold (MEMORY): {memory_kb:.1f} KB"
            )
        else:
            text = (
                f"<b>сессия пуста</b>\n\n"
                f"следующее сообщение = новая сессия\n\n"
                f"<b>память</b>\n"
                f"rules: {rules_kb:.1f} KB\n"
                f"warm: {decisions_kb:.1f} KB\n"
                f"hot: {hot_kb:.1f} KB\n"
                f"cold: {memory_kb:.1f} KB"
            )
        try:
            tg_api(token, "sendMessage", chat_id=chat_id, text=text, parse_mode="HTML")
        except Exception:
            pass
        return True

    if cmd in ("/reset", "/new"):
        force = args.strip().lower() == "force"
        if not sid_file.exists():
            try:
                tg_api(token, "sendMessage", chat_id=chat_id, text="<b>session already empty</b>", parse_mode="HTML")
            except Exception:
                pass
            return True

        old_sid = sid_file.read_text().strip()

        if force:
            sid_file.unlink(missing_ok=True)
            first_file.unlink(missing_ok=True)
            text = (
                f"<b>session reset (force)</b>\n\n"
                f"old: <code>{old_sid[:8]}...</code>\n"
                f"next message = new session"
            )
            try:
                tg_api(token, "sendMessage", chat_id=chat_id, text=text, parse_mode="HTML")
            except Exception:
                pass
            return True

        try:
            ack = tg_api(token, "sendMessage", chat_id=chat_id,
                         text="<i>handoff: сжимаю контекст...</i>", parse_mode="HTML")
            ack_msg_id = ack.get("result", {}).get("message_id")
        except Exception:
            ack_msg_id = None

        def _do_handoff_reset():
            handoff_prompt = (
                "СИСТЕМА: принц нажал /new — полный handoff перед сбросом сессии.\n"
                "Выполни ВСЕ 3 шага:\n\n"
                "1. HANDOFF: Прочитай core/hot/handoff.md (Read tool). "
                "Перезапиши его (Write tool) — оставь последние 10 записей из текущей сессии "
                "(что делали, что решили, что pending). Формат: '### YYYY-MM-DD HH:MM [тип]\\n**Принц:** ...\\n**Silvana:** ...'\n\n"
                "2. WARM: Прочитай core/warm/decisions.md (Read tool). "
                "Добавь в НАЧАЛО файла новую секцию с сегодняшней датой — "
                "ключевые решения, архитектурные изменения, новые правила из этой сессии. "
                "Не дублируй то что уже есть. Edit tool.\n\n"
                "3. MEMORY: Прочитай MEMORY.md (через auto-memory путь). "
                "Добавь/обнови записи: текущий focus, pending actions, "
                "важные решения которых нет в decisions.md. Edit tool.\n\n"
                "Ответь одной строкой: 'handoff: N в handoff, M в decisions, K в memory'"
            )
            import subprocess as _sp
            workspace = _get_workspace(agent, cfg)
            env = os.environ.copy()
            env["PATH"] = f"{Path.home()}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
            for k, v in (cfg.get("env") or {}).items():
                env[k] = v
            summary = "(не получен)"
            try:
                r = _sp.run(
                    ["claude", "-p", handoff_prompt, "--model", "sonnet",
                     "--output-format", "text", "--permission-mode", "bypassPermissions",
                     "--resume", old_sid],
                    cwd=workspace, env=env, capture_output=True, text=True, timeout=120,
                )
                summary = (r.stdout or r.stderr or "").strip()[:400]
            except Exception as e:
                summary = f"(ошибка handoff: {e})"
            sid_file.unlink(missing_ok=True)
            first_file.unlink(missing_ok=True)
            final_text = (
                f"<b>новая сессия</b>\n\n"
                f"старая: <code>{old_sid[:8]}...</code>\n"
                f"{escape_html(summary)}\n\n"
                f"следующее сообщение = свежий контекст"
            )
            if ack_msg_id:
                edit_message(token, chat_id, ack_msg_id, final_text)
            else:
                try:
                    tg_api(token, "sendMessage", chat_id=chat_id, text=final_text, parse_mode="HTML")
                except Exception:
                    pass

        import threading as _th
        _th.Thread(target=_do_handoff_reset, daemon=True).start()
        return True

    if cmd == "/compact":
        if not sid_file.exists():
            try:
                tg_api(token, "sendMessage", chat_id=chat_id, text="<b>no session</b> -- nothing to compact", parse_mode="HTML")
            except Exception:
                pass
            return True

        sid = sid_file.read_text().strip()
        try:
            ack = tg_api(token, "sendMessage", chat_id=chat_id, text="<i>compacting hot -> warm/decisions.md...</i>", parse_mode="HTML")
            ack_msg_id = ack.get("result", {}).get("message_id")
        except Exception:
            ack_msg_id = None

        def _do_compact():
            today = time.strftime("%Y-%m-%d")
            compact_prompt = (
                f"SYSTEM: manual hot->warm compact.\n"
                f"1. Read core/hot/recent.md (Read tool).\n"
                f"2. Extract key facts from last 24h and ADD to beginning of core/warm/decisions.md:\n"
                f"## {today}\n- fact 1\n- fact 2\n\n"
                f"3. Trim hot/recent.md: keep last 24h.\n"
                f"Extract: new preferences, decisions, pending actions, patterns. "
                f"Skip duplicates.\n"
                f"Reply: 'compact: N added, hot trimmed'."
            )
            import subprocess as _sp
            workspace = _get_workspace(agent, cfg)
            env = os.environ.copy()
            env["PATH"] = f"{Path.home()}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
            for k, v in (cfg.get("env") or {}).items():
                env[k] = v
            reply = "(timeout)"
            try:
                r = _sp.run(
                    ["claude", "-p", compact_prompt, "--model", "sonnet",
                     "--output-format", "text", "--permission-mode", "bypassPermissions",
                     "--resume", sid],
                    cwd=workspace, env=env, capture_output=True, text=True, timeout=180,
                )
                reply = (r.stdout or r.stderr or "").strip()[:400]
            except Exception as e:
                reply = f"error: {str(e)[:200]}"
            final_text = f"<b>compact done</b>\n\n<i>{escape_html(reply)}</i>"
            if ack_msg_id:
                edit_message(token, chat_id, ack_msg_id, final_text)
            else:
                try:
                    tg_api(token, "sendMessage", chat_id=chat_id, text=final_text, parse_mode="HTML")
                except Exception:
                    pass

        import threading as _th
        _th.Thread(target=_do_compact, daemon=True).start()
        return True

    if cmd in ("/stop", "/cancel"):
        proc = _ACTIVE_PROCS.get((agent, chat_id))
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                time.sleep(0.5)
                if proc.poll() is None:
                    proc.kill()
                text = "<b>stopped</b>\n\n<i>agent interrupted current task</i>"
            except Exception as e:
                text = f"<b>stop error</b>: {escape_html(str(e)[:100])}"
        else:
            text = "<b>nothing to stop</b> -- agent is idle"
        try:
            tg_api(token, "sendMessage", chat_id=chat_id, text=text, parse_mode="HTML")
        except Exception:
            pass
        return True

    if cmd in ("/model", "/toggle"):
        model_map = {
            "sonnet": "claude-sonnet-4-6",
            "opus": "claude-opus-4-7",
            "haiku": "claude-haiku-4-5-20251001",
        }
        alias_map = {v: k for k, v in model_map.items()}
        current_model = cfg.get("model", "sonnet")
        current_alias = alias_map.get(current_model, current_model)

        if cmd == "/toggle":
            alias = "opus" if current_alias == "sonnet" else "sonnet"
        else:
            alias = args.strip().split()[0] if args.strip() else ""
            if not alias:
                try:
                    tg_api(token, "sendMessage", chat_id=chat_id,
                           text=f"Current model: <b>{current_alias}</b> (<code>{current_model}</code>)\n\n"
                                f"Switch: <code>/model sonnet|opus|haiku</code> or <code>/toggle</code>",
                           parse_mode="HTML")
                except Exception:
                    pass
                return True
            if alias not in model_map:
                try:
                    tg_api(token, "sendMessage", chat_id=chat_id,
                           text="Usage: /model sonnet | opus | haiku")
                except Exception:
                    pass
                return True

        new_model = model_map[alias]
        cfg["model"] = new_model
        try:
            full_cfg = json.loads(CONFIG_PATH.read_text())
            full_cfg.setdefault("agents", {}).setdefault(agent, {})["model"] = new_model
            tmp_path = CONFIG_PATH.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(full_cfg, indent=2, ensure_ascii=False))
            os.replace(tmp_path, CONFIG_PATH)
            persisted = True
        except Exception as e:
            log.warning(f"{cmd} persist failed: {e}")
            persisted = False
        suffix = "" if persisted else " (in-memory only — persist failed, see logs)"
        try:
            tg_api(token, "sendMessage", chat_id=chat_id,
                   text=f"Model switched to <b>{alias}</b> (<code>{new_model}</code>){suffix}",
                   parse_mode="HTML")
        except Exception:
            pass
        log.info(f"[{agent}] {cmd} switched to {alias} ({new_model}) persisted={persisted}")
        return True

    if cmd == "/menu":
        # Onboarder is opt-in: only show wizard button when configured
        from_user = (cfg.get("_last_message_from_user") or {})  # set by process_update
        user_id = from_user.get("id", chat_id)
        onboarder_cfg = (cfg.get("onboarder") or {})
        onboarder_enabled = bool(onboarder_cfg.get("enabled"))

        rows: list[list[dict[str, str]]] = []
        rows.append([{"text": "💬 Чат с агентом", "callback_data": "menu:chat"}])
        if onboarder_enabled:
            rows.append([{"text": "🎓 Новый курс", "callback_data": "menu:onboard"}])
            rows.append([{"text": "📎 Загрузить YouTube cookies", "callback_data": "menu:cookies"}])
            current_compose = (onboarder_cfg.get("models") or {}).get("compose") or "opus"
            rows.append([{"text": f"⚙️ Модель курсов: {current_compose}",
                          "callback_data": "menu:compose_model"}])
        rows.append([{"text": "📊 Статус", "callback_data": "menu:status"}])

        current_mode = get_user_mode(agent, user_id)
        text = (
            f"<b>Меню</b>\n\n"
            f"Текущий режим: <code>{current_mode}</code>\n\n"
            "Выбери действие:"
        )
        send_message_with_buttons(token, chat_id, text, rows)
        return True

    if cmd == "/cancel":
        from_user = (cfg.get("_last_message_from_user") or {})
        user_id = from_user.get("id", chat_id)
        prev = get_user_mode(agent, user_id)
        set_user_mode(agent, user_id, MODE_CHAT)
        # Clear wizard state if any
        try:
            from onboarder import wizard as _wiz
            _wiz.clear_wizard_state(agent, user_id)
        except Exception:
            pass
        text = "<b>Окей, обратно в чат с агентом.</b>" if prev != MODE_CHAT else "<b>Уже в режиме чата.</b>"
        try:
            tg_api(token, "sendMessage", chat_id=chat_id, text=text, parse_mode="HTML")
        except Exception:
            pass
        return True

    if cmd == "/help":
        onboarder_enabled = bool((cfg.get("onboarder") or {}).get("enabled"))
        wizard_line = "<code>/menu</code> -- меню (включая 🎓 Новый курс)\n" if onboarder_enabled else ""
        text = (
            "<b>gateway commands</b>\n\n"
            "<code>/stop</code> or <code>/cancel</code> -- stop current agent task / выход из wizard\n"
            "<code>/status</code> -- session and memory status\n"
            "<code>/reset</code> -- reset session (saves important to MEMORY)\n"
            "<code>/reset force</code> -- reset without saving\n"
            "<code>/compact</code> -- manual memory compaction\n"
            "<code>/model</code> -- show current LLM model\n"
            "<code>/model sonnet|opus|haiku</code> -- switch LLM model\n"
            "<code>/toggle</code> -- toggle between sonnet and opus\n"
            f"{wizard_line}"
            "<code>/help</code> -- this help\n\n"
            "<i>auto-compact: daily 05:00 UTC</i>"
        )
        try:
            tg_api(token, "sendMessage", chat_id=chat_id, text=text, parse_mode="HTML")
        except Exception:
            pass
        return True

    return False


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def session_id_for(agent: str, chat_id: int) -> str:
    """Stable UUID per (agent, chat). Persisted on disk for --resume."""
    f = STATE_DIR / f"sid-{agent}-{chat_id}.txt"
    if f.exists():
        return f.read_text().strip()
    sid = str(uuid.uuid4())
    f.write_text(sid)
    return sid


def read_latest_memory_section(agent: str, cfg: dict) -> str:
    """Read latest section from core/MEMORY.md for post-reset injection."""
    workspace = _get_workspace(agent, cfg)
    mem_file = Path(workspace) / "core" / "MEMORY.md"
    if not mem_file.exists():
        return ""
    try:
        content = mem_file.read_text()
        # Find all ## sections and return the last one
        import re as _r
        matches = list(_r.finditer(r'^## .+$', content, _r.MULTILINE))
        if not matches:
            return content[:2000]
        last_start = matches[-1].start()
        # Read until next section or EOF
        return content[last_start:][:2000]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Claude invocation
# ---------------------------------------------------------------------------

def invoke_claude(
    agent: str, cfg: dict, chat_id: int, user_text: str
) -> tuple[str, int, int, list[str]]:
    """Run claude -p --resume <sid> and return (response_text, duration_ms, success, written_files)."""
    sid = session_id_for(agent, chat_id)
    workspace = _get_workspace(agent, cfg)
    model = cfg.get("model", "sonnet")

    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
    env.setdefault("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "400000")
    for k, v in cfg.get("env", {}).items():
        env[k] = v

    # Use --resume if session exists, --session-id for first turn
    sid_file = STATE_DIR / f"sid-{agent}-{chat_id}.first"
    is_first = not sid_file.exists()

    # Post-reset: inject latest MEMORY.md section so agent knows what was saved
    if is_first:
        latest_memory = read_latest_memory_section(agent, cfg)
        if latest_memory:
            user_text = (
                f"[after reset, latest entry in MEMORY.md]:\n{latest_memory}\n\n"
                f"[current user message]:\n{user_text}"
            )

    cmd = [
        "claude",
        "-p",
        user_text,
        "--model",
        model,
        "--output-format",
        "stream-json",
        "--input-format",
        "text",
        "--verbose",
        "--permission-mode",
        "bypassPermissions",
    ]

    # Enable extended thinking if configured (low|medium|high|xhigh|max).
    # Thinking blocks are rendered by _ProgressDisplay as a "думаю:" line,
    # which lets the student see the agent's reasoning in real time.
    effort = cfg.get("effort", "")
    if effort:
        cmd.extend(["--effort", effort])
        # Opus 4.7 ships encrypted thinking by default (signature only, empty
        # plain text). Force plain-text thinking so _ProgressDisplay can render
        # the "думаю:" line.
        cmd.extend(["--thinking-display", "summarized"])

    # Inject system reminder via --append-system-prompt (if configured)
    active_reminder = cfg.get("_active_system_reminder", "")
    if active_reminder:
        cmd.extend(["--append-system-prompt", active_reminder])

    if is_first:
        cmd.extend(["--session-id", sid])
    else:
        cmd.extend(["--resume", sid])

    timeout_sec = cfg.get("timeout_sec", 120)
    typing_cb = cfg.get("_typing_refresh_cb")
    status_cb = cfg.get("_status_update_cb")  # (status_text: str) -> None
    streaming_mode = cfg.get("streaming_mode", "partial")  # off | partial | progress
    # In "off" mode, disable status updates (no edit-in-place preview)
    if streaming_mode == "off":
        status_cb = None
    # Always create tracker for file tracking (sendDocument), even without status display
    tracker = _TaskBoundaryTracker(status_cb)
    t0 = time.time()
    last_activity = t0  # heartbeat: reset on every event from Claude
    last_typing = 0.0
    final_text = ""

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=workspace,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        # Mark session as registered immediately after Popen —
        # CLI already owns the session-id at this point
        if is_first:
            sid_file.touch()
        # Register for /stop command
        _ACTIVE_PROCS[(agent, chat_id)] = proc

        # Use select on proc.stdout for line-by-line reading with timeout
        import select as _select
        stdout_fd = proc.stdout.fileno() if proc.stdout else None
        buffer = ""

        while True:
            # Heartbeat timeout: kill only if no events for timeout_sec
            now = time.time()
            idle = now - last_activity
            if idle > timeout_sec:
                proc.kill()
                proc.communicate()
                dur_ms = int((now - t0) * 1000)
                log.error(
                    f"claude idle timeout after {idle:.0f}s silence "
                    f"(total {dur_ms}ms, limit={timeout_sec}s)"
                )
                return (
                    f"[gateway error: claude idle {int(idle)}s -- no activity]",
                    dur_ms,
                    0,
                    [],
                )

            # Refresh typing every 2s
            if time.time() - last_typing > 2:
                if typing_cb:
                    try:
                        typing_cb()
                    except Exception:
                        pass
                last_typing = time.time()

            # Read with timeout
            if stdout_fd is not None:
                ready, _, _ = _select.select([stdout_fd], [], [], 1.0)
                if ready:
                    line = proc.stdout.readline()
                    if not line:
                        # EOF -- process done
                        break
                    buffer += line
                    # Parse completed lines
                    while "\n" in buffer:
                        jline, buffer = buffer.split("\n", 1)
                        jline = jline.strip()
                        if not jline:
                            continue
                        try:
                            event = json.loads(jline)
                        except json.JSONDecodeError:
                            continue
                        last_activity = time.time()  # heartbeat: Claude is alive
                        _handle_stream_event(event, tracker)
                        # Capture final result text
                        if event.get("type") == "result":
                            final_text = event.get("result") or final_text
                else:
                    # No output yet -- check if process done
                    if proc.poll() is not None:
                        # Drain remaining
                        remaining = proc.stdout.read() if proc.stdout else ""
                        buffer += remaining
                        for jline in (buffer.strip().split("\n") if buffer.strip() else []):
                            jline = jline.strip()
                            if not jline:
                                continue
                            try:
                                event = json.loads(jline)
                                _handle_stream_event(event, tracker)
                                if event.get("type") == "result":
                                    final_text = event.get("result") or final_text
                            except json.JSONDecodeError:
                                pass
                        break

        proc.wait()
        dur_ms = int((time.time() - t0) * 1000)
        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            log.error(f"claude exit {proc.returncode}: {stderr[:500]}")
            if "already in use" in stderr:
                sid_file.touch()
                log.warning(f"auto-recovery: touched {sid_file.name} after 'already in use'")
            return (f"[gateway error: claude exit {proc.returncode}]", dur_ms, 0, [])
        return (
            final_text.strip(), dur_ms, 1,
            tracker.written_files if tracker else [],
        )
    except Exception as e:
        dur_ms = int((time.time() - t0) * 1000)
        log.exception(f"claude invoke failed: {e}")
        return (f"[gateway error: {e}]", dur_ms, 0, [])
    finally:
        _ACTIVE_PROCS.pop((agent, chat_id), None)


# ---------------------------------------------------------------------------
# Progress tracking (subagent labels, tool tags, status tracker)
# ---------------------------------------------------------------------------

SUBAGENT_LABELS = {
    "researcher": "searching and verifying sources",
    "content-writer": "writing draft",
    "content-orchestrator": "preparing content",
    "firebase-auditor": "auditing data",
    "code-reviewer": "code review",
    "general-purpose": "running research",
    "Explore": "exploring structure",
    "Plan": "building plan",
    "statusline-setup": "setting up status",
    "claude-code-guide": "checking docs",
}

# ASCII tool tags for real-time tool call display (NO emoji)
TOOL_TAGS: dict[str, str] = {
    "Read": "[R]",
    "Write": "[W]",
    "Edit": "[W]",
    "MultiEdit": "[W]",
    "Bash": "[B]",
    "Grep": "[G]",
    "Glob": "[G]",
    "WebFetch": "[F]",
    "WebSearch": "[S]",
    "Agent": "[A]",
    "TodoWrite": "[T]",
}
_TOOL_TAG_DEFAULT = "[.]"


def _summarize_tool_input(name: str, tinput: dict) -> str:
    """Extract a short human-readable summary of tool invocation."""
    s = ""
    if name in ("Read", "Write", "Edit", "MultiEdit"):
        fp = tinput.get("file_path") or tinput.get("path") or ""
        parts = fp.replace("\\", "/").rsplit("/", 2)
        s = "/".join(parts[-2:]) if len(parts) >= 2 else fp
    elif name == "Bash":
        s = (tinput.get("command") or "")[:40]
    elif name == "Grep":
        s = tinput.get("pattern") or ""
        if s:
            s = f'"{s}"'
    elif name == "Glob":
        s = tinput.get("pattern") or ""
        if s:
            s = f'"{s}"'
    elif name == "Agent":
        s = tinput.get("subagent_type") or tinput.get("description") or ""
    elif name == "WebFetch":
        url = tinput.get("url") or ""
        try:
            from urllib.parse import urlparse
            s = urlparse(url).netloc or url[:40]
        except Exception:
            s = url[:40]
    elif name == "WebSearch":
        s = (tinput.get("query") or "")[:40]
    else:
        s = str(tinput)[:30] if tinput else ""
    return escape_html(s[:40])


def _mask_secrets(s: str) -> str:
    """Mask IPs, URLs, tokens, secret paths in status messages for Telegram."""
    import re
    # Mask IPv4 addresses: 1.2.3.4 -> 1.***.***4
    s = re.sub(
        r'\b(\d{1,3})\.\d{1,3}\.\d{1,3}\.(\d{1,3})\b',
        r'\1.***.***.\2', s,
    )
    # Mask secret paths: ~/.secrets/... or /secrets/...
    s = re.sub(r'(~/?\.\w+/)secrets/\S+', r'\1secrets/***', s)
    # Mask API tokens/keys (long alphanumeric strings 20+ chars)
    s = re.sub(r'\b([A-Za-z0-9_-]{4})[A-Za-z0-9_-]{16,}([A-Za-z0-9_-]{4})\b', r'\1***\2', s)
    # Mask bot tokens: 1234567890:AAxx... -> 123***:AA***
    s = re.sub(r'\b(\d{3})\d{7,}:(AA\w{2})\w+', r'\1***:\2***', s)
    # Mask Supabase project URLs
    def _mask_host(m: re.Match) -> str:
        host = m.group(0)
        parts = host.split('.')
        if len(parts[0]) > 8:
            parts[0] = parts[0][:4] + '*****' + parts[0][-4:]
        if len(parts) > 1 and len(parts[-2]) > 5:
            parts[-2] = parts[-2][:4] + '***'
        return '.'.join(parts)
    s = re.sub(r'[a-z0-9]{10,}\.supabase\.co', _mask_host, s)
    return s


def _code(s: str) -> str:
    """Wrap text in <code> tags, escaping content."""
    return f"<code>{escape_html(s)}</code>"


def _humanize_tool(tname: str, tinput: dict) -> str | None:
    """Convert Claude tool invocation into human-readable status line (HTML)."""
    if tname == "Agent":
        sub = tinput.get("subagent_type", "?")
        label = SUBAGENT_LABELS.get(sub, f"running {escape_html(sub)}")
        return f"<b>{escape_html(label)}</b>"
    if tname == "Bash":
        cmd = (tinput.get("command") or "").strip()
        if cmd.startswith("curl"):
            return "calling API"
        if cmd.startswith("git"):
            return "git command"
        if cmd.startswith(("cat ", "tail ", "head ", "ls ", "grep ")):
            return "reading files"
        return f"running: {_code(_mask_secrets(cmd[:60]))}"
    if tname == "Read":
        path = tinput.get("file_path") or ""
        name = path.split("/")[-1] if path else ""
        return f"reading {_code(name)}" if name else "reading file"
    if tname == "Write":
        path = tinput.get("file_path") or ""
        name = path.split("/")[-1] if path else ""
        return f"creating {_code(name)}" if name else "creating file"
    if tname == "Edit":
        path = tinput.get("file_path") or ""
        name = path.split("/")[-1] if path else ""
        return f"editing {_code(name)}" if name else "editing file"
    if tname == "Glob":
        p = tinput.get("pattern") or ""
        return f"searching files: {_code(p[:40])}" if p else "searching files"
    if tname == "Grep":
        p = tinput.get("pattern") or ""
        return f"searching code: {_code(p[:40])}" if p else "searching code"
    if tname == "WebFetch":
        url = tinput.get("url") or ""
        return f"fetching web: {_code(url[:60])}"
    if tname == "WebSearch":
        q = tinput.get("query") or ""
        return f"web search: <i>{escape_html(q[:60])}</i>"
    if tname == "TodoWrite":
        return None  # handled specially
    return None


def _format_todos(todos: list) -> str | None:
    """Format TodoWrite list as HTML checklist."""
    if not todos:
        return None
    lines = []
    for t in todos[:8]:
        status = t.get("status", "pending")
        content = escape_html((t.get("content") or t.get("subject") or "")[:60])
        if status == "completed":
            lines.append(f"  [x] <s>{content}</s>")
        elif status == "in_progress":
            lines.append(f"  [>] <b>{content}</b>")
        else:
            lines.append(f"  [ ] {content}")
    if len(todos) > 8:
        lines.append(f"  ... +{len(todos) - 8} more")
    return "<b>plan:</b>\n" + "\n".join(lines)


class _StatusTracker:
    """Tracks assistant stream events and emits structured progress."""
    def __init__(self, status_cb):
        self.status_cb = status_cb
        self.step_num = 0
        self.current_subagent = None  # set when inside Agent tool turn
        self.lines: list[str] = []

    def _emit(self, line: str) -> None:
        self.lines.append(line)
        if self.status_cb:
            self.status_cb("\n".join(self.lines))

    def handle_event(self, event: dict) -> None:
        etype = event.get("type")
        if etype == "assistant":
            content = (event.get("message") or {}).get("content") or []
            for block in content:
                btype = block.get("type")
                if btype == "tool_use":
                    tname = block.get("name", "")
                    tinput = block.get("input") or {}
                    if tname == "TodoWrite":
                        todos = tinput.get("todos") or []
                        formatted = _format_todos(todos)
                        if formatted:
                            self._emit(formatted)
                        continue
                    line = _humanize_tool(tname, tinput)
                    if not line:
                        continue
                    if tname == "Agent":
                        sub = tinput.get("subagent_type", "?")
                        self.current_subagent = sub
                        self.step_num += 1
                        self._emit(f"<b>{self.step_num}.</b> {line}")
                    else:
                        if self.current_subagent:
                            self._emit(f"   <i>. {line}</i>")
                        else:
                            self.step_num += 1
                            self._emit(f"<b>{self.step_num}.</b> {line}")
        elif etype == "user":
            # tool_result event -- subagent may have finished
            content = (event.get("message") or {}).get("content") or []
            for block in content:
                if block.get("type") == "tool_result":
                    # Heuristic: if previous step was Agent, close the subagent context
                    if self.current_subagent:
                        self.current_subagent = None


def _handle_stream_event(event: dict, tracker: Any) -> None:
    """Pass event to the status tracker."""
    if tracker:
        tracker.handle_event(event)


def _progress_bar(done: int, total: int, width: int = 10) -> str:
    if total <= 0:
        return ""
    pct = min(100, int(done * 100 / total))
    filled = int(width * done / total)
    bar = "\u25b0" * filled + "\u25b1" * (width - filled)
    return f"{bar} {pct}%"


# File extensions that gateway can send as Telegram documents
SENDABLE_EXTENSIONS = {
    ".html", ".pdf", ".png", ".jpg", ".jpeg", ".csv", ".svg",
    ".pptx", ".ppt", ".xlsx", ".xls", ".docx", ".doc",
    ".zip", ".tar", ".gz", ".json", ".txt", ".py", ".md",
}


class _TaskBoundaryTracker:
    """Progress tracker: task boundaries + subagent dispatches.
    Clean display: thinking + plan + subagent steps.
    Individual tool calls (Read/Edit/Bash) collapsed to single activity line.

    Format:
      working -- 45s
      <thinking snippet>

      plan:
      [x] completed task
      [>] in-progress task
      [ ] pending task
      ▰▰▰▰▰▱▱▱▱▱ 50%

      steps:
       1 | researcher -- done
       2 | content-writer  <- now
       3 | delivery
    """
    def __init__(self, status_cb):
        self.status_cb = status_cb
        self.todos: list[dict] = []
        self.dispatches: list[dict] = []  # [{label, desc, status, summary}]
        self.pending_agents: dict[str, int] = {}  # tool_use_id -> dispatch index
        self.tool_calls: list[dict] = []  # [{tag, name, detail}] last N tool calls
        self._thinking: str = ""  # last thinking block, truncated to 3-4 lines
        self._start_time: float = time.time()
        self._last_tool_render: float = 0.0
        self.written_files: list[str] = []  # file paths from Write tool_use

    def _render(self) -> None:
        if not self.status_cb:
            return
        elapsed = int(time.time() - self._start_time)
        lines = [f"working -- {elapsed}s"]
        if self._thinking:
            lines.append("")
            lines.append(self._thinking)
        if self.tool_calls:
            lines.append("")
            lines.append(self._render_activity())
        if self.todos:
            lines.append("")
            lines.extend(self._render_todos())
        if self.dispatches:
            lines.append("")
            lines.extend(self._render_dispatches())
        body = "\n".join(lines).strip()
        if body:
            self.status_cb(f"<pre>{escape_html(body)}</pre>")

    def _render_activity(self) -> str:
        """Render the last N tool calls with tool names and short details.

        Shows a live stream of what the agent is doing:
            ▸ bash git push -u origin feature/...
            ▸ edit gateway.py
            ▸ read /tmp/foo.json

        Older tool calls beyond the window are collapsed into a single
        "+ N earlier" line so the block stays compact on mobile.
        """
        if not self.tool_calls:
            return ""
        total = len(self.tool_calls)
        window = 5
        recent = self.tool_calls[-window:]
        lines: list[str] = []
        if total > len(recent):
            lines.append(f"▸ ... +{total - len(recent)} earlier")
        for tc in recent:
            detail = tc.get("detail") or tc.get("name", "")
            detail = _mask_secrets(detail)
            lines.append(f"▸ {detail}")
        return "\n".join(lines)

    def _render_todos(self) -> str:
        done = sum(1 for t in self.todos if t.get("status") == "completed")
        total = len(self.todos)
        completed = [t for t in self.todos if t.get("status") == "completed"]
        in_progress = [t for t in self.todos if t.get("status") == "in_progress"]
        pending = [t for t in self.todos if t.get("status") == "pending"]
        visible: list[tuple[dict, str]] = []
        if completed:
            visible.append((completed[-1], "completed"))
        if done > 1:
            visible.insert(0, ({"content": f"... +{done - 1} done"}, "skip"))
        for t in in_progress:
            visible.append((t, "in_progress"))
        for t in pending[:2]:
            visible.append((t, "pending"))
        if len(pending) > 2:
            visible.append(({"content": f"... +{len(pending) - 2} more"}, "skip"))
        lines = [_progress_bar(done, total)]
        for t, st in visible:
            content = (t.get("content") or "")[:60]
            if st == "skip":
                lines.append(f"  {content}")
            elif st == "completed":
                lines.append(f"  x {content}")
            elif st == "in_progress":
                lines.append(f"  > {content}")
            else:
                lines.append(f"    {content}")
        return lines

    def _render_dispatches(self) -> list[str]:
        total = len(self.dispatches)
        single = total == 1
        lines: list[str] = [] if single else ["steps:"]
        for i, d in enumerate(self.dispatches[-4:], start=max(1, total - 3)):
            label = d["label"]
            status = d["status"]
            if single:
                if status == "done":
                    summary = (d.get("summary", "") or "")[:30]
                    marker = f" -- {summary}" if summary else ""
                    lines.append(f"x {label}{marker}")
                elif status == "running":
                    desc = (d.get("desc", "") or "")[:40]
                    desc_part = f" -- {desc}" if desc else ""
                    lines.append(f"> {label}{desc_part}")
            else:
                if status == "done":
                    summary = (d.get("summary", "") or "")[:30]
                    marker = f" -- {summary}" if summary else ""
                    lines.append(f" {i:>2} | x {label}{marker}")
                elif status == "running":
                    desc = (d.get("desc", "") or "")[:40]
                    desc_part = f" -- {desc}" if desc else ""
                    lines.append(f" {i:>2} | > {label}{desc_part}")
                else:
                    lines.append(f" {i:>2} |   {label}")
        done = sum(1 for d in self.dispatches if d["status"] == "done")
        if total > 1:
            lines.append(_progress_bar(done, total))
        return lines

    @staticmethod
    def _truncate_thinking(text: str, max_lines: int = 2, max_chars: int = 140) -> str:
        lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
        result = "\n".join(lines[-max_lines:])
        if len(result) > max_chars:
            result = result[:max_chars].rsplit(" ", 1)[0] + "..."
        return result

    def handle_event(self, event: dict) -> None:
        etype = event.get("type")
        if etype == "assistant":
            content = (event.get("message") or {}).get("content") or []
            for block in content:
                if block.get("type") == "thinking":
                    raw = block.get("thinking") or ""
                    if raw.strip():
                        self._thinking = self._truncate_thinking(raw)
                        self._render()
                    continue
                if block.get("type") != "tool_use":
                    continue
                tname = block.get("name", "")
                tinput = block.get("input") or {}
                tuid = block.get("id", "")
                # Track every tool call for real-time display
                tag = TOOL_TAGS.get(tname, _TOOL_TAG_DEFAULT)
                detail = _summarize_tool_input(tname, tinput)
                display_detail = f"{tname.lower()} {detail}" if detail else ""
                self.tool_calls.append({"tag": tag, "name": tname, "detail": display_detail})
                # Keep last 10 tool calls
                if len(self.tool_calls) > 10:
                    self.tool_calls = self.tool_calls[-10:]
                # Track written files for sendDocument
                if tname == "Write":
                    fp = tinput.get("file_path", "")
                    if fp:
                        ext = Path(fp).suffix.lower()
                        if ext in SENDABLE_EXTENSIONS:
                            self.written_files.append(fp)
                if tname == "TodoWrite":
                    self.todos = tinput.get("todos") or []
                    self._render()
                elif tname == "Agent":
                    sub = tinput.get("subagent_type", "?")
                    desc = (tinput.get("description") or "")[:60]
                    label = SUBAGENT_LABELS.get(sub, sub)
                    idx = len(self.dispatches)
                    self.dispatches.append({
                        "label": label, "desc": desc, "status": "running", "summary": "",
                    })
                    self.pending_agents[tuid] = idx
                    self._render()
                else:
                    now = time.time()
                    if now - self._last_tool_render >= 5.0:
                        self._last_tool_render = now
                        self._render()
        elif etype == "user":
            # tool_result for Agent = subagent finished
            content = (event.get("message") or {}).get("content") or []
            for block in content:
                if block.get("type") != "tool_result":
                    continue
                tuid = block.get("tool_use_id", "")
                if tuid in self.pending_agents:
                    idx = self.pending_agents.pop(tuid)
                    if idx < len(self.dispatches):
                        # Extract short summary
                        raw = block.get("content")
                        summary = ""
                        if isinstance(raw, str):
                            summary = raw
                        elif isinstance(raw, list):
                            for c in raw:
                                if isinstance(c, dict) and c.get("type") == "text":
                                    summary = c.get("text", "")
                                    break
                        first_line = next((ln for ln in summary.split("\n") if ln.strip()), "")
                        self.dispatches[idx]["status"] = "done"
                        self.dispatches[idx]["summary"] = first_line
                    self._render()


# ---------------------------------------------------------------------------
# Hot memory
# ---------------------------------------------------------------------------

HOT_SIZE_THRESHOLD = 15000  # chars, triggers compaction signal


def append_to_hot_memory(agent: str, cfg: dict, user_text: str, agent_response: str, source_tag: str) -> None:
    """Append lightweight turn summary to agent's hot/recent.md rolling journal.
    Uses fcntl.LOCK_EX to prevent interleaved writes from concurrent chat handlers.
    Emergency trim if file exceeds 20KB.
    """
    workspace = _get_workspace(agent, cfg)
    hot_file = Path(workspace) / "core" / "hot" / "recent.md"
    if not hot_file.parent.exists():
        return
    try:
        ts = time.strftime("%Y-%m-%d %H:%M")
        u_snippet = (user_text or "").replace("\n", " ")[:200]
        a_snippet = (agent_response or "(inline)").replace("\n", " ")[:200]
        entry = (
            f"\n### {ts} [{source_tag}]\n"
            f"**User:** {u_snippet}\n"
            f"**{agent.capitalize()}:** {a_snippet}\n"
        )
        # Atomic append with file lock
        with open(hot_file, "a") as f:
            _fcntl.flock(f.fileno(), _fcntl.LOCK_EX)
            try:
                f.write(entry)
                f.flush()
            finally:
                _fcntl.flock(f.fileno(), _fcntl.LOCK_UN)

        # Emergency trim if >20KB (compact failed to run or burst of messages)
        size = hot_file.stat().st_size
        if size > 20480:
            log.warning(f"[hot] {agent} recent.md {size}B -- emergency trim")
            # Keep last ~100 entries (roughly last 2-3 days)
            lines = hot_file.read_text().split("\n")
            # Find cutoff -- keep last 600 lines (entries are 4 lines each, 600/4=150 entries)
            if len(lines) > 600:
                header = "# Hot memory -- last 24h rolling journal\n"
                kept = lines[-600:]
                # Find first entry header to avoid truncated entry at top
                for i, ln in enumerate(kept):
                    if ln.startswith("### "):
                        kept = kept[i:]
                        break
                with open(hot_file, "w") as f:
                    _fcntl.flock(f.fileno(), _fcntl.LOCK_EX)
                    try:
                        f.write(header + "\n" + "\n".join(kept))
                    finally:
                        _fcntl.flock(f.fileno(), _fcntl.LOCK_UN)
    except Exception as e:
        log.warning(f"[hot] append failed: {e}")


# ---------------------------------------------------------------------------
# OpenViking semantic memory (optional)
# ---------------------------------------------------------------------------

def push_to_openviking(agent: str, cfg: dict, user_text: str, agent_response: str, chat_id: int) -> None:
    """Push conversation turn for L4 semantic memory extraction. Fire-and-forget.

    T-19 routing inversion (PLAN §B-4):

    * When ``cfg.l4_backend == "ov"`` (default) we run the legacy in-process
      OV push kept here byte-for-byte (pinned by
      ``silvana_cognee/tests/test_legacy_ov_push.py``).
    * When ``cfg.l4_backend`` is ``"cognee"`` or ``"dual"`` we delegate the
      full dispatch to ``silvana_cognee.dual_write.l4_write`` via the CLI
      bridge (``_l4_call``) and return. The CLI handles OV too, so there is
      no double-write.

    The canonical predicate for "is this tag eligible" lives in
    ``silvana_cognee.dual_write.should_push_to_l4``; callers gate on
    source_tag before invoking this function.
    """
    backend = _l4_backend(cfg)
    if backend != "ov":
        # Delegate to the cognee backend. Two transports:
        #   - ``http`` (M3): direct POST to shared Thrall. No local venv.
        #   - ``subprocess`` (default): shell out to the cognee venv CLI,
        #     which does the full OV+Cognee dispatch via dual_write.l4_write.
        source_category = _infer_source_category(user_text)
        payload = {
            "user_text": user_text,
            "agent_response": agent_response or "",
            "extra_meta": {},
        }
        if _l4_transport(cfg) == "http":
            _l4_http_call(
                "gateway-write", cfg, agent, chat_id, payload,
                extra={"source_category": source_category},
            )
        else:
            _l4_call(
                "gateway-write",
                [
                    "--agent", agent,
                    "--chat-id", str(chat_id),
                    # source_tag isn't passed into this function, so we infer it
                    # from the `[source:...]` prefix we embed on the user_text.
                    "--source-category", source_category,
                ],
                payload,
            )
        return

    # --- Legacy OV path (backend == "ov") -----------------------------------
    ov_url = cfg.get("openviking_url")
    if not ov_url:
        return
    ov_key_file = cfg.get("openviking_key_file")
    if not ov_key_file:
        return
    key_path = Path(expand(ov_key_file))
    if not key_path.exists():
        return
    key = key_path.read_text().strip()
    ov_account = cfg.get("openviking_account", "default")
    headers = {
        "X-API-Key": key,
        "X-OpenViking-Account": ov_account,
        "X-OpenViking-User": agent,
        "Content-Type": "application/json",
    }
    base = f"{ov_url.rstrip('/')}/api/v1"
    sid = None
    try:
        # Create session
        r = requests.post(f"{base}/sessions", headers=headers, json={}, timeout=10)
        if r.status_code != 200:
            log.warning(f"[ov] [{agent}/{chat_id}] create session failed: {r.status_code}")
            return
        sid = r.json().get("result", {}).get("session_id")
        if not sid:
            log.warning(f"[ov] [{agent}/{chat_id}] no session_id in response")
            return
        # user_text already has [source:...] prefix from process_update
        ts = time.strftime("%Y-%m-%d %H:%M")
        # Anti-pollution guard: strong instruction to OV LLM about forwarded content
        if "[source:forwarded" in user_text:
            guard = (
                "\n[extraction hint: this content was FORWARDED to the user from someone else. "
                "Do NOT extract as user's own preferences. Only extract as events/cases/entities about the third-party source.]\n"
            )
        elif "[source:external_media" in user_text:
            guard = (
                "\n[extraction hint: this is external media the user is sharing, not their own words. "
                "Do NOT extract as user's preferences.]\n"
            )
        else:
            guard = ""
        meta_prefix = f"[chat:{chat_id} agent:{agent} at {ts}]{guard}\n"
        # AUDIT [HIGH] fix: previously unchecked rc -> silent drop. Now we
        # log and stop the session early so the session DELETE still runs.
        r_user = requests.post(
            f"{base}/sessions/{sid}/messages",
            headers=headers,
            json={"role": "user", "content": meta_prefix + user_text[:3000]},
            timeout=10,
        )
        if r_user.status_code != 200:
            log.warning(f"[ov] [{agent}/{chat_id}] user message rc={r_user.status_code}")
            return
        if agent_response:
            r_asst = requests.post(
                f"{base}/sessions/{sid}/messages",
                headers=headers,
                json={"role": "assistant", "content": agent_response[:3000]},
                timeout=10,
            )
            if r_asst.status_code != 200:
                log.warning(f"[ov] [{agent}/{chat_id}] assistant message rc={r_asst.status_code}")
                return
        # Extract memories (runs LLM for structured extraction)
        ext = requests.post(f"{base}/sessions/{sid}/extract", headers=headers, json={}, timeout=60)
        extracted = ext.json().get("result", []) if ext.status_code == 200 else []
        log.info(f"[ov] [{agent}/{chat_id}] extracted {len(extracted)} memories")
    except Exception as e:
        log.warning(f"[ov] [{agent}/{chat_id}] push failed: {e}")
    finally:
        # Always clean up session to prevent leaks
        if sid:
            try:
                requests.delete(f"{base}/sessions/{sid}", headers=headers, timeout=5)
            except Exception:
                pass


def _infer_source_category(user_text: str) -> str:
    """Best-effort source_category recovery from the ``[source:...]`` prefix.

    The legacy signature of ``push_to_openviking`` does not carry the tag, so
    we read the bracket prefix the caller embedded earlier in the pipeline.
    Defaults to ``own_text`` — the bridge's safest whitelist bucket.
    """
    if not user_text:
        return "own_text"
    if "[source:forwarded" in user_text:
        return "forwarded"
    if "[source:external_media" in user_text:
        return "external_media"
    if "[source:own_voice" in user_text:
        return "own_voice"
    return "own_text"


def _auto_transcribe_group_voice(
    agent: str, cfg: dict, token: str, msg: dict
) -> None:
    """Auto-transcribe voice/audio/video_note in allowlisted groups.

    Downloads the file, transcribes via Groq Whisper, and replies with
    the transcript as italic HTML. Fire-and-forget via _L4_POOL.
    Does NOT interfere with normal message processing pipeline.
    """
    try:
        voice = (
            msg.get("voice")
            or msg.get("audio")
            or msg.get("video_note")
        )
        if not voice:
            return
        file_id = voice["file_id"]

        # Download using existing helper
        local_path = download_telegram_file(
            token, file_id, "voice", None
        )
        if not local_path:
            log.warning(
                f"[{agent}] auto-transcribe: download failed"
            )
            return

        # Transcribe using existing Groq Whisper helper
        transcript = transcribe_audio(local_path, agent_cfg=cfg)
        if not transcript or not transcript.strip():
            log.info(
                f"[{agent}] auto-transcribe: empty transcript"
            )
            return

        # Reply to the original voice message with italic transcript
        chat_id = msg["chat"]["id"]
        message_id = msg["message_id"]
        text = f"<i>{escape_html(transcript.strip())}</i>"
        tg_api(
            token, "sendMessage",
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
            reply_to_message_id=message_id,
            allow_sending_without_reply=True,
        )
        log.info(
            f"[{agent}] auto-transcribed voice in group {chat_id}"
        )
    except Exception:
        log.exception(
            f"[{agent}] auto-transcribe failed"
        )


def _build_group_message_content(msg: dict) -> tuple[str, str, int]:
    """Format a group message the same way legacy OV push did.

    Returned as ``(content, sender_name, chat_id)`` so both the in-process
    OV path and the CLI bridge can share the formatting.
    """
    from_user = msg.get("from") or {}
    sender_name = (
        from_user.get("first_name", "")
        + (" " + from_user.get("last_name", "") if from_user.get("last_name") else "")
    ).strip() or "Unknown"
    username = from_user.get("username", "")
    text = (msg.get("text") or msg.get("caption") or "").strip()
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(msg.get("date", 0)))

    lines = [
        f"[EdgeLab Chat] {ts}",
        f"From: {sender_name}" + (f" (@{username})" if username else ""),
    ]
    if text:
        lines.append(text)

    entities = msg.get("entities") or msg.get("caption_entities") or []
    links = []
    raw_text = msg.get("text") or msg.get("caption") or ""
    for ent in entities:
        if ent.get("type") == "url":
            url = raw_text[ent["offset"]:ent["offset"] + ent["length"]]
            links.append(url)
        elif ent.get("type") == "text_link":
            links.append(ent.get("url", ""))
    if links:
        lines.append("Links: " + ", ".join(links))

    media_types = []
    if msg.get("photo"):
        media_types.append("[Photo]")
    if msg.get("video"):
        media_types.append("[Video]")
    if msg.get("voice") or msg.get("audio"):
        media_types.append("[Voice]")
    if msg.get("document"):
        media_types.append("[Document]")
    if msg.get("sticker"):
        media_types.append("[Sticker]")
    if media_types:
        lines.append(" ".join(media_types))

    reply_msg = msg.get("reply_to_message")
    if reply_msg:
        quoted = (reply_msg.get("text") or reply_msg.get("caption") or "")[:100]
        if quoted:
            lines.append(f'Re: "{quoted}"')

    chat_id = (msg.get("chat") or {}).get("id", 0)
    return "\n".join(lines), sender_name, chat_id


def _push_group_message_to_ov(agent: str, cfg: dict, msg: dict) -> None:
    """Push a group chat message for L4 semantic logging. Fire-and-forget.

    T-19 routing inversion matches ``push_to_openviking``: when the L4
    backend is ``"ov"`` we run the in-process OV push (legacy shape). When
    the backend is ``"cognee"`` or ``"dual"`` we delegate to the CLI bridge
    (``_l4_call("gateway-write-group", ...)``) which owns the full OV+Cognee
    dispatch, and we skip the inline OV code below to avoid double-writes.
    """
    try:
        backend = _l4_backend(cfg)
        content, sender_name, chat_id = _build_group_message_content(msg)
        from_user = msg.get("from") or {}
        username = from_user.get("username", "") or sender_name

        if backend != "ov":
            # Keep the HTTP and subprocess payloads byte-identical so diffing
            # across transports during the M6 cutover stays meaningful. Python
            # side (silvana_cognee.dual_write.l4_write_group) injects
            # ``{"group": True, "user_handle": …}`` into extra_meta and pins
            # source_category="group_mention" — mirror that here.
            payload = {
                "user_handle": username,
                "message_text": content,
                "extra_meta": {
                    "sender_name": sender_name,
                    "user_handle": username,
                    "group": True,
                },
            }
            if _l4_transport(cfg) == "http":
                _l4_http_call(
                    "gateway-write-group", cfg, agent, chat_id, payload,
                    extra={"source_category": "group_mention"},
                )
            else:
                _l4_call(
                    "gateway-write-group",
                    [
                        "--agent", agent,
                        "--group-chat-id", str(chat_id),
                    ],
                    payload,
                )
            return

        # --- Legacy OV path (backend == "ov") -------------------------------
        ov_url = cfg.get("openviking_url")
        if not ov_url:
            return
        ov_key_file = cfg.get("openviking_key_file")
        if not ov_key_file:
            return
        key_path = Path(expand(ov_key_file))
        if not key_path.exists():
            return
        key = key_path.read_text().strip()
        ov_account = cfg.get("openviking_account", "default")
        ov_user = cfg.get("group_log_ov_user", "")
        if not ov_user:
            return  # feature disabled

        headers = {
            "X-API-Key": key,
            "X-OpenViking-Account": ov_account,
            "X-OpenViking-User": ov_user,
            "Content-Type": "application/json",
        }
        base = f"{ov_url.rstrip('/')}/api/v1"

        # OV session: create -> add message -> extract -> cleanup
        sid = None
        r = requests.post(f"{base}/sessions", headers=headers, json={}, timeout=10)
        if r.status_code != 200:
            log.warning(f"[ov-group] [{agent}/{chat_id}] create session rc={r.status_code}")
            return
        sid = r.json().get("result", {}).get("session_id")
        if not sid:
            log.warning(f"[ov-group] [{agent}/{chat_id}] no session_id in response")
            return
        try:
            # AUDIT [HIGH] fix: previously unchecked rc -> silent drop.
            r_msg = requests.post(
                f"{base}/sessions/{sid}/messages",
                headers=headers,
                json={"role": "user", "content": content[:3000]},
                timeout=10,
            )
            if r_msg.status_code != 200:
                log.warning(f"[ov-group] [{agent}/{chat_id}] message rc={r_msg.status_code}")
                return
            ext = requests.post(
                f"{base}/sessions/{sid}/extract",
                headers=headers, json={}, timeout=60,
            )
            extracted = ext.json().get("result", []) if ext.status_code == 200 else []
            log.info(
                f"[ov-group] [{agent}/{chat_id}] extracted {len(extracted)} memories "
                f"from {sender_name}"
            )
        finally:
            try:
                requests.delete(f"{base}/sessions/{sid}", headers=headers, timeout=5)
            except Exception:
                pass
    except Exception as e:
        log.warning(f"[ov-group] push failed for agent={agent}: {e}")


# ---------------------------------------------------------------------------
# Heartbeat (no-op / log-only, no external dependencies)
# ---------------------------------------------------------------------------

def record_heartbeat(
    agent: str, started_at_ms: int, dur_ms: int, status: str, chat_id: int
) -> None:
    """Log heartbeat info. Override this function to push to your analytics backend."""
    log.info(
        f"[heartbeat] agent={agent} chat={chat_id} status={status} "
        f"duration={dur_ms}ms started={started_at_ms}"
    )


# ---------------------------------------------------------------------------
# Main message processing
# ---------------------------------------------------------------------------

def process_update(agent: str, cfg: dict, token: str, update: dict, allowlist: list[int]) -> None:
    """Handle one Telegram update. Ignore non-message / non-allowlisted / non-addressed."""
    is_webhook = update.get("_webhook", False)
    msg = update.get("message") or update.get("channel_post")
    if not msg:
        return

    # Group/supergroup gating: check chat_id against allowlist_group_ids
    chat_type = (msg.get("chat") or {}).get("type", "private")
    is_group = chat_type in ("group", "supergroup")
    if not is_webhook and is_group:
        chat_id_check = msg["chat"]["id"]
        allowlist_groups = cfg.get("_allowlist_group_ids", [])
        if chat_id_check not in allowlist_groups:
            log.info(
                f"[{agent}] denied group chat_id={chat_id_check}"
            )
            return

    user_id = (msg.get("from") or {}).get("id")
    if not is_webhook and user_id not in allowlist:
        log.info(f"denied user_id={user_id} agent={agent}")
        return

    # Early transcription for voice in groups so is_addressed_to_agent can check transcript
    if is_group and not msg.get("_voice_transcript"):
        voice_obj = msg.get("voice") or msg.get("audio") or msg.get("video_note")
        if voice_obj:
            local = download_telegram_file(token, voice_obj["file_id"], "voice", None)
            if local:
                transcript = transcribe_audio(local, agent_cfg=cfg) or ""
                if transcript:
                    msg["_voice_transcript"] = transcript

    # Group-chat gating: only respond if addressed via @mention, name, or reply
    bot_username = cfg.get("_bot_username")
    if not is_webhook and not is_addressed_to_agent(agent, msg, bot_username, cfg):
        chat_type = (msg.get("chat") or {}).get("type", "?")
        log.info(f"[{agent}] group chat {chat_type}, not addressed, skip")
        return
    chat_id = msg["chat"]["id"]
    text = (msg.get("text") or msg.get("caption") or "").strip()
    message_id = msg.get("message_id")

    # Stash sender info on cfg so handle_command() can read user_id / username.
    # Cleared after dispatch to avoid leaking between messages.
    cfg["_last_message_from_user"] = msg.get("from") or {}

    # Handle gateway commands (/status, /reset, /help, /new, /menu, /cancel) -- don't go to claude
    if text.startswith("/"):
        parts = text.split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        if handle_command(token, chat_id, agent, cmd, args, cfg):
            log.info(f"[{agent}] command: {cmd} {args}".strip())
            cfg.pop("_last_message_from_user", None)
            return

    # Mode router: if user is in wizard mode, dispatch to onboarder, skip Claude.
    # Default mode is chat (existing behavior).
    if user_id is not None:
        mode = get_user_mode(agent, user_id)
        if mode == MODE_WIZARD:
            try:
                from onboarder import wizard as _wizard
                _wizard.handle_wizard_message(token, agent, cfg, chat_id, user_id, text, msg)
            except Exception as e:
                log.exception(f"[{agent}] wizard handler error: {e}")
                try:
                    tg_api(token, "sendMessage", chat_id=chat_id,
                           text=f"⚠️ Wizard error: {e}\nUse /cancel to return to chat.")
                except Exception:
                    pass
            cfg.pop("_last_message_from_user", None)
            return

    # Classify source for memory extraction provenance
    source_tag, source_label = classify_source(msg)

    # Handle media attachments
    media_note = ""
    media_ref = resolve_media_ref(msg)
    if media_ref:
        send_chat_action(token, chat_id, "typing")
        local = download_telegram_file(token, media_ref["file_id"], media_ref["type"], media_ref["file_name"])
        if local:
            mtype = media_ref["type"]
            if mtype in ("voice", "audio", "video_note"):
                # Use pre-transcribed text from producer if available
                transcript = msg.get("_voice_transcript") or ""
                if not transcript:
                    transcript = transcribe_audio(local, agent_cfg=cfg) or ""
                if transcript:
                    media_note = f"\n\n[Voice/audio transcript]: {transcript}"
                else:
                    media_note = f"\n\n[Audio transcription failed: {local}]"
            elif mtype == "video":
                # Video: attach path, claude Read tool can't play video but can reference
                media_note = f"\n\n[Video: {local} ({media_ref.get('mime_type') or 'video'})]"
            elif mtype == "photo":
                media_note = f"\n\n[Image: {local}] -- read via Read tool"
            elif mtype == "document":
                fname = media_ref.get("file_name") or local.name
                media_note = f"\n\n[File {fname}: {local}] -- read via Read tool (for PDF use pypdf or shell pdftotext)"
            elif mtype == "sticker":
                sticker_obj = msg.get("sticker") or {}
                sticker_emoji = sticker_obj.get("emoji", "")
                sticker_set = sticker_obj.get("set_name", "")
                sticker_uid = sticker_obj.get("file_unique_id", "")
                desc = _get_sticker_description(sticker_uid, sticker_emoji, sticker_set, local)
                media_note = f"\n\n[Sticker: {desc}]"
        else:
            media_note = f"\n\n[Failed to download {media_ref['type']} (possibly >20MB)]"

    if not text and not media_note:
        return  # nothing to process
    if not text:
        text = "(user sent attachment)"
    text = text + media_note

    # Ack reaction: eyes emoji to show message is being processed
    if message_id:
        set_reaction(token, chat_id, message_id, "\U0001f440")

    # Prepend source tag for OV memory extraction (invisible to agent context, used by gateway OV push)
    text_for_agent = text  # what agent sees
    # Forward context: agent sees who forwarded the message
    if source_tag == "forwarded":
        fwd_name = source_label.replace("forwarded from: ", "")
        text_for_agent = f"[Forwarded from: {fwd_name}]\n{text}"

    # Reply context (openclaw pattern: untrusted metadata block)
    # When user replies to a message, include its content so agent knows the reference.
    reply_msg = msg.get("reply_to_message")
    if isinstance(reply_msg, dict):
        reply_body = str(reply_msg.get("text") or reply_msg.get("caption") or "")
        reply_doc = reply_msg.get("document") or {}
        if isinstance(reply_doc, dict) and reply_doc and not reply_body:
            reply_body = f"[file: {reply_doc.get('file_name', '?')}]"
        # Fallback for media-only replies (photo/sticker/voice/video)
        if not reply_body:
            if reply_msg.get("photo"):
                reply_body = "[photo]"
            elif reply_msg.get("sticker"):
                reply_body = "[sticker]"
            elif reply_msg.get("voice"):
                reply_body = "[voice]"
            elif reply_msg.get("video") or reply_msg.get("video_note"):
                reply_body = "[video]"
            elif reply_msg.get("audio"):
                reply_body = "[audio]"
        reply_from = reply_msg.get("from") or {}
        is_bot_reply = bool(reply_from.get("is_bot", False)) if isinstance(reply_from, dict) else False
        sender_label = (
            "agent's previous message"
            if is_bot_reply
            else (reply_from.get("first_name") if isinstance(reply_from, dict) else None) or "unknown"
        )
        if not is_bot_reply and isinstance(reply_from, dict):
            reply_uname = reply_from.get("username")
            reply_uid = reply_from.get("id")
            if reply_uname:
                sender_label = f"{sender_label} (@{reply_uname})"
            if reply_uid:
                sender_label = f"{sender_label} [id:{reply_uid}]"
        if reply_body:
            truncated = len(reply_body) > 1200
            snippet = reply_body[:1200].replace("\x00", "")
            payload = {"sender": sender_label, "body": snippet}
            if truncated:
                payload["truncated"] = True
            reply_block = (
                "[Replied message (untrusted metadata, for context only):]\n"
                + json.dumps(payload, ensure_ascii=False)
                + "\n"
            )
            text_for_agent = reply_block + text_for_agent

    text_for_ov = f"[source:{source_tag} | {source_label}]\n{text}"

    # -----------------------------------------------------------------------
    # Group vs Private: choose streaming mode and system reminder per context
    # -----------------------------------------------------------------------
    chat_type = (msg.get("chat") or {}).get("type", "private")
    is_group = chat_type in ("group", "supergroup")

    if is_group:
        streaming_mode = cfg.get(
            "streaming_mode_group",
            cfg.get("streaming_mode", "off"),
        )
    else:
        streaming_mode = cfg.get(
            "streaming_mode_private",
            cfg.get("streaming_mode", "partial"),
        )

    _DEFAULT_GROUP_REMINDER = (
        "You are in a PUBLIC group chat. Rules:\n"
        "1. Answer concisely -- result only, no process\n"
        "2. Do NOT show: commands, file paths, logs, intermediate steps\n"
        "3. Do NOT reveal: private data, API keys, internal architecture\n"
        "4. Keep response under 500 characters unless asked for detail\n"
        "5. No code blocks unless specifically requested"
    )
    if is_group:
        active_reminder = cfg.get("system_reminder_group", "")
        if not active_reminder:
            active_reminder = _DEFAULT_GROUP_REMINDER
    else:
        active_reminder = cfg.get(
            "system_reminder_private",
            cfg.get("system_reminder", ""),
        )

    # Group context: prepend chat title and sender name so agent knows the source
    if is_group:
        chat_title = (msg.get("chat") or {}).get("title", "unknown")
        from_user = msg.get("from") or {}
        sender = from_user.get("first_name", "unknown")
        username = from_user.get("username")
        from_user_id = from_user.get("id")
        sender_label = f"{sender} (@{username})" if username else sender
        if from_user_id:
            sender_label = f"{sender_label} [id:{from_user_id}]"
        text_for_agent = (
            f"[Group: {chat_title} | From: {sender_label}]\n{text_for_agent}"
        )

    log.info(f"[{agent}] chat={chat_id} user={user_id} src={source_tag}: {text[:100]}")

    started_ms = int(time.time() * 1000)
    send_chat_action(token, chat_id, "typing")

    # No placeholder; status message created lazily on first real task-boundary event.
    status_msg_id = [None]
    last_edit_t = [0.0]
    last_sent = [""]

    def status_update(text_html: str) -> None:
        text_html = text_html[:3900]
        now = time.time()
        if text_html == last_sent[0] or now - last_edit_t[0] < 2.0:
            return
        last_edit_t[0] = now
        last_sent[0] = text_html
        if status_msg_id[0] is None:
            # Lazy create status message on first real event (TodoWrite or Agent dispatch)
            try:
                r = tg_api(token, "sendMessage", chat_id=chat_id, text=text_html, parse_mode="HTML")
                status_msg_id[0] = r.get("result", {}).get("message_id")
            except Exception:
                pass
        else:
            edit_message(token, chat_id, status_msg_id[0], text_html, html=True)

    invoke_cfg = {
        **cfg,
        "streaming_mode": streaming_mode,
        "_active_system_reminder": active_reminder,
        "_typing_refresh_cb": lambda: send_chat_action(token, chat_id, "typing"),
        "_status_update_cb": status_update,
    }
    response, dur_ms, status_int, written_files = invoke_claude(
        agent, invoke_cfg, chat_id, text_for_agent,
    )

    # Delete status message before sending final reply (keep it clean)
    if status_msg_id[0]:
        delete_message(token, chat_id, status_msg_id[0])

    if not response:
        response = "agent did not respond"
    status = "completed" if status_int else "error"

    record_heartbeat(agent, started_ms, dur_ms, status, chat_id)

    # Append to hot memory (always, regardless of source)
    append_to_hot_memory(agent, cfg, text_for_agent, response or "(inline)", source_tag)

    # Skip OV push if voice transcription failed (would pollute with error text)
    transcribe_failed = media_note and "transcription failed" in media_note

    # L4 push via bounded thread pool -- only for own content or forwarded (marked).
    # Canonical predicate lives in silvana_cognee.dual_write.should_push_to_l4;
    # the gateway keeps an inline whitelist here to avoid importing the cognee
    # venv at gateway start. Keep these two lists in sync.
    if source_tag in ("own_text", "own_voice", "forwarded") and not transcribe_failed:
        _L4_POOL.submit(
            push_to_openviking, agent, cfg, text_for_ov,
            response or "(answered inline)", chat_id
        )
    # external_media -> hot only, not L4 memory extraction (to avoid polluting preferences)

    if response:  # not already delivered via edit-in-place
        try:
            send_message(token, chat_id, response, reply_to=message_id)
            log.info(f"[{agent}] replied chat={chat_id} dur={dur_ms}ms")
        except Exception as e:
            log.exception(f"reply failed: {e}")

    # Send written files as documents (sendDocument)
    if written_files:
        workspace = Path(expand(cfg["workspace"])).resolve()
        for fpath in written_files:
            try:
                # Resolve relative paths against agent workspace, not gateway CWD
                raw = Path(fpath)
                p = (workspace / raw).resolve() if not raw.is_absolute() else raw.resolve()
                # Security: only send files within agent workspace
                if not p.is_relative_to(workspace):
                    log.warning(
                        f"[{agent}] send_document blocked:"
                        f" {fpath} outside workspace"
                    )
                    continue
                if p.exists() and p.stat().st_size > 0:
                    send_chat_action(token, chat_id, "upload_document")
                    send_document(
                        token, chat_id, str(p),
                        caption=f"<code>{escape_html(p.name)}</code>",
                    )
                    log.info(
                        f"[{agent}] sent document:"
                        f" {p.name} chat={chat_id}"
                    )
            except Exception as e:
                log.warning(
                    f"[{agent}] send_document failed"
                    f" for {fpath}: {e}"
                )


# ---------------------------------------------------------------------------
# Bot commands menu
# ---------------------------------------------------------------------------

_BOT_COMMANDS = [
    {"command": "new", "description": "Новая сессия (полный handoff)"},
    {"command": "status", "description": "Статус сессии и памяти"},
    {"command": "stop", "description": "Остановить текущую задачу"},
    {"command": "compact", "description": "Компактизация памяти"},
    {"command": "reset", "description": "Сброс без handoff (force)"},
    {"command": "model", "description": "Текущая модель / sonnet|opus|haiku"},
    {"command": "toggle", "description": "Переключить sonnet ↔ opus"},
    {"command": "help", "description": "Справка по командам"},
]


# ---------------------------------------------------------------------------
# Polling: producer-consumer architecture
# ---------------------------------------------------------------------------

def _is_oob_command(text: str) -> bool:
    """Check if message text is an out-of-band command for instant handling."""
    if not text or not text.startswith("/"):
        return False
    parts = text.split(None, 1)
    cmd = parts[0].lower()
    # Strip @botname suffix (e.g. /stop@mybotname)
    if "@" in cmd:
        cmd = cmd.split("@")[0]
    return cmd in _OOB_COMMANDS


def _handle_oob_command(
    agent: str, token: str, chat_id: int, text: str,
    cfg: dict | None = None,
) -> None:
    """Handle out-of-band command immediately from producer thread.

    Handles: /stop, /status, /reset force
    Non-force /reset and other commands are queued for consumer.
    """
    parts = text.split(None, 2)
    cmd = parts[0].lower()
    # Strip @botname suffix
    if "@" in cmd:
        cmd = cmd.split("@")[0]
    args = parts[1] if len(parts) > 1 else ""

    if cmd in ("/stop", "/cancel"):
        proc = _ACTIVE_PROCS.get((agent, chat_id))
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                time.sleep(0.5)
                if proc.poll() is None:
                    proc.kill()
                reply = (
                    "<b>stopped</b>\n\n"
                    "<i>agent interrupted current task</i>"
                )
            except Exception as e:
                reply = (
                    f"<b>stop error</b>: "
                    f"{escape_html(str(e)[:100])}"
                )
        else:
            reply = "<b>nothing to stop</b> -- agent is idle"
        try:
            tg_api(
                token, "sendMessage",
                chat_id=chat_id, text=reply, parse_mode="HTML",
            )
        except Exception:
            pass
        log.info(f"[{agent}] OOB /stop chat={chat_id}")
        return

    if cmd == "/status":
        handle_command(token, chat_id, agent, "/status", "", cfg=cfg or {})
        log.info(f"[{agent}] OOB /status chat={chat_id}")
        return

    if cmd == "/reset" and args.strip().lower() == "force":
        # Kill active subprocess if any
        proc = _ACTIVE_PROCS.pop((agent, chat_id), None)
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                time.sleep(0.5)
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
        # Delete session files
        sid_file = STATE_DIR / f"sid-{agent}-{chat_id}.txt"
        first_file = STATE_DIR / f"sid-{agent}-{chat_id}.first"
        old_sid = ""
        if sid_file.exists():
            old_sid = sid_file.read_text().strip()
        sid_file.unlink(missing_ok=True)
        first_file.unlink(missing_ok=True)
        reply = (
            "<b>session reset (force)</b>\n\n"
            f"old: <code>{escape_html(old_sid[:8])}...</code>\n"
            "next message = new session"
        )
        try:
            tg_api(
                token, "sendMessage",
                chat_id=chat_id, text=reply, parse_mode="HTML",
            )
        except Exception:
            pass
        log.info(f"[{agent}] OOB /reset force chat={chat_id}")
        return


def _init_bot_metadata(agent: str, cfg: dict, token: str) -> None:
    """Cache bot_username and register commands menu (once per agent)."""
    if cfg.get("_bot_username"):
        return
    try:
        info = tg_api(token, "getMe")
        bot_username = (info.get("result") or {}).get("username")
        cfg["_bot_username"] = bot_username
        try:
            tg_api(token, "setMyCommands", commands=_BOT_COMMANDS)
        except Exception as e:
            log.warning(f"[{agent}] setMyCommands failed: {e}")
    except Exception:
        pass


def polling_producer(
    agent: str, cfg: dict, allowlist: list[int], offset_file: Path
) -> None:
    """Daemon thread: polls Telegram, routes OOB commands immediately,
    queues regular messages for consumer.

    OOB commands (/stop, /status, /reset force) are handled instantly
    even when the consumer thread is blocked on invoke_claude().
    """
    token = _resolve_telegram_token(cfg)
    _init_bot_metadata(agent, cfg, token)
    msg_queue = _MSG_QUEUES[agent]
    poll_interval_sec = 1  # producer polls every 1s for responsiveness

    log.info(f"[{agent}] producer thread started")

    while not _SHUTDOWN_EVENT.is_set():
        # Read offset
        offset = 0
        if offset_file.exists():
            try:
                offset = int(offset_file.read_text().strip() or "0")
            except Exception:
                offset = 0

        try:
            r = tg_api(
                token, "getUpdates",
                offset=offset, timeout=0, limit=10,
            )
        except Exception as e:
            log.warning(f"[{agent}] producer getUpdates failed: {e}")
            time.sleep(poll_interval_sec)
            continue

        updates = r.get("result", [])
        for upd in updates:
            new_offset = upd["update_id"] + 1
            offset_file.write_text(str(new_offset))

            # Handle callback queries (inline button presses)
            cq = upd.get("callback_query")
            if cq:
                dispatch_callback_query(token, agent, cfg, cq)
                continue

            msg = upd.get("message") or upd.get("channel_post")
            if not msg:
                continue

            # Group/supergroup gating: check chat_id against allowlist_group_ids
            chat_type = (msg.get("chat") or {}).get("type", "private")
            is_group = chat_type in ("group", "supergroup")
            if is_group:
                group_chat_id = msg["chat"]["id"]
                allowlist_groups = cfg.get("_allowlist_group_ids", [])
                if group_chat_id not in allowlist_groups:
                    log.info(
                        f"[{agent}] producer denied group "
                        f"chat_id={group_chat_id}"
                    )
                    continue

            # Log ALL messages from allowlisted groups to L4 semantic memory
            # (fire-and-forget, before user_id check -- logs even
            # non-allowlisted users' messages for group context). The L4
            # dispatcher (silvana_cognee.dual_write.l4_write_group) is the
            # canonical predicate source; see _push_group_message_to_ov.
            if is_group and cfg.get("group_log_ov_user"):
                _L4_POOL.submit(
                    _push_group_message_to_ov, agent, cfg, msg
                )

            user_id = (msg.get("from") or {}).get("id")
            if user_id not in allowlist:
                log.info(
                    f"[{agent}] producer denied user_id={user_id}"
                )
                continue

            # Per-topic routing: if agent has topic_routing config, only accept
            # messages from specified topics in specified groups
            topic_routing = cfg.get("topic_routing")
            if topic_routing:
                chat_id_str = str(msg["chat"]["id"])
                thread_id = msg.get("message_thread_id")
                if chat_id_str in topic_routing:
                    allowed_topics = topic_routing[chat_id_str]
                    if thread_id is None or str(thread_id) not in allowed_topics:
                        continue  # message not in routed topic for this agent

            # Early voice transcription in groups so is_addressed_to_agent can check transcript
            if is_group and not msg.get("_voice_transcript"):
                voice_obj = msg.get("voice") or msg.get("audio") or msg.get("video_note")
                if voice_obj:
                    local = download_telegram_file(token, voice_obj["file_id"], "voice", None)
                    if local:
                        transcript = transcribe_audio(local, agent_cfg=cfg) or ""
                        if transcript:
                            msg["_voice_transcript"] = transcript

            # Group-chat gating
            bot_username = cfg.get("_bot_username")
            if not is_addressed_to_agent(agent, msg, bot_username, cfg):
                continue

            chat_id = msg["chat"]["id"]
            text = (
                msg.get("text") or msg.get("caption") or ""
            ).strip()

            # Out-of-band commands: handle immediately in producer
            if _is_oob_command(text):
                # /reset without 'force' goes to consumer (it needs
                # blocking Claude save). Only /reset force is OOB.
                parts = text.split(None, 2)
                cmd = parts[0].lower()
                if "@" in cmd:
                    cmd = cmd.split("@")[0]
                args = parts[1].lower() if len(parts) > 1 else ""

                if cmd == "/new" or (cmd == "/reset" and args != "force"):
                    # /new and non-force /reset -> queue for consumer (blocking handoff)
                    msg_queue.put(upd)
                    continue

                try:
                    _handle_oob_command(agent, token, chat_id, text, cfg=cfg)
                except Exception:
                    log.exception(
                        f"[{agent}] OOB command failed: {text}"
                    )
                continue

            # Regular message -> queue for consumer
            msg_queue.put(upd)

        time.sleep(poll_interval_sec)

    log.info(f"[{agent}] producer thread stopped")


def message_consumer(
    agent: str, cfg: dict, token: str, allowlist: list[int]
) -> None:
    """Daemon thread: takes messages from queue, invokes Claude.

    Blocks on queue.get() when no messages. Only one message processed
    at a time per agent (serial execution).
    """
    msg_queue = _MSG_QUEUES[agent]
    log.info(f"[{agent}] consumer thread started")

    while not _SHUTDOWN_EVENT.is_set():
        try:
            # Block with timeout so we can check shutdown event
            upd = msg_queue.get(timeout=2)
        except queue.Empty:
            continue

        try:
            process_update(agent, cfg, token, upd, allowlist)
        except Exception:
            log.exception(f"[{agent}] consumer update processing failed")
        finally:
            msg_queue.task_done()

    log.info(f"[{agent}] consumer thread stopped")


# ---------------------------------------------------------------------------
# Webhook API -- lightweight HTTP server for external triggers
# ---------------------------------------------------------------------------

from http.server import HTTPServer, BaseHTTPRequestHandler as _BaseHandler


class _WebhookHandler(_BaseHandler):
    """Handle POST /hooks/agent -- inject message into agent queue."""

    # Set by main() before server starts
    gateway_cfg: dict = {}
    gateway_agents: dict = {}
    webhook_token: str = ""

    def log_message(self, fmt: str, *args: Any) -> None:
        log.debug(f"[webhook] {fmt % args}")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/hooks/agent":
            self._reply(404, {"error": "not found"})
            return

        # Auth check
        auth = self.headers.get("Authorization", "")
        expected = f"Bearer {self.webhook_token}" if self.webhook_token else ""
        if self.webhook_token and auth != expected:
            self._reply(401, {"error": "unauthorized"})
            return

        # Parse body
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
        except Exception:
            self._reply(400, {"error": "invalid json"})
            return

        agent_id = body.get("agentId", "")
        message = body.get("message", "")
        if not agent_id or not message:
            self._reply(400, {"error": "agentId and message required"})
            return

        if agent_id not in self.gateway_agents:
            self._reply(404, {"error": f"agent '{agent_id}' not found"})
            return

        # Inject into agent queue as synthetic update
        q = _MSG_QUEUES.get(agent_id)
        if not q:
            self._reply(503, {"error": f"agent '{agent_id}' queue not ready"})
            return

        # Build a synthetic Telegram-like update
        chat_id = body.get("chatId") or body.get("to")
        synthetic_msg = {
            "message_id": 0,
            "from": {"id": 0, "first_name": "webhook", "is_bot": False},
            "chat": {"id": int(chat_id) if chat_id else 0, "type": "private"},
            "date": int(time.time()),
            "text": message,
        }
        q.put({"update_id": 0, "message": synthetic_msg, "_webhook": True})
        log.info(f"[webhook] injected message for {agent_id}: {message[:80]}")
        self._reply(200, {"ok": True, "agent": agent_id})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            agents_status = {
                a: "online" for a in self.gateway_agents
            }
            self._reply(200, {"status": "ok", "agents": agents_status})
        else:
            self._reply(404, {"error": "not found"})

    def _reply(self, code: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start_webhook_server(
    cfg: dict, agents: dict, port: int, token: str
) -> None:
    """Start webhook HTTP server in daemon thread."""
    _WebhookHandler.gateway_cfg = cfg
    _WebhookHandler.gateway_agents = agents
    _WebhookHandler.webhook_token = token

    server = HTTPServer(("127.0.0.1", port), _WebhookHandler)
    t = threading.Thread(
        target=server.serve_forever,
        name="webhook-server",
        daemon=True,
    )
    t.start()
    log.info(f"[webhook] HTTP server on 127.0.0.1:{port}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _menu_callback_handler(token: str, agent: str, cfg: dict, cq: dict) -> None:
    """Handle inline buttons from /menu (callback_data: 'menu:chat' | 'menu:onboard' | 'menu:status')."""
    cq_id = cq.get("id", "")
    data = cq.get("data", "")
    msg = cq.get("message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    user_id = (cq.get("from") or {}).get("id")
    if chat_id is None or user_id is None:
        answer_callback_query(token, cq_id)
        return

    action = data.split(":", 1)[1] if ":" in data else ""

    if action == "chat":
        set_user_mode(agent, user_id, MODE_CHAT)
        try:
            from onboarder import wizard as _wiz
            _wiz.clear_wizard_state(agent, user_id)
        except Exception:
            pass
        answer_callback_query(token, cq_id, "Режим: чат с агентом")
        try:
            tg_api(token, "sendMessage", chat_id=chat_id,
                   text="<b>💬 Чат с агентом активен.</b>\nПиши как обычно.",
                   parse_mode="HTML")
        except Exception:
            pass
        return

    if action == "onboard":
        if not (cfg.get("onboarder") or {}).get("enabled"):
            answer_callback_query(token, cq_id, "Онбордер выключен в config.json", show_alert=True)
            return
        set_user_mode(agent, user_id, MODE_WIZARD)
        answer_callback_query(token, cq_id, "Запускаю wizard…")
        try:
            from onboarder import wizard as _wiz
            _wiz.start_wizard(token, agent, cfg, chat_id, user_id)
        except Exception as e:
            log.exception(f"[{agent}] start_wizard failed: {e}")
            try:
                tg_api(token, "sendMessage", chat_id=chat_id,
                       text=f"⚠️ Не удалось запустить wizard: {e}")
            except Exception:
                pass
        return

    if action == "cookies":
        if not (cfg.get("onboarder") or {}).get("enabled"):
            answer_callback_query(token, cq_id, "Онбордер выключен в config.json", show_alert=True)
            return
        set_user_mode(agent, user_id, MODE_WIZARD)
        answer_callback_query(token, cq_id, "Жду файл cookies.txt…")
        try:
            from onboarder import wizard as _wiz
            _wiz.start_cookies_upload(token, agent, cfg, chat_id, user_id)
        except Exception as e:
            log.exception(f"[{agent}] start_cookies_upload failed: {e}")
            try:
                tg_api(token, "sendMessage", chat_id=chat_id,
                       text=f"⚠️ Не удалось запустить загрузку cookies: {e}")
            except Exception:
                pass
        return

    if action == "status":
        # Reuse /status text path
        from_msg = {"chat": {"id": chat_id}, "from": {"id": user_id}}
        cfg["_last_message_from_user"] = from_msg["from"]
        try:
            handle_command(token, chat_id, agent, "/status", "", cfg)
        finally:
            cfg.pop("_last_message_from_user", None)
        answer_callback_query(token, cq_id)
        return

    if action == "compose_model":
        # Show submenu: pick model for course composition
        onb = (cfg.get("onboarder") or {})
        current = (onb.get("models") or {}).get("compose") or "opus"
        answer_callback_query(token, cq_id)
        send_message_with_buttons(
            token, chat_id,
            text=(f"⚙️ <b>Модель для генерации курсов</b>\n\n"
                  f"Текущая: <code>{current}</code>\n\n"
                  f"Выбери:\n"
                  f"• <b>Opus</b> — лучшее качество прозы, медленнее\n"
                  f"• <b>Sonnet</b> — быстрее и дешевле, качество ниже\n"
                  f"• <b>Haiku</b> — самая быстрая, для простых тем"),
            buttons=[[
                {"text": f"{'✓ ' if current=='opus' else ''}Opus", "callback_data": "menu:set_compose:opus"},
                {"text": f"{'✓ ' if current=='sonnet' else ''}Sonnet", "callback_data": "menu:set_compose:sonnet"},
                {"text": f"{'✓ ' if current=='haiku' else ''}Haiku", "callback_data": "menu:set_compose:haiku"},
            ]],
        )
        return

    if action.startswith("set_compose:"):
        new_model = action.split(":", 1)[1]
        if new_model not in ("sonnet", "opus", "haiku"):
            answer_callback_query(token, cq_id, "Неизвестная модель", show_alert=True)
            return
        try:
            full_cfg = json.loads(CONFIG_PATH.read_text())
            full_cfg.setdefault("agents", {}).setdefault(agent, {}) \
                .setdefault("onboarder", {}).setdefault("models", {})["compose"] = new_model
            tmp_path = CONFIG_PATH.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(full_cfg, indent=2, ensure_ascii=False))
            os.replace(tmp_path, CONFIG_PATH)
            # Update in-memory cfg too so the next phase 2 picks it up immediately
            cfg.setdefault("onboarder", {}).setdefault("models", {})["compose"] = new_model
            persisted = True
        except Exception as e:
            log.warning(f"set_compose persist failed: {e}")
            persisted = False
        suffix = "" if persisted else " ⚠️ (in-memory only — не сохранилось в config.json)"
        answer_callback_query(token, cq_id, f"Модель: {new_model}")
        try:
            tg_api(token, "sendMessage", chat_id=chat_id,
                   text=f"✅ Модель курсов: <code>{new_model}</code>{suffix}",
                   parse_mode="HTML")
        except Exception:
            pass
        log.info(f"[{agent}] onboarder.models.compose → {new_model} persisted={persisted}")
        return

    answer_callback_query(token, cq_id)


def main() -> None:
    if not CONFIG_PATH.exists():
        log.error(f"config not found: {CONFIG_PATH}")
        log.error("create config.json in the working directory (see docstring for format)")
        sys.exit(1)

    cfg = json.loads(CONFIG_PATH.read_text())
    # Accept both key names: 'allowed_user_ids' (Day 1 format) and
    # 'allowlist_user_ids' (legacy). Day 1 takes priority.
    allowlist = cfg.get("allowed_user_ids") or cfg.get("allowlist_user_ids", [])
    allowlist_groups = cfg.get("allowed_group_ids") or cfg.get("allowlist_group_ids", [])
    agents = {
        k: v for k, v in cfg["agents"].items() if v.get("enabled")
    }

    if not agents:
        log.error("no enabled agents in config")
        sys.exit(1)

    # Register inline-keyboard handlers for /menu (chat | onboard | status)
    register_callback_handler("menu:", _menu_callback_handler)
    try:
        from onboarder import wizard as _wiz
        _wiz.register_callbacks(register_callback_handler)
    except ImportError:
        log.info("onboarder module not present; wizard mode disabled")
    except Exception as e:
        log.warning(f"onboarder.wizard.register_callbacks failed: {e}")

    log.info(
        f"gateway started (producer-consumer), "
        f"agents={list(agents.keys())}, allowlist={allowlist}, "
        f"allowlist_groups={allowlist_groups}"
    )

    # Start webhook API server (optional)
    webhook_port = cfg.get("webhook_port", 0)
    if webhook_port:
        webhook_token = cfg.get("webhook_token", "")
        _start_webhook_server(cfg, agents, webhook_port, webhook_token)

    offsets = {a: STATE_DIR / f"offset-{a}.txt" for a in agents}
    threads: list[threading.Thread] = []

    for agent, acfg in agents.items():
        # Inject group allowlist into per-agent config for access in handlers
        acfg["_allowlist_group_ids"] = allowlist_groups

        # Initialize per-agent message queue
        _MSG_QUEUES[agent] = queue.Queue()

        # Read token for consumer thread
        token = _resolve_telegram_token(acfg)

        # Producer thread: polls Telegram, handles OOB commands
        t_prod = threading.Thread(
            target=polling_producer,
            args=(agent, acfg, allowlist, offsets[agent]),
            name=f"producer-{agent}",
            daemon=True,
        )
        t_prod.start()
        threads.append(t_prod)

        # Consumer thread: processes queued messages
        t_cons = threading.Thread(
            target=message_consumer,
            args=(agent, acfg, token, allowlist),
            name=f"consumer-{agent}",
            daemon=True,
        )
        t_cons.start()
        threads.append(t_cons)

    # Main thread waits for shutdown signal
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("gateway shutting down...")
        _SHUTDOWN_EVENT.set()
        for t in threads:
            t.join(timeout=5)
        log.info("gateway stopped")


if __name__ == "__main__":
    main()
