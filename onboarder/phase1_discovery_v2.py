"""Phase 1 v2: hybrid Claude + yt-dlp discovery pipeline.

This module is OPT-IN — gated by `cfg["agents"][AGENT]["onboarder"]["phase1_use_new_pipeline"]`.
When false (default), `phase1_discovery._run` runs the original yt-dlp-search-driven
pipeline unchanged. When true, the dispatcher in `phase1_discovery._run` routes
to `_run_v2` below.

Rollback path: flip the config flag back to false and restart the gateway.
No git revert needed — old code is untouched.

Pipeline (vs v1):
    OLD:                                          NEW:
    1. generate_search_queries (Claude)           1. Claude discovery — 30 channel handles
    2. yt-dlp ytsearch (×6 queries)               2. yt-dlp size filter — drop !in [500..70K]
    3. parallel get_channel_metadata              3. yt-dlp /videos pull — filter dur/year
    4. score_channels (Claude)                    4. Claude select — pick 12-15 from real list
    5. per-channel select_videos (Claude)         5. Loop steps 1-4 with exclusion list if < count
    6. enrich (Whisper/compose) → Sheet → button  6. enrich (Whisper/compose) → Sheet → button

Steps 1-4 here REPLACE old steps 1-5; step 6 (enrich + Sheet + button) is
copied from old `_run` to keep new module isolated. F4 rejection feedback,
F6 contiguous course_idx, batch mode, F2 noise routing — all preserved.

Tested standalone in /home/jarvis/projects/jarvis-telegram-gateway/_test_new_pipeline.py
on 2 topics (TMJ, edema) — produced 5 courses with 12-15 videos each, 0 hallucinations.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from . import (_secrets, llm, phase1_enrich, proxy_pool, rejections, sheets,
               state as _state, whisper, youtube_dl as ytdl)
from .proxy_pool import CookiesNeededError

log = logging.getLogger("gateway")


# ---------------------------------------------------------------------------
# Tuning constants (overridable via onb["phase1_v2_*"] keys)
# ---------------------------------------------------------------------------

DEFAULT_TARGET_COURSES = 5          # what wizard passes as `count`
DEFAULT_MAX_ITERATIONS = 3          # discovery retries with exclusion list
DEFAULT_CANDIDATES_PER_ITER = 30    # how many channels Claude returns per iteration
DEFAULT_MIN_SUBSCRIBERS = 500
DEFAULT_MAX_SUBSCRIBERS = 70_000
DEFAULT_MIN_DURATION_SEC = 360      # 6 minutes
DEFAULT_MAX_DURATION_SEC = 1500     # 25 minutes
DEFAULT_MIN_YEAR = 2016
DEFAULT_MIN_VIDEOS_PER_COURSE = 6
DEFAULT_MAX_VIDEOS_PER_CHANNEL = 80
DEFAULT_PROXY_RETRIES = 3

CLAUDE_DISCOVERY_TIMEOUT_SEC = 1500  # 25 min — Opus + WebSearch can take ~5-10 min
CLAUDE_SELECT_TIMEOUT_SEC = 600      # 10 min — short prompt, no WebSearch

# Model + effort for Step 1 (semantic discovery via WebSearch)
DISCOVERY_MODEL = "opus"
DISCOVERY_EFFORT = "high"
# Model + effort for Step 4 (pick from list — no WebSearch needed)
SELECT_MODEL = "sonnet"
SELECT_EFFORT = "medium"


def _cfg_int(onb: dict, key: str, default: int) -> int:
    try:
        v = int(onb.get(key) if onb.get(key) is not None else default)
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Claude CLI wrapper (separate from llm._call_json so we can pass --effort
# without changing the shared helper used by Whisper/compose etc.)
# ---------------------------------------------------------------------------

def _claude_call(prompt: str, *, model: str, effort: str,
                 timeout: int) -> str:
    """Invoke `claude -p` with the given prompt + effort. Returns stdout text.

    Same subprocess flags as llm._call_json but adds --effort.
    Raises RuntimeError on non-zero exit / empty stdout.
    """
    with tempfile.TemporaryDirectory(prefix="onboarder-claude-v2-") as tmpdir:
        env = os.environ.copy()
        env.setdefault(
            "PATH",
            f"{Path.home()}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
        )
        try:
            r = subprocess.run(
                [
                    "claude", "-p",
                    "--model", model,
                    "--effort", effort,
                    "--output-format", "text",
                    "--permission-mode", "bypassPermissions",
                ],
                input=prompt,
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"claude CLI timed out after {timeout}s (model={model} effort={effort})"
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "claude CLI not found in PATH. Ensure Claude Code is installed "
                "and CLAUDE_CODE_OAUTH_TOKEN is set in the gateway's env."
            ) from e
    if r.returncode != 0:
        raise RuntimeError(
            f"claude CLI exit {r.returncode}: stderr={(r.stderr or '')[:300]}"
        )
    text = (r.stdout or "").strip()
    if not text:
        raise RuntimeError(
            f"claude CLI returned empty stdout. stderr={(r.stderr or '')[:300]!r}"
        )
    return text


def _extract_json(text: str) -> Any:
    """Strip markdown fences, find first JSON object/array, tolerate trailing prose."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    else:
        text = text.strip()
        if not (text.startswith("{") or text.startswith("[")):
            idx = min(
                (i for i in (text.find("{"), text.find("[")) if i >= 0),
                default=-1,
            )
            if idx >= 0:
                text = text[idx:]
    obj, _ = json.JSONDecoder().raw_decode(text)
    return obj


# ---------------------------------------------------------------------------
# Step 1 — Claude discovers candidate channel handles
# ---------------------------------------------------------------------------

_DISCOVERY_PROMPT_TMPL = """You are a YouTube niche channel scout.

=== TOPIC ===
{topic}

=== AUDIENCE ===
{audience}

{pain_block}

=== YOUR TASK ===
Find {count} candidate YouTube channels that COULD work for a self-help course on this topic.

You ONLY need to return CHANNEL HANDLES or channel URLs. You do NOT need to verify size, videos, or details — a downstream pipeline will validate everything.

CRITICAL: We want SMALL niche channels (target: {min_subs}-{max_subs} subscribers). NOT mainstream.
- Skip Bob and Brad, Jeremy Ethier, Athlean-X, Yoga With Adriene, anyone you know from general wellness knowledge
- Skip channels appearing on 'best YouTube channel' listicles
- Skip commercial brands with aggressive sales funnels
- Target solo practitioners, niche clinicians, hidden gems

We will VERIFY actual subscriber count after you return — don't worry if you can't estimate exactly. Just give us your best {count} guesses based on search.

{exclusion_block}{rejection_block}

OUTPUT FORMAT (strict JSON, nothing else):
{{
  "candidates": [
    {{"channel_name": "...", "handle_or_url": "@handle or /channel/UC..."}}
  ],
  "queries_tried": ["..."],
  "channels_rejected_as_mainstream": ["name (reason)"],
  "notes": "anything relevant about confidence"
}}

Rules:
- Return up to {count} candidates. Fewer is fine if you can't find that many small ones.
- handle_or_url must be a real YouTube identifier you saw in search results
- Do NOT fabricate URLs
- Use WebSearch aggressively — try many query angles
- Do NOT propose any channel from the EXCLUSION list above (if any)

Return ONLY the JSON.
"""


def _build_pain_block(pain: str) -> str:
    if not pain:
        return ""
    return f"=== PAIN POINTS ===\n{pain}\n\n"


def _build_exclusion_block(names: list[str], max_names: int = 200) -> str:
    if not names:
        return ""
    take = names[:max_names]
    lines = "\n".join(f"- {n}" for n in take)
    return (
        "=== EXCLUSION LIST — DO NOT PROPOSE THESE ===\n"
        "We have already tried (or already use) these channels. Find DIFFERENT ones.\n"
        f"{lines}\n\n"
    )


def _build_rejection_block(rej_records: list[dict]) -> str:
    """F4 rejection feedback — tell Claude to avoid topically-similar courses."""
    if not rej_records:
        return ""
    lines = ["=== PREVIOUSLY REJECTED COURSES (avoid similar) ==="]
    # Dedup by (channel, title), take last 20 meaningful
    seen: set = set()
    pool: list[dict] = []
    for r in reversed(rej_records):
        ch = (r.get("channel") or "").strip()
        ct = (r.get("course_title") or "").strip()
        reason = (r.get("reason") or "").strip()
        if not (ch or ct):
            continue  # skip empty entries (known F3 bug)
        key = (ch.lower(), ct.lower())
        if key in seen:
            continue
        seen.add(key)
        pool.append(r)
        if len(pool) >= 20:
            break
    if not pool:
        return ""
    for r in pool:
        ch = r.get("channel", "")
        ct = r.get("course_title", "")
        reason = r.get("reason", "")
        bits = []
        if ch:
            bits.append(f"channel «{ch}»")
        if ct:
            bits.append(f"course «{ct}»")
        head = " / ".join(bits) or "(unknown)"
        if reason:
            lines.append(f'- {head}: "{reason[:200]}"')
        else:
            lines.append(f"- {head}")
    lines.append("Treat as strong negative signal — avoid channels with similar angle/style.\n")
    return "\n".join(lines) + "\n"


def _step1_discover(topic: str, audience: str, pain: str,
                    count: int, exclude_names: list[str],
                    rej_records: list[dict],
                    min_subs: int, max_subs: int) -> dict:
    """Single Claude discovery call. Returns parsed JSON with .candidates list."""
    prompt = _DISCOVERY_PROMPT_TMPL.format(
        topic=topic,
        audience=audience or "(not specified — infer from topic)",
        pain_block=_build_pain_block(pain),
        count=count,
        min_subs=min_subs,
        max_subs=max_subs,
        exclusion_block=_build_exclusion_block(exclude_names),
        rejection_block=_build_rejection_block(rej_records),
    )
    raw = _claude_call(
        prompt,
        model=DISCOVERY_MODEL,
        effort=DISCOVERY_EFFORT,
        timeout=CLAUDE_DISCOVERY_TIMEOUT_SEC,
    )
    try:
        return _extract_json(raw)
    except Exception as e:
        log.error(f"phase1_v2: discovery JSON parse failed: {e}; raw[:300]={raw[:300]!r}")
        raise RuntimeError(f"discovery LLM returned non-JSON: {e}")


# ---------------------------------------------------------------------------
# Step 2 — yt-dlp size filter (with retry × N)
# ---------------------------------------------------------------------------

def _normalize_channel_url(handle_or_url: str) -> str:
    s = (handle_or_url or "").strip()
    if not s:
        return ""
    if s.startswith("http"):
        return s
    if s.startswith("@"):
        return f"https://www.youtube.com/{s}"
    if s.startswith("/channel/") or s.startswith("channel/"):
        return f"https://www.youtube.com/{s.lstrip('/')}"
    if re.match(r"^UC[A-Za-z0-9_-]{22}$", s):
        return f"https://www.youtube.com/channel/{s}"
    if s.startswith("/c/") or s.startswith("c/"):
        return f"https://www.youtube.com/{s.lstrip('/')}"
    return f"https://www.youtube.com/@{s.lstrip('@')}"


def _ytdlp_channel_info(url: str, max_retries: int) -> dict | None:
    """Get channel metadata via existing ytdl helper, with retries on transient errors.

    yt-dlp uses the proxy pool configured globally (ydl_opts). Failures are
    typically 502 Bad Gateway from a single proxy — yt-dlp already handles
    intra-call retries; our retry loop is a backstop for cross-call resilience.
    """
    last_err: str = ""
    for attempt in range(max_retries):
        try:
            meta = ytdl.get_channel_metadata(url)
            if meta and meta.get("channel_id"):
                return meta
            last_err = "empty metadata"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        # Small backoff between attempts to give yt-dlp's internal proxy
        # rotation a chance to land on a working IP.
        if attempt + 1 < max_retries:
            time.sleep(2)
    log.warning(f"phase1_v2: channel_info failed for {url} after {max_retries} attempts: {last_err}")
    return None


def _step2_filter_size(candidates: list[dict],
                       min_subs: int, max_subs: int,
                       seen_channel_ids: set[str],
                       proxy_retries: int,
                       on_progress: Any) -> tuple[list[dict], dict[str, int]]:
    """Apply size filter + Sheet dedup. Returns (kept, stats)."""
    kept: list[dict] = []
    stats = {"too_big": 0, "too_small": 0, "already_in_sheet": 0, "errored": 0}
    for c in candidates:
        name = (c.get("channel_name") or "?").strip()
        url = _normalize_channel_url(c.get("handle_or_url", ""))
        if not url:
            continue
        meta = _ytdlp_channel_info(url, max_retries=proxy_retries)
        if not meta or not meta.get("channel_id"):
            stats["errored"] += 1
            on_progress(f"❓ <i>{name[:40]}</i> — нет ответа (3× прокси)")
            continue
        cid = meta["channel_id"]
        fc = int(meta.get("subscribers") or 0)
        cname = meta.get("channel_name") or name
        if cid in seen_channel_ids:
            stats["already_in_sheet"] += 1
            on_progress(f"♻️ {cname[:40]} (уже в Sheet)")
            continue
        if fc < min_subs:
            stats["too_small"] += 1
            on_progress(f"⤵️ {cname[:40]} — {fc:,} subs (мало)")
            continue
        if fc > max_subs:
            stats["too_big"] += 1
            on_progress(f"⤴️ {cname[:40]} — {fc:,} subs (много)")
            continue
        kept.append({
            "channel_id": cid,
            "channel_name": cname,
            "channel_name_claude": name,
            "subscribers": fc,
            "description": meta.get("description", ""),
            "url": meta.get("channel_url") or url,
        })
        on_progress(f"✅ {cname[:40]} — {fc:,} subs")
    return kept, stats


# ---------------------------------------------------------------------------
# Step 3 — yt-dlp /videos enumeration + filter
# ---------------------------------------------------------------------------

def _v2_list_videos_full(channel_id: str, max_results: int) -> list[dict]:
    """List channel videos with FULL per-entry metadata (duration, upload_date).

    Why this exists: youtube_dl.list_channel_videos uses
    `extract_flat: "in_playlist"` which returns only id+title — duration is
    None/0 in flat mode for most YouTube responses. Our v2 step 3 NEEDS
    duration to apply the 6-25 min filter; without it every video fails as
    "too short" (duration=0 < 360s).

    Use ytdl._ydl() which defaults to extract_flat=False (full per-entry
    extraction). Slower than flat mode (~1-2 sec per video × N = ~2-3 min
    for 80 videos) but it's the only way to get duration without paying
    for YouTube Data API quota.

    Returns the same shape as ytdl.list_channel_videos.
    """
    url = ytdl._channel_videos_url(channel_id)
    # _ydl() default already has extract_flat=False; we just cap entries.
    # ignoreerrors lets us continue past unavailable/private videos in the list.
    try:
        with ytdl._ydl({"playlistend": max(1, max_results),
                        "ignoreerrors": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        log.warning(f"phase1_v2: list_videos_full failed for {url}: {e}")
        return []
    entries = (info or {}).get("entries") or []
    out: list[dict] = []
    for e in entries:
        if not e:
            continue
        vid = e.get("id") or ""
        if not vid:
            continue
        out.append({
            "video_id": vid,
            "title": e.get("title") or "",
            "duration_sec": int(e.get("duration") or 0),
            "view_count": int(e.get("view_count") or 0),
            "upload_date": e.get("upload_date") or "",
            "url": e.get("webpage_url") or f"https://youtu.be/{vid}",
            "description": (e.get("description") or "")[:500],
        })
    return out


def _step3_enumerate(channels: list[dict],
                     min_dur_sec: int, max_dur_sec: int,
                     min_year: int, max_videos: int,
                     seen_video_ids: set[str],
                     on_progress: Any) -> list[dict]:
    """For each channel, pull recent videos + apply hard filters.

    Returns the channels list enriched with a `videos: list[dict]` field
    containing video_id, title, duration_sec, upload_year, view_count.
    Empty videos list = no valid content — that channel will fail step 4.
    """
    out: list[dict] = []
    for ch in channels:
        try:
            # Use the v2-local full-metadata helper. ytdl.list_channel_videos
            # uses flat-playlist mode which strips durations to 0, making the
            # 6-25 min filter reject everything. _v2_list_videos_full does
            # per-entry extraction so duration is preserved.
            raw_videos = _v2_list_videos_full(
                ch["channel_id"], max_results=max_videos,
            )
        except Exception as e:
            log.warning(f"phase1_v2: list videos failed for {ch['channel_name']!r}: {e}")
            raw_videos = []
        valid: list[dict] = []
        for v in raw_videos:
            vid = v.get("video_id")
            if not vid or vid in seen_video_ids:
                continue
            dur = int(v.get("duration_sec") or 0)
            if dur < min_dur_sec or dur > max_dur_sec:
                continue
            date_str = (v.get("upload_date") or "")
            try:
                year = int(date_str[:4]) if date_str[:4].isdigit() else 0
            except (ValueError, IndexError):
                year = 0
            if year < min_year:
                continue
            valid.append({
                "video_id": vid,
                "title": v.get("title") or "",
                "duration_sec": dur,
                "upload_year": year,
                "view_count": int(v.get("view_count") or 0),
                "url": v.get("url") or f"https://youtu.be/{vid}",
                "upload_date": date_str,
            })
        out.append({**ch, "videos": valid, "total_videos_seen": len(raw_videos)})
        on_progress(
            f"📹 {ch['channel_name'][:35]} — {len(valid)}/{len(raw_videos)} видео "
            f"подходят (6-25min, ≥2016)"
        )
    return out


# ---------------------------------------------------------------------------
# Step 4 — Claude picks 12-15 best videos PER channel from real list
# ---------------------------------------------------------------------------

_SELECT_PROMPT_TMPL = """You are a curriculum designer building a course on:

TOPIC: {topic}
AUDIENCE: {audience}

Below is the FULL list of valid videos from one YouTube channel (already filtered:
{min_dur}-{max_dur} min, {min_year}+, public, not in our Sheet yet).
Your job: pick the BEST videos for a coherent learning progression. Order them
pedagogically (foundations → core → progression).

CHANNEL: {channel_name} ({subscribers} subs)

VIDEOS (ground truth from yt-dlp — these are real, no fabrication needed):
{video_list}

TARGET COURSE SIZE:
- IDEAL: 12-15 videos
- ACCEPTABLE: {min_videos}-30 videos
- MINIMUM: {min_videos} videos. If you cannot find at least {min_videos} topically-relevant videos,
  this channel is NOT a good fit — return an empty selected_video_ids list with
  confidence='low' and a clear note. Don't pretend a course exists with 3-5 stretched videos.

When in doubt about a video's fit, INCLUDE it if it's adjacent to the topic — we'd
rather have a 12-video course with some loose fits than a 5-video course of perfect
fits. The minimum {min_videos} is a HARD rule.

OUTPUT FORMAT (strict JSON, nothing else):
{{
  "course_title_en": "...",
  "course_rationale": "1-2 sentences why this set works",
  "selected_video_ids": ["video_id_1", "video_id_2", ...],
  "ordering_logic": "brief explanation of progression",
  "confidence": "high|medium|low",
  "notes": "anything to flag"
}}

Rules:
- selected_video_ids must be 0 or {min_videos}-30 ids — NEVER 1-{min_minus1}
- DO NOT invent ids — pick only from the list above
- If channel does not have {min_videos} topically-fitting videos → return [] with confidence='low' and notes

Return ONLY the JSON.
"""


def _step4_select(channels: list[dict], topic: str, audience: str,
                  min_dur_sec: int, max_dur_sec: int, min_year: int,
                  min_videos: int, on_progress: Any) -> list[dict]:
    """For each channel with ≥ min_videos valid videos, ask Claude to select.

    Returns list of "course candidates" with selected video_ids + course_title.
    Drops anything Claude rejects or returns < min_videos.
    """
    courses: list[dict] = []
    min_dur_min = min_dur_sec // 60
    max_dur_min = max_dur_sec // 60
    for ch in channels:
        videos = ch.get("videos") or []
        if len(videos) < min_videos:
            log.info(
                f"phase1_v2: skip @{ch['channel_name']}: only {len(videos)} valid "
                f"videos (< {min_videos} minimum)"
            )
            continue
        video_list_str = "\n".join(
            f"- id={v['video_id']} dur={v['duration_sec']}s year={v['upload_year']} "
            f"views={v['view_count']} | {v['title'][:80]}"
            for v in videos
        )
        prompt = _SELECT_PROMPT_TMPL.format(
            topic=topic,
            audience=audience or "(not specified — infer from topic)",
            min_dur=min_dur_min,
            max_dur=max_dur_min,
            min_year=min_year,
            channel_name=ch["channel_name"],
            subscribers=ch["subscribers"],
            video_list=video_list_str,
            min_videos=min_videos,
            min_minus1=min_videos - 1,
        )
        try:
            raw = _claude_call(
                prompt,
                model=SELECT_MODEL,
                effort=SELECT_EFFORT,
                timeout=CLAUDE_SELECT_TIMEOUT_SEC,
            )
            data = _extract_json(raw)
        except Exception as e:
            log.warning(f"phase1_v2: select failed for {ch['channel_name']!r}: {e}")
            on_progress(
                f"⚠️ {ch['channel_name'][:35]} — select упал ({str(e)[:60]})"
            )
            continue
        raw_ids = data.get("selected_video_ids") or []
        valid_id_set = {v["video_id"] for v in videos}
        seen_now: set = set()
        confirmed: list[str] = []
        for vid in raw_ids:
            if vid in valid_id_set and vid not in seen_now:
                confirmed.append(vid)
                seen_now.add(vid)
        if len(confirmed) < min_videos:
            on_progress(
                f"⤵️ {ch['channel_name'][:35]} — отобрано {len(confirmed)}/{min_videos} "
                f"({data.get('confidence', '?')})"
            )
            continue
        # Build the rich selected list with title/duration for downstream enrich.
        videos_by_id = {v["video_id"]: v for v in videos}
        selected_full = [videos_by_id[vid] for vid in confirmed if vid in videos_by_id]
        courses.append({
            **ch,
            "selected_videos": selected_full,
            "course_title": data.get("course_title_en", "") or "Untitled Course",
            "rationale": data.get("course_rationale", ""),
            "claude_confidence": data.get("confidence", "?"),
        })
        on_progress(
            f"✅ {ch['channel_name'][:35]} — {len(confirmed)} видео "
            f"({data.get('confidence', '?')}): "
            f"{(data.get('course_title_en') or '')[:50]}"
        )
    return courses


# ---------------------------------------------------------------------------
# Per-course write helper (mirrors lines 524-713 of old phase1_discovery._run)
# ---------------------------------------------------------------------------

# Imported lazily from phase1_discovery to avoid circular import — see _run_v2.
def _send(*args, **kwargs):  # type: ignore[no-redef]
    from . import phase1_discovery as _p1
    _p1._send(*args, **kwargs)


def _send_noise(*args, **kwargs):  # type: ignore[no-redef]
    from . import phase1_discovery as _p1
    _p1._send_noise(*args, **kwargs)


def _send_per_course_phase2_button(*args, **kwargs):  # type: ignore[no-redef]
    from . import phase1_discovery as _p1
    _p1._send_per_course_phase2_button(*args, **kwargs)


def _html_escape(s: str) -> str:
    from . import phase1_discovery as _p1
    return _p1._html_escape(s)


def _process_one_course(*, course_idx: int, run_id: str, ch_record: dict,
                        topic: str, pain: str, audience: str,
                        target_count: int,
                        client: Any, sheet_id: str,
                        active_video_ids: set[str], blocked_channel_ids: set[str],
                        openai_key: str, youtube_cookies_file: str | None,
                        rotator: Any, parallel_per_course: int,
                        compose_model: str, mark_cuts_word_level: bool,
                        streaming_enabled: bool,
                        token: str, chat_id: int) -> tuple[bool, str | None, int]:
    """Run enrich for ONE course (one channel + its Claude-selected videos),
    write Sheet rows, send per-course Phase 2 button.

    Returns (success, skip_reason, videos_written).

    Mirrors old phase1_discovery._run lines 524-713 but takes a pre-selected
    channel + lesson list as input (instead of running select_videos here).
    """
    ch_name = ch_record["channel_name"]
    ch_id = ch_record["channel_id"]
    # Build the lesson list in the shape phase1_enrich.enrich_course expects:
    # [{"video_id", "title", ...}]
    selected_full = ch_record["selected_videos"]
    # Late-dedup: drop any video already done/processing in any run
    dedup_lessons: list[dict] = []
    course_skipped = 0
    for v in selected_full:
        vid = v.get("video_id")
        if vid and vid in active_video_ids:
            course_skipped += 1
            log.info(f"phase1_v2 dedup skip {vid} in {ch_name}")
            continue
        dedup_lessons.append(v)
    if not dedup_lessons:
        _send_noise(
            token, chat_id,
            f"⚠️ Канал «{_html_escape(ch_name)}» пропущен — все видео уже обрабатывались.",
        )
        return False, f"{ch_name}: all videos were duplicates", 0
    if len(dedup_lessons) < 5:
        _send_noise(
            token, chat_id,
            f"⚠️ Канал «{_html_escape(ch_name)}» — после дедупа осталось "
            f"{len(dedup_lessons)} видео (< 5). Пропускаю.",
        )
        return False, f"{ch_name}: {len(dedup_lessons)}/5 unique after dedup", 0

    # Build videos_metadata for enrich (it expects {video_id: full_dict}).
    videos_metadata = {v["video_id"]: v for v in selected_full}
    # Convert lesson dicts into the shape enrich_course expects.
    # CRITICAL: `order` is 1-based lesson position. enrich_course uses it as
    # lesson_idx in the Sheet rows. Without it the curriculum reverts to
    # arbitrary ordering.
    enrich_lessons = [
        {
            "video_id": v["video_id"],
            "title": v.get("title", ""),
            "order": i + 1,
            "reason": "",
            "duration_sec": v.get("duration_sec", 0),
            "url": v.get("url", ""),
        }
        for i, v in enumerate(dedup_lessons)
    ]

    streaming_row_map: dict[int, int] = {}
    if streaming_enabled:
        streaming_row_map = phase1_enrich.pre_allocate_for_streaming(
            client, sheet_id, run_id=run_id, course_idx=course_idx,
            channel_id=ch_id, channel_name=ch_name,
            selected_videos=enrich_lessons,
            videos_metadata=videos_metadata,
        )

    try:
        enriched = phase1_enrich.enrich_course(
            course_idx=course_idx, run_id=run_id,
            channel_id=ch_id,
            channel_name=ch_name,
            channel_description=ch_record.get("description", ""),
            course_topic_input=topic,
            course_title_from_llm=ch_record.get("course_title", ""),
            selected_videos=enrich_lessons,
            videos_metadata=videos_metadata,
            openai_key=openai_key,
            cookies_file=youtube_cookies_file,
            rotator=rotator,
            on_progress=lambda msg: _send(token, chat_id, _html_escape(msg)),
            on_progress_noise=lambda msg: _send_noise(token, chat_id, _html_escape(msg)),
            max_parallel=parallel_per_course,
            compose_model=compose_model,
            pain=pain, audience=audience,
            sheets_client=(client if streaming_enabled else None),
            sheet_id=(sheet_id if streaming_enabled else None),
            lesson_row_map=(streaming_row_map if streaming_enabled else None),
            streaming_describe=streaming_enabled,
            mark_cuts_word_level=mark_cuts_word_level,
        )
    except CookiesNeededError:
        raise
    except Exception as e:
        log.warning(f"phase1_v2: enrich failed for course {course_idx}: {e}", exc_info=True)
        _send(token, chat_id,
              f"⚠️ Курс {course_idx} ({_html_escape(ch_name)}): обогащение упало "
              f"(<code>{_html_escape(str(e))[:160]}</code>). Пропускаю.")
        return False, f"{ch_name}: enrich error: {str(e)[:80]}", 0

    if not enriched.get("videos"):
        _send(token, chat_id,
              f"⚠️ Курс {course_idx}: ни одно видео не довелось до конца. Пропускаю.")
        return False, f"{ch_name}: no videos survived enrichment", 0

    # Build course rows for the Sheet (identical shape to old _run).
    final_course_title = enriched.get("course_title") or ch_record.get("course_title", "")
    full_course_title = f"Курс {course_idx}: {ch_name} — {final_course_title}"

    course_desc = enriched.get("course_description", "")
    course_tagline = enriched.get("course_tagline", "")
    course_what = enriched.get("course_what_you_learn", "")
    course_target = enriched.get("course_target_audience", "")
    author_name = enriched.get("author_name", "") or ch_name
    author_bio = enriched.get("author_bio", "")
    author_expertise = enriched.get("author_expertise", "")
    course_about = enriched.get("course_about", "")
    course_plan = enriched.get("course_plan", "")
    course_science = enriched.get("course_science", "")
    course_desc_ru = enriched.get("course_description_ru", "")
    course_tagline_ru = enriched.get("course_tagline_ru", "")
    course_what_ru = enriched.get("course_what_you_learn_ru", "")
    course_target_ru = enriched.get("course_target_audience_ru", "")
    author_bio_ru = enriched.get("author_bio_ru", "")
    author_expertise_ru = enriched.get("author_expertise_ru", "")

    if streaming_enabled and streaming_row_map:
        phase1_enrich.finalize_streaming_row1(
            client, sheet_id,
            lesson_row_map=streaming_row_map,
            enriched=enriched,
            full_course_title=full_course_title,
            final_course_title=final_course_title,
            author_name_fallback=ch_name,
        )
        written_count = len(streaming_row_map)
        _send_per_course_phase2_button(
            token, chat_id, sheets.sheet_url(sheet_id),
            run_id=run_id, course_idx=course_idx, count=target_count,
            course_title=final_course_title, video_count=written_count,
            channel_name=ch_name,
        )
        return True, None, written_count

    # Non-streaming: batch-write all rows then send button.
    course_rows: list[dict] = []
    for v in enriched["videos"]:
        is_first = (v["lesson_idx"] == 1)
        course_rows.append({
            "course": full_course_title if is_first else f"Курс {course_idx}",
            "lesson_idx": v["lesson_idx"],
            "channel": ch_name,
            "lesson_title": v["title"],
            "url": v["url"],
            "duration_sec": v.get("duration_sec", 0),
            "video_id": v["video_id"],
            "channel_id": ch_id,
            "course_idx": course_idx,
            "lesson_description": v.get("lesson_description", ""),
            "lesson_description_ru": v.get("lesson_description_ru", ""),
            "transcript_excerpt": v.get("transcript_excerpt", ""),
            "course_description": course_desc if is_first else "",
            "course_tagline": course_tagline if is_first else "",
            "course_what_you_learn": course_what if is_first else "",
            "course_target_audience": course_target if is_first else "",
            "author_name": author_name if is_first else "",
            "author_bio": author_bio if is_first else "",
            "author_expertise": author_expertise if is_first else "",
            "course_title": (final_course_title if is_first else ""),
            "course_about": course_about if is_first else "",
            "course_plan": course_plan if is_first else "",
            "course_science": course_science if is_first else "",
            "course_description_ru": course_desc_ru if is_first else "",
            "course_tagline_ru": course_tagline_ru if is_first else "",
            "course_what_you_learn_ru": course_what_ru if is_first else "",
            "course_target_audience_ru": course_target_ru if is_first else "",
            "author_bio_ru": author_bio_ru if is_first else "",
            "author_expertise_ru": author_expertise_ru if is_first else "",
        })

    with sheets.sheet_lock():
        latest_seen = sheets.get_active_video_ids(client, sheet_id)
        latest_blocked = sheets.get_seen_channel_ids(client, sheet_id)
        rows_to_write = [
            r for r in course_rows
            if r.get("video_id") not in latest_seen
            and r.get("channel_id") not in latest_blocked
        ]
        late_skipped = len(course_rows) - len(rows_to_write)
        if rows_to_write:
            sheets.append_lesson_rows(client, sheet_id, run_id=run_id, rows=rows_to_write)
    if late_skipped:
        log.info(f"phase1_v2: late dedup dropped {late_skipped} rows for course {course_idx}")

    if not rows_to_write:
        _send(token, chat_id,
              f"⚠️ Курс {course_idx} ({_html_escape(ch_name)}) пропущен — "
              f"все {len(course_rows)} видео уже в Sheet (поздний дубликат).")
        return False, f"{ch_name}: late dedup", 0

    _send_per_course_phase2_button(
        token, chat_id, sheets.sheet_url(sheet_id),
        run_id=run_id, course_idx=course_idx, count=target_count,
        course_title=final_course_title, video_count=len(rows_to_write),
        channel_name=ch_name,
    )
    return True, None, len(rows_to_write)


# ---------------------------------------------------------------------------
# Main entry — _run_v2 (same signature as _run)
# ---------------------------------------------------------------------------

def _run_v2(token: str, agent: str, cfg: dict, chat_id: int, user_id: int,
            topic: str, count: int, onb: dict,
            *, pain: str = "", audience: str = "",
            thread_id: int = 0,
            batch_run_id: str | None = None,
            batch_course_idx_offset: int = 0,
            batch_silent_finish: bool = False) -> tuple[int, int]:
    """v2 Phase 1: Claude-driven discovery + yt-dlp validation.

    Same contract as old `_run` — returns `(successful_courses, slots_used)`.
    For v2, slots_used == successful_courses (no "burnt" slots — F6 contiguous
    by design since we never increment course_idx on failure).
    """
    # ── 1. Resolve secrets and open Sheet (mirrors old _run lines 161-194)
    sa_path = _secrets.resolve_path(onb, "google_service_account")
    sheet_id = onb.get("google_sheet_id") or ""
    if not sheet_id:
        raise RuntimeError("config: onboarder.google_sheet_id not set")

    whisper_provider = (onb.get("whisper_provider") or "openai").lower()
    whisper.set_default_provider(whisper_provider)
    if whisper_provider == "groq":
        openai_key = _secrets.resolve(onb, "groq_api_key", env="GROQ_API_KEY")
    else:
        openai_key = _secrets.resolve(onb, "openai_api_key", env="OPENAI_API_KEY")
    youtube_cookies_file = onb.get("youtube_cookies_file") or None
    proxy_pool_list = proxy_pool.normalise_pool(
        onb.get("youtube_proxies") or onb.get("youtube_proxy")
    )
    parallel_per_course = int(onb.get("phase1_parallel_per_course") or 4)
    compose_model = (onb.get("models") or {}).get("compose") or llm.DEFAULT_MODEL_QUALITY
    streaming_enabled = bool(onb.get("streaming_sheet_writes", False))
    mark_cuts_word_level = bool(onb.get("mark_cuts_word_level", False))

    # v2-specific knobs (overridable from config)
    target_courses = max(1, int(count) or DEFAULT_TARGET_COURSES)
    max_iterations = _cfg_int(onb, "phase1_v2_max_iterations", DEFAULT_MAX_ITERATIONS)
    candidates_per_iter = _cfg_int(onb, "phase1_v2_candidates_per_iter",
                                   DEFAULT_CANDIDATES_PER_ITER)
    min_subs = _cfg_int(onb, "phase1_v2_min_subscribers", DEFAULT_MIN_SUBSCRIBERS)
    max_subs = _cfg_int(onb, "phase1_v2_max_subscribers", DEFAULT_MAX_SUBSCRIBERS)
    min_dur_sec = _cfg_int(onb, "phase1_v2_min_duration_sec", DEFAULT_MIN_DURATION_SEC)
    max_dur_sec = _cfg_int(onb, "phase1_v2_max_duration_sec", DEFAULT_MAX_DURATION_SEC)
    min_year = _cfg_int(onb, "phase1_v2_min_year", DEFAULT_MIN_YEAR)
    min_videos_per_course = _cfg_int(onb, "phase1_v2_min_videos_per_course",
                                     DEFAULT_MIN_VIDEOS_PER_COURSE)
    max_videos_per_channel = _cfg_int(onb, "phase1_v2_max_videos_per_channel",
                                      DEFAULT_MAX_VIDEOS_PER_CHANNEL)
    proxy_retries = _cfg_int(onb, "phase1_v2_proxy_retries", DEFAULT_PROXY_RETRIES)

    client = sheets.open_client(sa_path)

    # ── 2. Dedup state from Sheet (channels + videos already used)
    sheets.ensure_lessons_tab(client, sheet_id)
    active_video_ids = sheets.get_active_video_ids(client, sheet_id)
    blocked_channel_ids = sheets.get_seen_channel_ids(client, sheet_id)
    log.info(
        f"phase1_v2[{user_id}] {len(active_video_ids)} videos and "
        f"{len(blocked_channel_ids)} channels previously seen in Sheet"
    )

    # Collect channel NAMES from Sheet to seed exclusion list for Claude.
    ws = sheets.ensure_lessons_tab(client, sheet_id)
    rows = ws.get_all_values()
    header = rows[0] if rows else []
    try:
        idx_ch = header.index("channel")
    except ValueError:
        idx_ch = -1
    seen_channel_names: set[str] = set()
    if idx_ch >= 0:
        for row in rows[1:]:
            if len(row) > idx_ch:
                name = (row[idx_ch] or "").strip()
                if name:
                    seen_channel_names.add(name)

    # ── 3. F4 rejection feedback
    rejection_log = rejections.load_recent_rejections(n=20)
    if rejection_log:
        log.info(
            f"phase1_v2[{user_id}] loaded {len(rejection_log)} rejections for LLM"
        )

    run_id = batch_run_id or sheets.make_run_id()
    _state.update(
        agent, user_id, thread_id=int(thread_id or 0),
        run_id=run_id, step="phase1_running",
        topic=topic, count=count, pain=pain, audience=audience,
        chat_id=chat_id, sheet_url=sheets.sheet_url(sheet_id),
    )

    # ── 4. Init proxy rotator (same UX as old _run)
    rotator = proxy_pool.ProxyRotator(proxy_pool_list, cookies_file=youtube_cookies_file)
    if proxy_pool_list:
        _send_noise(token, chat_id,
                    f"🔍 Проверяю {len(proxy_pool_list)} прокси на YouTube…")
        last_progress = [time.time()]
        probe_results: list[str] = []

        def _on_probe(idx: int, total: int, name: str, ok: bool) -> None:
            probe_results.append(f"{'✅' if ok else '❌'} {idx}/{total} {name}")
            now = time.time()
            if ok or now - last_progress[0] > 15 or idx == total:
                _send_noise(token, chat_id, "\n".join(probe_results[-12:]))
                last_progress[0] = now

        try:
            rotator.init(on_progress=_on_probe)
            _send_noise(
                token, chat_id,
                f"✅ Прокси готов: <code>{proxy_pool._proxy_label(rotator.current)}</code>",
            )
        except proxy_pool.NoWorkingProxyError as e:
            raise RuntimeError(
                f"Ни один прокси не прошёл проверку YouTube.\n\n{str(e)[:600]}\n\n"
                "Обнови cookies (/menu → 📎 Загрузить cookies) и попробуй ещё раз."
            )
    else:
        _send_noise(token, chat_id, "⚠️ Прокси не настроен — пробую напрямую с VPS-IP")

    # ── 5. Discovery loop ────────────────────────────────────────────────
    excluded_names: list[str] = sorted(seen_channel_names)
    selected_courses: list[dict] = []
    iteration_history: list[dict] = []
    skip_reasons: list[str] = []

    _send_noise(
        token, chat_id,
        f"🧭 <b>Phase 1 v2 — гибридный pipeline.</b>\n"
        f"Цель: <b>{target_courses}</b> курса(ов), до <b>{max_iterations}</b> "
        f"итераций discovery.\n"
        f"Фильтры: <code>{min_subs}-{max_subs} subs</code>, "
        f"<code>{min_dur_sec//60}-{max_dur_sec//60} мин</code>, "
        f"<code>≥{min_year}</code> год, "
        f"<code>≥{min_videos_per_course}</code> видео на курс.",
    )

    for iteration in range(1, max_iterations + 1):
        if len(selected_courses) >= target_courses:
            break

        _send_noise(
            token, chat_id,
            f"\n🔁 <b>Итерация {iteration}/{max_iterations}</b> "
            f"({len(selected_courses)}/{target_courses} курсов уже найдено).",
        )

        # ── Step 1: Claude discovery
        _send_noise(
            token, chat_id,
            f"🤖 <i>Claude ищет {candidates_per_iter} каналов "
            f"(исключая {len(excluded_names)} уже опробованных)…</i>",
        )
        try:
            disc = _step1_discover(
                topic=topic, audience=audience, pain=pain,
                count=candidates_per_iter,
                exclude_names=excluded_names,
                rej_records=rejection_log,
                min_subs=min_subs, max_subs=max_subs,
            )
        except Exception as e:
            log.warning(f"phase1_v2 iter {iteration}: discovery failed: {e}")
            _send_noise(token, chat_id,
                        f"⚠️ Discovery упал: <code>{_html_escape(str(e))[:200]}</code>")
            iteration_history.append({"iter": iteration, "discovery": "failed"})
            continue
        candidates = disc.get("candidates") or []
        if not candidates:
            _send_noise(token, chat_id,
                        "ℹ️ Claude не вернул кандидатов — пробую следующую итерацию.")
            iteration_history.append({"iter": iteration, "candidates": 0})
            continue

        _send_noise(
            token, chat_id,
            f"📋 Получено <b>{len(candidates)}</b> кандидатов от Claude.",
        )

        # Track every candidate name for next iteration's exclusion (incl. ones we'll
        # drop in step 2 — don't ask Claude for them again).
        for c in candidates:
            n = (c.get("channel_name") or "").strip()
            if n and n not in excluded_names:
                excluded_names.append(n)

        # ── Step 2: size + Sheet dedup
        _send_noise(token, chat_id,
                    f"📊 <i>Проверяю размер каналов через yt-dlp (retry ×{proxy_retries})…</i>")
        size_progress: list[str] = []

        def _on_size(line: str) -> None:
            size_progress.append(line)
            if len(size_progress) % 6 == 0:
                _send_noise(token, chat_id, "\n".join(size_progress[-6:]))

        surviving, size_stats = _step2_filter_size(
            candidates, min_subs=min_subs, max_subs=max_subs,
            seen_channel_ids=blocked_channel_ids,
            proxy_retries=proxy_retries,
            on_progress=_on_size,
        )
        if size_progress and len(size_progress) % 6 != 0:
            _send_noise(token, chat_id, "\n".join(size_progress[-6:]))
        _send_noise(
            token, chat_id,
            f"📊 Размер: KEEP={len(surviving)} / "
            f"too_big={size_stats['too_big']} / "
            f"too_small={size_stats['too_small']} / "
            f"in_sheet={size_stats['already_in_sheet']} / "
            f"err={size_stats['errored']}",
        )
        # Mark every probed channel_id as seen so next iteration doesn't re-probe.
        for ch in surviving:
            blocked_channel_ids.add(ch["channel_id"])

        if not surviving:
            iteration_history.append({"iter": iteration, "after_size": 0, **size_stats})
            continue

        # ── Step 3: video enumeration + filter
        _send_noise(token, chat_id,
                    f"📹 <i>Тяну видео ({max_videos_per_channel} последних) "
                    f"с {len(surviving)} каналов, фильтрую…</i>")
        enum_progress: list[str] = []

        def _on_enum(line: str) -> None:
            enum_progress.append(line)
            if len(enum_progress) % 4 == 0:
                _send_noise(token, chat_id, "\n".join(enum_progress[-4:]))

        enriched = _step3_enumerate(
            surviving,
            min_dur_sec=min_dur_sec, max_dur_sec=max_dur_sec, min_year=min_year,
            max_videos=max_videos_per_channel,
            seen_video_ids=active_video_ids,
            on_progress=_on_enum,
        )
        if enum_progress and len(enum_progress) % 4 != 0:
            _send_noise(token, chat_id, "\n".join(enum_progress[-4:]))

        # ── Step 4: Claude select
        _send_noise(token, chat_id,
                    f"🎯 <i>Claude выбирает лучшие видео для курсов…</i>")
        select_progress: list[str] = []

        def _on_select(line: str) -> None:
            select_progress.append(line)
            if len(select_progress) % 3 == 0:
                _send_noise(token, chat_id, "\n".join(select_progress[-3:]))

        iter_courses = _step4_select(
            enriched, topic=topic, audience=audience,
            min_dur_sec=min_dur_sec, max_dur_sec=max_dur_sec, min_year=min_year,
            min_videos=min_videos_per_course,
            on_progress=_on_select,
        )
        if select_progress and len(select_progress) % 3 != 0:
            _send_noise(token, chat_id, "\n".join(select_progress[-3:]))

        # Accumulate; trim to target if iteration overshoots.
        for c in iter_courses:
            if len(selected_courses) >= target_courses:
                break
            selected_courses.append(c)
            for v in c["selected_videos"]:
                active_video_ids.add(v["video_id"])

        iteration_history.append({
            "iter": iteration,
            "candidates": len(candidates),
            "after_size": len(surviving),
            "with_videos": sum(1 for e in enriched if e["videos"]),
            "courses_added": len(iter_courses),
            **size_stats,
        })

        if len(selected_courses) >= target_courses:
            _send_noise(
                token, chat_id,
                f"🎉 Достигли цели: <b>{len(selected_courses)}</b> курса(ов).",
            )
            break

    if not selected_courses:
        diag = "\n".join(
            f"  iter {h['iter']}: candidates={h.get('candidates', '?')}, "
            f"after_size={h.get('after_size', 0)}, "
            f"with_videos={h.get('with_videos', 0)}, "
            f"courses={h.get('courses_added', 0)}"
            for h in iteration_history
        ) or "  (no iterations completed)"
        raise RuntimeError(
            f"Ни один из проверенных каналов не дал валидной подборки уроков "
            f"за {len(iteration_history)} итераций.\n\n{diag}\n\n"
            "Попробуй другую тему, расширь критерии (subs/duration/year) "
            "или проверь что прокси работает."
        )

    _send_noise(
        token, chat_id,
        f"🧱 <b>Discovery завершён.</b> Курсов отобрано: "
        f"<b>{len(selected_courses)}</b>. Начинаю enrichment (Whisper + Claude)…",
    )

    # ── 6. Per-course enrich + Sheet write + Phase 2 button
    successful_courses = 0
    total_videos_written = 0
    course_summaries: list[str] = []
    for ch_record in selected_courses:
        if successful_courses >= target_courses:
            break
        tentative_idx = batch_course_idx_offset + successful_courses + 1
        _send(
            token, chat_id,
            f"🎬 <b>Курс {tentative_idx}</b> — канал «{_html_escape(ch_record['channel_name'])}» "
            f"({ch_record['subscribers']:,} subs): "
            f"<i>{_html_escape(ch_record.get('course_title', '?'))[:80]}</i>",
        )
        try:
            ok, skip_reason, written = _process_one_course(
                course_idx=tentative_idx, run_id=run_id,
                ch_record=ch_record,
                topic=topic, pain=pain, audience=audience,
                target_count=target_courses,
                client=client, sheet_id=sheet_id,
                active_video_ids=active_video_ids,
                blocked_channel_ids=blocked_channel_ids,
                openai_key=openai_key, youtube_cookies_file=youtube_cookies_file,
                rotator=rotator, parallel_per_course=parallel_per_course,
                compose_model=compose_model,
                mark_cuts_word_level=mark_cuts_word_level,
                streaming_enabled=streaming_enabled,
                token=token, chat_id=chat_id,
            )
        except CookiesNeededError:
            raise

        if ok:
            successful_courses += 1
            total_videos_written += written
            course_summaries.append(
                f"{tentative_idx}. {ch_record.get('course_title', '?')} ({written} уроков)"
            )
        elif skip_reason:
            skip_reasons.append(skip_reason)

    if successful_courses == 0:
        diag = ""
        if skip_reasons:
            diag = "\n\nПричины пропуска:\n" + "\n".join(f"  • {r}" for r in skip_reasons[:10])
        raise RuntimeError(
            "Discovery нашёл каналы, но ни один курс не прошёл enrichment. " + diag
        )

    slots_used = successful_courses  # v2: no burnt slots — course_idx is contiguous

    if batch_silent_finish:
        return (successful_courses, slots_used)

    _state.update(
        agent, user_id, thread_id=int(thread_id or 0),
        step="awaiting_approval",
        courses_summary=course_summaries,
    )
    _send(
        token, chat_id,
        f"🏁 <b>Phase 1 завершён.</b>\n\n"
        f"Курсов: <b>{successful_courses}</b>"
        + (f" (хотели {target_courses})" if successful_courses < target_courses else "")
        + f"\nВидео: <b>{total_videos_written}</b>\n"
        + f"Итераций discovery: <b>{len(iteration_history)}</b>\n\n"
        + "Каждый курс выше — со своей кнопкой "
        + "<b>«🚀 Запустить Курс N»</b>. Жми когда проверил Sheet.",
    )
    return (successful_courses, slots_used)
