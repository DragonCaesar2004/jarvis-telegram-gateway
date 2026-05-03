"""Claude LLM calls for the onboarder pipeline.

Four jobs:
    1. score_channels()   — rank candidate YouTube channels against criteria + topic
    2. select_videos()    — pick a coherent course of 6-12 videos from one channel's video list
    3. mark_cuts()        — given a Whisper transcript with word timestamps, return [(start_s, end_s, reason)]
                            for intro/outro/promo/off-topic segments to remove
    4. compose_course()   — final course title, excerpt, aboutContent + author bio

Implementation: subprocess to `claude -p` CLI (Claude Code). Uses the OAuth
token already configured for the gateway's Max subscription, so no separate
Anthropic API key is needed. Costs are charged against the Max quota.

Defaults to Sonnet (fast & cheap quota-wise) for classification, Opus for prose.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("gateway")

# CLI model aliases (claude -p --model <alias>)
DEFAULT_MODEL_FAST = "sonnet"
DEFAULT_MODEL_QUALITY = "opus"

# Default subprocess timeout — should fit longest prompt round-trip.
# Channel scoring on 15 channels: ~15s. Video selection: ~30s. Course composition: ~60s.
CLAUDE_CLI_TIMEOUT_SEC = 180


# ---------------------------------------------------------------------------
# Subprocess wrapper
# ---------------------------------------------------------------------------

def _call_json(*, model: str, system: str, user: str,
               max_tokens: int = 4096,  # accepted for API compat; CLI ignores
               timeout: int = CLAUDE_CLI_TIMEOUT_SEC) -> Any:
    """Invoke `claude -p` and parse the response as JSON.

    The system prompt is prepended to the user prompt because Claude Code CLI
    in `-p` mode doesn't take a separate system-message flag in all versions.
    Using --append-system-prompt would be cleaner but isn't supported on every
    install path.
    """
    del max_tokens  # CLI handles token budget itself
    full_prompt = f"{system.strip()}\n\n---\n\n{user.strip()}\n\nReturn ONLY the JSON, no commentary."

    # Run from an isolated tmpdir so Claude Code doesn't pick up workspace state.
    with tempfile.TemporaryDirectory(prefix="onboarder-claude-") as tmpdir:
        env = os.environ.copy()
        env.setdefault("PATH", f"{Path.home()}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin")
        try:
            r = subprocess.run(
                [
                    "claude", "-p", full_prompt,
                    "--model", model,
                    "--output-format", "text",
                    "--permission-mode", "bypassPermissions",
                ],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"claude CLI timed out after {timeout}s") from e
        except FileNotFoundError as e:
            raise RuntimeError(
                "claude CLI not found in PATH. Ensure Claude Code is installed "
                "and CLAUDE_CODE_OAUTH_TOKEN is set in the gateway's env."
            ) from e

        if r.returncode != 0:
            raise RuntimeError(
                f"claude CLI exit {r.returncode}: stderr={r.stderr[:500]!r} "
                f"stdout={r.stdout[:500]!r}"
            )

    text = (r.stdout or "").strip()
    if not text:
        raise RuntimeError(f"claude CLI returned empty stdout. stderr={r.stderr[:500]!r}")

    # Strip code fences if model wrapped JSON in ```json ... ```
    if text.startswith("```"):
        # Drop opening fence (with optional language tag) and trailing fence
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

    # Some Claude responses include leading prose before JSON; try to find first { or [
    if not (text.startswith("{") or text.startswith("[")):
        for opener in ("{", "["):
            idx = text.find(opener)
            if idx != -1:
                text = text[idx:]
                break

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error(f"llm: JSON parse failed. Raw: {text[:500]}")
        raise ValueError(f"LLM returned non-JSON: {e}")


# ---------------------------------------------------------------------------
# Phase 1: channel scoring
# ---------------------------------------------------------------------------

SCORE_CHANNELS_SYSTEM = """You are an expert at evaluating YouTube channels for educational course material.

Given a topic and a list of candidate channels with their metadata, score each channel from 0.0 to 1.0 on its fit for building a coherent online course on that topic.

Scoring factors (apply your judgment):
- Subscriber count and video count are within configured min/max
- Recent activity (videos within max_video_age_months)
- Channel focus aligns with the topic — not a generic vlog/news channel
- Educational format (explainers, tutorials) preferred over reactions/promos
- Language is in preferred_languages
- Channel is not primarily a sales funnel for the author's own paid products

Return ONLY valid JSON, no prose:
[
  {"channel_id": "UC...", "score": 0.87, "reason": "focused educational content on AI marketing, regular publishing, RU"},
  ...
]
"""


def score_channels(*, topic: str, criteria: dict[str, Any],
                   channels: list[dict[str, Any]], model: str = DEFAULT_MODEL_FAST) -> list[dict[str, Any]]:
    """Return list of {channel_id, score, reason} sorted by score desc.

    `channels` items shape (from yt-dlp metadata):
        {"channel_id": str, "name": str, "subscribers": int, "video_count": int,
         "description": str, "language": str | None,
         "recent_videos": [{"title": str, "published_at": str}, ...]}
    """
    user = json.dumps({"topic": topic, "criteria": criteria, "channels": channels},
                      ensure_ascii=False, indent=2)
    parsed = _call_json(model=model, system=SCORE_CHANNELS_SYSTEM, user=user)
    if not isinstance(parsed, list):
        raise ValueError(f"score_channels: expected list, got {type(parsed).__name__}")
    parsed.sort(key=lambda x: x.get("score", 0), reverse=True)
    return parsed


# ---------------------------------------------------------------------------
# Phase 1: video selection within a channel
# ---------------------------------------------------------------------------

SELECT_VIDEOS_SYSTEM = """You build coherent online courses by curating videos from a single YouTube channel.

Given a channel's recent video list and a course topic, select 6-12 videos that together form a logical learning progression on that topic. Skip:
- Off-topic videos (channel may cover multiple themes)
- Promotional/announcement videos
- Live streams and Q&A sessions (unless clearly structured)
- Videos shorter than preferred_video_length_min minutes or longer than preferred_video_length_max
- Duplicates / reposts / "best of" compilations

Order the selected videos for course delivery (basics → advanced).

Return ONLY valid JSON, no prose:
{
  "course_title": "Short, descriptive course title (e.g. 'AI for marketers: practical foundations')",
  "lessons": [
    {"video_id": "abc123", "title": "(may keep original or improve)", "order": 0, "reason": "intro"},
    ...
  ]
}

If the channel doesn't have enough on-topic material for a coherent 6+ video course, return:
{"course_title": null, "lessons": [], "skip_reason": "..."}
"""


def select_videos(*, topic: str, criteria: dict[str, Any],
                  channel_name: str, videos: list[dict[str, Any]],
                  model: str = DEFAULT_MODEL_FAST) -> dict[str, Any]:
    """Return {course_title, lessons[]} or {course_title: null, lessons: [], skip_reason}.

    `videos` items shape (yt-dlp listing):
        {"video_id": str, "title": str, "duration_sec": int, "published_at": str,
         "view_count": int, "description": str (truncated)}
    """
    user = json.dumps({
        "topic": topic, "criteria": criteria,
        "channel_name": channel_name, "videos": videos,
    }, ensure_ascii=False, indent=2)
    parsed = _call_json(model=model, system=SELECT_VIDEOS_SYSTEM, user=user,
                        max_tokens=8192)
    if not isinstance(parsed, dict):
        raise ValueError(f"select_videos: expected dict, got {type(parsed).__name__}")
    return parsed


# ---------------------------------------------------------------------------
# Phase 2: cut markers from transcript
# ---------------------------------------------------------------------------

MARK_CUTS_SYSTEM = """You edit educational videos by identifying segments to remove.

Given a Whisper transcript with word-level timestamps, return time ranges (in seconds) to CUT OUT. Target:
- Channel intros (logo animations, "Hi everyone, welcome back to my channel")
- Outros ("Like and subscribe", "Hit the notification bell", "See you next time")
- Mid-roll promotional segments (mentions of the host's paid course, sponsorship reads, "join my Patreon")
- Off-topic personal stories that don't serve the course topic
- Excessive filler ("ums" alone — keep, but cut 30+ seconds of repeated false starts)

Be CONSERVATIVE: if unsure, KEEP the segment. The course topic context is provided so you can judge relevance.

Return ONLY valid JSON, no prose:
[
  {"start": 0.0, "end": 18.4, "reason": "intro animation + greeting"},
  {"start": 754.2, "end": 770.1, "reason": "subscribe call mid-video"},
  ...
]

Use exact timestamps from the transcript. Cuts must not overlap. Empty list if nothing to cut.
"""


def mark_cuts(*, course_topic: str, transcript: dict[str, Any],
              model: str = DEFAULT_MODEL_FAST) -> list[dict[str, Any]]:
    """Return list of {start, end, reason} time ranges to remove."""
    user = json.dumps({"course_topic": course_topic, "transcript": transcript},
                      ensure_ascii=False)
    parsed = _call_json(model=model, system=MARK_CUTS_SYSTEM, user=user,
                        max_tokens=4096)
    if not isinstance(parsed, list):
        raise ValueError(f"mark_cuts: expected list, got {type(parsed).__name__}")
    return parsed


# ---------------------------------------------------------------------------
# Phase 2: course composition (description + author bio)
# ---------------------------------------------------------------------------

COMPOSE_COURSE_SYSTEM = """You write course landing-page copy in English for an online learning platform.

Given a course topic, the channel/author info, and short transcripts of each lesson, produce:
- excerpt: 1-2 sentence hook (max 200 chars), conversational tone
- aboutContent: 2-3 short paragraphs, what the student will learn and why it matters
- author_bio: 2-3 sentences about the instructor, focused on credibility (years experience, specialty)

Tone: confident, practical, no hype. No exclamation marks. No "transform your life" cliches.

Return ONLY valid JSON, no prose:
{
  "excerpt": "...",
  "aboutContent": "...\\n\\n...",
  "author_bio": "..."
}
"""


def compose_course(*, course_topic: str, course_title: str,
                   channel_name: str, channel_description: str,
                   lesson_transcripts: list[str],
                   model: str = DEFAULT_MODEL_QUALITY) -> dict[str, str]:
    """Return {excerpt, aboutContent, author_bio}. Uses Opus by default for prose quality."""
    # Truncate each lesson transcript to ~500 words to stay within token budget
    trimmed = []
    for t in lesson_transcripts:
        words = t.split()
        trimmed.append(" ".join(words[:500]))

    user = json.dumps({
        "course_topic": course_topic, "course_title": course_title,
        "channel_name": channel_name, "channel_description": channel_description,
        "lesson_transcripts": trimmed,
    }, ensure_ascii=False)
    parsed = _call_json(model=model, system=COMPOSE_COURSE_SYSTEM, user=user,
                        max_tokens=2048)
    if not isinstance(parsed, dict):
        raise ValueError(f"compose_course: expected dict, got {type(parsed).__name__}")
    for key in ("excerpt", "aboutContent", "author_bio"):
        if key not in parsed:
            raise ValueError(f"compose_course: missing '{key}'")
    return parsed
