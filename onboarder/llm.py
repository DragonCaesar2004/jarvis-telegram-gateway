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
        # Pipe prompt via stdin to avoid OS ARG_MAX limit on long transcripts.
        # claude -p reads stdin when no positional prompt argument is given.
        try:
            r = subprocess.run(
                [
                    "claude", "-p",
                    "--model", model,
                    "--output-format", "text",
                    "--permission-mode", "bypassPermissions",
                ],
                input=full_prompt,
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
# Phase 2: full course composition via delimiter-based template
# ---------------------------------------------------------------------------

COMPOSE_FULL_SYSTEM = """You are an expert course content writer for an online education platform called TrueLifeFlow. Create a complete, structured course description from the materials provided.

I will give you:
- Transcribed video lessons (each video = exactly one lesson — DO NOT invent extra lessons)
- Channel/author info
- Course topic + course title

Fill in the template below. Follow rules EXACTLY.

## TEMPLATE RULES

Sections start with `===SECTION===`. Sub-blocks use `---block---` or `-- lesson: NAME --`. Comments starting with `#` are ignored.

### ===AUTHOR===
- name: Real instructor name from materials
- bio: 3-7 sentences, third person, highlights expertise. NO links, NO URLs, NO social media handles. Self-contained text only. Markdown bold/italic OK.

### ===COURSE===
- title: If a course title is provided, use it EXACTLY. Else create 3-10 word engaging title.
- isAdult: true if 18+, else false

### ===EXCERPT===
2-4 sentences. Plain text. Hook + value proposition.

### ===ABOUT===
Markdown. 150-400 words. Compelling opening paragraph, **bold** for benefits, bullet lists for outcomes, who it's for, motivating CTA at end.

### ===PLAN===
Format:
```
---section: Section Title---
- Topic
- Another topic
```
3-8 sections, 2-5 topics each. Marketing roadmap, NOT lesson list. Topics can span lines (continuation lines without `- ` prefix).

### ===SCIENCE===
Format:
```
headline: One sentence (10-18 words) positioning the course's method as research-backed
subtitle: One clarifying sentence (10-20 words) mentioning the specific topic

---institution---
name: Real institution / journal name
style: serif|serif-italic|serif-caps|serif-wide|serif-bold|serif-smallcaps|sans|sans-caps|sans-bold|sans-thin|display|mono

---institution---
name: ...
style: ... (use a DIFFERENT style than above)

---institution---
name: ...
style: ... (DIFFERENT third style)

---stat---
value: 74% (or 3.5x or "5 YEARS")
description: One-sentence outcome
citation: Source · Year

---stat---
value: ...
description: ...
citation: ...

---stat---
value: ...
description: ...
citation: ...
```
EXACTLY 3 institutions + EXACTLY 3 stats. Real, credible publications/institutions related to the course topic. Don't invent fake sources. Use 3 DIFFERENT styles. Good trios: `serif-italic + serif-wide + sans-caps`, or `serif-caps + sans-bold + serif-smallcaps`.

### ===CURRICULUM===
Format:
```
---section: Section Title---

-- lesson: Your Lesson Title --
description: 1-2 sentences

-- lesson: Another Lesson --
description: ...
```
**CRITICAL: total lessons MUST equal number of transcribed videos provided. Each video = exactly one lesson, in order.** Write your own lesson titles (don't copy YouTube titles), 3-8 words. Group into sections by learning theme. Add `[BONUS]` to bonus section titles.

### ===TESTIMONIALS===
Format:
```
---review---
name: Diverse first+last name
text: Specific review (1-4 sentences, mention concrete techniques)
rating: 5
```
Generate 5-8 testimonials. All rating: 5. Vary tone, length. Be specific to actual course content.

### ===COLLECTION===
- name: Pick from existing or suggest new (1-3 words):
  - Fitness & Health
  - Mindfulness
  - Dance & Movement
  - Creativity & Arts
  - Relationships & Intimacy
  - New
  - Men's Sexual Health

## OUTPUT RULES

Return ONLY the filled template. Start with `===AUTHOR===`. End after `===COLLECTION===`. No markdown code fences around it. No commentary."""


# Section names in order
_TEMPLATE_SECTIONS = ["AUTHOR", "COURSE", "EXCERPT", "ABOUT", "PLAN", "SCIENCE",
                      "CURRICULUM", "TESTIMONIALS", "COLLECTION"]


def _split_top_sections(text: str) -> dict[str, str]:
    """Split text by `===SECTION===` markers."""
    import re
    sections: dict[str, str] = {}
    pattern = re.compile(r"^===([A-Z_]+)===\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()
    return sections


def _parse_kv(block: str) -> dict[str, str]:
    """Parse simple `key: value` blocks (multiline values supported until next key or EOF)."""
    out: dict[str, str] = {}
    current_key: str | None = None
    for line in block.splitlines():
        if line.strip().startswith("#"):
            continue
        # New key starts with `word:` at column 0 (no leading spaces)
        if ":" in line and not line.startswith((" ", "\t")) and " " not in line.split(":", 1)[0]:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            current_key = key
            out[key] = val
        elif current_key is not None and line.strip():
            # continuation
            out[current_key] = (out[current_key] + "\n" + line).strip()
    return out


def _parse_plan(block: str) -> list[dict[str, Any]]:
    """Parse PLAN section: ---section: Title--- followed by `- topic` lines."""
    import re
    sections: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    cur_topic: str | None = None
    section_pat = re.compile(r"^---section:\s*(.+?)\s*---$")
    for raw in block.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        m = section_pat.match(line.strip())
        if m:
            if cur:
                if cur_topic is not None:
                    cur["items"].append({"title": cur_topic.strip()})
                sections.append(cur)
            cur = {"title": m.group(1), "items": []}
            cur_topic = None
            continue
        if cur is None:
            continue
        if line.strip().startswith("- "):
            if cur_topic is not None:
                cur["items"].append({"title": cur_topic.strip()})
            cur_topic = line.strip()[2:]
        elif cur_topic is not None:
            cur_topic = cur_topic + " " + line.strip()
    if cur:
        if cur_topic is not None:
            cur["items"].append({"title": cur_topic.strip()})
        sections.append(cur)
    return sections


def _parse_science(block: str) -> dict[str, Any] | None:
    """Parse SCIENCE section into sciencePlan dict, or None if blank."""
    import re
    if not block.strip():
        return None
    headline = ""
    subtitle = ""
    institutions: list[dict[str, str]] = []
    stats: list[dict[str, str]] = []
    cur_kind: str | None = None
    cur: dict[str, str] | None = None

    def flush():
        nonlocal cur, cur_kind
        if cur is None:
            return
        if cur_kind == "institution":
            institutions.append(cur)
        elif cur_kind == "stat":
            stats.append(cur)
        cur = None

    for raw in block.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("#") or not line.strip():
            continue
        if line.strip() == "---institution---":
            flush()
            cur_kind = "institution"
            cur = {"name": "", "logo": None, "style": "serif"}
            continue
        if line.strip() == "---stat---":
            flush()
            cur_kind = "stat"
            cur = {"value": "", "description": "", "citation": ""}
            continue
        if cur is None:
            # top-level kv (headline/subtitle)
            if line.startswith("headline:"):
                headline = line.split(":", 1)[1].strip()
            elif line.startswith("subtitle:"):
                subtitle = line.split(":", 1)[1].strip()
            continue
        # inside institution or stat
        if ":" in line:
            k, _, v = line.partition(":")
            cur[k.strip()] = v.strip()
    flush()
    if not headline and not institutions and not stats:
        return None
    return {
        "enabled": True,
        "headline": headline,
        "subtitle": subtitle,
        "institutions": institutions[:3],
        "stats": stats[:3],
    }


def _parse_curriculum(block: str) -> list[dict[str, Any]]:
    """Parse CURRICULUM into [{title, isBonus, lessons:[{title, description}]}]."""
    import re
    sections: list[dict[str, Any]] = []
    cur_section: dict[str, Any] | None = None
    cur_lesson: dict[str, str] | None = None
    section_pat = re.compile(r"^---section:\s*(.+?)\s*---$")
    lesson_pat = re.compile(r"^--\s*lesson:\s*(.+?)\s*--$")

    def flush_lesson():
        nonlocal cur_lesson
        if cur_lesson is not None and cur_section is not None:
            cur_section["lessons"].append(cur_lesson)
            cur_lesson = None

    for raw in block.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("#"):
            continue
        ms = section_pat.match(line.strip())
        if ms:
            flush_lesson()
            if cur_section is not None:
                sections.append(cur_section)
            title = ms.group(1)
            is_bonus = "[BONUS]" in title.upper()
            title = title.replace("[BONUS]", "").replace("[bonus]", "").strip()
            cur_section = {"title": title, "isBonus": is_bonus, "lessons": []}
            continue
        ml = lesson_pat.match(line.strip())
        if ml:
            flush_lesson()
            cur_lesson = {"title": ml.group(1), "description": ""}
            continue
        if cur_lesson is not None and line.startswith("description:"):
            cur_lesson["description"] = line.split(":", 1)[1].strip()
        elif cur_lesson is not None and line.strip() and not line.strip().startswith("---"):
            # Continuation of description
            cur_lesson["description"] = (cur_lesson["description"] + " " + line.strip()).strip()
    flush_lesson()
    if cur_section is not None:
        sections.append(cur_section)
    return sections


def _parse_testimonials(block: str) -> list[dict[str, Any]]:
    """Parse TESTIMONIALS into [{authorName, text, rating}]."""
    out: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for raw in block.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("#"):
            continue
        if line.strip() == "---review---":
            if cur and cur.get("authorName") and cur.get("text"):
                out.append(cur)
            cur = {"authorName": "", "text": "", "rating": 5}
            continue
        if cur is None:
            continue
        if line.startswith("name:"):
            cur["authorName"] = line.split(":", 1)[1].strip()
        elif line.startswith("text:"):
            cur["text"] = line.split(":", 1)[1].strip()
        elif line.startswith("rating:"):
            try:
                cur["rating"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                cur["rating"] = 5
        elif cur.get("text") and line.strip() and not line.strip().startswith(("---", "name:", "rating:")):
            cur["text"] = (cur["text"] + "\n" + line).strip()
    if cur and cur.get("authorName") and cur.get("text"):
        out.append(cur)
    return out


def parse_template(text: str) -> dict[str, Any]:
    """Parse the LLM's filled template into a structured dict ready for NMS API."""
    sections = _split_top_sections(text)

    author_kv = _parse_kv(sections.get("AUTHOR", ""))
    course_kv = _parse_kv(sections.get("COURSE", ""))
    is_adult = course_kv.get("isAdult", "false").strip().lower() in ("true", "yes", "1")

    return {
        "author": {
            "name": author_kv.get("name", "").strip(),
            "bio": author_kv.get("bio", "").strip(),
        },
        "course": {
            "title": course_kv.get("title", "").strip(),
            "isAdult": is_adult,
            "excerpt": sections.get("EXCERPT", "").strip(),
            "aboutContent": sections.get("ABOUT", "").strip(),
        },
        "planSections": _parse_plan(sections.get("PLAN", "")),
        "sciencePlan": _parse_science(sections.get("SCIENCE", "")),
        "curriculum": _parse_curriculum(sections.get("CURRICULUM", "")),
        "testimonials": _parse_testimonials(sections.get("TESTIMONIALS", "")),
        "collectionName": _parse_kv(sections.get("COLLECTION", "")).get("name", "").strip(),
    }


def compose_full_course(*, course_topic: str, course_title: str,
                        channel_name: str, channel_description: str,
                        lesson_transcripts: list[str],
                        model: str = DEFAULT_MODEL_QUALITY,
                        timeout: int = 600) -> dict[str, Any]:
    """Generate full course content via template + parser. Returns structured dict."""
    # Trim each transcript to ~700 words to stay within budget but keep enough context
    trimmed_lessons = []
    for i, t in enumerate(lesson_transcripts, start=1):
        words = (t or "").split()
        trimmed_lessons.append(f"--- Video {i} transcript ---\n" + " ".join(words[:700]))

    materials = (
        f"COURSE TOPIC: {course_topic}\n"
        f"COURSE TITLE (use exactly if you keep one): {course_title}\n"
        f"INSTRUCTOR / CHANNEL: {channel_name}\n"
        f"CHANNEL DESCRIPTION: {channel_description[:500] if channel_description else '(none)'}\n\n"
        f"NUMBER OF VIDEO LESSONS: {len(lesson_transcripts)} "
        f"(curriculum MUST contain exactly this many lessons in this order)\n\n"
        + "\n\n".join(trimmed_lessons)
    )

    full_prompt = COMPOSE_FULL_SYSTEM + "\n\n---\n\n## MATERIALS\n\n" + materials

    # Use the same _call subprocess machinery as _call_json, but expect plain text (template).
    import os, subprocess, tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="onboarder-claude-") as tmpdir:
        env = os.environ.copy()
        env.setdefault("PATH", f"{Path.home()}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin")
        try:
            r = subprocess.run(
                ["claude", "-p",
                 "--model", model,
                 "--output-format", "text",
                 "--permission-mode", "bypassPermissions"],
                input=full_prompt,
                cwd=tmpdir, env=env,
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"compose_full_course: claude CLI timed out after {timeout}s") from e

    if r.returncode != 0 or not (r.stdout or "").strip():
        raise RuntimeError(f"compose_full_course: claude exit {r.returncode}, stderr={r.stderr[:300]!r}")

    text = r.stdout.strip()
    # Strip optional code fences
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rsplit("```", 1)[0].strip()

    parsed = parse_template(text)

    # Sanity validation
    if not parsed["author"]["name"] or not parsed["author"]["bio"]:
        raise ValueError(f"compose_full_course: missing author.name or author.bio. Got: {parsed['author']}")
    if not parsed["course"]["title"]:
        raise ValueError("compose_full_course: missing course.title")
    if not parsed["curriculum"]:
        raise ValueError("compose_full_course: empty curriculum")

    return parsed


# Backward-compat shim — phase2 will be updated to use compose_full_course
def compose_course(*, course_topic: str, course_title: str,
                   channel_name: str, channel_description: str,
                   lesson_transcripts: list[str],
                   model: str = DEFAULT_MODEL_QUALITY) -> dict[str, str]:
    """Legacy: returns only {excerpt, aboutContent, author_bio} for back-compat."""
    full = compose_full_course(
        course_topic=course_topic, course_title=course_title,
        channel_name=channel_name, channel_description=channel_description,
        lesson_transcripts=lesson_transcripts, model=model,
    )
    return {
        "excerpt": full["course"]["excerpt"],
        "aboutContent": full["course"]["aboutContent"],
        "author_bio": full["author"]["bio"],
    }
