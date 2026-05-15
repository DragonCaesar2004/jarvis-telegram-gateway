"""Claude LLM calls for the onboarder pipeline.

Four jobs:
    1. score_channels()   — rank candidate YouTube channels against criteria + topic
    2. select_videos()    — pick a coherent course of 5-30 videos from one channel's video list
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
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("gateway")

# CLI model aliases (claude -p --model <alias>)
DEFAULT_MODEL_FAST = "sonnet"
DEFAULT_MODEL_QUALITY = "opus"


# ---------------------------------------------------------------------------
# Output-language hints
# ---------------------------------------------------------------------------
# The operator (and Liza, who'll be running the bot day-to-day) reviews the
# Sheet in Russian and the platform locale matches the customer's language.
# When the topic the user typed is Cyrillic-heavy, we tell every Claude prompt
# to write its user-facing text (descriptions, bios, titles, plans, taglines,
# testimonials) in Russian. Default is English.

def detect_topic_lang(*texts: str) -> str:
    """Best-effort language detection from a few short strings.

    Returns ISO code: 'ru' if any input is mostly Cyrillic, otherwise 'en'.
    """
    for t in texts:
        if not t:
            continue
        letters = [c for c in t if c.isalpha()]
        if not letters:
            continue
        cyr = sum(1 for c in letters if 'а' <= c.lower() <= 'я' or c.lower() == 'ё')
        if cyr / len(letters) > 0.3:
            return 'ru'
    return 'en'


def _lang_instruction(output_lang: str) -> str:
    """Return an extra system-prompt block forcing the output language.

    Caller appends the result to the system prompt. Always returns a non-empty
    block so the LLM cannot drift to the input language just because the source
    materials (transcripts, channel names) are non-English.
    """
    lang = (output_lang or 'en').lower()
    if lang == 'ru':
        return (
            "\n\nLANGUAGE REQUIREMENT (override examples below): write ALL "
            "user-facing text in Russian (Cyrillic). This applies to: lesson "
            "descriptions, course titles, taglines, excerpts, about content, "
            "plan section titles and topics, science headline/subtitle, "
            "curriculum lesson titles and descriptions, testimonials, author "
            "name (transliterate if originally non-Russian — e.g. \"Greg "
            "Doucette\" stays as is, but the bio is in Russian), author bio, "
            "expertise. Keep technical identifiers (URLs, IDs, code) in their "
            "original form. Russian examples / English examples in the prompt "
            "below are STYLE references only — you must produce Russian output."
        )
    # Default: English. We MUST be explicit — without this block, the LLM
    # mirrors the source-material language (e.g. when transcripts are Russian
    # or the channel is Russian, output drifts to Russian).
    return (
        "\n\nLANGUAGE REQUIREMENT — CRITICAL, OVERRIDES ANY EXAMPLES BELOW:\n"
        "Write 100% of user-facing text in ENGLISH (Latin script only). This "
        "is an absolute requirement. The output is published on an "
        "English-language platform — Cyrillic characters anywhere in the "
        "user-facing fields will break the landing page.\n"
        "\n"
        "Applies to EVERY field, no exceptions:\n"
        "- author.name → use ROMAN/LATIN transliteration "
        "(\"Андрей Курпатов\" → \"Andrey Kurpatov\", \"Полина Киржева\" → "
        "\"Polina Kirzheva\"). Do NOT keep Cyrillic in author.name.\n"
        "- author.bio, author.expertise → English prose\n"
        "- course.title, course.excerpt, course.aboutContent → English\n"
        "- planSections (titles + topics) → English\n"
        "- sciencePlan (headline, subtitle, institutions, stats) → English\n"
        "- curriculum (every lesson title + description) → English. DO NOT "
        "copy Russian video titles from the transcripts. Write fresh English "
        "lesson titles describing what the student will learn.\n"
        "- testimonials (every name + text) → English. Generate English-"
        "sounding student names (\"Sarah Mitchell\", \"David Chen\"), not "
        "transliterated Russian ones.\n"
        "- collectionName → English\n"
        "\n"
        "Translate concepts from Russian/other-language transcripts into "
        "natural, fluent English — do NOT keep Russian phrasing, idioms, or "
        "Cyrillic words anywhere in user-facing output. Keep technical "
        "identifiers (URLs, IDs, code blocks) in their original form."
    )


def _pain_audience_block(pain: str, audience: str) -> str:
    """Return an extra system-prompt block describing operator-specified
    course direction and (optionally) target audience. Empty when both blank.

    `pain` is named for legacy reasons — the wizard now uses it as a
    free-form COURSE DESCRIPTION (what topics the course should cover, what
    pain it solves, what outcome the student gets). Either interpretation
    flows through identically: it's a strong directive about the course's
    direction. Older runs that stored a pure pain sentence still work.
    """
    pain = (pain or "").strip()
    audience = (audience or "").strip()
    if not pain and not audience:
        return ""
    parts = ["\n\n## OPERATOR-SPECIFIED COURSE DIRECTION\n"]
    if pain:
        parts.append(
            f"- Course description / what the course MUST cover: \"{pain}\"\n"
            "  Treat this as the canonical specification. The course's topics, "
            "lesson selection, sequencing, depth, and tone must all serve it."
        )
    if audience:
        parts.append(f"- Target audience: \"{audience}\"")
    parts.append(
        "- Bias EVERY judgment (channel scoring, video selection, lesson "
        "descriptions, course title/excerpt/about, curriculum structure, "
        "author bio framing, testimonials voice) toward this exact direction. "
        "Penalize generic or off-spec material — it doesn't help the student."
    )
    return "\n".join(parts)

# Default subprocess timeout — should fit longest prompt round-trip.
# Channel scoring on 15 channels: ~15s. Video selection: ~30s. Course composition: ~60s.
CLAUDE_CLI_TIMEOUT_SEC = 180

# Retry budget for transient Anthropic API errors (503/529/timeout/network).
# Anthropic occasionally returns 503 under load — without retries, the whole
# Phase 1 / Phase 2 worker dies. With these we wait, retry, and only surface
# the error if all attempts fail.
_CLI_RETRYABLE_MARKERS = (
    "503", "529", "Service is currently unavailable", "rate_limit_error",
    "Internal server error", "Overloaded", "overloaded_error", "EAI_AGAIN",
    "Connection reset", "Connection aborted", "ConnectTimeout",
    "Read timed out", "API_ERROR_500", "APIError",
)
_CLI_RETRY_BACKOFFS = (5, 15, 45)  # seconds; total ~65s of waiting before failing


def _is_retryable_cli_failure(stderr: str, stdout: str) -> bool:
    blob = (stderr or "") + "\n" + (stdout or "")
    return any(m in blob for m in _CLI_RETRYABLE_MARKERS)


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

    last_err: str = ""
    for attempt, backoff in enumerate([0] + list(_CLI_RETRY_BACKOFFS)):
        if backoff:
            log.warning(f"llm._call_json retry {attempt} after {backoff}s "
                        f"(prev error: {last_err[:200]})")
            time.sleep(backoff)
        # Run from an isolated tmpdir so Claude Code doesn't pick up workspace state.
        with tempfile.TemporaryDirectory(prefix="onboarder-claude-") as tmpdir:
            env = os.environ.copy()
            env.setdefault("PATH", f"{Path.home()}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin")
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
            except subprocess.TimeoutExpired:
                last_err = f"claude CLI timed out after {timeout}s"
                continue  # retry on timeout — Anthropic might just be slow
            except FileNotFoundError as e:
                raise RuntimeError(
                    "claude CLI not found in PATH. Ensure Claude Code is installed "
                    "and CLAUDE_CODE_OAUTH_TOKEN is set in the gateway's env."
                ) from e

        if r.returncode == 0 and (r.stdout or "").strip():
            break  # success
        last_err = f"exit {r.returncode}: stderr={r.stderr[:300]} stdout={r.stdout[:200]}"
        # Only retry on transient API errors. Permanent failures (auth, schema)
        # surface immediately so we don't waste a minute on something doomed.
        if not _is_retryable_cli_failure(r.stderr or "", r.stdout or ""):
            raise RuntimeError(f"claude CLI {last_err}")
    else:
        raise RuntimeError(f"claude CLI exhausted retries: {last_err}")

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
                   channels: list[dict[str, Any]], model: str = DEFAULT_MODEL_FAST,
                   pain: str = "", audience: str = "") -> list[dict[str, Any]]:
    """Return list of {channel_id, score, reason} sorted by score desc.

    `channels` items shape (from yt-dlp metadata):
        {"channel_id": str, "name": str, "subscribers": int, "video_count": int,
         "description": str, "language": str | None,
         "recent_videos": [{"title": str, "published_at": str}, ...]}
    """
    payload: dict[str, Any] = {"topic": topic, "criteria": criteria, "channels": channels}
    if pain:
        payload["target_pain"] = pain
    if audience:
        payload["target_audience"] = audience
    user = json.dumps(payload, ensure_ascii=False, indent=2)
    system = SCORE_CHANNELS_SYSTEM + _pain_audience_block(pain, audience)
    # 15-min timeout: with HARD_CAP_CHANNELS_TO_CHECK at 100, the prompt can
    # carry 50-100 channels (30K+ tokens). Default 180s isn't enough for Claude
    # to process this much input; we raise specifically for score_channels so
    # other callers keep their tighter timeouts.
    parsed = _call_json(model=model, system=system, user=user, timeout=900)  # 15 min for score_channels
    if not isinstance(parsed, list):
        raise ValueError(f"score_channels: expected list, got {type(parsed).__name__}")
    parsed.sort(key=lambda x: x.get("score", 0), reverse=True)
    return parsed


# ---------------------------------------------------------------------------
# Phase 1: video selection within a channel
# ---------------------------------------------------------------------------

SELECT_VIDEOS_SYSTEM = """You build coherent online courses by curating videos from a single YouTube channel.

Given a channel's recent video list and a course topic, select 5-30 videos that together form a STRUCTURED LEARNING PROGRESSION on that exact topic.

## Selection bar (be strict — quality matters far more than count)

INCLUDE only videos that:
1. **Directly serve the topic.** The title clearly signals it covers the core subject or a sub-topic any reasonable curriculum would include. "Tangentially related" is not enough — if a video would feel out of place in a paid course on this topic, exclude it.
2. **Build on each other.** Taken together, the selected videos cover the topic from foundations to advanced application without major gaps. A reader who watches them in order should leave with a coherent mental model, not a grab-bag of tips.
3. **Aren't redundant.** If two videos cover the same material, keep the better one (clearer title, longer/more thorough, more recent).

EXCLUDE videos that are:
- Off-topic (channels often cover multiple themes — only pick those that fit THIS course's topic)
- Promotional / announcement-only (sponsorship reads, "I'm starting a new course", product launches)
- Live streams, Q&As, podcasts, or unstructured interviews (unless they're explicitly framed as standalone lessons)
- Outside preferred_video_length_min..preferred_video_length_max
- Duplicates / reposts / "best of" compilations / Shorts (under ~3 min)

## Sequencing

Order the selected videos for course delivery. Default arc: foundations → core concepts → application → advanced. If a different ordering serves the topic better (chronological, problem-by-problem, anatomy-by-anatomy), use it — but justify it with the per-lesson `reason` field. Each `reason` should say WHAT this lesson contributes to the overall progression, not just describe the video.

## Length rules

- **Minimum: 5 videos.** If fewer than 5 on the channel meet the bar AND together form a coherent progression, return skip_reason with a null course_title. DO NOT pad with off-topic or low-quality videos to reach the minimum — better to skip the channel.
- **Maximum: 30 videos.** If the channel has more on-topic material than fits, pick the strongest 30 that together still form a clean progression.
- **Aim for the smallest count that COMPLETELY covers the topic.** Don't inflate the course with filler. A tight 8-video course beats a bloated 20-video one.

## Output

Return ONLY valid JSON, no prose:
{
  "course_title": "Short, outcome-oriented course title (e.g. 'AI for marketers: practical foundations' — names what the student can do after)",
  "lessons": [
    {"video_id": "abc123", "title": "(may keep original or rewrite for clarity)", "order": 0, "reason": "What this lesson contributes to the progression — e.g. 'establishes the core decision-matrix used in lessons 4-7'"},
    ...
  ]
}

If the channel doesn't have enough on-topic material for a coherent 5+ video course, return:
{"course_title": null, "lessons": [], "skip_reason": "Concrete reason — e.g. 'only 2 videos directly cover this topic; the rest are reaction/lifestyle content'"}
"""


SELECT_VIDEOS_VIDEO_CAP = 300
"""Hard upper bound on how many videos we hand to Claude per channel.

Beyond ~300 entries the prompt grows into the 50k+ token range, which is
fine for Claude's context window but reliably pushes the per-call latency
past our 5-minute timeout when Anthropic is under load. Channels with
fewer videos pass through untouched; channels with more get the freshest
SELECT_VIDEOS_VIDEO_CAP videos (yt-dlp returns newest-first), which is
plenty of material for picking a coherent 5-30 lesson course.
"""

SELECT_VIDEOS_TIMEOUT_SEC = 300


def select_videos(*, topic: str, criteria: dict[str, Any],
                  channel_name: str, videos: list[dict[str, Any]],
                  model: str = DEFAULT_MODEL_FAST,
                  pain: str = "", audience: str = "") -> dict[str, Any]:
    """Return {course_title, lessons[]} or {course_title: null, lessons: [], skip_reason}.

    `videos` items shape (yt-dlp listing):
        {"video_id": str, "title": str, "duration_sec": int, "published_at": str,
         "view_count": int, "description": str (truncated)}
    """
    # Cap the candidate list to keep prompt size and round-trip time sane.
    # yt-dlp returns newest-first; trim the tail.
    capped_videos = videos[:SELECT_VIDEOS_VIDEO_CAP]
    if len(videos) > SELECT_VIDEOS_VIDEO_CAP:
        log.info(f"select_videos: capping {len(videos)} → "
                 f"{SELECT_VIDEOS_VIDEO_CAP} videos for channel {channel_name!r}")

    payload: dict[str, Any] = {
        "topic": topic, "criteria": criteria,
        "channel_name": channel_name, "videos": capped_videos,
    }
    if pain:
        payload["target_pain"] = pain
    if audience:
        payload["target_audience"] = audience
    user = json.dumps(payload, ensure_ascii=False, indent=2)
    system = SELECT_VIDEOS_SYSTEM + _pain_audience_block(pain, audience)
    parsed = _call_json(model=model, system=system, user=user,
                        max_tokens=8192,
                        timeout=SELECT_VIDEOS_TIMEOUT_SEC)
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
# Phase 1 (post-transcribe): per-lesson descriptions, batched per course
# ---------------------------------------------------------------------------

DESCRIBE_LESSONS_SYSTEM = """You write concise, landing-page-quality lesson descriptions for an online course.

I will give you:
- Course topic and title
- A list of lessons in order, each with its title and transcript excerpt

For each lesson, return a 2-4 sentence description that:
- Tells a prospective student what they will learn or be able to do after the lesson
- References at least one concrete concept, technique, or example from the transcript
- Is written in plain English, third person, no fluff (no "in this lesson we will…")
- Does not invent material that isn't in the transcript

Return ONLY valid JSON, no prose:
[
  {"order": 0, "description": "Introduces the four-quadrant decision matrix used throughout the rest of the course, with a worked example on choosing between SEO and paid acquisition."},
  ...
]

Keep order matching the input. Empty/garbage transcript → return a generic 1-sentence description rather than failing.
"""


def describe_lessons(*, course_topic: str, course_title: str,
                     lessons: list[dict[str, Any]],
                     model: str = DEFAULT_MODEL_FAST,
                     output_lang: str = "en",
                     pain: str = "", audience: str = "") -> list[dict[str, Any]]:
    """Generate per-lesson descriptions in one batch call.

    `lessons` items: {order: int, title: str, transcript: str}. Transcripts are
    truncated to ~500 words inside this function to keep the prompt cheap.

    `output_lang`: 'en' (default) or 'ru'. Russian forces all description text
    into Cyrillic for operator review.
    `pain` / `audience`: optional targeting strings — when provided, every
    description should be framed in terms of solving that pain for that audience.

    Returns list of {order, description} aligned to input order.
    """
    if not lessons:
        return []

    trimmed: list[dict[str, Any]] = []
    for l in lessons:
        text = (l.get("transcript") or "").strip()
        words = text.split()
        trimmed.append({
            "order": int(l.get("order", 0)),
            "title": l.get("title", "")[:200],
            "transcript_excerpt": " ".join(words[:500]),
        })

    payload: dict[str, Any] = {
        "course_topic": course_topic,
        "course_title": course_title,
        "lessons": trimmed,
    }
    if pain:
        payload["target_pain"] = pain
    if audience:
        payload["target_audience"] = audience
    user = json.dumps(payload, ensure_ascii=False, indent=2)

    system = (DESCRIBE_LESSONS_SYSTEM
              + _pain_audience_block(pain, audience)
              + _lang_instruction(output_lang))
    parsed = _call_json(model=model, system=system, user=user,
                        max_tokens=4096)
    if not isinstance(parsed, list):
        raise ValueError(f"describe_lessons: expected list, got {type(parsed).__name__}")

    # Defensive: backfill any missing entries
    by_order = {int(p.get("order", -1)): str(p.get("description", "")).strip()
                for p in parsed if isinstance(p, dict)}
    out: list[dict[str, Any]] = []
    for l in lessons:
        order = int(l.get("order", 0))
        out.append({"order": order,
                    "description": by_order.get(order, "")})
    return out


# ---------------------------------------------------------------------------
# Phase 1: batch translation to Russian for the Sheet review pass
# ---------------------------------------------------------------------------
# Operator reviews the Sheet in Russian. When the source language of a course
# is English (or any non-Russian), we ALSO need a Russian rendering of every
# lesson description, the course description, and the author bio so the
# operator can sanity-check what's about to land on the platform without
# bouncing through Google Translate.
#
# Stored format in the Sheet: original + a separator + Russian, so a reviewer
# sees both in a single cell. When the source is already Russian we skip the
# duplication.

TRANSLATE_BATCH_SYSTEM = """You translate landing-page copy from any source language into natural Russian.

You will receive a JSON array of items, each with `id` and `text`. Translate every `text` into Russian and return a JSON array with the same `id`s in the same order. Strictly preserve markdown formatting (**bold**, *italic*, ### headings, bullet lists), URLs (don't translate), and proper names (people, brands, institutions — transliterate only when the Russian convention does so). Don't summarize or shorten — translate fully. Don't add commentary.

Output ONLY valid JSON in this exact shape:
[
  {"id": "lesson_1", "text": "<Russian translation>"},
  {"id": "course_description", "text": "<...>"},
  ...
]
"""


_RU_TRANSLATION_SEP = "\n\n— Перевод на русский —\n\n"


def translate_batch_to_russian(items: list[dict[str, str]],
                               model: str = DEFAULT_MODEL_FAST,
                               timeout: int = 180) -> dict[str, str]:
    """Translate many short texts in one Claude call.

    `items`: [{"id": str, "text": str}, ...]. Empty texts are skipped.

    Returns a {id: russian_text} dict. On failure / partial misses, the missing
    ids are absent — caller should fall back to the source text.
    """
    payload = [{"id": str(it["id"]), "text": str(it.get("text") or "").strip()}
               for it in items if (it.get("text") or "").strip()]
    if not payload:
        return {}
    user = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        parsed = _call_json(model=model, system=TRANSLATE_BATCH_SYSTEM, user=user,
                            timeout=timeout, max_tokens=8192)
    except Exception as e:
        log.warning(f"translate_batch_to_russian failed: {e}")
        return {}
    if not isinstance(parsed, list):
        log.warning(f"translate_batch_to_russian: non-list result: {parsed!r}")
        return {}
    out: dict[str, str] = {}
    for p in parsed:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "").strip()
        ptext = str(p.get("text") or "").strip()
        if pid and ptext:
            out[pid] = ptext
    return out


def join_with_russian(original: str, russian: str | None) -> str:
    """Combine source text and its Russian translation into one Sheet cell.

    If the russian field is missing or equals the original (translation
    failed), returns the original alone. Otherwise returns
    "original<sep>russian".
    """
    original = (original or "").strip()
    russian = (russian or "").strip()
    if not original:
        return russian or ""
    if not russian or russian == original:
        return original
    return f"{original}{_RU_TRANSLATION_SEP}{russian}"


# ---------------------------------------------------------------------------
# Phase 1: deep author research with WebSearch (Claude CLI tool)
# ---------------------------------------------------------------------------

RESEARCH_AUTHOR_SYSTEM = """You research public figures and online educators to produce trustworthy biographies.

You will be given a YouTube channel name, the channel's about-text, sample video titles, and a course topic. Use the WebSearch tool freely to find:
- The author's real first and last name (the channel may be a personal brand or a pseudonym)
- LinkedIn profile, personal website, Wikipedia, faculty page, or major publications by them
- Concrete credentials: degrees, employers, books, talks, peer-reviewed work, certifications
- Recent activity that confirms they're still active in the field

Synthesize the findings into a structured profile. If WebSearch returns nothing useful, fall back to what's evident from the channel description and titles, but flag that with `confidence: "low"`.

Return ONLY valid JSON, no prose:
{
  "name": "Real full name (or best-effort guess; e.g. channel handle if unknown)",
  "bio": "3-5 sentences, third person, factual, no links or URLs, no emojis. Markdown bold/italic OK. Specific credentials over generic praise.",
  "expertise": "Comma-separated list of 3-6 expertise areas relevant to the course topic",
  "confidence": "high | medium | low",
  "sources": ["url1", "url2", ...]
}

Hard rules:
- NEVER invent credentials, degrees, or affiliations. If unsure, omit.
- NEVER include URLs or social handles in `bio` (sources go in the sources array).
- If the channel is clearly a faceless brand, set name to the brand and write the bio in the brand's voice.
"""


def research_author(*, channel_name: str, channel_description: str,
                    sample_video_titles: list[str], course_topic: str,
                    model: str = DEFAULT_MODEL_QUALITY,
                    timeout: int = 240,
                    output_lang: str = "en",
                    pain: str = "", audience: str = "") -> dict[str, Any]:
    """Deep author research via Claude CLI (uses WebSearch under the hood when needed).

    Returns {name, bio, expertise, confidence, sources}. On failure, returns a
    low-confidence stub built from `channel_name` + `channel_description` so the
    pipeline can keep going without aborting the whole course.
    """
    user_payload: dict[str, Any] = {
        "channel_name": channel_name,
        "channel_description": (channel_description or "")[:1500],
        "sample_video_titles": sample_video_titles[:8],
        "course_topic": course_topic,
    }
    if pain:
        user_payload["target_pain"] = pain
    if audience:
        user_payload["target_audience"] = audience
    user = json.dumps(user_payload, ensure_ascii=False, indent=2)

    system = (RESEARCH_AUTHOR_SYSTEM
              + _pain_audience_block(pain, audience)
              + _lang_instruction(output_lang))
    try:
        parsed = _call_json(model=model, system=system, user=user,
                            timeout=timeout)
    except Exception as e:
        log.warning(f"llm.research_author: failed for {channel_name}: {e}")
        return _author_fallback(channel_name, channel_description)

    if not isinstance(parsed, dict):
        log.warning(f"llm.research_author: non-dict result for {channel_name}: {parsed!r}")
        return _author_fallback(channel_name, channel_description)

    return {
        "name": str(parsed.get("name") or channel_name).strip(),
        "bio": str(parsed.get("bio") or "").strip(),
        "expertise": str(parsed.get("expertise") or "").strip(),
        "confidence": str(parsed.get("confidence") or "low").strip().lower(),
        "sources": [s for s in (parsed.get("sources") or []) if isinstance(s, str)][:10],
    }


def _author_fallback(channel_name: str, channel_description: str) -> dict[str, Any]:
    """Cheap fallback when WebSearch / LLM fails — keeps pipeline alive."""
    desc = (channel_description or "").strip()[:300]
    bio = (desc if desc else
           f"{channel_name} runs an educational YouTube channel covering this topic.")
    return {
        "name": channel_name,
        "bio": bio,
        "expertise": "",
        "confidence": "low",
        "sources": [],
    }


# ---------------------------------------------------------------------------
# Phase 2: full course composition via delimiter-based template
# ---------------------------------------------------------------------------

COMPOSE_FULL_SYSTEM = """You are an expert course content writer for an online education platform called TrueLifeFlow. Your task is to create a complete, structured course description based on the materials I provide.

I will give you:
- Transcribed video lessons from the instructor (each video = exactly one lesson in the curriculum)
- Information about the instructor (links to their website, social media, bio, etc.)
- Any additional context about the course topic

**IMPORTANT: The number of lessons in the CURRICULUM section must EXACTLY match the number of video transcriptions I provide. Do NOT invent extra lessons.**

Based on these materials, fill in the template below. Follow the rules EXACTLY.

---

## TEMPLATE RULES

The template uses delimiter-based sections. Each section starts with `===SECTION_NAME===`. Do NOT change the delimiters — the system parses them automatically.

Lines starting with `#` are comments and will be ignored by the parser. Do NOT add comments to your output — only fill in the actual data.

### Section-by-section instructions:

**===AUTHOR===**
- `name:` — Full name of the course instructor. Use the real name from the provided materials.
- `bio:` — A compelling author biography (3-7 sentences). Write in third person. Highlight their expertise, credentials, experience, and why they're qualified to teach this topic. Can be multiline. Supports markdown (**bold**, *italic*) for emphasis. **DO NOT include any links, URLs, or references to external sources** (no website links, social media handles, YouTube channels, Instagram, etc). The bio must be self-contained text only.

**===COURSE===**
- `title:` — If a course title is provided in the materials, use it EXACTLY as given — do not change, rephrase, or "improve" it. If no title is provided, create an engaging, clear title (3-10 words) that communicates what the student will learn.
- `isAdult:` — Write `true` if the course contains adult/sensitive content (18+), otherwise `false`.

**===EXCERPT===**
Write a short, compelling description (2-4 sentences). This appears on course cards and in the hero section. It should hook the reader and clearly state the value proposition. No markdown here — plain text only.

**===ABOUT===**
Write a detailed course description in markdown format. This is the main "About" section on the course page. Structure it well:
- Start with a compelling opening paragraph about what the course offers
- Use **bold** for key benefits
- Use bullet lists for features, outcomes, or what's included
- Mention who will benefit from this course
- End with a motivating call-to-action sentence
- Length: 150-400 words
- Supports full markdown: **bold**, *italic*, ### headings, - lists, [links](url)

**===PLAN===**
Academic plan — a structured overview of what the course covers. This is displayed as an accordion on the course page. Use this format:

```
---section: Section Title---
- Topic or lesson title (can span
  multiple lines if needed)
- Another topic
- Third topic
```

Create 3-8 plan sections, each with 2-5 topics. Each topic starts with `- `. If a topic title is long, continuation lines (without `- ` prefix) are appended to the current topic. The plan should give a high-level roadmap — it's NOT the same as the lesson curriculum. Think of it as a marketing overview of the knowledge areas covered.

**===SCIENCE===**
An optional social-proof band titled "The science behind this program". It appears between the course description and the academic plan on the landing. Use this format:

```
headline: Backed by peer-reviewed research from the world's leading medical institutions
subtitle: Every method reflects what 20+ years of clinical trials show about <topic>.

---institution---
name: Harvard Medical School
style: serif-caps

---institution---
name: JAMA
style: sans-caps

---institution---
name: The New England Journal of Medicine
style: serif-italic

---stat---
value: 74%
description: Short outcome description tied to the course topic.
citation: Source · Year or study name

---stat---
value: 36%
description: Another outcome stat.
citation: Registry or journal · key detail

---stat---
value: 5 YEARS
description: Time-based outcome (can be a duration instead of a percent).
citation: Journal · year
```

Rules:
- `headline:` — one big sentence (10-18 words) that positions the course's method as grounded in serious research. Rewrite per-course, don't reuse the same sentence.
- `subtitle:` — one clarifying sentence (10-20 words) mentioning the specific topic (e.g. "knee pain and recovery", "sleep and circadian rhythm", "anxiety and nervous-system regulation"). Should feel specific, not generic.
- **Institutions (EXACTLY 3):** pick real, credible publications / medical institutions / research bodies that have actually published work on the course's topic. Examples: *The New England Journal of Medicine*, *JAMA*, *Harvard Medical School*, *Mayo Clinic*, *American Psychological Association*, *Nature*, *Lancet*, *British Journal of Sports Medicine*, *University of Oxford*, *Stanford Medicine*. Choose what fits the subject matter — don't invent fake sources. Always output exactly 3 `---institution---` blocks, no more and no less.

    Each institution block has two fields:
    - `name:` — the institution's display name (use the form it normally writes itself as — e.g. `The New England Journal of Medicine`, `JAMA`, `HARVARD MEDICAL SCHOOL`, `Mayo Clinic`).
    - `style:` — a typography hint that will render the name on the landing. Pick the style that best matches the institution's visual identity and that makes the three wordmarks look visually distinct from each other (so the row reads like a press-mention strip, not three copies of the same font). Allowed values:
      - `serif` — classic serif, semi-bold. Good default for medical journals.
      - `serif-italic` — italic serif, masthead feel. Good for fashion/lifestyle (*Vogue*, *Glamour*, *The New England Journal of Medicine*).
      - `serif-caps` — serif uppercase with tracking. Good for universities in formal form (*Harvard Medical School*, *Oxford*, *Stanford Medicine*).
      - `serif-wide` — large serif with wide letter-spacing. Good for acronym-style journal marks (*JAMA*, *BMJ*, *NEJM* when abbreviated).
      - `serif-bold` — bold serif, no caps. Good for names like *Lancet*, *Cell*.
      - `serif-smallcaps` — serif with true small-caps (mixed capital heights). Good for elegant academic marks.
      - `sans` — neutral sans-serif. Fallback when nothing else fits.
      - `sans-caps` — bold sans in caps with tracking. Good for brand-style titles (*Men's Health*, *Forbes*, *Fortune*, *Harvard* in the "HARVARD MEDICAL SCHOOL" form).
      - `sans-bold` — very heavy sans-serif. Good for journals with bold wordmarks (*Nature*, *Science*).
      - `sans-thin` — light sans in caps, widely tracked. Good for minimalist/contemporary outlets (*Wired*, *The Atlantic*).
      - `display` — italic serif with tight letter-spacing. Reserve for distinctive masthead logos.
      - `mono` — monospaced caps. Rare, reserve for tech/developer publications (*MIT Technology Review*).

      Use a **different style for each of the three** institutions so the row doesn't look monotone. Good trios:
      - `serif-italic` + `serif-wide` + `sans-caps` (matches the classic "medical journal + JAMA + Harvard" look)
      - `serif-caps` + `sans-bold` + `serif-smallcaps`
      - `sans-thin` + `serif-italic` + `serif-wide`
- **Stats (exactly 3):** each stat has a `value` (e.g. `74%`, `3.5x`, `5 YEARS`, `68%`), a `description` (concrete, one-sentence outcome relevant to the course topic), and a `citation` (plausible source with name and year or study identifier). Pick real published findings where possible; if you must approximate, keep numbers within the range reported in published literature on the topic.
- If the course topic doesn't lend itself to clinical stats (e.g. a pure creativity course), leave the SCIENCE section blank — the parser will skip it and the block won't appear on the landing.

**===CURRICULUM===**
The actual lesson structure. Organize lessons into sections. Use this format:

```
---section: Section Title---

-- lesson: Lesson Title --
description: Brief description of what this lesson covers (1-2 sentences)

-- lesson: Another Lesson --
description: What this lesson teaches
```

**CRITICAL RULE: The total number of lessons MUST EXACTLY match the number of transcribed videos I provide.** Each transcribed video = exactly one lesson. Do NOT invent, split, or merge lessons. If I give you 4 video transcriptions, the curriculum must contain exactly 4 lessons total — no more, no less.

Other rules:
- **Write your own lesson title** — DO NOT copy the video title from the transcription header. Read what the lesson is actually about and craft a clear, compelling, course-appropriate title (3-8 words). The title should describe the practice/skill/topic, not the YouTube video brand. Example: instead of "10 Minute Heart Coherence Breathwork I The Perfect Breath" write "Heart Coherence Breathing for Calm Focus".
- **Write your own section titles** — DO NOT just group videos by topic name. Create section titles that describe the learning stage or theme (e.g. "Foundations of Breathwork", "Calming the Nervous System", "Advanced Practices"). Sections should feel like chapters of a structured course.
- Group the lessons into logical sections, but do NOT add extra lessons that don't correspond to a real video.
- Each lesson MUST have a title and description based on the actual transcription content (so the title is informed by what the lesson teaches, not the original YouTube title).
- If a section is bonus content, add `[BONUS]` to the section title: `---section: Bonus Materials [BONUS]---`

**===TESTIMONIALS===**
Fake but realistic student reviews for this course. Use this format:

```
---review---
name: Student Name
text: Their review text (can be multiline).
The review should feel authentic and specific.
rating: 5
```

Rules:
- Generate 5-8 testimonials
- Each review MUST have: `name:` (required), `text:` (required, multiline ok), `rating:` (always 5)
- Use diverse, realistic first+last names
- Make reviews specific to the course content — mention particular lessons, techniques, or outcomes
- Vary review length (1-4 sentences) and tone (enthusiastic, thoughtful, grateful, practical)
- Do NOT use generic phrases like "great course" — be specific about what the student learned or how it helped them

**===COLLECTION===**
- `name:` — Choose the most appropriate collection (category) from the list below, or suggest a new one if none fit.

**Existing collections:**
  - Fitness & Health
  - Mindfulness
  - Dance & Movement
  - Creativity & Arts
  - Relationships & Intimacy
  - New
  - My new collection
  - Men's Sexual Health

If none of the existing collections fit, write a new collection name that best describes this course's category. Keep it short (1-3 words).

---

## OUTPUT FORMAT

Return ONLY the filled template — no extra text, no explanations, no markdown code fences around it. Start directly with `===AUTHOR===` and end after `===COLLECTION===`."""


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


# Public aliases — Phase 2 calls these to parse operator-edited Sheet cells
parse_plan_from_sheet = _parse_plan
parse_science_from_sheet = _parse_science


def format_plan_for_sheet(plan_sections: list[dict[str, Any]] | None) -> str:
    """Inverse of _parse_plan — render plan_sections back to delimiter format.

    Output the operator sees / edits in Sheet:
        ---section: Foundations---
        - Anatomy of the knee
        - Common surgical approaches
        ---section: Recovery---
        - Week 1-2 protocols
    """
    if not plan_sections:
        return ""
    lines: list[str] = []
    for sec in plan_sections:
        title = (sec.get("title") or "").strip()
        if not title:
            continue
        lines.append(f"---section: {title}---")
        for item in (sec.get("items") or []):
            t = (item.get("title") or "").strip()
            if t:
                lines.append(f"- {t}")
    return "\n".join(lines)


def format_science_for_sheet(science_plan: dict[str, Any] | None) -> str:
    """Inverse of _parse_science — render sciencePlan back to delimiter format.

    Output:
        headline: ...
        subtitle: ...
        ---institution---
        name: Harvard Medical School
        style: serif-caps
        ---stat---
        value: 74%
        description: ...
        citation: ...
    """
    if not science_plan or not science_plan.get("enabled", True):
        return ""
    lines: list[str] = []
    headline = (science_plan.get("headline") or "").strip()
    subtitle = (science_plan.get("subtitle") or "").strip()
    if headline:
        lines.append(f"headline: {headline}")
    if subtitle:
        lines.append(f"subtitle: {subtitle}")
    for inst in (science_plan.get("institutions") or []):
        lines.append("---institution---")
        lines.append(f"name: {(inst.get('name') or '').strip()}")
        lines.append(f"style: {(inst.get('style') or 'serif').strip()}")
    for stat in (science_plan.get("stats") or []):
        lines.append("---stat---")
        lines.append(f"value: {(stat.get('value') or '').strip()}")
        lines.append(f"description: {(stat.get('description') or '').strip()}")
        lines.append(f"citation: {(stat.get('citation') or '').strip()}")
    return "\n".join(lines)


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
                        timeout: int = 600,
                        output_lang: str = "en",
                        pain: str = "", audience: str = "") -> dict[str, Any]:
    """Generate full course content via template + parser. Returns structured dict.

    `pain` / `audience`: when provided, the course title, excerpt, about, plan,
    and curriculum should all foreground how the course solves this pain for
    this audience (instead of producing a generic course on the topic).
    """
    # Trim each transcript to ~700 words to stay within budget but keep enough context
    trimmed_lessons = []
    for i, t in enumerate(lesson_transcripts, start=1):
        words = (t or "").split()
        trimmed_lessons.append(f"--- Video {i} transcript ---\n" + " ".join(words[:700]))

    pain_audience_lines = ""
    if pain:
        pain_audience_lines += f"TARGET PAIN: {pain}\n"
    if audience:
        pain_audience_lines += f"TARGET AUDIENCE: {audience}\n"

    # When no title is provided (URL mode), tell the LLM to generate one from
    # the lesson transcripts instead of fabricating a forced "COURSE TITLE: "
    # blank line that confuses generation.
    if course_title.strip():
        title_line = f"COURSE TITLE (use exactly if you keep one): {course_title}\n"
    else:
        title_line = (
            "COURSE TITLE: (NOT PROVIDED — generate one from the lesson "
            "transcripts below: clear, outcome-oriented, 3-10 words)\n"
        )

    materials = (
        f"COURSE TOPIC: {course_topic}\n"
        f"{title_line}"
        f"{pain_audience_lines}"
        f"INSTRUCTOR / CHANNEL: {channel_name}\n"
        f"CHANNEL DESCRIPTION: {channel_description[:500] if channel_description else '(none)'}\n\n"
        f"NUMBER OF VIDEO LESSONS: {len(lesson_transcripts)} "
        f"(curriculum MUST contain exactly this many lessons in this order)\n\n"
        + "\n\n".join(trimmed_lessons)
    )

    system = (COMPOSE_FULL_SYSTEM
              + _pain_audience_block(pain, audience)
              + _lang_instruction(output_lang))
    full_prompt = system + "\n\n---\n\n## MATERIALS\n\n" + materials

    # Same retry-on-503/timeout machinery as _call_json, but we expect plain
    # text (the template), not JSON.
    import os, subprocess, tempfile
    from pathlib import Path

    last_err: str = ""
    r = None
    for attempt, backoff in enumerate([0] + list(_CLI_RETRY_BACKOFFS)):
        if backoff:
            log.warning(f"compose_full_course retry {attempt} after {backoff}s "
                        f"(prev error: {last_err[:200]})")
            time.sleep(backoff)
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
            except subprocess.TimeoutExpired:
                last_err = f"compose_full_course timed out after {timeout}s"
                continue
        if r is not None and r.returncode == 0 and (r.stdout or "").strip():
            break
        last_err = (f"exit {r.returncode if r else '?'}: "
                    f"stderr={(r.stderr if r else '')[:300]} "
                    f"stdout={(r.stdout if r else '')[:200]}")
        if r is None or not _is_retryable_cli_failure(r.stderr or "", r.stdout or ""):
            raise RuntimeError(f"compose_full_course: {last_err}")
    else:
        raise RuntimeError(f"compose_full_course exhausted retries: {last_err}")

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
