"""
Test of proposed 4-step Phase 1 redesign pipeline.

Does NOT touch prod:
- no Sheet writes
- no gateway state
- no Phase 2 triggers
- no config changes

Just demonstrates: Claude discovery → yt-dlp filter → yt-dlp enumerate → Claude select.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

CONFIG_PATH = "/home/jarvis/projects/jarvis-telegram-gateway/config.json"

cfg = json.load(open(CONFIG_PATH))
ONB = cfg["agents"]["operations"]["onboarder"]
PROXIES = ONB["youtube_proxies"]
YTDLP_CMD = ["/home/jarvis/projects/jarvis-telegram-gateway/venv/bin/python", "-m", "yt_dlp"]

# Rotate through proxies for resilience
_proxy_idx = 0
def next_proxy() -> str:
    global _proxy_idx
    p = PROXIES[_proxy_idx % len(PROXIES)]
    _proxy_idx += 1
    return p


# Cap on Claude calls — for safety during testing
CLAUDE_TIMEOUT_DISCOVERY = 1500
CLAUDE_TIMEOUT_SELECT = 600


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def claude_call(prompt: str, model: str = "sonnet", effort: str = "high",
                timeout: int = 600) -> str:
    """Run claude -p with the given prompt, return stdout."""
    cmd = ["claude", "-p",
           "--model", model,
           "--effort", effort,
           "--output-format", "text",
           "--permission-mode", "bypassPermissions"]
    log(f"  claude {model} effort={effort} (~{timeout}s budget)")
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"claude exit {r.returncode}: stderr={r.stderr[:200]}")
    return r.stdout.strip()


def extract_json(text: str) -> dict:
    """Strip markdown fences and find first JSON object. Tolerates trailing prose."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    else:
        text = text.strip()
        if not text.startswith("{") and not text.startswith("["):
            idx = min(
                (i for i in (text.find("{"), text.find("[")) if i >= 0),
                default=-1,
            )
            if idx >= 0:
                text = text[idx:]
    # raw_decode tolerates trailing prose after the JSON object
    obj, _ = json.JSONDecoder().raw_decode(text)
    return obj


# ---------------------------------------------------------------------------
# Step 1 — Claude DISCOVERY: only channel handles
# ---------------------------------------------------------------------------

DISCOVERY_PROMPT_TEMPLATE = """You are a YouTube niche channel scout.

=== TOPIC ===
{topic}

=== AUDIENCE ===
{audience}

=== YOUR TASK ===
Find {count} candidate YouTube channels that COULD work for a self-help course on this topic.

You ONLY need to return CHANNEL HANDLES or channel URLs. You do NOT need to verify size, videos, or details — a downstream pipeline will validate everything.

CRITICAL: We want SMALL niche channels (target: 500-70,000 subscribers). NOT mainstream.
- Skip Bob and Brad, Jeremy Ethier, Athlean-X, Yoga With Adriene, Adam Fields DC, anyone you know from general wellness knowledge
- Skip channels appearing on 'best YouTube channel' lists
- Skip commercial brands (Mandible Coach, Faceology, Vivos, Glowinface, Mouth-tape brands)
- Target solo practitioners, niche clinicians, hidden gems

We will VERIFY actual subscriber count after you return — don't worry if you can't estimate exactly. Just give us your best {count} guesses based on search.

{exclusion_block}

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
- Do NOT propose any channel from the EXCLUSION list above

Return ONLY the JSON.
"""


def step1_discovery(topic: str, audience: str, count: int = 15,
                    exclude: list[str] | None = None) -> dict:
    log(f"STEP 1: Claude discovery — looking for {count} channels"
        + (f" (excluding {len(exclude)} already-tried)" if exclude else ""))
    exclusion_block = ""
    if exclude:
        excl_lines = "\n".join(f"- {name}" for name in exclude[:200])
        exclusion_block = (
            "=== EXCLUSION LIST — DO NOT PROPOSE THESE ===\n"
            "We have already tried these and they did not pass our pipeline filters "
            "(too big, too small, no valid videos, etc.). Find DIFFERENT ones.\n\n"
            f"{excl_lines}\n"
        )
    prompt = DISCOVERY_PROMPT_TEMPLATE.format(
        topic=topic, audience=audience, count=count,
        exclusion_block=exclusion_block,
    )
    t0 = time.time()
    raw = claude_call(prompt, model="opus", effort="high",
                      timeout=CLAUDE_TIMEOUT_DISCOVERY)
    dt = time.time() - t0
    log(f"  → done in {dt:.0f}s, {len(raw)} bytes")
    try:
        data = extract_json(raw)
    except Exception as e:
        log(f"  PARSE ERR: {e}; raw[:300]={raw[:300]!r}")
        raise
    cands = data.get("candidates", [])
    log(f"  → {len(cands)} candidates")
    for c in cands[:20]:
        log(f"     • {c.get('channel_name', '?')}  ({c.get('handle_or_url', '?')})")
    if data.get("notes"):
        log(f"  notes: {data['notes'][:200]}")
    return data


# ---------------------------------------------------------------------------
# Step 2 — yt-dlp channel size filter
# ---------------------------------------------------------------------------

def normalize_channel_url(handle_or_url: str) -> str:
    s = handle_or_url.strip()
    if s.startswith("http"):
        return s
    if s.startswith("@"):
        return f"https://www.youtube.com/{s}"
    if s.startswith("/channel/") or s.startswith("channel/"):
        s = s.lstrip("/")
        return f"https://www.youtube.com/{s}"
    if re.match(r"^UC[A-Za-z0-9_-]{22}$", s):
        return f"https://www.youtube.com/channel/{s}"
    return f"https://www.youtube.com/@{s.lstrip('@')}"


def _ytdlp_channel_info(url: str, max_retries: int = 3) -> tuple[str, int, str] | None:
    last_err = ""
    for attempt in range(max_retries):
        try:
            r = subprocess.run(
                YTDLP_CMD + [
                    "--print", "%(channel_id)s|%(channel_follower_count)s|%(channel)s",
                    "--playlist-items", "1",
                    "--proxy", next_proxy(),
                    url,
                ],
                capture_output=True, text=True, timeout=60,
            )
            line = (r.stdout or "").strip().split("\n")[-1]
            parts = line.split("|", 2)
            if len(parts) == 3 and parts[0]:
                cid, fcount_str, cname = parts
                fc = int(fcount_str) if fcount_str.isdigit() else 0
                return cid, fc, cname
            last_err = f"bad_output"
        except subprocess.TimeoutExpired:
            last_err = "timeout"
        except Exception as e:
            last_err = f"{type(e).__name__}"
    return None


def step2_filter_size(candidates: list[dict], min_subs: int = 500,
                       max_subs: int = 70_000,
                       seen_channel_ids: set[str] | None = None) -> list[dict]:
    log(f"STEP 2: yt-dlp size filter ({min_subs}-{max_subs} subs) — 3 proxy retries")
    seen = seen_channel_ids or set()
    kept = []
    for c in candidates:
        url = normalize_channel_url(c["handle_or_url"])
        info = _ytdlp_channel_info(url, max_retries=3)
        if info is None:
            log(f"  ERR  {c['channel_name'][:30]:30s} (3 proxies failed)")
            continue
        cid, fc, cname = info
        if cid in seen:
            log(f"  SKIP @{c['channel_name'][:30]:30s} {fc:>7} subs  (already in Sheet)")
            continue
        if min_subs <= fc <= max_subs:
            log(f"  KEEP @{c['channel_name'][:30]:30s} {fc:>7} subs  cid={cid}")
            kept.append({
                "channel_name_claude": c["channel_name"],
                "channel_name_real": cname,
                "channel_id": cid,
                "subscribers": fc,
                "url": url,
            })
        else:
            reason = "too small" if fc < min_subs else "too big"
            log(f"  DROP @{c['channel_name'][:30]:30s} {fc:>7} subs ({reason})")
    log(f"STEP 2 result: {len(kept)}/{len(candidates)} channels passed")
    return kept


# ---------------------------------------------------------------------------
# Step 3 — yt-dlp video enumeration + filter
# ---------------------------------------------------------------------------

def step3_enumerate_videos(channels: list[dict], min_dur_sec: int = 300,
                            max_dur_sec: int = 1800, min_year: int = 2016,
                            max_videos_per_channel: int = 30,
                            seen_video_ids: set[str] | None = None) -> list[dict]:
    log(f"STEP 3: enumerate + filter videos "
        f"(dur {min_dur_sec//60}-{max_dur_sec//60}min, year≥{min_year}, "
        f"top {max_videos_per_channel} per channel)")
    seen_vids = seen_video_ids or set()
    out = []
    for ch in channels:
        # Use canonical channel-id URL — handle/c URLs sometimes fail to resolve
        url = f"https://www.youtube.com/channel/{ch['channel_id']}/videos"
        log(f"  pulling /videos for @{ch['channel_name_real']} ({ch['subscribers']} subs)")
        try:
            # NO --flat-playlist: we need full metadata (duration, upload_date).
            # Limit to top N recent videos to keep test reasonable.
            r = subprocess.run(
                YTDLP_CMD + [
                    "--playlist-end", str(max_videos_per_channel),
                    "--print", "%(id)s|%(title)s|%(duration)s|%(upload_date)s|%(view_count)s",
                    "--proxy", next_proxy(),
                    url,
                ],
                capture_output=True, text=True, timeout=600,
            )
            lines = (r.stdout or "").strip().split("\n")
            valid_videos = []
            total_seen = 0
            dup_count = 0
            for line in lines:
                if "|" not in line:
                    continue
                parts = line.split("|", 4)
                if len(parts) < 5:
                    continue
                vid, title, dur_str, date_str, views_str = parts
                total_seen += 1
                try:
                    dur = int(dur_str) if dur_str.isdigit() else 0
                    year = int(date_str[:4]) if date_str and date_str[:4].isdigit() else 0
                    views = int(views_str) if views_str.isdigit() else 0
                except (ValueError, IndexError):
                    continue
                if vid in seen_vids:
                    dup_count += 1
                    continue
                if dur < min_dur_sec or dur > max_dur_sec:
                    continue
                if year < min_year:
                    continue
                valid_videos.append({
                    "video_id": vid,
                    "title": title,
                    "duration_sec": dur,
                    "upload_year": year,
                    "view_count": views,
                })
            extra = f" ({dup_count} already in Sheet)" if dup_count else ""
            log(f"    {len(valid_videos)}/{total_seen} videos passed filters{extra}")
            out.append({**ch, "videos": valid_videos, "total_videos_seen": total_seen})
        except subprocess.TimeoutExpired:
            log(f"    TIMEOUT pulling videos for @{ch['channel_name_real']}")
            out.append({**ch, "videos": [], "total_videos_seen": 0})
        except Exception as e:
            log(f"    ERR @{ch['channel_name_real']}: {type(e).__name__} {e}")
            out.append({**ch, "videos": [], "total_videos_seen": 0})
    return out


# ---------------------------------------------------------------------------
# Step 4 — Claude selects best videos PER channel from REAL list
# ---------------------------------------------------------------------------

SELECT_PROMPT_TEMPLATE = """You are a curriculum designer building a course on:

TOPIC: {topic}
AUDIENCE: {audience}

Below is the FULL list of valid videos from one YouTube channel (already filtered: 6-25 min, 2016+, public).
Your job: pick the BEST videos for a coherent learning progression. Order them pedagogically (foundations → core → progression).

CHANNEL: {channel_name} ({subscribers} subs)

VIDEOS (ground truth from yt-dlp — these are real, no fabrication needed):
{video_list}

TARGET COURSE SIZE:
- IDEAL: 12-15 videos
- ACCEPTABLE: 6-30 videos
- MINIMUM: 6 videos. If you cannot find at least 6 topically-relevant videos, this channel is NOT a good fit — return an empty selected_video_ids list with confidence='low' and a clear note. Don't pretend a course exists with 3-5 stretched videos.

When in doubt about a video's fit, INCLUDE it if it's adjacent to the topic — we'd rather have a 12-video course with some loose fits than a 5-video course of perfect fits. The minimum 6 is a HARD rule.

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
- selected_video_ids must be 0 or 6-30 ids — NEVER 1-5
- DO NOT invent ids — pick only from the list
- If channel does not have 6 topically-fitting videos → return [] with confidence='low' and notes explaining why

Return ONLY the JSON.
"""


def step4_select(channels: list[dict], topic: str, audience: str) -> list[dict]:
    log(f"STEP 4: Claude per-channel selection")
    results = []
    for ch in channels:
        if len(ch["videos"]) < 6:
            log(f"  SKIP @{ch['channel_name_real']}: only {len(ch['videos'])} valid videos")
            continue
        video_list_str = "\n".join(
            f"- id={v['video_id']} dur={v['duration_sec']}s year={v['upload_year']} "
            f"views={v['view_count']} | {v['title'][:80]}"
            for v in ch["videos"]
        )
        prompt = SELECT_PROMPT_TEMPLATE.format(
            topic=topic, audience=audience,
            channel_name=ch["channel_name_real"],
            subscribers=ch["subscribers"],
            video_list=video_list_str,
        )
        log(f"  selecting for @{ch['channel_name_real']} (from {len(ch['videos'])} candidates)")
        try:
            raw = claude_call(prompt, model="sonnet", effort="medium",
                              timeout=CLAUDE_TIMEOUT_SELECT)
            data = extract_json(raw)
            sel_ids = data.get("selected_video_ids", [])
            valid_ids = {v["video_id"] for v in ch["videos"]}
            # dedup + keep only ids that exist in our ground-truth list
            seen: set = set()
            confirmed: list = []
            for vid in sel_ids:
                if vid in valid_ids and vid not in seen:
                    confirmed.append(vid)
                    seen.add(vid)
            dups = len(sel_ids) - len(confirmed)
            if dups:
                log(f"    WARN dropped {dups} duplicate/fabricated ids")
            log(f"    selected {len(confirmed)} videos: {data.get('course_title_en', '?')[:60]}")
            results.append({
                **ch,
                "selected_ids": confirmed,
                "course_title": data.get("course_title_en", ""),
                "rationale": data.get("course_rationale", ""),
                "claude_confidence": data.get("confidence", "?"),
            })
        except Exception as e:
            log(f"    ERR Claude select: {type(e).__name__} {e}")
            continue
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TOPIC = """Root-cause TMJ relief — jaw pain, face alignment, headache relief,
tongue posture, neck release, breathing correction, and facial tension/asymmetry.
Pain points: jaw pain, jaw tightness, clicking, popping, teeth grinding, clenching,
tension headaches, migraines, face asymmetry, neck and shoulder tension, mouth
breathing, poor sleep. Course content: myofunctional exercises, osteopathic jaw
release, tongue posture training, jaw mobility, neck release, breathing correction,
exercises for facial tension and asymmetry. NOT just jaw massage — a root-cause
systemic approach combining tongue, jaw muscles, neck, breathing and tension habits.

Language: English.
Outcome: less tension, better jaw mobility, fewer headaches, more relaxed and
symmetric face."""

AUDIENCE = """People who wake up with jaw pain, clench their teeth, wear mouth guards,
suffer from clicking, headaches, or facial tension. They want self-help exercises
they can do alone at home — NOT clinical material for osteopaths/physiotherapists."""


TARGET_COURSES = 5
MAX_ITERATIONS = 3
MIN_DUR_SEC = 360   # 6 minutes
MAX_DUR_SEC = 1500  # 25 minutes
MIN_YEAR = 2016
MAX_VIDEOS_PER_CHANNEL = 80
DISCOVERY_BATCH_SIZE = 30  # was 15 — give Claude more room to find small channels


def _load_sheet_dedup() -> tuple[set[str], set[str], set[str]]:
    """Return (seen_video_ids, seen_channel_ids, seen_channel_names) from Sheet."""
    sys.path.insert(0, "/home/jarvis/projects/jarvis-telegram-gateway")
    from onboarder import sheets  # type: ignore
    sa = ONB["google_service_account_file"].replace("~", str(Path.home()))
    client = sheets.open_client(sa)
    seen_video_ids = sheets.get_active_video_ids(client, ONB["google_sheet_id"])
    seen_channel_ids = sheets.get_seen_channel_ids(client, ONB["google_sheet_id"])
    # Also extract channel NAMES so Claude can be told "don't propose these"
    ws = sheets.ensure_lessons_tab(client, ONB["google_sheet_id"])
    rows = ws.get_all_values()
    header = rows[0]
    ri = {h: i for i, h in enumerate(header)}
    seen_channel_names: set[str] = set()
    for row in rows[1:]:
        name = row[ri["channel"]] if ri["channel"] < len(row) else ""
        if name.strip():
            seen_channel_names.add(name.strip())
    return seen_video_ids, seen_channel_ids, seen_channel_names


def main() -> None:
    log("=" * 70)
    log("TEST PIPELINE START")
    log("=" * 70)
    log(f"TOPIC: {TOPIC.strip()[:80]}...")
    log(f"AUDIENCE: {AUDIENCE.strip()[:80]}...")
    log(f"Target: {TARGET_COURSES} courses, up to {MAX_ITERATIONS} discovery iterations")
    log("")

    # ── Load dedup state from Sheet (channels/videos already used in any prior run)
    log("Loading dedup state from Sheet...")
    try:
        seen_video_ids, seen_channel_ids, seen_channel_names = _load_sheet_dedup()
        log(f"  seen_video_ids={len(seen_video_ids)} "
            f"seen_channel_ids={len(seen_channel_ids)} "
            f"seen_channel_names={len(seen_channel_names)}")
    except Exception as e:
        log(f"  WARN: dedup load failed ({e}); proceeding without Sheet dedup")
        seen_video_ids = set()
        seen_channel_ids = set()
        seen_channel_names = set()
    log("")

    accumulated_courses: list[dict] = []
    # Seed exclusion list with channel names already in Sheet
    excluded_names: list[str] = sorted(seen_channel_names)
    iteration_history: list[dict] = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        log("─" * 70)
        log(f"ITERATION {iteration} / {MAX_ITERATIONS}  "
            f"(have {len(accumulated_courses)}/{TARGET_COURSES} courses)")
        log("─" * 70)

        # ── Step 1 (with exclusion list of already-tried names)
        try:
            discovery_data = step1_discovery(
                TOPIC, AUDIENCE,
                count=DISCOVERY_BATCH_SIZE,
                exclude=excluded_names if excluded_names else None,
            )
        except Exception as e:
            log(f"  Step 1 failed: {e}")
            break
        candidates = discovery_data.get("candidates", [])
        if not candidates:
            log(f"  no candidates returned — stopping loop")
            break
        log("")

        # ── Step 2 (with Sheet channel_id dedup)
        surviving = step2_filter_size(
            candidates, min_subs=500, max_subs=70_000,
            seen_channel_ids=seen_channel_ids,
        )
        log("")

        # Track every candidate name (whether kept or dropped) for next iteration's exclusion
        for c in candidates:
            n = c.get("channel_name", "").strip()
            if n and n not in excluded_names:
                excluded_names.append(n)

        if not surviving:
            log(f"  iteration {iteration}: 0 channels survived size filter")
            iteration_history.append({
                "iteration": iteration,
                "candidates": len(candidates),
                "surviving_size": 0,
                "courses_added": 0,
            })
            continue

        # ── Step 3 (with Sheet video_id dedup)
        enriched = step3_enumerate_videos(
            surviving,
            min_dur_sec=MIN_DUR_SEC,
            max_dur_sec=MAX_DUR_SEC,
            min_year=MIN_YEAR,
            max_videos_per_channel=MAX_VIDEOS_PER_CHANNEL,
            seen_video_ids=seen_video_ids,
        )
        # As we accept new course videos, add them to seen_video_ids so that
        # subsequent iterations within this run don't re-pick the same videos.
        log("")

        # ── Step 4
        raw_selected = step4_select(enriched, TOPIC, AUDIENCE)
        # Enforce hard 6-video minimum at the pipeline level (independent of
        # Claude's prompt compliance). Anything <6 → not a real course.
        kept_courses = [s for s in raw_selected if len(s["selected_ids"]) >= 6]
        dropped_below_min = [s for s in raw_selected if len(s["selected_ids"]) < 6]
        for d in dropped_below_min:
            log(f"  DROP course from @{d['channel_name_real']}: "
                f"only {len(d['selected_ids'])} videos selected (< 6 minimum)")
        accumulated_courses.extend(kept_courses)
        # Add channel_ids and chosen video_ids from VALID courses to dedup sets
        for s in kept_courses:
            seen_channel_ids.add(s["channel_id"])
            for vid in s["selected_ids"]:
                seen_video_ids.add(vid)
        # Add channel_ids from DROPPED courses to exclusion as well so Claude
        # doesn't re-propose them in the next iteration
        for d in dropped_below_min:
            seen_channel_ids.add(d["channel_id"])
            if d["channel_name_real"] not in excluded_names:
                excluded_names.append(d["channel_name_real"])
        # Also add channels that passed step 2 but had 0 valid videos in step 3
        # (they're already in seen but log them as known dead-ends)
        for e in enriched:
            if not e["videos"]:
                seen_channel_ids.add(e["channel_id"])
                if e["channel_name_real"] not in excluded_names:
                    excluded_names.append(e["channel_name_real"])

        iteration_history.append({
            "iteration": iteration,
            "candidates": len(candidates),
            "surviving_size": len(surviving),
            "with_videos": sum(1 for e in enriched if e["videos"]),
            "courses_kept": len(kept_courses),
            "courses_dropped_below_min": len(dropped_below_min),
        })
        log("")
        log(f"Iteration {iteration}: {len(kept_courses)} valid courses kept "
            f"({len(dropped_below_min)} dropped <6 videos). "
            f"Total: {len(accumulated_courses)}/{TARGET_COURSES}")
        log("")

        if len(accumulated_courses) >= TARGET_COURSES:
            log(f"✓ Reached target of {TARGET_COURSES} courses — stopping loop")
            break

    # ── Final report
    log("=" * 70)
    log("FINAL RESULT")
    log("=" * 70)
    log(f"Iterations run: {len(iteration_history)}")
    for h in iteration_history:
        log(f"  iter {h['iteration']}: {h['candidates']} cands → "
            f"{h['surviving_size']} passed size → "
            f"{h.get('with_videos', '?')} had videos → "
            f"{h.get('courses_kept', '?')} courses kept "
            f"({h.get('courses_dropped_below_min', 0)} dropped <6)")
    log("")
    log(f"Courses ready: {len(accumulated_courses)} (target {TARGET_COURSES})")

    for s in accumulated_courses[:TARGET_COURSES]:
        log("")
        log(f"  COURSE: {s['course_title']}")
        log(f"    channel: @{s['channel_name_real']} ({s['subscribers']} subs)")
        log(f"    videos: {len(s['selected_ids'])}")
        log(f"    confidence: {s['claude_confidence']}")
        log(f"    rationale: {s['rationale'][:200]}")
        for vid in s["selected_ids"]:
            vinfo = next((v for v in s["videos"] if v["video_id"] == vid), None)
            if vinfo:
                log(f"      • https://youtube.com/watch?v={vid}  "
                    f"({vinfo['duration_sec']//60}min, {vinfo['upload_year']})  "
                    f"{vinfo['title'][:60]}")

    # Save raw for inspection
    save_path = "/tmp/test_pipeline_result.json"
    with open(save_path, "w") as f:
        json.dump({
            "iteration_history": iteration_history,
            "excluded_names_final": excluded_names,
            "final_courses": accumulated_courses,
        }, f, indent=2, default=str)
    log(f"\nFull result saved to: {save_path}")


if __name__ == "__main__":
    main()
