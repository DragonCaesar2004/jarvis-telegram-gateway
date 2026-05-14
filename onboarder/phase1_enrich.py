"""Phase 1 enrichment: download → transcribe → cut markers → describe → research author → compose course.

Heavy work that used to live in Phase 2 has moved here so the operator sees
real, landing-page-ready descriptions in the Lessons sheet BEFORE approving
videos for production. After this step the Sheet contains:

    - Per-lesson description (3-5 sentences, real)
    - Course-level description, tagline, target audience, "what you learn"
    - Researched author bio + name + expertise (deep research via Claude WebSearch)
    - Transcript excerpt for review

What's persisted to pipeline.db (so Phase 2 doesn't re-run Whisper / Claude):
    video_cuts(video_id)        — cut timecodes + working transcript + detected lang
    course_compose(run_id, idx) — full curriculum/plan/science/testimonials payload

Parallelism:
    Within a course: up to N videos download+transcribe at the same time.
    Across courses: handled by the caller (phase1_discovery loops sequentially today).

Failure handling:
    - One video failing (download / Whisper / mark_cuts) does NOT abort the course.
      The video is dropped from the curriculum; Telegram message reports it.
    - All videos in course failing → enrich returns {"videos": [], ...}, caller
      decides whether to skip the course entirely.
    - Author research / compose failing → fall back to lightweight stubs so the
      run still produces a Sheet (operator can edit by hand).
    - Proxy pool exhaustion (CookiesNeededError) is propagated up — the wizard
      pauses Phase 1 and asks the user to upload fresh cookies, same UX as Phase 2.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from . import cache, ffmpeg_cut, llm, pipeline_db, proxy_pool, whisper
from .elevenlabs_dub import _iso as _lang_iso
from .proxy_pool import CookiesNeededError, ProxyRotator

log = logging.getLogger("gateway")

DEFAULT_PARALLEL_PER_COURSE = 4
TRANSCRIPT_EXCERPT_CHARS = 500


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_course(*, course_idx: int, run_id: str,
                  channel_id: str, channel_name: str,
                  channel_description: str,
                  course_topic_input: str,
                  course_title_from_llm: str,
                  selected_videos: list[dict[str, Any]],
                  videos_metadata: dict[str, dict[str, Any]],
                  openai_key: str,
                  cookies_file: str | None,
                  rotator: ProxyRotator | None,
                  on_progress: Callable[[str], None] | None = None,
                  max_parallel: int = DEFAULT_PARALLEL_PER_COURSE,
                  compose_model: str = llm.DEFAULT_MODEL_QUALITY,
                  pain: str = "",
                  audience: str = "",
                  ) -> dict[str, Any]:
    """Heavy lift: download, transcribe, mark cuts, describe, research author, compose.

    `selected_videos`: items from llm.select_videos lessons, shape:
        {video_id: str, title: str, order: int, reason: str}
    `videos_metadata`: video_id → {title, url, duration_sec, view_count, ...}
        (filled by phase1_discovery from the channel listing).

    Returns:
        {
          "videos": [
            {
              "video_id", "title", "url", "duration_sec", "lesson_idx",
              "transcript_excerpt", "lesson_description",
              "detected_lang", "cuts_count",
            },
            ...
          ],
          "course_title": str,         # may be overridden by compose
          "course_description": str,   # course excerpt
          "course_tagline": str,
          "course_what_you_learn": str,
          "course_target_audience": str,
          "author_name": str,
          "author_bio": str,
          "author_expertise": str,
          "compose_ok": bool,          # False if compose failed and we used stubs
        }
    """

    def _emit(msg: str) -> None:
        if on_progress:
            try:
                on_progress(msg)
            except Exception:
                pass
        log.info(f"phase1_enrich[course={course_idx}] {msg}")

    # ── 1. Parallel download + transcribe + mark cuts ────────────────────
    _emit(f"⬇️  Скачиваю и транскрибирую {len(selected_videos)} видео "
          f"(до {max_parallel} параллельно)…")

    processed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    progress_lock = threading.Lock()

    def _on_video_done(result: dict[str, Any]) -> None:
        with progress_lock:
            if result.get("ok"):
                processed.append(result)
                _emit(f"  ✓ {result['title'][:50]}  ({result.get('detected_lang') or '?'}, "
                      f"{result.get('cuts_count', 0)} вырезок)")
            else:
                failed.append(result)
                _emit(f"  ✗ {result['title'][:50]}: {result.get('error', '?')[:120]}")

    with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as ex:
        futures = []
        for v in selected_videos:
            vid = v.get("video_id")
            if not vid:
                continue
            meta = videos_metadata.get(vid) or {}
            futures.append(ex.submit(
                _process_video_for_enrich,
                video_id=vid,
                title=v.get("title") or meta.get("title", ""),
                url=meta.get("url") or f"https://youtu.be/{vid}",
                course_topic=course_title_from_llm or course_topic_input,
                cookies_file=cookies_file,
                rotator=rotator,
                openai_key=openai_key,
            ))
        cookies_needed: CookiesNeededError | None = None
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except CookiesNeededError as e:
                cookies_needed = e
                # Cancel remaining futures — pool exhausted
                for f in futures:
                    f.cancel()
                continue
            except Exception as e:
                # Bubbled-up unexpected error inside the worker — convert to a
                # row in `failed` so the rest of the course can finish.
                log.error(f"phase1_enrich: worker crashed: {e}", exc_info=True)
                continue
            _on_video_done(result)

    if cookies_needed is not None:
        # Propagate so phase1_discovery pauses and asks for cookies refresh
        raise cookies_needed

    # Sort by `order` from the LLM's curriculum sequence
    selected_order: dict[str, int] = {
        v["video_id"]: int(v.get("order", i)) for i, v in enumerate(selected_videos)
    }
    processed.sort(key=lambda r: selected_order.get(r["video_id"], 99))

    if not processed:
        _emit(f"⚠️ Курс {course_idx}: ни одно видео не транскрибировалось. "
              f"Сдаюсь по этому курсу.")
        return {
            "videos": [],
            "course_title": course_title_from_llm,
            "course_description": "",
            "course_tagline": "",
            "course_what_you_learn": "",
            "course_target_audience": "",
            "author_name": channel_name,
            "author_bio": "",
            "author_expertise": "",
            "compose_ok": False,
        }

    # Admin payload is ALWAYS English (source-of-truth on the platform).
    # If the operator wants to review in Russian, the *_ru columns of the
    # Lessons sheet are populated separately via translate_batch_to_russian
    # below. The operator's input language doesn't affect output language —
    # all landing copy, descriptions, plan, science, etc. ship in EN.
    output_lang = "en"
    _emit("🌐 Язык вывода: en (всегда; RU перевод — в *_ru колонках для ревью)")

    # Lesson descriptions will be extracted from compose_full_course's curriculum
    # (avoids a redundant separate Claude call — compose generates them as part of
    # the full course structure, based on the same transcripts, at higher quality).
    lesson_descriptions: dict[str, str] = {}

    # ── 2. Author research (deep, with WebSearch) — cached per channel_id ──
    # Same channel → same author. Bio rarely changes, so we cache forever in
    # pipeline_db.author_research. Saves 30-90s of Claude+WebSearch on every
    # repeat run of the same channel.
    cached_author = pipeline_db.get_author_research(channel_id)
    if cached_author and cached_author.get("bio"):
        author = cached_author
        _emit(f"  ✓ author из кэша: {author.get('name')} "
              f"(cached @ {author.get('cached_at', '')[:10]})")
    else:
        _emit(f"🔍 Ищу информацию об авторе «{channel_name}» (Claude WebSearch, {output_lang})…")
        author = llm.research_author(
            channel_name=channel_name,
            channel_description=channel_description,
            sample_video_titles=[p["title"] for p in processed[:8]],
            course_topic=course_title_from_llm or course_topic_input,
            output_lang=output_lang,
            pain=pain, audience=audience,
        )
        _emit(f"  ✓ author: {author.get('name')} (confidence={author.get('confidence')})")
        # Save to cache for future runs of the same channel
        try:
            pipeline_db.save_author_research(
                channel_id,
                name=author.get("name", ""),
                bio=author.get("bio", ""),
                expertise=author.get("expertise", ""),
                confidence=author.get("confidence", "low"),
                sources=author.get("sources", []),
            )
        except Exception as e:
            log.warning(f"phase1_enrich: save_author_research failed: {e}")

    # ── 3. Full course compose (curriculum/plan/science/testimonials/...) ─
    _emit(f"✍️  Собираю полное описание курса (Claude, {output_lang})…")
    composed_full: dict[str, Any] | None = None
    compose_ok = False
    try:
        composed_full = llm.compose_full_course(
            course_topic=course_title_from_llm or course_topic_input,
            course_title=course_title_from_llm,
            channel_name=channel_name,
            channel_description=channel_description,
            lesson_transcripts=[p["working_transcript"] for p in processed],
            model=compose_model,
            output_lang=output_lang,
            pain=pain, audience=audience,
        )
        compose_ok = True
    except Exception as e:
        log.warning(f"phase1_enrich: compose_full_course failed: {e}")

    # Override the LLM-generated author block with deep-research result
    if composed_full and author.get("bio"):
        composed_full["author"]["name"] = author.get("name") or composed_full["author"].get("name", "")
        composed_full["author"]["bio"] = author["bio"]

    # ── 3.5 Extract lesson descriptions from compose curriculum ──────────
    # compose_full_course already wrote per-lesson descriptions in CURRICULUM;
    # use those instead of a separate describe_lessons call.
    if composed_full and composed_full.get("curriculum"):
        flat_lessons = [
            lesson
            for sec in composed_full["curriculum"]
            for lesson in sec.get("lessons", [])
        ]
        for i, p in enumerate(processed):
            if i < len(flat_lessons):
                desc = (flat_lessons[i].get("description") or "").strip()
                if desc:
                    lesson_descriptions[p["video_id"]] = desc
        if not any(lesson_descriptions.values()):
            log.warning("phase1_enrich: no lesson descriptions extracted from curriculum")

    # Persist composed payload (Phase 2 reads back instead of re-running compose)
    if composed_full:
        try:
            pipeline_db.save_course_compose(
                run_id=run_id, course_idx=course_idx,
                composed={
                    **composed_full,
                    "author_expertise": author.get("expertise", ""),
                    "author_confidence": author.get("confidence", "low"),
                    "author_sources": author.get("sources", []),
                },
            )
        except Exception as e:
            log.warning(f"phase1_enrich: save_course_compose failed: {e}")

    # ── 5. Build per-video enriched rows for the caller ──────────────────
    # Title priority: compose → operator-provided LLM-selected title → channel name.
    # URL mode passes course_title_from_llm="" so compose generates it from
    # transcripts; channel_name is the safety net if compose also failed.
    final_course_title = (
        (composed_full or {}).get("course", {}).get("title")
        or course_title_from_llm
        or channel_name
    )
    course_description = (composed_full and composed_full["course"].get("excerpt")) or ""
    course_about = (composed_full and composed_full["course"].get("aboutContent")) or ""

    # Tagline + what_you_learn + target_audience: extract concise variants from compose
    course_tagline = _extract_tagline(course_description, course_about)
    course_what_you_learn = _extract_what_you_learn(composed_full)
    course_target_audience = _extract_target_audience(course_about)

    # ── 5.5. Russian translations for operator review ───────────────────
    # If the source language is already Russian, skip — the originals are
    # already in Russian. Otherwise batch every description / bio / tagline
    # / what-you-learn into a single Claude translate call to keep latency
    # and quota cost low.
    russian_map: dict[str, str] = {}
    if output_lang != "ru":
        translation_items: list[dict[str, str]] = []
        if course_description:
            translation_items.append({"id": "course_description",
                                      "text": course_description})
        if course_tagline:
            translation_items.append({"id": "course_tagline",
                                      "text": course_tagline})
        if course_what_you_learn:
            translation_items.append({"id": "course_what_you_learn",
                                      "text": course_what_you_learn})
        if course_target_audience:
            translation_items.append({"id": "course_target_audience",
                                      "text": course_target_audience})
        author_bio_src = (author.get("bio") or "")
        if author_bio_src:
            translation_items.append({"id": "author_bio", "text": author_bio_src})
        author_expertise_src = (author.get("expertise") or "")
        if author_expertise_src:
            translation_items.append({"id": "author_expertise",
                                      "text": author_expertise_src})
        for p in processed:
            ld = lesson_descriptions.get(p["video_id"], "")
            if ld:
                translation_items.append({"id": f"lesson_{p['video_id']}",
                                          "text": ld})

        if translation_items:
            _emit(f"🌐 Перевожу {len(translation_items)} описаний на русский (один батч)…")
            try:
                russian_map = llm.translate_batch_to_russian(translation_items)
                _emit(f"  ✓ переведено: {len(russian_map)}/{len(translation_items)}")
            except Exception as e:
                log.warning(f"phase1_enrich: russian batch translate failed: {e}")

    enriched_videos: list[dict[str, Any]] = []
    for lesson_idx, p in enumerate(processed, start=1):
        excerpt = (p["working_transcript"] or "")[:TRANSCRIPT_EXCERPT_CHARS]
        ld_src = lesson_descriptions.get(p["video_id"], "")
        ld_ru = russian_map.get(f"lesson_{p['video_id']}", "")
        enriched_videos.append({
            "video_id": p["video_id"],
            "title": p["title"],
            "url": p["url"],
            "duration_sec": p["duration_sec"],
            "lesson_idx": lesson_idx,
            "transcript_excerpt": excerpt,
            # Source-of-truth original (will be sent to admin if operator
            # doesn't edit it in Sheet). Russian goes to its own _ru column.
            "lesson_description": ld_src,
            "lesson_description_ru": ld_ru,
            "detected_lang": p["detected_lang"],
            "cuts_count": p["cuts_count"],
        })

    if failed:
        _emit(f"⚠️ В курсе {course_idx} пропущено {len(failed)} видео из-за ошибок "
              f"(остальные {len(enriched_videos)} прошли).")

    # Pre-render the structured plan/science back to the delimiter format the
    # operator edits in Sheet (single multiline cell). Phase 2 parses these
    # back via llm.parse_plan_from_sheet / parse_science_from_sheet if the
    # operator edited them.
    course_plan_text = ""
    course_science_text = ""
    if composed_full:
        try:
            course_plan_text = llm.format_plan_for_sheet(
                composed_full.get("planSections") or []
            )
        except Exception as e:
            log.warning(f"phase1_enrich: format_plan_for_sheet failed: {e}")
        try:
            course_science_text = llm.format_science_for_sheet(
                composed_full.get("sciencePlan")
            )
        except Exception as e:
            log.warning(f"phase1_enrich: format_science_for_sheet failed: {e}")

    return {
        "videos": enriched_videos,
        "course_title": final_course_title,
        # Original-language fields (canon — Phase 2 sends these to admin
        # unless the operator overrides them in the Sheet).
        "course_description": course_description,
        "course_tagline": course_tagline,
        "course_what_you_learn": course_what_you_learn,
        "course_target_audience": course_target_audience,
        "author_name": author.get("name") or channel_name,
        "author_bio": author.get("bio") or "",
        "author_expertise": author.get("expertise") or "",
        # Full landing payload — operator can edit any of these in Sheet.
        "course_about": course_about,
        "course_plan": course_plan_text,
        "course_science": course_science_text,
        # Russian translations (Sheet-only, review aid).
        "course_description_ru": russian_map.get("course_description", ""),
        "course_tagline_ru": russian_map.get("course_tagline", ""),
        "course_what_you_learn_ru": russian_map.get("course_what_you_learn", ""),
        "course_target_audience_ru": russian_map.get("course_target_audience", ""),
        "author_bio_ru": russian_map.get("author_bio", ""),
        "author_expertise_ru": russian_map.get("author_expertise", ""),
        "compose_ok": compose_ok,
    }


# ---------------------------------------------------------------------------
# Per-video pipeline (called from a worker thread)
# ---------------------------------------------------------------------------

def _process_video_for_enrich(*, video_id: str, title: str, url: str,
                              course_topic: str, cookies_file: str | None,
                              rotator: ProxyRotator | None,
                              openai_key: str) -> dict[str, Any]:
    """One video: download to cache → working transcribe → mark cuts → save to db.

    Returns a dict with `ok=True` on success or `ok=False` + `error` on failure.
    Raises CookiesNeededError ONLY when proxy pool is exhausted (caller handles).
    """
    try:
        cached = cache.cached_path(video_id)
        if not cache.is_cached(video_id):
            cached = _download_with_rotation(
                video_id=video_id, url=url, output_path=cached,
                cookies_file=cookies_file, rotator=rotator,
            )

        working = whisper.transcribe(
            api_key=openai_key, file_path=cached,
            with_word_timestamps=True,
        )
        text = working.get("text", "")
        detected_iso = _lang_iso((working.get("language") or "").lower())
        try:
            duration_sec = int(ffmpeg_cut.probe_duration(cached))
        except Exception:
            duration_sec = int(working.get("duration") or 0)

        cuts: list[dict[str, Any]] = []
        try:
            cuts = llm.mark_cuts(course_topic=course_topic, transcript=working) or []
        except Exception as e:
            log.warning(f"phase1_enrich: mark_cuts failed for {video_id}: {e}")
            cuts = []

        try:
            pipeline_db.save_cuts(
                video_id, cuts=cuts, working_transcript=text,
                detected_lang=detected_iso, duration_sec=duration_sec,
                # Persist full Whisper segments (with word-level timestamps) so
                # Phase 2 can derive cleaned-timeline segments via segment_shift
                # without re-running Whisper on the cut video.
                segments=working.get("segments") or [],
            )
        except Exception as e:
            log.warning(f"phase1_enrich: save_cuts failed for {video_id}: {e}")

        return {
            "ok": True,
            "video_id": video_id,
            "title": title,
            "url": url,
            "duration_sec": duration_sec,
            "working_transcript": text,
            "detected_lang": detected_iso,
            "cuts_count": len(cuts),
        }
    except CookiesNeededError:
        raise
    except Exception as e:
        log.error(f"phase1_enrich: video {video_id} failed: {e}", exc_info=True)
        return {
            "ok": False,
            "video_id": video_id,
            "title": title,
            "error": str(e),
        }


MAX_TRUNCATED_PROXY_ROTATIONS = 5
"""How many different proxies to try when a download truncates mid-stream.

yt-dlp already retries ~10 times within a single proxy before giving up
("Giving up after 10 retries"). When that happens, the proxy itself is likely
flaky for this specific video (bandwidth cap, server-side throttle, or
mid-stream connection reset). Rotating to a fresh proxy usually recovers."""


def _download_with_rotation(*, video_id: str, url: str, output_path: Path,
                            cookies_file: str | None,
                            rotator: ProxyRotator | None) -> Path:
    """Download with thread-aware proxy rotation.

    Handles two distinct failure modes:
      1. Bot-check ("Sign in to confirm you're not a bot") — rotate until pool
         exhausted, then raise CookiesNeededError so wizard prompts for cookies.
      2. Truncated download ("Giving up after N retries", "X bytes read, Y more
         expected") — rotate up to MAX_TRUNCATED_PROXY_ROTATIONS different proxies,
         then give up and raise FFmpegError (caller skips this one video).
    """
    truncated_tries = 0
    while True:
        proxy = rotator.current if rotator else None
        try:
            return ffmpeg_cut.download_video(
                url=url, output_path=output_path,
                cookies_file=cookies_file, proxy=proxy,
            )
        except ffmpeg_cut.FFmpegError as e:
            if _is_bot_check(e) and rotator is not None:
                new_proxy = rotator.rotate_if_still(proxy)
                if new_proxy is None:
                    raise CookiesNeededError(
                        f"Все {len(rotator.pool)} прокси из пула заблокированы "
                        f"YouTube'ом в Phase 1 (видео {video_id}). Cookies "
                        f"скорее всего тоже устарели."
                    ) from e
                continue
            if _is_truncated_download(e) and rotator is not None:
                truncated_tries += 1
                if truncated_tries > MAX_TRUNCATED_PROXY_ROTATIONS:
                    log.warning(
                        f"phase1_enrich: video {video_id} truncated on "
                        f"{truncated_tries} different proxies, giving up"
                    )
                    raise
                new_proxy = rotator.rotate_if_still(proxy)
                if new_proxy is None:
                    # Pool exhausted before hitting the rotation budget — surface
                    # the truncation error, not CookiesNeededError (cookies are
                    # not the issue here).
                    raise
                log.info(
                    f"phase1_enrich: video {video_id} truncated, trying proxy "
                    f"#{truncated_tries + 1}/{MAX_TRUNCATED_PROXY_ROTATIONS + 1}"
                )
                continue
            # Any other FFmpeg/yt-dlp error: surface immediately.
            raise


def _is_bot_check(err: Exception) -> bool:
    s = str(err).lower()
    return "sign in to confirm" in s or "not a bot" in s


def _is_truncated_download(err: Exception) -> bool:
    """Match yt-dlp's mid-stream truncation errors."""
    s = str(err).lower()
    return (
        "giving up after" in s
        or "bytes read" in s and "more expected" in s
        or "connection reset" in s
        or "incomplete read" in s
        or "remote end closed connection" in s
    )


# ---------------------------------------------------------------------------
# Helpers — derive tagline / target / what-you-learn from the composed payload
# ---------------------------------------------------------------------------

def _extract_tagline(course_description: str, about: str) -> str:
    """Single-sentence tagline: first sentence of description (or about)."""
    src = (course_description or about or "").strip()
    if not src:
        return ""
    # Take the first sentence (period or newline boundary)
    for sep in ("\n\n", "\n", ". "):
        if sep in src:
            head = src.split(sep, 1)[0].strip()
            if 20 < len(head) < 200:
                return head + ("" if head.endswith(".") else ".")
    return src[:200]


def _extract_what_you_learn(composed: dict[str, Any] | None) -> str:
    """Pull bullet-list outcomes from the ABOUT section (markdown bullets)."""
    if not composed:
        return ""
    about = composed.get("course", {}).get("aboutContent", "")
    bullets: list[str] = []
    for line in (about or "").splitlines():
        s = line.strip()
        if s.startswith(("- ", "* ", "• ")):
            bullets.append(s[2:].strip())
        if len(bullets) >= 6:
            break
    return "\n".join(f"• {b}" for b in bullets[:6])


def _extract_target_audience(about: str) -> str:
    """Best-effort 'who it's for' line. Looks for a 'who is this for' marker."""
    if not about:
        return ""
    lower = about.lower()
    for marker in ("who is this for", "for whom", "who it's for", "this course is for"):
        idx = lower.find(marker)
        if idx == -1:
            continue
        chunk = about[idx:idx + 400]
        # Clip at the next blank line
        for sep in ("\n\n", "\n"):
            if sep in chunk[len(marker):]:
                return chunk.split(sep, 1)[0].strip()
        return chunk.strip()
    return ""
