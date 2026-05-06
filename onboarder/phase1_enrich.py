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

    # Detect output language ONCE per course from topic + pain + course title.
    # Pain is often the most descriptive — operator types it in their own
    # language, so it's the strongest signal.
    output_lang = llm.detect_topic_lang(pain, course_topic_input, course_title_from_llm)
    _emit(f"🌐 Язык вывода: {output_lang}")

    # ── 2. Per-lesson descriptions (single batched Claude call) ──────────
    _emit(f"📝 Пишу описания {len(processed)} уроков (Claude, {output_lang})…")
    lesson_descriptions: dict[str, str] = {}
    try:
        descs = llm.describe_lessons(
            course_topic=course_title_from_llm or course_topic_input,
            course_title=course_title_from_llm,
            lessons=[
                {"order": i, "title": p["title"], "transcript": p["working_transcript"]}
                for i, p in enumerate(processed)
            ],
            output_lang=output_lang,
            pain=pain, audience=audience,
        )
        for i, d in enumerate(descs):
            if i < len(processed):
                lesson_descriptions[processed[i]["video_id"]] = d.get("description", "")
    except Exception as e:
        log.warning(f"phase1_enrich: describe_lessons failed: {e}")

    # ── 3. Author research (deep, with WebSearch) ────────────────────────
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

    # ── 4. Full course compose (curriculum/plan/science/testimonials/...) ─
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
    final_course_title = (composed_full and composed_full["course"]["title"]
                          or course_title_from_llm)
    course_description = (composed_full and composed_full["course"].get("excerpt")) or ""
    course_about = (composed_full and composed_full["course"].get("aboutContent")) or ""

    # Tagline + what_you_learn + target_audience: extract concise variants from compose
    course_tagline = _extract_tagline(course_description, course_about)
    course_what_you_learn = _extract_what_you_learn(composed_full)
    course_target_audience = _extract_target_audience(course_about)

    enriched_videos: list[dict[str, Any]] = []
    for lesson_idx, p in enumerate(processed, start=1):
        excerpt = (p["working_transcript"] or "")[:TRANSCRIPT_EXCERPT_CHARS]
        enriched_videos.append({
            "video_id": p["video_id"],
            "title": p["title"],
            "url": p["url"],
            "duration_sec": p["duration_sec"],
            "lesson_idx": lesson_idx,
            "transcript_excerpt": excerpt,
            "lesson_description": lesson_descriptions.get(p["video_id"], ""),
            "detected_lang": p["detected_lang"],
            "cuts_count": p["cuts_count"],
        })

    if failed:
        _emit(f"⚠️ В курсе {course_idx} пропущено {len(failed)} видео из-за ошибок "
              f"(остальные {len(enriched_videos)} прошли).")

    return {
        "videos": enriched_videos,
        "course_title": final_course_title,
        "course_description": course_description,
        "course_tagline": course_tagline,
        "course_what_you_learn": course_what_you_learn,
        "course_target_audience": course_target_audience,
        "author_name": author.get("name") or channel_name,
        "author_bio": author.get("bio") or "",
        "author_expertise": author.get("expertise") or "",
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


def _download_with_rotation(*, video_id: str, url: str, output_path: Path,
                            cookies_file: str | None,
                            rotator: ProxyRotator | None) -> Path:
    """Download with thread-aware proxy rotation. Mirrors phase2's helper."""
    while True:
        proxy = rotator.current if rotator else None
        try:
            return ffmpeg_cut.download_video(
                url=url, output_path=output_path,
                cookies_file=cookies_file, proxy=proxy,
            )
        except ffmpeg_cut.FFmpegError as e:
            if not _is_bot_check(e) or rotator is None:
                raise
            new_proxy = rotator.rotate_if_still(proxy)
            if new_proxy is None:
                raise CookiesNeededError(
                    f"Все {len(rotator.pool)} прокси из пула заблокированы YouTube'ом "
                    f"в Phase 1 (видео {video_id}). Cookies скорее всего тоже устарели."
                ) from e


def _is_bot_check(err: Exception) -> bool:
    s = str(err).lower()
    return "sign in to confirm" in s or "not a bot" in s


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
