"""HTTP client for NewMindStart agent API (`/api/agent/onboard-course`).

Sends a finished course assembly (author + course + lessons with already-uploaded
Bunny videoKeys) to NewMindStart, which creates the DRAFT course in Postgres.

The endpoint is documented in:
  newmindstart/src/app/api/agent/onboard-course/route.ts

Auth: Authorization: Bearer ${AGENT_API_TOKEN}
"""

from __future__ import annotations

import logging
from typing import Any

import requests

log = logging.getLogger("gateway")


class NMSError(RuntimeError):
    """Wraps any HTTP/JSON error from the NMS endpoint with response detail."""


def create_draft_course(*, endpoint: str, token: str,
                        author: dict[str, Any],
                        course: dict[str, Any],
                        lessons: list[dict[str, Any]] | None = None,
                        curriculum: list[dict[str, Any]] | None = None,
                        section_title: str | None = None,
                        plan_sections: list[dict[str, Any]] | None = None,
                        science_plan: dict[str, Any] | None = None,
                        testimonials: list[dict[str, Any]] | None = None,
                        collection_name: str | None = None,
                        timeout: int = 60) -> dict[str, Any]:
    """POST a fully-assembled course to NewMindStart. Returns {courseId, slug, adminUrl, ...}.

    Pass EITHER `lessons` (single section) OR `curriculum` (multi-section).

    Args:
        endpoint: Base URL like "https://truelifeflow.com".
        token:    Bearer token (matches AGENT_API_TOKEN on the NMS .env).
        author:   {"name", "bio", "avatar"?}
        course:   {"title", "excerpt"?, "aboutContent"?, "isAdult"?}
        lessons:  flat list (single section) of {"title","videoKey","videoLibraryId",...}
        curriculum: list of sections [{"title","isBonus","lessons":[...]}]
        plan_sections: [{"title","items":[{"title": str}]}]
        science_plan: {"enabled","headline","subtitle","institutions","stats"}
        testimonials: [{"authorName","text","rating"}]
        collection_name: existing or new collection title
    """
    if not lessons and not curriculum:
        raise NMSError("must provide either lessons or curriculum")

    url = endpoint.rstrip("/") + "/api/agent/onboard-course"
    body: dict[str, Any] = {
        "author": author,
        "course": course,
    }
    if curriculum:
        body["curriculum"] = curriculum
    elif lessons:
        body["lessons"] = lessons
    if section_title:
        body["sectionTitle"] = section_title
    if plan_sections:
        body["planSections"] = plan_sections
    if science_plan:
        body["sciencePlan"] = science_plan
    if testimonials:
        body["testimonials"] = testimonials
    if collection_name:
        body["collectionName"] = collection_name

    try:
        r = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=body,
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise NMSError(f"network error calling {url}: {e}") from e

    try:
        data = r.json()
    except ValueError:
        data = None

    if r.status_code >= 400:
        detail = ""
        if isinstance(data, dict):
            detail = data.get("error") or data.get("result_message") or ""
        raise NMSError(
            f"NMS {r.status_code}: {detail or r.text[:300]}"
        )

    if not isinstance(data, dict) or not isinstance(data.get("data"), dict):
        raise NMSError(f"NMS unexpected response shape: {str(data)[:300]}")

    payload = data["data"]
    for key in ("courseId", "slug", "adminUrl"):
        if key not in payload:
            raise NMSError(f"NMS response missing '{key}': {payload}")
    return payload


def append_lessons_to_course(*, endpoint: str, token: str,
                             course_id: str,
                             lessons: list[dict[str, Any]],
                             section_title: str | None = None,
                             timeout: int = 60) -> dict[str, Any]:
    """POST extra lessons to an EXISTING course.

    Calls `/api/agent/courses/{course_id}/append-lessons`. Used for the
    recovery flow: when a Phase 2 run fails on some videos (e.g. Whisper
    mis-detected language → translate 400), the operator can fix the bug,
    re-process JUST the failed videos, and append them to the partial
    course instead of deleting and re-creating from scratch.

    `course_id` may be a Prisma id OR a slug — NMS resolves both.
    `section_title` is optional. If matches an existing section
    (case-insensitive), lessons land there; otherwise they go into the
    last existing section.

    Returns: {courseId, sectionId, appendedLessonIds, appendedCount, adminUrl}.
    """
    if not lessons:
        raise NMSError("append_lessons_to_course: lessons[] is empty")

    url = (
        endpoint.rstrip("/")
        + f"/api/agent/courses/{course_id}/append-lessons"
    )
    body: dict[str, Any] = {"lessons": lessons}
    if section_title:
        body["sectionTitle"] = section_title

    try:
        r = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=body,
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise NMSError(f"network error calling {url}: {e}") from e

    try:
        data = r.json()
    except ValueError:
        data = None

    if r.status_code >= 400:
        detail = ""
        if isinstance(data, dict):
            detail = data.get("error") or data.get("result_message") or ""
        raise NMSError(f"NMS {r.status_code}: {detail or r.text[:300]}")

    if not isinstance(data, dict) or not isinstance(data.get("data"), dict):
        raise NMSError(f"NMS unexpected response shape: {str(data)[:300]}")

    payload = data["data"]
    for key in ("courseId", "appendedLessonIds", "appendedCount", "adminUrl"):
        if key not in payload:
            raise NMSError(f"NMS response missing '{key}': {payload}")
    return payload
