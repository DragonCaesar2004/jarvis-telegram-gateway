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
                        lessons: list[dict[str, Any]],
                        section_title: str | None = None,
                        timeout: int = 60) -> dict[str, Any]:
    """POST a fully-assembled course to NewMindStart. Returns {courseId, slug, adminUrl}.

    Args:
        endpoint: Base URL like "https://truelifeflow.com" (no trailing slash needed).
        token:    Bearer token (matches AGENT_API_TOKEN on the NMS .env).
        author:   {"name": str, "bio": str, "avatar"?: str}
        course:   {"title": str, "excerpt"?: str, "aboutContent"?: str, "isAdult"?: bool}
        lessons:  list of {"title", "order"?, "description"?, "videoKey",
                           "videoLibraryId", "duration"?, "transcriptEn"?,
                           "originalLang"?, "wasDubbed"?}
        section_title: defaults to "Course content" (NMS picks if None).
    """
    url = endpoint.rstrip("/") + "/api/agent/onboard-course"
    body: dict[str, Any] = {
        "author": author,
        "course": course,
        "lessons": lessons,
    }
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
