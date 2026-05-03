"""Google Sheets I/O for the onboarder pipeline.

Layout (see SETUP.md for human setup):
    Tab `Criteria` -- Param/Value rows; bot reads at start of every run.
    Tab `Runs`     -- one row per wizard run; bot writes status updates.
    Tab `Run-{ts}` -- per-run sheet with course/lesson candidates and Approved checkbox.

All functions take a `client` (authorized gspread.Client) and `sheet_id` (str) so
they can be unit-tested with a fake client without touching the real Sheet.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("gateway")

# Header rows (must match SETUP.md). Lowercase identifiers used internally.
RUN_TAB_HEADER = [
    "Курс",          # course title (e.g. "Курс 1: AI Hub — Основы AI для маркетинга")
    "Lesson №",
    "Канал",
    "Название",
    "Ссылка на видео",
    "Длит",          # human-readable like "18 мин"
    "Approved",      # TRUE / FALSE checkbox
    # Hidden columns (after Approved) used by the bot for cross-step linking:
    "_video_id",     # YouTube video id
    "_channel_id",   # YouTube channel id
    "_duration_sec", # numeric duration
    "_course_idx",   # 1..N course index
    "_lesson_idx",   # 1..M lesson index within course
]

RUNS_INDEX_HEADER = ["timestamp", "topic", "count", "status", "sheet_tab", "courses"]

CRITERIA_DEFAULTS: dict[str, Any] = {
    "min_subscribers": 5000,
    "max_subscribers": 500000,
    "min_videos_on_channel": 20,
    "max_videos_on_channel": 1000,
    "max_video_age_months": 24,
    "preferred_languages": ["ru", "en"],
    "preferred_video_length_min": 10,
    "preferred_video_length_max": 35,
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def open_client(service_account_file: str) -> Any:
    """Authorize gspread with a service-account JSON key. Lazy import gspread."""
    import gspread
    from google.oauth2.service_account import Credentials

    sa_path = Path(service_account_file).expanduser()
    if not sa_path.exists():
        raise FileNotFoundError(f"service account JSON not found: {sa_path}")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
    ]
    creds = Credentials.from_service_account_file(str(sa_path), scopes=scopes)
    return gspread.authorize(creds)


def open_spreadsheet(client: Any, sheet_id: str) -> Any:
    """Open the spreadsheet by ID. Caller handles SpreadsheetNotFound."""
    return client.open_by_key(sheet_id)


# ---------------------------------------------------------------------------
# Criteria tab
# ---------------------------------------------------------------------------

def read_criteria(client: Any, sheet_id: str) -> dict[str, Any]:
    """Read tab `Criteria` and merge with defaults. Missing keys fall back to defaults.

    Type coercion:
        - integer-looking strings → int
        - comma-separated strings (preferred_languages) → list[str]
    """
    ss = open_spreadsheet(client, sheet_id)
    try:
        ws = ss.worksheet("Criteria")
    except Exception as e:
        log.warning(f"sheets: Criteria tab not found, using defaults ({e})")
        return dict(CRITERIA_DEFAULTS)

    rows = ws.get_all_values()
    parsed: dict[str, Any] = dict(CRITERIA_DEFAULTS)
    for row in rows[1:]:  # skip header
        if len(row) < 2:
            continue
        key = (row[0] or "").strip()
        val = (row[1] or "").strip()
        if not key:
            continue
        parsed[key] = _coerce(key, val)
    return parsed


def _coerce(key: str, val: str) -> Any:
    if key.endswith("_languages") or "," in val:
        return [v.strip() for v in val.split(",") if v.strip()]
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


# ---------------------------------------------------------------------------
# Runs index tab
# ---------------------------------------------------------------------------

def append_run(client: Any, sheet_id: str, *, run_id: str, topic: str,
               count: int, status: str, sheet_tab: str = "") -> int:
    """Append a row to `Runs`. Returns 1-based row number."""
    ss = open_spreadsheet(client, sheet_id)
    ws = _ensure_ws(ss, "Runs", header=RUNS_INDEX_HEADER)
    row = [run_id, topic, str(count), status, sheet_tab, ""]
    ws.append_row(row, value_input_option="USER_ENTERED")
    # gspread's append_row doesn't return the row index, so we approximate:
    return len(ws.get_all_values())


def update_run_status(client: Any, sheet_id: str, run_id: str, *,
                      status: str | None = None,
                      sheet_tab: str | None = None,
                      courses: str | None = None) -> None:
    """Find the run row by run_id and update specified columns. Silent if not found."""
    ss = open_spreadsheet(client, sheet_id)
    try:
        ws = ss.worksheet("Runs")
    except Exception:
        log.warning("sheets: Runs tab missing on update_run_status")
        return
    rows = ws.get_all_values()
    for idx, row in enumerate(rows[1:], start=2):  # skip header, 1-based
        if row and row[0] == run_id:
            updates = []
            if status is not None:
                updates.append({"range": f"D{idx}", "values": [[status]]})
            if sheet_tab is not None:
                updates.append({"range": f"E{idx}", "values": [[sheet_tab]]})
            if courses is not None:
                updates.append({"range": f"F{idx}", "values": [[courses]]})
            if updates:
                ws.batch_update(updates, value_input_option="USER_ENTERED")
            return
    log.warning(f"sheets: run_id {run_id} not found in Runs tab")


# ---------------------------------------------------------------------------
# Per-run tab (Run-{ts})
# ---------------------------------------------------------------------------

def create_run_tab(client: Any, sheet_id: str, tab_name: str) -> Any:
    """Create the Run-* tab with header. If exists, returns existing worksheet."""
    ss = open_spreadsheet(client, sheet_id)
    return _ensure_ws(ss, tab_name, header=RUN_TAB_HEADER, rows=200, cols=12)


def append_lesson_rows(client: Any, sheet_id: str, tab_name: str,
                       rows: list[dict[str, Any]]) -> None:
    """Append per-lesson rows to the Run-* tab.

    Each row dict keys (case-sensitive):
        course, lesson_idx, channel, title, url, duration_sec,
        video_id, channel_id, course_idx
    The Approved column is auto-set to FALSE (user ticks via Sheet UI).
    """
    if not rows:
        return
    ss = open_spreadsheet(client, sheet_id)
    ws = ss.worksheet(tab_name)
    values = []
    for r in rows:
        dur_sec = int(r.get("duration_sec") or 0)
        values.append([
            r.get("course", ""),
            r.get("lesson_idx", ""),
            r.get("channel", ""),
            r.get("title", ""),
            r.get("url", ""),
            _format_duration(dur_sec),
            "FALSE",
            r.get("video_id", ""),
            r.get("channel_id", ""),
            dur_sec,
            r.get("course_idx", ""),
            r.get("lesson_idx", ""),
        ])
    ws.append_rows(values, value_input_option="USER_ENTERED")


def read_approved_rows(client: Any, sheet_id: str, tab_name: str) -> list[dict[str, Any]]:
    """Read the Run-* tab and return only rows where Approved column is truthy.

    Returns dicts with the same keys as append_lesson_rows() input, plus
    the (possibly user-edited) `title`. Approval is checked case-insensitively
    against TRUE / true / 1 / x / yes / да.
    """
    ss = open_spreadsheet(client, sheet_id)
    ws = ss.worksheet(tab_name)
    rows = ws.get_all_values()
    out: list[dict[str, Any]] = []
    if len(rows) < 2:
        return out
    for row in rows[1:]:
        # pad to header length
        padded = row + [""] * (len(RUN_TAB_HEADER) - len(row))
        approved = (padded[6] or "").strip().lower()
        if approved not in ("true", "1", "x", "yes", "да"):
            continue
        try:
            duration_sec = int(padded[9] or 0)
        except ValueError:
            duration_sec = 0
        out.append({
            "course": padded[0],
            "lesson_idx_display": padded[1],
            "channel": padded[2],
            "title": padded[3],            # user may have edited
            "url": padded[4],
            "video_id": padded[7],
            "channel_id": padded[8],
            "duration_sec": duration_sec,
            "course_idx": _safe_int(padded[10]),
            "lesson_idx": _safe_int(padded[11]),
        })
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_ws(ss: Any, title: str, *, header: list[str], rows: int = 100, cols: int = 12) -> Any:
    """Return existing worksheet by title, or create with header if missing."""
    try:
        return ss.worksheet(title)
    except Exception:
        ws = ss.add_worksheet(title=title, rows=rows, cols=cols)
        ws.append_row(header, value_input_option="USER_ENTERED")
        return ws


def _format_duration(sec: int) -> str:
    if sec <= 0:
        return ""
    m, s = divmod(int(sec), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}ч {m}м"
    return f"{m} мин"


def _safe_int(v: str | int) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


def make_run_id() -> str:
    """e.g. 2026-05-04T12-34-56"""
    return time.strftime("%Y-%m-%dT%H-%M-%S")


def run_tab_name(run_id: str) -> str:
    return f"Run-{run_id}"


def sheet_tab_url(sheet_id: str, gid: int | None = None) -> str:
    base = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"
    return f"{base}#gid={gid}" if gid is not None else base
