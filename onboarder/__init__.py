"""Onboarder: wizard-driven YouTube → course pipeline.

Module structure:
    wizard.py            -- Telegram state machine (form, callbacks, dispatch)
    state.py             -- per-user wizard state JSON + per-run pipeline.db
    sheets.py            -- Google Sheets I/O (Criteria, Runs, Run-* layouts)
    youtube_dl.py        -- yt-dlp wrappers (search, channel listing, video download)
    whisper.py           -- OpenAI Whisper API (transcription with language detection)
    elevenlabs_dub.py    -- ElevenLabs Dubbing API (translate+dub video)
    ffmpeg_cut.py        -- FFmpeg: cut segments by timecodes
    bunny.py             -- Bunny Stream upload (videoKey/library_id back)
    llm.py               -- Anthropic Claude API: scoring, classification, descriptions
    nms_client.py        -- POST /api/agent/onboard-course (Bearer auth)
    phase1_discovery.py  -- channel search → score → per-channel video selection
    phase2_production.py -- per-video: download → transcribe → cut → dub → upload → assemble draft

Default state: chat (existing Claude behavior). Wizard mode is opt-in via /menu → 🎓 Новый курс.
"""

from . import wizard  # noqa: F401  (re-export for `from onboarder import wizard`)

__all__ = ["wizard"]
