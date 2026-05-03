"""Resolve secrets uniformly: explicit file path in config, OR env var fallback.

Pattern:
    resolve(cfg["onboarder"], "anthropic_api_key", env="ANTHROPIC_API_KEY")
        Tries cfg["anthropic_api_key_file"] first; if missing, reads $ANTHROPIC_API_KEY.
        Returns the secret value (string) or raises FileNotFoundError with both paths
        tried so the operator sees exactly what's missing.

For libraries that take a file path (not value), use resolve_path() instead.
"""

from __future__ import annotations

import os
from pathlib import Path


def _expand(path: str) -> Path:
    return Path(path).expanduser()


def resolve(cfg: dict, key: str, *, env: str | None = None) -> str:
    """Read secret value. cfg may have either '{key}' (inline value) or '{key}_file' (path)."""
    inline = cfg.get(key)
    if isinstance(inline, str) and inline.strip():
        return inline.strip()

    file_key = f"{key}_file"
    file_path = cfg.get(file_key)
    if isinstance(file_path, str) and file_path.strip():
        p = _expand(file_path)
        if p.exists():
            val = p.read_text().strip()
            if val:
                return val
            raise ValueError(f"secrets: {file_key} ({p}) is empty")
        # Fall through to env if file is configured but missing
        env_val = os.environ.get(env) if env else None
        if env_val:
            return env_val.strip()
        raise FileNotFoundError(f"secrets: {file_key} not found at {p} and ${env} not set")

    if env:
        env_val = os.environ.get(env)
        if env_val:
            return env_val.strip()

    raise FileNotFoundError(
        f"secrets: neither cfg[{key!r}], cfg[{file_key!r}] nor env var ${env} provides a value"
    )


def resolve_path(cfg: dict, key: str) -> str:
    """For libraries that need a file path, not a value (e.g. Google service account)."""
    file_key = f"{key}_file"
    file_path = cfg.get(file_key) or cfg.get(key)
    if not isinstance(file_path, str) or not file_path.strip():
        raise FileNotFoundError(f"secrets: cfg[{file_key!r}] not configured")
    p = _expand(file_path)
    if not p.exists():
        raise FileNotFoundError(f"secrets: {file_key} → {p} does not exist")
    return str(p)
