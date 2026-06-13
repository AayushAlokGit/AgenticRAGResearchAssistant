"""Configuration loading.

Reads the versioned ``config/default.yaml`` and resolves the relative paths inside it
(``./corpus``, ``./chromadb_data`` …) against the project root, so scripts behave the
same no matter what directory they're launched from.

Kept tiny on purpose: this is the one place that knows where config lives and how to
turn a config-relative path into an absolute one.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


def project_root() -> Path:
    """Return the repo root by walking up until we find ``pyproject.toml``."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Could not locate project root (no pyproject.toml above {here})")


@lru_cache(maxsize=None)
def load_config(config_path: str | None = None) -> dict:
    """Load and parse the YAML config. Cached so repeated calls are free.

    Uses ``yaml.safe_load`` (never ``load``) — the config is trusted, but safe_load is
    the correct default and sidesteps YAML's footguns (e.g. the 'Norway problem').
    """
    path = Path(config_path) if config_path else project_root() / "config" / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    """Resolve a config path (e.g. ``./corpus``) against the project root.

    Absolute paths are returned unchanged.
    """
    p = Path(relative)
    return p if p.is_absolute() else (project_root() / p)
