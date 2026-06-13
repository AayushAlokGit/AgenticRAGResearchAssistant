"""Where eval run records are written.

One convention for every eval: ``eval_runs/<metric>/<timestamp>.json`` (gitignored).
The timestamp is local-time and human-readable (``DD_MM_YYYY_HH_MM_SS``) so the files
sort and read naturally when you're diffing runs by hand.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from agentic_rag.config import resolve_path


def eval_run_path(metric: str) -> Path:
    """Create ``eval_runs/<metric>/`` if needed and return a fresh timestamped json path."""
    out_dir = resolve_path("./eval_runs") / metric
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    return out_dir / f"{stamp}.json"
