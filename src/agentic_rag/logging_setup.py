"""Run logging: console output + a per-run timestamped file under logs/<category>/.

Each top-level run (an ingestion, an eval) calls ``configure_run_logging(category)`` once
at startup. It creates ``logs/<category>/<category>_<UTC-timestamp>.log`` and wires the
root logger to write there (DEBUG, full detail) and to the console (INFO, clean — just the
message, so output reads like the old print()s). Library modules just do
``logging.getLogger(__name__)`` and log; they don't know or care where it goes.

``logs/`` is gitignored — these are run artifacts, like ``eval_runs/``.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from agentic_rag.config import resolve_path

# File gets full structured context; console stays clean (message only) for readability.
_FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_CONSOLE_FORMAT = "%(message)s"

# Third-party loggers that are noisy at INFO and would drown our own messages.
_NOISY_LIBRARIES = ["httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers",
                    "transformers", "huggingface_hub", "groq"]


def configure_run_logging(category: str) -> Path:
    """Set up logging for one run and return the path of its log file.

    Args:
        category: subfolder + filename prefix, e.g. "ingestion" or "evals".
    """
    logs_dir = resolve_path("./logs") / category
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = logs_dir / f"{category}_{stamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Drop any handlers from a previous call so a second run in the same process doesn't
    # double-log or keep writing to the previous file.
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
    root.addHandler(console_handler)

    # Quiet noisy libraries so the console shows our messages, not request-by-request spam.
    # (Their DEBUG/INFO still won't reach the file either, which is fine — we want OUR trace.)
    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger(__name__).info("run log -> %s", log_path)
    return log_path
