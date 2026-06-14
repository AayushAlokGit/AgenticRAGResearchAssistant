"""Load the seed eval set (``evals/datasets/seed.yaml``) into typed objects.

One ``EvalQuestion`` per entry. Kept separate from scoring so any scorer (retrieval
recall now; answer-quality later) consumes the same parsed dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import yaml

from agentic_rag.config import load_config, resolve_path


@dataclass
class EvalQuestion:
    id: str
    type: str                      # factual | multi_hop | abstention
    question: str
    match: str                     # "any" | "all" — how to score expected_sources
    expected_sources: List[str]    # corpus filenames; empty for abstention
    should_abstain: bool           # true → answer is NOT in the corpus
    expected_answer: str
    notes: str


def eval_dataset_version(path: Optional[str] = None) -> str:
    """The dataset's declared ``version`` (str, ``'unknown'`` if absent).

    Stamped into the run-record path + config so runs against different dataset versions
    don't get compared as if they were the same baseline.
    """
    dataset_path = resolve_path(path or load_config()["eval"]["dataset"])
    raw = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    return str(raw.get("version", "unknown"))


def load_eval_dataset(path: Optional[str] = None) -> List[EvalQuestion]:
    """Parse the YAML eval set referenced by ``config.eval.dataset`` (or an override)."""
    dataset_path = resolve_path(path or load_config()["eval"]["dataset"])
    raw = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    return [
        EvalQuestion(
            id=q["id"],
            type=q.get("type", ""),
            question=q["question"],
            match=q.get("match", "any"),
            expected_sources=q.get("expected_sources") or [],
            should_abstain=bool(q.get("should_abstain", False)),
            expected_answer=q.get("expected_answer", ""),
            notes=q.get("notes", ""),
        )
        for q in raw["questions"]
    ]
