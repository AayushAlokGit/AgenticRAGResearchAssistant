"""Load an eval set (e.g. ``evals/datasets/seed.yaml``) into typed objects.

One ``EvalQuestion`` per entry. Kept separate from scoring so any scorer (retrieval
recall, answer-correctness, the agentic trajectory harness) consumes the same parsed
dataset.

Two kinds of dataset share this schema:
  - ``seed.yaml`` — the frozen RAG→generate regression set (outcome fields only).
  - ``agentic.yaml`` — the agent-capability set, which ALSO carries the optional
    ``capability`` tag and ``trajectory`` block (the L1 trajectory assertions; see
    ``docs/evals/AGENT_EVAL_SETS.md``). Both extras are optional, so the seed set
    parses unchanged and an outcome-only agentic question stays simple.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import yaml

from agentic_rag.config import load_config, resolve_path


@dataclass
class TrajectoryExpectation:
    """A LOOSE, per-question expectation about the agent's PATH — the L1 assertion level.

    Each field is an OPTIONAL invariant any good trajectory should satisfy (assert the
    property, not the path — see AGENT_EVAL_SETS.md A3). The harness checks only the
    fields that are set, and scores the block as a SEPARATE diagnostic (PASS/FAIL),
    never folded into answer-correctness. All map directly onto the run's ``Trajectory``:

      expects_tool         -> this tool was invoked at least once (tool_calls[tool] >= 1)
      max_rounds           -> effort ceiling: rounds_used <= max_rounds
      min_searches         -> decomposition pressure: tool_calls['search'] >= min_searches
      expects_exit         -> the loop ended this way (exit_reason == finish|budget|oscillation)

    NOTE (DD-027): ``expects_tool`` binds to the tool's CODE NAME on purpose — accepted
    coupling while the tool set is small; validated against the live catalogue at load so
    a rename fails loudly. Promote to a capability label if/when tool names churn (A2).
    """
    expects_tool: Optional[str] = None
    max_rounds: Optional[int] = None
    min_searches: Optional[int] = None
    expects_exit: Optional[str] = None


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
    # Agentic extras (optional; absent on the seed regression set):
    capability: Optional[str] = None                       # taxonomy tag (tool_selection, …)
    trajectory: Optional[TrajectoryExpectation] = None     # L1 path assertions, if any


def eval_dataset_version(path: Optional[str] = None) -> str:
    """The dataset's declared ``version`` (str, ``'unknown'`` if absent).

    Stamped into the run-record path + config so runs against different dataset versions
    don't get compared as if they were the same baseline.
    """
    dataset_path = resolve_path(path or load_config()["eval"]["dataset"])
    raw = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    return str(raw.get("version", "unknown"))


def _parse_trajectory(raw: Optional[dict]) -> Optional[TrajectoryExpectation]:
    """Parse an optional ``trajectory:`` block into a TrajectoryExpectation (None if absent)."""
    if not raw:
        return None
    return TrajectoryExpectation(
        expects_tool=raw.get("expects_tool"),
        max_rounds=raw.get("max_rounds"),
        min_searches=raw.get("min_searches"),
        expects_exit=raw.get("expects_exit"),
    )


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
            capability=q.get("capability"),
            trajectory=_parse_trajectory(q.get("trajectory")),
        )
        for q in raw["questions"]
    ]


# Exit reasons the agent loop can report (agent/loop.py); an eval typo here should fail loud.
_VALID_EXIT_REASONS = {"finish", "budget", "oscillation"}


def validate_trajectory_expectations(questions: List[EvalQuestion]) -> None:
    """Fail loudly if any trajectory assertion references something that can't exist (DD-027).

    The guardrail that makes the accepted ``expects_tool`` name-coupling SAFE: a renamed or
    mistyped tool/exit is caught here, at load, instead of silently never-matching for a
    whole run. Validates against the FULL tool catalogue (not the current arm's registry) —
    a tool absent from one A/B arm is legitimate (that arm simply can't satisfy it), but a
    name no tool ever had is a bug. Lazily imports the catalogue so the generic loader stays
    decoupled from the agent unless trajectory assertions are actually used.
    """
    from agentic_rag.agent.tools import known_tool_names

    known_tools = set(known_tool_names())
    for q in questions:
        traj = q.trajectory
        if traj is None:
            continue
        if traj.expects_tool is not None and traj.expects_tool not in known_tools:
            raise ValueError(
                f"{q.id}: trajectory.expects_tool={traj.expects_tool!r} is not a known tool "
                f"(known: {sorted(known_tools)}). Fix the eval set or the registry (DD-027)."
            )
        if traj.expects_exit is not None and traj.expects_exit not in _VALID_EXIT_REASONS:
            raise ValueError(
                f"{q.id}: trajectory.expects_exit={traj.expects_exit!r} is not a valid exit reason "
                f"(valid: {sorted(_VALID_EXIT_REASONS)})."
            )
