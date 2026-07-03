"""Agentic capability eval — the trajectory-aware harness for ``agentic.yaml``.

WHY A SEPARATE HARNESS (not an extension of answer_correctness.py)
-----------------------------------------------------------------
``answer_correctness.py`` is the REGRESSION harness for the frozen ``seed.yaml`` set: it
grades a RAG→generate pipeline on OUTCOME (right answer / right abstention). This harness
grades the AGENT — so it keeps the outcome meter (reusing the same judge + abstention
check, no duplication) and adds the two things that only make sense for an agent:

  1. a per-question TRAJECTORY assertion (L1) — did a good path satisfy the loose invariant
     the eval declares (used the right tool, stayed efficient, took enough hops, stopped by
     choice)? Scored as a SEPARATE diagnostic axis, never folded into correctness — a good
     answer by a bad/lucky path is still a yellow flag, not a pass (AGENT_EVAL_SETS.md A3/A4).
  2. a CAPABILITY-sliced report — scores grouped by what each question tests
     (tool_selection, decomposition, …) so "tool-selection 2/3" replaces one blurred number.

Keeping it in its own module leaves the regression harness focused and stable, and gives
the agentic concerns a clearly-named entry point (the "new thin module", per the design).

Run (needs an ingested store + the agent ON in config + provider keys):
    python -m agentic_rag.evals.agentic_eval
    python -m agentic_rag.evals.agentic_eval --ids a05,a06     # just the expand_document cases
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from agentic_rag.agent.loop import build_answerer
from agentic_rag.config import load_config
from agentic_rag.evals.answer_correctness import (UNGRADED, aggregate_trajectory,
                                                  is_abstention, judge_correctness)
from agentic_rag.evals.faithfulness import check_citations, judge_faithfulness
from agentic_rag.evals.dataset import (EvalQuestion, TrajectoryExpectation, eval_dataset_version,
                                       load_eval_dataset, validate_trajectory_expectations)
from agentic_rag.evals.runs import (agent_config_snapshot, context_config_snapshot, eval_run_path,
                                    retrieval_config_snapshot, usage_record)
from agentic_rag.llm.provider import Usage, build_llm, role_model
from agentic_rag.logging_setup import configure_run_logging
from agentic_rag.rag.answer import Trajectory, assemble_context, load_prompt
from agentic_rag.rag.vector_store import Hit
from agentic_rag.rag.retriever import build_retriever

logger = logging.getLogger(__name__)

DEFAULT_AGENTIC_DATASET = "evals/datasets/agentic.yaml"


# ───────────────────────────── trajectory assertions (the new axis) ─────────────────────────

@dataclass
class TrajectoryCheck:
    """One L1 invariant checked against the actual run trajectory."""
    key: str           # human label, e.g. "expects_tool=expand_document"
    expected: str      # what the eval asserted, e.g. ">=1"
    actual: object     # what the run did, e.g. 0
    passed: bool


def check_trajectory(expectation: TrajectoryExpectation, traj: Trajectory) -> List[TrajectoryCheck]:
    """Evaluate every SET field of a trajectory expectation against the run's Trajectory.

    Only the fields the eval bothered to set are checked (assert the property, not the
    path). Each maps onto a Trajectory field — see TrajectoryExpectation for the mapping.
    """
    checks: List[TrajectoryCheck] = []

    if expectation.expects_tool is not None:
        used = traj.tool_calls.get(expectation.expects_tool, 0)
        checks.append(TrajectoryCheck(
            key=f"expects_tool={expectation.expects_tool}", expected=">=1",
            actual=used, passed=used >= 1))

    if expectation.max_rounds is not None:
        checks.append(TrajectoryCheck(
            key="max_rounds", expected=f"<={expectation.max_rounds}",
            actual=traj.rounds_used, passed=traj.rounds_used <= expectation.max_rounds))

    if expectation.min_searches is not None:
        searches = traj.tool_calls.get("search", 0)
        checks.append(TrajectoryCheck(
            key="min_searches", expected=f">={expectation.min_searches}",
            actual=searches, passed=searches >= expectation.min_searches))

    if expectation.expects_exit is not None:
        checks.append(TrajectoryCheck(
            key="expects_exit", expected=expectation.expects_exit,
            actual=traj.exit_reason, passed=traj.exit_reason == expectation.expects_exit))

    return checks


# ───────────────────────────── per-question result ─────────────────────────────

@dataclass
class AgenticResult:
    q: EvalQuestion
    answer: str
    abstained: bool
    verdict: Optional[str]                 # CORRECT/PARTIALLY_CORRECT/INCORRECT/UNGRADED, or None
    judge_reason: str = ""
    # Faithfulness axis (generator groundedness, reference-free): None when the answer abstained
    # (no claims → trivially faithful → skipped) or faithfulness was switched off for the run.
    faith_verdict: Optional[str] = None    # SUPPORTED/PARTIALLY_SUPPORTED/UNSUPPORTED/UNGRADED, or None
    faith_reason: str = ""
    unsupported_citations: List[str] = field(default_factory=list)  # cited but NOT among retrieved chunks
    retrieved: List[Hit] = field(default_factory=list)
    trajectory: Optional[Trajectory] = None
    traj_checks: List[TrajectoryCheck] = field(default_factory=list)
    # traj_pass: True/False once asserted+measured; None means "not scored on trajectory"
    # (either the question declared no assertions, or the run produced no trajectory).
    traj_pass: Optional[bool] = None
    traj_unassertable: bool = False        # had assertions but no trajectory (naive run)
    controller_usage: Usage = field(default_factory=Usage)
    generator_usage: Usage = field(default_factory=Usage)
    judge_usage: Usage = field(default_factory=Usage)


def outcome_ok(r: AgenticResult) -> bool:
    """Did this question get the OUTCOME right (correct answer, or correct abstention)?"""
    if r.q.should_abstain:
        return r.abstained
    return r.verdict == "CORRECT"


def describe(r: AgenticResult) -> str:
    """Short live-progress string: outcome status + a trajectory tag."""
    if r.q.should_abstain:
        outcome = "abstained OK" if r.abstained else "FAILED TO ABSTAIN"
    elif r.abstained:
        outcome = "FALSE ABSTENTION"
    else:
        outcome = r.verdict or "?"
    if r.traj_pass is None:
        traj = "traj n/a" if not r.traj_unassertable else "traj UNASSERTABLE"
    else:
        traj = "traj OK" if r.traj_pass else "traj MISS"
    return f"{outcome:<17} | {traj}"


# ───────────────────────────── the run ─────────────────────────────

def run(save: bool = True, limit: Optional[int] = None, dataset: str = DEFAULT_AGENTIC_DATASET,
        ids: Optional[List[str]] = None, run_faithfulness: bool = True,
        cache_enabled: Optional[bool] = None) -> dict:
    config = load_config()
    version = eval_dataset_version(dataset)

    # Same build-once pattern as answer_correctness: expensive deps constructed once and
    # reused. controller (routing brain) + generator (answer writer) may be different models
    # (DD-025); judge is a different family (no self-eval bias, separate token bucket).
    # cache_enabled=None follows config.cache.enabled; False forces the cold, uncached A/B control.
    retriever = build_retriever(config)
    generator_llm = build_llm(config, role="generator", cache_enabled=cache_enabled)
    controller_llm = build_llm(config, role="controller", cache_enabled=cache_enabled)
    judge_llm = build_llm(config, role="judge", cache_enabled=cache_enabled)
    answerer = build_answerer(config, retriever, generator_llm, controller_llm=controller_llm)
    judge_prompt = load_prompt(config, "judge_correctness")
    faith_prompt = load_prompt(config, "judge_faithfulness")   # reuses the SAME judge_llm (judge role)

    if not bool(config.get("agent", {}).get("enabled", False)):
        logger.warning("agent.enabled is FALSE — this harness grades the AGENT, so trajectory "
                       "assertions will be UNASSERTABLE (no loop runs). Outcome is still graded.")

    logger.info(f"agentic eval dataset version: v{version}")
    questions = load_eval_dataset(dataset)
    validate_trajectory_expectations(questions)   # DD-027 guardrail: fail loud on a bad tool/exit
    if ids:
        wanted = set(ids)
        questions = [q for q in questions if q.id in wanted]
    if limit is not None:
        questions = questions[:limit]

    logger.info(f"Scoring {len(questions)} agentic question(s) (agent loop + judge)...\n")
    results: List[AgenticResult] = []
    for i, q in enumerate(questions, start=1):
        generated = answerer.answer(q.question)
        abstained = is_abstention(generated.answer)

        # Outcome (reused judge): only judge answerable questions that actually answered.
        verdict = None
        reason = ""
        judge_usage = Usage()
        if not q.should_abstain and not abstained:
            verdict, reason, judge_usage = judge_correctness(
                judge_llm, judge_prompt, q.question, q.expected_answer, generated.answer)

        # Faithfulness (generator groundedness): a SECOND, reference-free judge call against the
        # EXACT context the generator saw. Fires on ANY answered question — including one that
        # SHOULD have abstained but answered anyway, which is exactly the fabrication this catches.
        # Cost folds into judge_usage (the instrument bucket), so the existing cost line stays right.
        faith_verdict = None
        faith_reason = ""
        unsupported: List[str] = []
        if run_faithfulness and not abstained:
            context = assemble_context(generated.retrieved)
            faith_verdict, faith_reason, faith_usage = judge_faithfulness(
                judge_llm, faith_prompt, q.question, context, generated.answer)
            _, unsupported = check_citations(generated.answer, generated.retrieved)
            judge_usage = judge_usage + faith_usage

        # Trajectory (the new axis): check only if the eval declared assertions.
        traj_checks: List[TrajectoryCheck] = []
        traj_pass: Optional[bool] = None
        traj_unassertable = False
        if q.trajectory is not None:
            if generated.trajectory is not None:
                traj_checks = check_trajectory(q.trajectory, generated.trajectory)
                traj_pass = all(c.passed for c in traj_checks)
            else:
                traj_unassertable = True   # assertions exist but the run has no trajectory

        result = AgenticResult(
            q=q, answer=generated.answer, abstained=abstained, verdict=verdict, judge_reason=reason,
            faith_verdict=faith_verdict, faith_reason=faith_reason, unsupported_citations=unsupported,
            retrieved=generated.retrieved, trajectory=generated.trajectory, traj_checks=traj_checks,
            traj_pass=traj_pass, traj_unassertable=traj_unassertable,
            controller_usage=generated.controller_usage, generator_usage=generated.generator_usage,
            judge_usage=judge_usage)
        results.append(result)
        logger.info(f"  [{i:2d}/{len(questions)}] {q.id} {q.capability or '-':<22} {describe(result)}")

    summary = report(results, config)
    if save:
        persist(summary, results, config, version)
    return summary


# ───────────────────────────── the report ─────────────────────────────

def report(results: List[AgenticResult], config: dict) -> dict:
    answerable = [r for r in results if not r.q.should_abstain]
    abstention = [r for r in results if r.q.should_abstain]

    answered = [r for r in answerable if not r.abstained]
    false_abstentions = [r for r in answerable if r.abstained]
    correct = [r for r in answered if r.verdict == "CORRECT"]
    partial = [r for r in answered if r.verdict == "PARTIALLY_CORRECT"]
    incorrect = [r for r in answered if r.verdict == "INCORRECT"]
    ungraded = [r for r in answered if r.verdict == UNGRADED]
    abstained_correctly = [r for r in abstention if r.abstained]

    n_graded = len(answerable) - len(ungraded)
    # Two correctness metrics, recorded SEPARATELY so they can't be confused:
    #  - STRICT (end_to_end_success): only full-CORRECT counts, over graded ANSWERABLE Qs. This is
    #    the headline; it understates a system whose errors are INCOMPLETE rather than WRONG.
    #  - PARTIAL-CREDIT: CORRECT=1.0, PARTIALLY_CORRECT=0.5 — sensitive to PARTIAL<->CORRECT moves a
    #    strict metric is blind to. `*_answerable` is over answerable Qs (parallel to strict);
    #    `*_overall` also folds in correct abstentions, over ALL graded Qs.
    end_to_end = len(correct) / n_graded if n_graded else 0.0
    partial_credit_answerable = (len(correct) + 0.5 * len(partial)) / n_graded if n_graded else 0.0
    n_scored = n_graded + len(abstention)
    partial_credit_overall = ((len(correct) + 0.5 * len(partial) + len(abstained_correctly)) / n_scored
                              if n_scored else 0.0)

    generator_model = role_model(config, "generator")
    judge_model = role_model(config, "judge")
    logger.info(f"\n=== Agentic Eval (generator={generator_model}, judge={judge_model}) ===")
    if generator_model == judge_model:
        logger.info("note: same model generates and grades (self-eval bias); output is non-deterministic\n")
    else:
        logger.info("note: judge is a different model (no self-eval bias); output is non-deterministic\n")

    # ---- OUTCOME axis ----
    logger.info("OUTCOME FAILURES:")
    any_fail = False
    for r in false_abstentions:
        any_fail = True
        logger.info(f"  {r.q.id}  FALSE ABSTENTION (answerable, but refused)")
    for r in incorrect:
        any_fail = True
        logger.info(f"  {r.q.id}  INCORRECT — {_last_line(r.judge_reason)}")
    for r in partial:
        any_fail = True
        logger.info(f"  {r.q.id}  PARTIAL — {_last_line(r.judge_reason)}")
    for r in abstention:
        if not r.abstained:
            any_fail = True
            logger.info(f"  {r.q.id}  FAILED TO ABSTAIN (answered an out-of-corpus question)")
    if not any_fail:
        logger.info("  (none)")
    if ungraded:
        logger.info("UNGRADED (judge gave no verdict; excluded from the rate):")
        for r in ungraded:
            logger.info(f"  {r.q.id}  (re-run to grade)")

    logger.info(f"\nOUTCOME SUMMARY")
    logger.info(f"  answerable ({len(answerable)}): answered {len(answered)}  false-abstention {len(false_abstentions)}")
    logger.info(f"    of answered -> CORRECT={len(correct)} PARTIAL={len(partial)} INCORRECT={len(incorrect)} UNGRADED={len(ungraded)}")
    logger.info(f"    end-to-end success (STRICT, full-correct/answerable): {len(correct)}/{n_graded} = {end_to_end:.3f}   [excludes {len(ungraded)} ungraded]")
    logger.info(f"    partial-credit (CORRECT=1, PARTIAL=0.5): answerable={partial_credit_answerable:.3f}  overall(incl. abstentions)={partial_credit_overall:.3f}")
    logger.info(f"  abstention ({len(abstention)}): abstained correctly {len(abstained_correctly)}/{len(abstention)}")

    # ---- TRAJECTORY axis (the diagnostic; separate from outcome) ----
    asserted = [r for r in results if r.traj_pass is not None]
    traj_passed = [r for r in asserted if r.traj_pass]
    traj_failed = [r for r in asserted if not r.traj_pass]
    unassertable = [r for r in results if r.traj_unassertable]
    traj_rate = len(traj_passed) / len(asserted) if asserted else None

    logger.info(f"\nTRAJECTORY DIAGNOSTIC (separate axis — a miss is a flag, not an outcome failure):")
    if not asserted:
        logger.info("  (no trajectory assertions scored)")
    else:
        logger.info(f"  passed {len(traj_passed)}/{len(asserted)}" +
                    (f" = {traj_rate:.3f}" if traj_rate is not None else ""))
        for r in traj_failed:
            misses = "; ".join(f"{c.key} expected {c.expected}, got {c.actual}"
                               for c in r.traj_checks if not c.passed)
            outcome_tag = "outcome OK" if outcome_ok(r) else "outcome FAIL"
            logger.info(f"  {r.q.id}  TRAJ MISS [{outcome_tag}] — {misses}")
    if unassertable:
        logger.info(f"  UNASSERTABLE ({len(unassertable)}): had assertions but no trajectory "
                    f"(agent off?) — {', '.join(r.q.id for r in unassertable)}")

    # ---- FAITHFULNESS axis (generator groundedness; reference-free) ----
    # Orthogonal to correctness: this is the meter that catches the "correct but ungrounded"
    # cell (a right-looking answer the retrieved context doesn't actually support). Present only
    # when faithfulness ran (faith_verdict set); absent on a --no-faithfulness run.
    faith_summary = faithfulness_report(results)

    # ---- CAPABILITY slices (outcome + trajectory, grouped by what each Q tests) ----
    per_capability = capability_slices(results)
    logger.info(f"\nBY CAPABILITY (outcome | trajectory):")
    for cap, s in per_capability.items():
        traj_part = f"{s['traj_passed']}/{s['traj_asserted']}" if s["traj_asserted"] else "—"
        logger.info(f"  {cap:<24} outcome {s['outcome_ok']}/{s['total']}   trajectory {traj_part}")

    # ---- CONTEXT COST axis (the silent one — context-eng wins show here, not in correctness) ----
    context_cost = context_cost_report(results)
    logger.info("\nCONTEXT COST (the silent axis — context-eng wins show here, not in correctness):")
    logger.info(f"  structural (provider-neutral): ~{context_cost['avg_evidence_chunks']:.1f} evidence chunk(s)/Q | "
                f"~{context_cost['avg_evidence_chars']:.0f} chars/Q assembled")
    logger.info(f"  billed (prompt tokens): controller ~{context_cost['avg_controller_prompt_tokens']:.0f}/Q | "
                f"generator ~{context_cost['avg_generator_prompt_tokens']:.0f}/Q")

    # ---- Cost + aggregate trajectory (reused) ----
    ctrl_total, gen_total, judge_total = Usage(), Usage(), Usage()
    for r in results:
        ctrl_total = ctrl_total + r.controller_usage
        gen_total = gen_total + r.generator_usage
        judge_total = judge_total + r.judge_usage
    n = len(results) or 1
    logger.info(f"\n  cost (tokens): controller {ctrl_total.total_tokens} in {ctrl_total.calls} call(s) "
                f"(~{ctrl_total.calls / n:.1f}/Q) | generator {gen_total.total_tokens} in {gen_total.calls} call(s) "
                f"| judge {judge_total.total_tokens} [instrument]")
    # LATENCY (wall-clock) — the operational meter the cache moves most. A warm cache re-run drops
    # this toward ~0 (hits are lookups, not API calls), so it's the number that makes caching visible.
    # Production latency = controller + generator (judge is instrument-only); reported per-Q so it
    # reads as an A/B gauge. NOTE: sums per-call time, so it reflects SERIAL wall-clock (our eval is
    # sequential); it would overcount if calls ran concurrently.
    system_latency = ctrl_total.latency_s + gen_total.latency_s
    logger.info(f"  latency (wall-clock s): controller {ctrl_total.latency_s:.1f} | generator {gen_total.latency_s:.1f} "
                f"| system ~{system_latency / n:.2f}/Q | judge {judge_total.latency_s:.1f} [instrument]")
    traj_aggregate = aggregate_trajectory(results)   # reused; reads r.trajectory
    guardrails = None
    if traj_aggregate is not None:
        logger.info(f"  trajectory totals: ~{traj_aggregate['avg_rounds']:.1f} round(s)/Q | "
                    f"tool calls {traj_aggregate['tool_calls']} | exits {traj_aggregate['exit_reasons']} | "
                    f"tool_errors {traj_aggregate['tool_errors']} | redundant {traj_aggregate['redundant_searches']}")
        # GUARDRAILS (Module 5, Safety ring) — both flag-first, so this line should read all-clear on
        # the champion: every answer grounded, zero spend_cap exits. A non-zero here is the signal.
        traj_n = traj_aggregate["questions"]
        ungrounded = traj_aggregate["ungrounded_answers"]
        spend_cap_exits = traj_aggregate["exit_reasons"].get("spend_cap", 0)
        guardrails = {
            "grounded_answers": traj_n - ungrounded,
            "questions": traj_n,
            "ungrounded_answers": ungrounded,
            "ungrounded_citations": traj_aggregate["ungrounded_citations"],
            "spend_cap_exits": spend_cap_exits,
        }
        detail = f" (fabricated: {traj_aggregate['ungrounded_citations']})" if ungrounded else ""
        logger.info(f"  guardrails: grounding {traj_n - ungrounded}/{traj_n} answers grounded{detail} "
                    f"| spend_cap exits {spend_cap_exits}")

    return {
        "generator_model": generator_model,
        "judge_model": judge_model,
        "outcome": {
            "answerable": len(answerable),
            "answered": len(answered),
            "false_abstention": len(false_abstentions),
            "correct": len(correct),
            "partial": len(partial),
            "incorrect": len(incorrect),
            "ungraded": len(ungraded),
            "graded": n_graded,
            "end_to_end_success": end_to_end,                       # STRICT: full-CORRECT / answerable
            "correctness_partial_credit_answerable": partial_credit_answerable,  # CORRECT=1, PARTIAL=0.5 / answerable
            "correctness_partial_credit_overall": partial_credit_overall,        # + correct abstentions, / all graded
            "abstention_total": len(abstention),
            "abstained_correctly": len(abstained_correctly),
        },
        "trajectory_diagnostic": {
            "asserted": len(asserted),
            "passed": len(traj_passed),
            "failed": len(traj_failed),
            "pass_rate": traj_rate,
            "unassertable": len(unassertable),
        },
        "faithfulness": faith_summary,   # None on a --no-faithfulness run
        "by_capability": per_capability,
        "trajectory_totals": traj_aggregate,
        "context_cost": context_cost,    # the silent context-engineering axis (structural + billed)
        "latency_s": {                   # operational meter (Module 5) — the cache moves this most
            "controller": round(ctrl_total.latency_s, 2),
            "generator": round(gen_total.latency_s, 2),
            "system_per_q": round((ctrl_total.latency_s + gen_total.latency_s) / n, 3),
            "judge": round(judge_total.latency_s, 2),
        },
        "guardrails": guardrails,        # Safety ring (Module 5) — grounding flags + spend_cap exits; None on naive
    }


def faithfulness_report(results: List[AgenticResult]) -> Optional[dict]:
    """Log the FAITHFULNESS axis and return its summary (or None if faithfulness didn't run).

    Mirrors faithfulness.py's report so the agentic run and the standalone faithfulness run report
    the same shape. 'judged' = every answered question faithfulness ran on (including a question
    that wrongly answered instead of abstaining — fabrication is what this is meant to catch). The
    rate excludes UNGRADED (an unparseable judge verdict is a missing measurement, not a failure).
    """
    judged = [r for r in results if r.faith_verdict is not None]
    if not judged:
        return None

    supported = [r for r in judged if r.faith_verdict == "SUPPORTED"]
    partial = [r for r in judged if r.faith_verdict == "PARTIALLY_SUPPORTED"]
    unsupported = [r for r in judged if r.faith_verdict == "UNSUPPORTED"]
    ungraded = [r for r in judged if r.faith_verdict == UNGRADED]
    cite_hallucinations = [r for r in judged if r.unsupported_citations]
    n_graded = len(judged) - len(ungraded)
    rate = len(supported) / n_graded if n_graded else 0.0

    logger.info("\nFAITHFULNESS (generator groundedness — reference-free; orthogonal to correctness):")
    any_unfaithful = False
    for r in unsupported:
        any_unfaithful = True
        logger.info(f"  {r.q.id}  UNSUPPORTED — {_last_line(r.faith_reason)}")
    for r in partial:
        any_unfaithful = True
        logger.info(f"  {r.q.id}  PARTIALLY_SUPPORTED — {_last_line(r.faith_reason)}")
    for r in cite_hallucinations:
        any_unfaithful = True
        logger.info(f"  {r.q.id}  CITED-NOT-RETRIEVED: {', '.join(r.unsupported_citations)}")
    if not any_unfaithful:
        logger.info("  (none)")
    if ungraded:
        logger.info(f"  UNGRADED ({len(ungraded)}): {', '.join(r.q.id for r in ungraded)} (re-run to grade)")
    logger.info(f"  faithfulness (fully SUPPORTED): {len(supported)}/{n_graded} = {rate:.3f}   "
                f"[excludes {len(ungraded)} ungraded] | citation hallucinations {len(cite_hallucinations)}/{len(judged)}")

    return {
        "judged": len(judged),
        "supported": len(supported),
        "partial": len(partial),
        "unsupported": len(unsupported),
        "ungraded": len(ungraded),
        "graded": n_graded,
        "faithfulness": rate,
        "citation_hallucinations": len(cite_hallucinations),
    }


def context_cost_report(results: List[AgenticResult]) -> dict:
    """The CONTEXT-ENGINEERING cost axis: how big a window we assembled per question, and
    what it billed. Context-eng wins are often SILENT on correctness but show here — a
    dedup/prune/compact lever shrinks the assembled context while the answer stays the same.
    Two meter kinds, on purpose:
      - STRUCTURAL (provider-neutral): evidence chunks + chars actually assembled. What the
        context levers directly move; comparable across providers (counts-not-dollars, runs.py)
        — so this is the clean A/B gauge.
      - BILLED (provider-specific): prompt/context tokens paid, split controller (accumulates
        over rounds) vs generator (the final-answer window) — the reality check, but it shifts
        if you swap models, so it's noisier as a gauge.
    Averaged per question so it reads as an A/B gauge.
    """
    n = len(results) or 1
    total_chunks = sum(len(r.retrieved) for r in results)
    total_chars = sum(len(h.text) for r in results for h in r.retrieved)
    ctrl_prompt = sum(r.controller_usage.prompt_tokens for r in results)
    gen_prompt = sum(r.generator_usage.prompt_tokens for r in results)
    return {
        "avg_evidence_chunks": total_chunks / n,
        "avg_evidence_chars": total_chars / n,
        "avg_controller_prompt_tokens": ctrl_prompt / n,
        "avg_generator_prompt_tokens": gen_prompt / n,
    }


def capability_slices(results: List[AgenticResult]) -> dict:
    """Group results by capability tag; report outcome and trajectory pass per slice.

    This is what turns one blurred score into 'tool_selection 2/3, decomposition 2/2' —
    the whole reason the set is organized by capability (AGENT_EVAL_SETS.md A2).
    """
    slices: dict = {}
    for r in results:
        cap = r.q.capability or "(untagged)"
        s = slices.setdefault(cap, {"total": 0, "outcome_ok": 0, "traj_asserted": 0, "traj_passed": 0})
        s["total"] += 1
        if outcome_ok(r):
            s["outcome_ok"] += 1
        if r.traj_pass is not None:
            s["traj_asserted"] += 1
            if r.traj_pass:
                s["traj_passed"] += 1
    return slices


def _last_line(text: str) -> str:
    """The judge's final line — its verdict rationale — for the one-line failure log."""
    return text.splitlines()[-1] if text else ""


# ───────────────────────────── persistence ─────────────────────────────

def persist(summary: dict, results: List[AgenticResult], config: dict, version: str) -> None:
    """Write a self-describing run record to eval_runs/agentic/v<version>/<timestamp>.json."""
    path = eval_run_path("agentic", version)

    ctrl_total, gen_total, judge_total = Usage(), Usage(), Usage()
    per_question = []
    for r in results:
        ctrl_total = ctrl_total + r.controller_usage
        gen_total = gen_total + r.generator_usage
        judge_total = judge_total + r.judge_usage
        per_question.append({
            "id": r.q.id,
            "type": r.q.type,
            "capability": r.q.capability,
            "question": r.q.question,
            "should_abstain": r.q.should_abstain,
            "abstained": r.abstained,
            "verdict": r.verdict,
            "outcome_ok": outcome_ok(r),
            "answer": r.answer,
            "expected_answer": r.q.expected_answer,
            "judge_reason": r.judge_reason,
            "faith_verdict": r.faith_verdict,
            "faith_reason": r.faith_reason,
            "unsupported_citations": r.unsupported_citations,
            "trajectory": r.trajectory.as_dict() if r.trajectory else None,
            "trajectory_pass": r.traj_pass,
            "trajectory_checks": [
                {"key": c.key, "expected": c.expected, "actual": c.actual, "passed": c.passed}
                for c in r.traj_checks
            ],
            "usage": usage_record(r.controller_usage, r.generator_usage, r.judge_usage),
            "retrieved": [{"source": h.source, "chunk_index": h.chunk_index,
                           "score": round(h.score, 4), "retrieved_by": h.retrieved_by,
                           "text": h.text} for h in r.retrieved],
        })

    run_config = {
        "dataset_version": version,
        "generator_model": role_model(config, "generator"),
        "judge_model": role_model(config, "judge"),
        "temperature": config["llm"]["temperature"],
        "max_tokens": config["llm"]["max_tokens"],
        "retrieval": retrieval_config_snapshot(config),
        "agent": agent_config_snapshot(config),
        "context": context_config_snapshot(config),
    }
    record = {
        "timestamp": path.stem,
        "metric": "agentic",
        "config": run_config,
        "summary": summary,
        "usage": usage_record(ctrl_total, gen_total, judge_total),
        "per_question": per_question,
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info(f"\n[saved] {path}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Score the agentic capability set (outcome + trajectory).")
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N questions.")
    parser.add_argument("--dataset", default=DEFAULT_AGENTIC_DATASET,
                        help="Agentic eval dataset YAML (default: evals/datasets/agentic.yaml).")
    parser.add_argument("--ids", default=None, help="Comma-separated question ids (e.g. a05,a06).")
    parser.add_argument("--no-save", action="store_true", help="Don't write a JSON run record.")
    parser.add_argument("--no-faithfulness", action="store_true",
                        help="Skip the faithfulness judge call (cheaper; outcome + trajectory only).")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force the LLM response cache OFF (overrides config.cache.enabled) — "
                             "the cold, uncached A/B control for measuring the cache's effect.")
    args = parser.parse_args()
    ids = [token.strip() for token in args.ids.split(",")] if args.ids else None
    configure_run_logging("evals/agentic")
    run(save=not args.no_save, limit=args.limit, dataset=args.dataset, ids=ids,
        run_faithfulness=not args.no_faithfulness,
        cache_enabled=False if args.no_cache else None)


if __name__ == "__main__":
    main()
