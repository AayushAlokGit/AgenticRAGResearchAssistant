"""Sequenced memory eval — the harness that can actually see episodic memory (Module 4, slice 1).

`agentic_eval.py` scores independent one-shot questions and so cannot detect memory. This
harness runs a SESSION (one dataset's questions IN ORDER with a persistent `EpisodicStore`)
TWICE — memory OFF (baseline) vs ON — then diffs three meters:

  COST WIN  : rounds + tokens on the REPEATS, ON vs OFF (should drop)
  OUTCOME   : correctness, ON vs OFF (must NOT drop)
  PRECISION : DISTRACTORS must stay correct ON (no false-recall to the seed's different answer)

Recall is threshold-free (DD-045): the store only ranks episodes; the agent decides whether to
use the hint, so precision is read from OUTCOME, not a cutoff (recall similarity is diagnostic).

Session fields: `session: 1|2` triggers a simulated RESTART at the boundary (store reloaded from
disk); `role` (seed/repeat/distractor/novel/chain) drives the per-role report. The staleness
session is NOT wired — it needs a corpus fixture swap + re-ingest mid-session (see run_session).

Run (needs an ingested store + agent ON in config + provider keys):
    python -m agentic_rag.evals.memory_eval --session recall
    python -m agentic_rag.evals.memory_eval --session persistence
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from agentic_rag.agent.loop import DEFAULT_TOOLS, build_agent_deps, run_agent
from agentic_rag.config import load_config, resolve_path
from agentic_rag.evals.answer_correctness import is_abstention, judge_correctness
from agentic_rag.llm.provider import Usage, build_llm, role_model
from agentic_rag.logging_setup import configure_run_logging
from agentic_rag.memory.episodic import EpisodicStore
from agentic_rag.rag.answer import load_prompt
from agentic_rag.rag.embeddings import LocalEmbedder
from agentic_rag.rag.ingest import ingest
from agentic_rag.rag.retriever import build_retriever

logger = logging.getLogger(__name__)

SESSIONS = {
    "recall": "evals/datasets/memory_session.yaml",                 # capability A + B
    "persistence": "evals/datasets/memory_session_persistence.yaml",  # capability C (restart)
    "accumulation": "evals/datasets/memory_session_accumulation.yaml",  # capability E (headroom)
    "consolidation": "evals/datasets/memory_session_consolidation.yaml",  # capability E slice 2 (3-way)
    "staleness": "evals/datasets/memory_session_staleness.yaml",     # capability D (corpus swap → forget)
}


# ───────────────────────────── dataset (memory-specific fields) ─────────────────────────────

@dataclass
class MemoryQuestion:
    """One session question, including the memory-specific fields agentic.yaml's loader ignores."""
    id: str
    role: str                      # seed | repeat | distractor | novel | chain
    question: str
    expected_answer: str
    expected_sources: List[str]
    should_abstain: bool
    type: str = ""
    relates_to: Optional[str] = None
    expect_recall: Optional[bool] = None
    expect_fresh: Optional[bool] = None   # staleness: answer MUST reflect the current source, not a cached one
    session: int = 1               # restart-group (persistence); 1 if absent
    notes: str = ""


def load_session(path: str) -> List[MemoryQuestion]:
    """Parse a session YAML, keeping the memory fields (we read the raw YAML, not the shared loader)."""
    raw = yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))
    questions = []
    for q in raw["questions"]:
        questions.append(MemoryQuestion(
            id=q["id"],
            role=q.get("role", ""),
            question=q["question"],
            expected_answer=q.get("expected_answer", ""),
            expected_sources=q.get("expected_sources") or [],
            should_abstain=bool(q.get("should_abstain", False)),
            type=q.get("type", ""),
            relates_to=q.get("relates_to"),
            expect_recall=q.get("expect_recall"),
            expect_fresh=q.get("expect_fresh"),
            session=int(q.get("session", 1)),
            notes=q.get("notes", ""),
        ))
    return questions


# ───────────────────────────── per-question result ─────────────────────────────

@dataclass
class QResult:
    id: str
    role: str
    question: str
    memory_on: bool
    answer: str
    abstained: bool
    verdict: Optional[str]         # CORRECT/PARTIALLY_CORRECT/INCORRECT/UNGRADED, or None (abstention)
    outcome_ok: bool
    rounds: int
    tokens: int                    # controller + generator (the cost of answering this question)
    recall_sim: Optional[float] = None      # top recalled similarity (ON only) — diagnostic
    recall_of: Optional[str] = None         # preview of the recalled past question (ON only)
    expect_fresh: Optional[bool] = None     # staleness: answer must reflect the CURRENT source
    judge_reason: str = ""
    retrieved: List[dict] = field(default_factory=list)   # the chunks the agent gathered (for debugging)


def serialize_hits(retrieved) -> List[dict]:
    """Flatten the gathered chunks (source + text + score + which action found it) for the run record."""
    return [{"source": hit.source, "chunk_index": hit.chunk_index, "score": round(hit.score, 4),
             "retrieved_by": hit.retrieved_by, "text": hit.text}
            for hit in retrieved]


# ───────────────────────────── one pass over the session ─────────────────────────────

@dataclass
class Deps:
    retriever: object
    generator_llm: object
    controller_llm: object
    judge_llm: object
    judge_prompt: str
    react_prompt: str
    answer_prompt: str
    consolidate_prompt: str        # used only by the consolidation session (merge prompt)
    registry: object
    store: object
    top_k: int
    max_rounds: int
    answer_char_budget: int
    ordering: str
    recall_k: int
    memory_path: object
    embedder: object
    config: dict        # kept so a pass can rebuild the retriever after a staleness corpus swap


def run_pass(questions: List[MemoryQuestion], deps: Deps, memory_on: bool,
             staleness: bool = False, consolidate_at_boundary: bool = False) -> List[QResult]:
    """Run the whole session in order, once, with memory ON or OFF.

    ON: a single EpisodicStore persists across the session (cleared to empty at the start);
    a change in the `session` field triggers a simulated RESTART (reload from disk). OFF:
    no store at all (the unchanged champion answer path).

    `consolidate_at_boundary` (capability E, consolidation slice): at the session boundary, merge
    the session-1 episodes into synthesis records BEFORE the restart. This is the only difference
    between the "episodic-ON" pass (False — plain recall) and the "consolidated-ON" pass (True),
    so the episodic→consolidated diff isolates exactly what the MERGE adds over plain recall.

    `staleness` (capability D): mutate the CORPUS across the session boundary — stage the v1
    fixture before session 1, swap to v2 at the boundary, remove it in `finally`. The corpus
    changes in BOTH passes (the world changed regardless of memory), so this is independent of
    `memory_on`. After each mutation the retriever is rebuilt (BM25 reindex). The point of the
    test: in session 2 a stale ON answer (cached v1) loses to the fresh corpus (v2).
    """
    memory_store = None
    if memory_on:
        memory_store = EpisodicStore(deps.memory_path, deps.embedder)
        memory_store.clear()   # every ON pass starts from an empty store

    corpus_dir = resolve_path(deps.config["corpus"]["root"]) if staleness else None
    if staleness:
        stage_fixture(1, corpus_dir)               # session-1 corpus = v1
        deps.retriever = build_retriever(deps.config)

    results: List[QResult] = []
    current_session = None
    try:
        for q in questions:
            boundary = current_session is not None and q.session != current_session
            # At a session boundary: optionally CONSOLIDATE the session-1 episodes, then drop RAM
            # and reload from disk (memory only). Consolidate BEFORE reload so the synthesis is
            # saved and comes back on the reload (and so it survives the simulated restart).
            if boundary and memory_on:
                if consolidate_at_boundary:
                    created = memory_store.consolidate(deps.generator_llm, deps.consolidate_prompt)
                    logger.info("memory-eval: --- CONSOLIDATE (session %s -> %s): %d synthesis record(s) created ---",
                                current_session, q.session, created)
                logger.info("memory-eval: --- SIMULATED RESTART (session %s -> %s): reloading store from disk ---",
                            current_session, q.session)
                memory_store.reload()
            # Corpus change at the boundary (staleness): v1 -> v2, then rebuild the retriever.
            if boundary and staleness:
                logger.info("memory-eval: --- CORPUS CHANGE (session %s -> %s): swapping fixture v1 -> v2 ---",
                            current_session, q.session)
                stage_fixture(2, corpus_dir)
                deps.retriever = build_retriever(deps.config)
            current_session = q.session

            # Diagnostic ONLY: what would be recalled (sim + which past question). Does not gate anything.
            recall_sim = None
            recall_of = None
            if memory_on:
                top = memory_store.read(q.question, k=1)
                if top:
                    episode, sim = top[0]
                    recall_sim = round(sim, 3)
                    recall_of = episode.get("question", "")[:60]

            generated = run_agent(
                q.question, deps.retriever, deps.generator_llm, deps.react_prompt, deps.answer_prompt,
                deps.top_k, deps.max_rounds, deps.answer_char_budget, deps.registry, deps.store,
                controller_llm=deps.controller_llm, ordering=deps.ordering,
                memory_store=memory_store, recall_k=deps.recall_k)

            abstained = is_abstention(generated.answer)
            verdict = None
            reason = ""
            if not q.should_abstain and not abstained:
                verdict, reason, _ = judge_correctness(
                    deps.judge_llm, deps.judge_prompt, q.question, q.expected_answer, generated.answer)
            outcome_ok = abstained if q.should_abstain else (verdict == "CORRECT")

            rounds = generated.trajectory.rounds_used if generated.trajectory else 0
            tokens = generated.controller_usage.total_tokens + generated.generator_usage.total_tokens
            results.append(QResult(
                id=q.id, role=q.role, question=q.question, memory_on=memory_on, answer=generated.answer,
                abstained=abstained, verdict=verdict, outcome_ok=outcome_ok, rounds=rounds, tokens=tokens,
                recall_sim=recall_sim, recall_of=recall_of, expect_fresh=q.expect_fresh, judge_reason=reason,
                retrieved=serialize_hits(generated.retrieved)))
            tag = "ABSTAIN" if abstained else (verdict or "-")
            recall_tag = f" recall~{recall_sim}" if recall_sim is not None else ""
            logger.info("  [%s] %-10s mem=%s  %-9s rounds=%d tokens=%d%s",
                        q.id, q.role, "ON " if memory_on else "OFF", tag, rounds, tokens, recall_tag)
    finally:
        if staleness:
            cleanup_fixture(corpus_dir)            # restore the corpus to baseline even on error
            deps.retriever = build_retriever(deps.config)
    return results


# ───────────────────────────── the report (ON vs OFF, three meters) ─────────────────────────────

def report(off: List[QResult], on: List[QResult]) -> dict:
    """Diff the OFF and ON passes on the three memory meters, by role."""
    off_by_id = {r.id: r for r in off}
    on_by_id = {r.id: r for r in on}
    ids = [r.id for r in off]

    def role_of(rid):  # role is identical across passes
        return off_by_id[rid].role

    # ---- COST WIN: repeats only (rounds + tokens, OFF vs ON) ----
    repeats = [rid for rid in ids if role_of(rid) == "repeat"]
    logger.info("\n=== COST WIN (repeats: should be CHEAPER ON) ===")
    cost_rows = []
    for rid in repeats:
        o, n = off_by_id[rid], on_by_id[rid]
        d_rounds = o.rounds - n.rounds
        d_tokens = o.tokens - n.tokens
        cost_rows.append({"id": rid, "off_rounds": o.rounds, "on_rounds": n.rounds,
                          "off_tokens": o.tokens, "on_tokens": n.tokens,
                          "recall_sim": n.recall_sim,
                          "outcome_off": o.outcome_ok, "outcome_on": n.outcome_ok})
        logger.info("  %s  rounds %d->%d (%+d)  tokens %d->%d (%+d)  recall~%s  outcome %s->%s",
                    rid, o.rounds, n.rounds, -d_rounds, o.tokens, n.tokens, -d_tokens,
                    n.recall_sim, _ok(o.outcome_ok), _ok(n.outcome_ok))
    if repeats:
        avg_off_rounds = sum(off_by_id[r].rounds for r in repeats) / len(repeats)
        avg_on_rounds = sum(on_by_id[r].rounds for r in repeats) / len(repeats)
        avg_off_tokens = sum(off_by_id[r].tokens for r in repeats) / len(repeats)
        avg_on_tokens = sum(on_by_id[r].tokens for r in repeats) / len(repeats)
        logger.info("  AVG  rounds %.1f->%.1f  tokens %.0f->%.0f", avg_off_rounds, avg_on_rounds,
                    avg_off_tokens, avg_on_tokens)
    else:
        avg_off_rounds = avg_on_rounds = avg_off_tokens = avg_on_tokens = 0.0

    # ---- PRECISION: distractors must stay correct ON (no false-recall) ----
    distractors = [rid for rid in ids if role_of(rid) == "distractor"]
    logger.info("\n=== PRECISION (distractors: must stay CORRECT ON despite a near-duplicate prior) ===")
    false_recalls = []
    for rid in distractors:
        o, n = off_by_id[rid], on_by_id[rid]
        broke = o.outcome_ok and not n.outcome_ok   # memory turned a right answer wrong
        if broke:
            false_recalls.append(rid)
        logger.info("  %s  outcome %s->%s  recall~%s  %s", rid, _ok(o.outcome_ok), _ok(n.outcome_ok),
                    n.recall_sim, "<-- FALSE RECALL" if broke else "ok")
    precision_ok = len(distractors) - len(false_recalls)

    # ---- OUTCOME guard + NO-HARM (novel): correctness must not drop anywhere ----
    logger.info("\n=== OUTCOME (must NOT drop ON) ===")
    regressions = [rid for rid in ids if off_by_id[rid].outcome_ok and not on_by_id[rid].outcome_ok]
    off_correct = sum(1 for r in off if r.outcome_ok)
    on_correct = sum(1 for r in on if r.outcome_ok)
    logger.info("  outcome_ok OFF %d/%d -> ON %d/%d", off_correct, len(off), on_correct, len(on))
    if regressions:
        for rid in regressions:
            logger.info("  REGRESSION %s (%s): correct OFF, wrong ON", rid, role_of(rid))
    else:
        logger.info("  no correctness regressions")

    # ---- FRESHNESS (staleness session only): changed-fact repeats must serve the NEW source ON ----
    # Under slice 1 (a write/read cache with NO freshness check) these are EXPECTED to fail — the
    # cached v1 answer wins over the changed v2 corpus. That failure is the point: it motivates forget.
    fresh_tests = [rid for rid in ids if on_by_id[rid].expect_fresh]
    stale_served = []
    if fresh_tests:
        logger.info("\n=== FRESHNESS (staleness: changed-fact repeats must serve the NEW source ON, not a cached answer) ===")
        for rid in fresh_tests:
            o, n = off_by_id[rid], on_by_id[rid]
            served_stale = o.outcome_ok and not n.outcome_ok   # fresh OFF, wrong ON = the cached stale answer won
            if served_stale:
                stale_served.append(rid)
            label = "<-- STALE-SERVED" if served_stale else ("fresh" if n.outcome_ok else "wrong (both passes)")
            logger.info("  %s  fresh-corpus OFF %s -> ON %s  recall~%s  %s",
                        rid, _ok(o.outcome_ok), _ok(n.outcome_ok), n.recall_sim, label)
        controls = [rid for rid in ids if on_by_id[rid].expect_fresh is False and role_of(rid) == "repeat"]
        for rid in controls:
            n = on_by_id[rid]
            logger.info("  [control] %s  ON %s  recall~%s  (UNCHANGED doc — safe recall should still hold)",
                        rid, _ok(n.outcome_ok), n.recall_sim)

    return {
        "n_questions": len(ids),
        "cost_win": {
            "repeats": len(repeats),
            "avg_rounds_off": avg_off_rounds, "avg_rounds_on": avg_on_rounds,
            "avg_tokens_off": avg_off_tokens, "avg_tokens_on": avg_on_tokens,
            "rows": cost_rows,
        },
        "precision": {
            "distractors": len(distractors),
            "stayed_correct": precision_ok,
            "false_recalls": false_recalls,
        },
        "outcome": {
            "off_correct": off_correct, "on_correct": on_correct,
            "total": len(ids), "regressions": regressions,
        },
        "freshness": {
            "fresh_tests": fresh_tests,
            "stale_served": stale_served,   # expected NON-empty under slice 1 (no forgetting yet)
        },
    }


def _ok(flag: bool) -> str:
    return "OK" if flag else "X"


# ───────────────────────────── staleness corpus swap (capability D) ─────────────────────────────
# The staleness session mutates the corpus mid-run: a fictional fixture doc is added (v1), swapped
# to a CHANGED version (v2) at the session boundary, then removed. It's copied to a STABLE filename
# so the content-hash tracker sees a *changed* file on swap (not a new one) and re-embeds it. After
# every mutation the caller must rebuild the retriever — the BM25 index is built once at construction
# and would otherwise keep serving the old text.

STALENESS_FIXTURE_DIR = "evals/fixtures/staleness"
STALENESS_STABLE_NAME = "acme_search_service.md"   # the name the corpus + the eval's expected_sources use


def stage_fixture(version: int, corpus_dir) -> None:
    """Copy fixture v{version} to the stable corpus filename and re-ingest (add or change)."""
    src = resolve_path(f"{STALENESS_FIXTURE_DIR}/acme_search_service.v{version}.md")
    dst = corpus_dir / STALENESS_STABLE_NAME
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    logger.info("staleness: staged fixture v%d -> %s", version, dst.name)
    ingest()


def cleanup_fixture(corpus_dir) -> None:
    """Remove the fixture from the corpus and re-ingest (purges its chunks → corpus back to baseline)."""
    dst = corpus_dir / STALENESS_STABLE_NAME
    if dst.exists():
        dst.unlink()
        logger.info("staleness: removed fixture %s", dst.name)
    ingest()


# ───────────────────────────── orchestration ─────────────────────────────

def build_deps(config: dict) -> Deps:
    retriever = build_retriever(config)
    generator_llm = build_llm(config, role="generator")
    controller_llm = build_llm(config, role="controller")
    judge_llm = build_llm(config, role="judge")
    agent_cfg = config.get("agent", {})
    registry, store = build_agent_deps(config, agent_cfg.get("tools", DEFAULT_TOOLS))
    memory_cfg = config.get("memory", {})
    return Deps(
        retriever=retriever, generator_llm=generator_llm, controller_llm=controller_llm,
        judge_llm=judge_llm, judge_prompt=load_prompt(config, "judge_correctness"),
        react_prompt=load_prompt(config, "agent_react"),
        answer_prompt=load_prompt(config, "answer_with_citations"),
        consolidate_prompt=load_prompt(config, "consolidate"),
        registry=registry, store=store, top_k=config["retrieval"]["top_k"],
        max_rounds=agent_cfg.get("max_rounds", 5),
        answer_char_budget=agent_cfg.get("answer_char_budget", 0),
        ordering=config.get("context", {}).get("ordering", "arrival"),
        recall_k=memory_cfg.get("recall_k", 1),
        memory_path=resolve_path(memory_cfg.get("path", "./chromadb_data/episodic_memory.json")),
        embedder=LocalEmbedder(config["embedding"]["model"]),
        config=config,
    )


def run_session(session: str, save: bool = True) -> dict:
    if session not in SESSIONS:
        raise SystemExit(f"unknown session {session!r}. Available: {sorted(SESSIONS)}.")
    config = load_config()
    if not bool(config.get("agent", {}).get("enabled", False)):
        raise SystemExit("agent.enabled is FALSE — the memory eval needs the agent loop ON.")

    questions = load_session(SESSIONS[session])
    deps = build_deps(config)

    if session == "consolidation":
        return _run_consolidation_session(questions, deps, config, save)

    staleness = session == "staleness"   # this session mutates the corpus across the boundary
    logger.info("=== MEMORY EVAL: session=%s (%d questions) — OFF baseline pass ===", session, len(questions))
    off = run_pass(questions, deps, memory_on=False, staleness=staleness)
    logger.info("\n=== MEMORY EVAL: session=%s — ON pass ===", session)
    on = run_pass(questions, deps, memory_on=True, staleness=staleness)

    summary = report(off, on)
    if save:
        persist(session, summary, {"off": off, "on": on}, config)
    return summary


def _run_consolidation_session(questions: List[MemoryQuestion], deps: Deps, config: dict, save: bool) -> dict:
    """Capability E (consolidation): THREE passes so the win can be attributed.

    OFF (no memory) / EPISODIC-ON (recall, no merge) / CONSOLIDATED-ON (merge at the boundary).
    OFF→episodic shows what plain recall buys; EPISODIC→consolidated isolates what the MERGE adds
    (on the broad questions, episodic recalls only one facet while consolidated recalls the whole).
    """
    n = len(questions)
    logger.info("=== MEMORY EVAL: session=consolidation (%d questions) — pass 1/3: OFF (no memory) ===", n)
    off = run_pass(questions, deps, memory_on=False)
    logger.info("\n=== consolidation — pass 2/3: EPISODIC-ON (recall, NO consolidation) ===")
    episodic = run_pass(questions, deps, memory_on=True, consolidate_at_boundary=False)
    logger.info("\n=== consolidation — pass 3/3: CONSOLIDATED-ON (consolidate at the session boundary) ===")
    consolidated = run_pass(questions, deps, memory_on=True, consolidate_at_boundary=True)

    logger.info("\n######## DIFF A: OFF -> EPISODIC-ON (what plain recall adds) ########")
    rep_off_episodic = report(off, episodic)
    logger.info("\n######## DIFF B: EPISODIC-ON -> CONSOLIDATED-ON (what the MERGE adds — the headline) ########")
    rep_episodic_consolidated = report(episodic, consolidated)

    summary = {"off_vs_episodic": rep_off_episodic, "episodic_vs_consolidated": rep_episodic_consolidated}
    if save:
        persist("consolidation", summary,
                {"off": off, "episodic": episodic, "consolidated": consolidated}, config)
    return summary


def persist(session: str, summary: dict, passes: dict, config: dict) -> None:
    out_dir = resolve_path("./eval_runs/memory")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{session}_{stamp}.json"

    def rows(passi):
        return [{"id": r.id, "role": r.role, "question": r.question, "memory_on": r.memory_on,
                 "verdict": r.verdict, "outcome_ok": r.outcome_ok, "expect_fresh": r.expect_fresh,
                 "rounds": r.rounds, "tokens": r.tokens, "recall_sim": r.recall_sim,
                 "recall_of": r.recall_of, "answer": r.answer, "retrieved": r.retrieved}
                for r in passi]

    record = {
        "timestamp": stamp,
        "metric": "memory",
        "session": session,
        "config": {
            "generator_model": role_model(config, "generator"),
            "judge_model": role_model(config, "judge"),
            "recall_k": config.get("memory", {}).get("recall_k", 1),
            "max_rounds": config.get("agent", {}).get("max_rounds"),
        },
        "summary": summary,
        "passes": {name: rows(results) for name, results in passes.items()},
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("\n[saved] %s", path)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Sequenced memory eval (OFF vs ON, three meters).")
    parser.add_argument("--session", default="recall", choices=sorted(SESSIONS),
                        help="Which session to run (default: recall).")
    parser.add_argument("--no-save", action="store_true", help="Don't write a JSON run record.")
    args = parser.parse_args()
    configure_run_logging("evals/memory")
    run_session(args.session, save=not args.no_save)


if __name__ == "__main__":
    main()
