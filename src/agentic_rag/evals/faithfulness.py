"""Faithfulness eval — rung 2, the reference-FREE half of answer quality.

Where ``answer_correctness.py`` asks "is the answer RIGHT?" (vs a known ``expected_answer``),
this asks "is the answer GROUNDED?" — is every claim in the generated answer supported by
the CHUNKS WE ACTUALLY RETRIEVED, with no smuggled-in outside knowledge? It needs no
reference answer, only the answer + the context the generator saw. The full conceptual
treatment (the orthogonal correctness-vs-faithfulness 2x2, the dangerous "correct but
ungrounded" cell, why this is the metric that can run as a live production guardrail) is in
docs/evals/ANSWER_QUALITY.md — this module is the implementation its Part 6 points to.

Two signals per answered question:

  1. Faithfulness verdict (LLM-as-judge, reference-FREE): a judge reads the retrieved
     CONTEXT and the CANDIDATE answer and returns SUPPORTED / PARTIALLY_SUPPORTED /
     UNSUPPORTED. The judge is told to treat ONLY the context as ground truth and NOT to
     rescue a claim with world knowledge — that instruction is the whole metric.

  2. Citation grounding (DETERMINISTIC, no LLM): every [filename] the answer cites must be
     among the chunks that were actually retrieved. A cited source that wasn't retrieved is
     a citation hallucination — cheap to catch, and something the correctness eval never
     looks at. This mirrors the deterministic abstention check in answer_correctness.

Scope: only ANSWERED questions are judged. An abstention ("Not enough information.") makes
no claims, so it is trivially faithful and skipped. A question that SHOULD have abstained
but answered anyway is judged here too — that's exactly the case faithfulness should flag.

Honest caveats (shared with answer_correctness):
  - Generation + judging are non-deterministic; treat small deltas as noise.
  - The judge is a different model from the generator (DD-013) — no self-eval bias + its own
    Groq daily-token bucket. If config points both roles at one model, the bias returns.
  - A holistic verdict is coarser than RAGAS-style claim decomposition (one verdict for the
    whole answer, not per-claim). It's the cheap first version; decomposition is the measured
    upgrade if this proves too lenient (see ANSWER_QUALITY.md / the granularity fork).
  - An unparseable judge response is UNGRADED (a missing measurement), excluded from the
    rate — not scored UNSUPPORTED. An instrument failure must not look like a real failure.

Run (needs an ingested store + GROQ_API_KEY):
    python -m agentic_rag.evals.faithfulness
    python -m agentic_rag.evals.faithfulness --limit 5    # quick subset
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from agentic_rag.config import load_config
from agentic_rag.evals.answer_correctness import is_abstention, UNGRADED, JUDGE_MAX_ATTEMPTS
from agentic_rag.evals.dataset import EvalQuestion, load_eval_dataset, eval_dataset_version
from agentic_rag.evals.runs import agent_config_snapshot, eval_run_path, retrieval_config_snapshot
from agentic_rag.logging_setup import configure_run_logging
from agentic_rag.agent.loop import build_answerer
from agentic_rag.llm.provider import build_llm, role_model
from agentic_rag.rag.answer import assemble_context, load_prompt
from agentic_rag.rag.retriever import build_retriever
from agentic_rag.rag.vector_store import Hit

logger = logging.getLogger(__name__)

# A citation in an answer looks like [SOME_FILE.md]. We pull the bracket contents and keep
# the ones that look like a filename (have an extension, no internal spaces) — see
# extract_citations. The corpus is .md today, but matching any "name.ext" keeps it general.
CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


@dataclass
class FaithResult:
    q: EvalQuestion
    answer: str
    abstained: bool
    verdict: Optional[str]                  # SUPPORTED/PARTIALLY_SUPPORTED/UNSUPPORTED/UNGRADED, or None if abstained
    judge_reason: str = ""
    cited_sources: List[str] = field(default_factory=list)        # filenames the answer cited
    unsupported_citations: List[str] = field(default_factory=list)  # cited but NOT retrieved
    retrieved: List[Hit] = field(default_factory=list)            # chunks fed to the generator, for diagnosis


# ───────────────────────────── the LLM judge (reference-free) ─────────────────────────────

def parse_faithfulness(raw: str) -> str:
    """Pull the verdict label off the judge's first non-empty line.

    Order matters: UNSUPPORTED and PARTIALLY_SUPPORTED both CONTAIN the substring
    'SUPPORTED', so check the more specific labels before the bare one.
    """
    first_line = ""
    for line in raw.strip().splitlines():
        if line.strip():
            first_line = line.strip().upper()
            break
    if "UNSUPPORTED" in first_line:
        return "UNSUPPORTED"
    if "PARTIAL" in first_line:
        return "PARTIALLY_SUPPORTED"
    if "SUPPORTED" in first_line:
        return "SUPPORTED"
    # No verdict token — a missing measurement, NOT a conservative UNSUPPORTED. The caller
    # excludes UNGRADED from the rate so an instrument failure isn't counted as ungrounded.
    return UNGRADED


def judge_faithfulness(llm, judge_prompt: str, question: str, context: str, candidate: str):
    """Judge groundedness, re-asking a few times if the judge returns no parseable verdict.

    The judge sees the QUESTION (for relevance), the retrieved CONTEXT (the only ground
    truth it may use), and the CANDIDATE answer — but NO reference answer. That's what makes
    this reference-free and runnable where no gold answer exists.
    """
    user_message = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"CANDIDATE ANSWER:\n{candidate}"
    )
    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": user_message},
    ]

    last_raw = ""
    for attempt in range(1, JUDGE_MAX_ATTEMPTS + 1):
        raw = llm.complete(messages)
        if raw is None:        # provider can return None content on an empty completion
            raw = ""
        last_raw = raw.strip()
        verdict = parse_faithfulness(raw)
        if verdict != UNGRADED:
            return verdict, last_raw
        logger.warning("faithfulness judge returned no parseable verdict (attempt %d/%d): %r",
                       attempt, JUDGE_MAX_ATTEMPTS, last_raw[:80])
    return UNGRADED, last_raw


# ───────────────────────── the deterministic citation check ─────────────────────────

def extract_citations(answer: str) -> List[str]:
    """Filenames the answer cited, e.g. [LOCAL_EMBEDDING_SERVICE_DOCUMENTATION.md] -> that name.

    A bracket may hold several comma-separated cites. We keep only filename-like tokens (an
    extension, no internal whitespace) so prose-in-brackets isn't mistaken for a citation.
    """
    cited = []
    for bracket in CITATION_RE.findall(answer):
        for part in bracket.split(","):
            token = part.strip()
            if "." in token and " " not in token:
                cited.append(token)
    return unique_in_order(cited)


def unique_in_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def check_citations(answer: str, retrieved: List[Hit]) -> tuple[List[str], List[str]]:
    """Return (cited_sources, unsupported_citations).

    A citation is unsupported when the answer cites a filename that was NOT among the chunks
    actually retrieved for this question — a citation hallucination. Deterministic, no LLM.
    """
    retrieved_sources = {hit.source for hit in retrieved}
    cited = extract_citations(answer)
    unsupported = [name for name in cited if name not in retrieved_sources]
    return cited, unsupported


# ───────────────────────────── running ─────────────────────────────

def run(save: bool = True, limit: Optional[int] = None, dataset: Optional[str] = None,
        ids: Optional[List[str]] = None) -> dict:
    config = load_config()
    version = eval_dataset_version(dataset)

    # Build once, reuse across questions (embedder/store/bm25 are expensive). Generator and
    # judge are SEPARATE models (DD-013): separate Groq buckets + a different-family judge.
    retriever = build_retriever(config)
    generator_llm = build_llm(config, role="generator")
    judge_llm = build_llm(config, role="judge")
    # The answerer is the naive-vs-agentic switch (config agent.enabled). Built once, reused.
    answerer = build_answerer(config, retriever, generator_llm)
    judge_prompt = load_prompt(config, "judge_faithfulness")

    logger.info(f"eval dataset version: v{version}")
    questions = load_eval_dataset(dataset)
    if ids:
        wanted = set(ids)
        selected = []
        for q in questions:
            if q.id in wanted:
                selected.append(q)
        questions = selected
    if limit is not None:
        questions = questions[:limit]

    logger.info(f"Scoring {len(questions)} questions (generate + faithfulness-judge LLM calls)...\n")
    results = []
    for i, q in enumerate(questions, start=1):
        generated = answerer.answer(q.question)
        abstained = is_abstention(generated.answer)

        verdict = None
        reason = ""
        cited: List[str] = []
        unsupported: List[str] = []
        if not abstained:
            # Judge groundedness against the EXACT context the generator saw.
            context = assemble_context(generated.retrieved)
            verdict, reason = judge_faithfulness(judge_llm, judge_prompt, q.question, context, generated.answer)
            cited, unsupported = check_citations(generated.answer, generated.retrieved)

        result = FaithResult(q, generated.answer, abstained, verdict, reason, cited, unsupported, generated.retrieved)
        results.append(result)
        status = "abstained (skipped)" if abstained else verdict
        cite_flag = f"  CITE-HALLUC: {', '.join(unsupported)}" if unsupported else ""
        logger.info(f"  [{i:2d}/{len(questions)}] {q.id} {q.type:<10} {status}{cite_flag}")

    summary = report(results, config)
    if save:
        persist(summary, results, config, version)
    return summary


# ───────────────────────────── printing + saving ─────────────────────────────

def report(results: List[FaithResult], config: dict) -> dict:
    judged = [r for r in results if not r.abstained]
    abstained = [r for r in results if r.abstained]

    supported = [r for r in judged if r.verdict == "SUPPORTED"]
    partial = [r for r in judged if r.verdict == "PARTIALLY_SUPPORTED"]
    unsupported = [r for r in judged if r.verdict == "UNSUPPORTED"]
    ungraded = [r for r in judged if r.verdict == UNGRADED]
    citation_hallucinations = [r for r in judged if r.unsupported_citations]

    generator_model = role_model(config, "generator")
    judge_model = role_model(config, "judge")
    same_model = generator_model == judge_model
    logger.info(f"\n=== Faithfulness (generator={generator_model}, judge={judge_model}) ===")
    if same_model:
        logger.info("note: same model generates and grades (self-eval bias); LLM output is non-deterministic\n")
    else:
        logger.info("note: judge is a different model (no self-eval bias); LLM output is non-deterministic\n")

    # List what you act on: ungrounded answers + citation hallucinations.
    logger.info("FAILURES:")
    any_failure = False
    for r in unsupported:
        any_failure = True
        logger.info(f"  {r.q.id}  UNSUPPORTED — {r.judge_reason.splitlines()[-1] if r.judge_reason else ''}")
    for r in partial:
        any_failure = True
        logger.info(f"  {r.q.id}  PARTIALLY_SUPPORTED — {r.judge_reason.splitlines()[-1] if r.judge_reason else ''}")
    for r in citation_hallucinations:
        any_failure = True
        logger.info(f"  {r.q.id}  CITED-NOT-RETRIEVED: {', '.join(r.unsupported_citations)}")
    if not any_failure:
        logger.info("  (none)")

    if ungraded:
        logger.info("UNGRADED (judge gave no verdict; excluded from the rate):")
        for r in ungraded:
            logger.info(f"  {r.q.id}  (re-run to grade)")

    # Faithfulness rate over the GRADED answered set: exclude ungraded from the denominator.
    n_judged = len(judged)
    n_graded = n_judged - len(ungraded)
    faithfulness = len(supported) / n_graded if n_graded else 0.0

    logger.info(f"\nSUMMARY")
    logger.info(f"  answered ({n_judged}); abstained/skipped ({len(abstained)})")
    logger.info(f"    of answered -> SUPPORTED={len(supported)} PARTIAL={len(partial)} UNSUPPORTED={len(unsupported)} UNGRADED={len(ungraded)}")
    logger.info(f"    faithfulness (fully SUPPORTED): {len(supported)}/{n_graded} = {faithfulness:.3f}   [excludes {len(ungraded)} ungraded]")
    logger.info(f"    citation hallucinations (cited a non-retrieved source): {len(citation_hallucinations)}/{n_judged}")

    return {
        "generator_model": generator_model,
        "judge_model": judge_model,
        "answered": n_judged,
        "abstained": len(abstained),
        "supported": len(supported),
        "partial": len(partial),
        "unsupported": len(unsupported),
        "ungraded": len(ungraded),
        "graded": n_graded,  # answered minus ungraded; the faithfulness denominator
        "faithfulness": faithfulness,
        "citation_hallucinations": len(citation_hallucinations),
    }


def persist(summary: dict, results: List[FaithResult], config: dict, version: str) -> None:
    """Write a run record to eval_runs/faithfulness/v<version>/<timestamp>.json (gitignored)."""
    path = eval_run_path("faithfulness", version)

    per_question = []
    for r in results:
        # The retrieved chunks (source + text) ARE the ground truth the judge ruled against,
        # so record them: an UNSUPPORTED verdict can be re-checked offline by reading the
        # exact passages and seeing whether the disputed claim was really absent.
        retrieved = []
        for hit in r.retrieved:
            retrieved.append({
                "source": hit.source,
                "chunk_index": hit.chunk_index,
                "score": round(hit.score, 4),
                "text": hit.text,
            })
        per_question.append({
            "id": r.q.id,
            "type": r.q.type,
            "abstained": r.abstained,
            "verdict": r.verdict,
            "answer": r.answer,
            "judge_reason": r.judge_reason,
            "cited_sources": r.cited_sources,
            "unsupported_citations": r.unsupported_citations,
            "expected_sources": r.q.expected_sources,
            "retrieved": retrieved,
        })

    run_config = {
        "dataset_version": version,
        "generator_model": role_model(config, "generator"),
        "judge_model": role_model(config, "judge"),
        "temperature": config["llm"]["temperature"],
        "max_tokens": config["llm"]["max_tokens"],
        "retrieval": retrieval_config_snapshot(config),
        "agent": agent_config_snapshot(config),
    }

    record = {
        "timestamp": path.stem,
        "metric": "faithfulness",
        "config": run_config,
        "summary": summary,
        "per_question": per_question,
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info(f"\n[saved] {path}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Score answer faithfulness (groundedness) over the seed eval set.")
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N questions (quick check).")
    parser.add_argument("--dataset", default=None,
                        help="Eval dataset YAML to score against (default: config eval.dataset).")
    parser.add_argument("--ids", default=None,
                        help="Comma-separated question ids to score (e.g. q22,q26,q41) — a targeted subset.")
    parser.add_argument("--no-save", action="store_true", help="Don't write a JSON run record.")
    args = parser.parse_args()
    ids = [token.strip() for token in args.ids.split(",")] if args.ids else None
    configure_run_logging("evals/faithfulness")
    run(save=not args.no_save, limit=args.limit, dataset=args.dataset, ids=ids)


if __name__ == "__main__":
    main()
