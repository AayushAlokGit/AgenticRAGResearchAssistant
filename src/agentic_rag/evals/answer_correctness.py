"""Answer-correctness eval — rung 2: LLM-as-judge + deterministic abstention check.

Where retrieval recall (rung 1) asks "did we find the right document?", this asks the
end-to-end question: "did the system produce the RIGHT ANSWER, or correctly refuse?"

It measures CORRECTNESS (answer vs the known-correct reference) + abstention — NOT
faithfulness (answer grounded in the retrieved context). Those are orthogonal axes; see
docs/evals/ANSWER_QUALITY.md. Faithfulness is a separate, later eval.

Two signals per question:

  1. Abstention (deterministic, no LLM): does the answer start with "Not enough
     information."? Abstention questions (q10-q12) SHOULD abstain; answerable questions
     should NOT (a false abstention — e.g. q01, whose fact-chunk isn't retrieved — is a
     failure the retrieval recall metric can't see).

  2. Answer correctness (LLM-as-judge, reference-based): for answerable questions that did
     answer, a judge compares the generated answer to the verified `expected_answer` and
     returns CORRECT / PARTIALLY_CORRECT / INCORRECT.

Honest caveats:
  - The judge is the SAME Groq model that generates the answers (self-evaluation bias).
    Acceptable for a baseline; a stronger/different judge model is a later upgrade.
  - Generation and judging are non-deterministic (even at temperature 0), so this metric
    has run-to-run variance — unlike the deterministic retrieval recall. Treat small
    deltas as noise; look for clear movement.
  - Faithfulness (is the answer grounded in the retrieved context, no hallucination?) is a
    SEPARATE reference-free judge, added as the next layer.

Run (needs an ingested store + GROQ_API_KEY):
    python -m agentic_rag.evals.answer_correctness
    python -m agentic_rag.evals.answer_correctness --limit 5    # quick subset
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from agentic_rag.config import load_config
from agentic_rag.evals.dataset import EvalQuestion, load_eval_dataset
from agentic_rag.evals.runs import eval_run_path
from agentic_rag.logging_setup import configure_run_logging
from agentic_rag.llm.provider import build_llm
from agentic_rag.rag.answer import generate_answer, load_prompt
from agentic_rag.rag.retriever import build_retriever
from agentic_rag.rag.vector_store import Hit

logger = logging.getLogger(__name__)

ABSTENTION_PHRASE = "not enough information"


@dataclass
class QAResult:
    q: EvalQuestion
    answer: str
    abstained: bool
    verdict: Optional[str]      # CORRECT/PARTIALLY_CORRECT/INCORRECT, or None if not judged
    judge_reason: str = ""
    retrieved: List[Hit] = field(default_factory=list)   # chunks fed to the generator, for diagnosis


def is_abstention(answer: str) -> bool:
    """True if the answer is the fixed refusal phrase the prompt instructs."""
    return answer.strip().lower().startswith(ABSTENTION_PHRASE)


def parse_verdict(raw: str) -> str:
    """Pull the verdict label off the judge's first non-empty line.

    Order matters: check PARTIAL and INCORRECT before CORRECT, since 'PARTIALLY_CORRECT'
    and 'INCORRECT' both contain the substring 'CORRECT'.
    """
    first_line = ""
    for line in raw.strip().splitlines():
        if line.strip():
            first_line = line.strip().upper()
            break
    if "PARTIAL" in first_line:
        return "PARTIALLY_CORRECT"
    if "INCORRECT" in first_line:
        return "INCORRECT"
    if "CORRECT" in first_line:
        return "CORRECT"
    return "INCORRECT"  # unparseable -> grade conservatively


def judge_correctness(llm, judge_prompt: str, question: str, reference: str, candidate: str):
    user_message = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"CANDIDATE ANSWER:\n{candidate}"
    )
    raw = llm.complete([
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": user_message},
    ])
    return parse_verdict(raw), raw.strip()


def describe(result: QAResult) -> str:
    """Short live-progress status string for one question."""
    if result.q.should_abstain:
        return "abstained OK" if result.abstained else "FAILED TO ABSTAIN"
    if result.abstained:
        return "FALSE ABSTENTION"
    return result.verdict


def run(save: bool = True, limit: Optional[int] = None) -> dict:
    config = load_config()

    # Build everything ONCE and reuse across questions (the embedder/store/bm25 are
    # expensive to construct; the LLM is reused for both generation and judging).
    retriever = build_retriever(config)
    llm = build_llm(config)
    system_prompt = load_prompt(config, "answer_with_citations")
    judge_prompt = load_prompt(config, "judge_correctness")
    top_k = config["retrieval"]["top_k"]

    questions = load_eval_dataset()
    if limit is not None:
        questions = questions[:limit]

    logger.info(f"Scoring {len(questions)} questions (generate + judge LLM calls)...\n")
    results = []
    for i, q in enumerate(questions, start=1):
        generated = generate_answer(q.question, retriever, llm, system_prompt, top_k)
        abstained = is_abstention(generated.answer)

        verdict = None
        reason = ""
        if not q.should_abstain and not abstained:
            verdict, reason = judge_correctness(llm, judge_prompt, q.question, q.expected_answer, generated.answer)

        result = QAResult(q, generated.answer, abstained, verdict, reason, generated.retrieved)
        results.append(result)
        logger.info(f"  [{i:2d}/{len(questions)}] {q.id} {q.type:<10} {describe(result)}")

    summary = report(results, config)
    if save:
        persist(summary, results, config)
    return summary


def report(results: List[QAResult], config: dict) -> dict:
    answerable = []
    abstention = []
    for r in results:
        if r.q.should_abstain:
            abstention.append(r)
        else:
            answerable.append(r)

    # Answerable breakdown.
    false_abstentions = [r for r in answerable if r.abstained]
    answered = [r for r in answerable if not r.abstained]
    correct = [r for r in answered if r.verdict == "CORRECT"]
    partial = [r for r in answered if r.verdict == "PARTIALLY_CORRECT"]
    incorrect = [r for r in answered if r.verdict == "INCORRECT"]

    # Abstention breakdown.
    abstained_correctly = [r for r in abstention if r.abstained]

    model = config["llm"]["models"]["groq"]
    logger.info(f"\n=== Answer Correctness (generator = judge = {model}) ===")
    logger.info("note: same model generates and grades (self-eval bias); LLM output is non-deterministic\n")

    # List the failures explicitly — that's what you act on.
    logger.info("FAILURES:")
    any_failure = False
    for r in false_abstentions:
        any_failure = True
        logger.info(f"  {r.q.id}  FALSE ABSTENTION (answerable, but refused)")
    for r in incorrect:
        any_failure = True
        logger.info(f"  {r.q.id}  INCORRECT — {r.judge_reason.splitlines()[-1] if r.judge_reason else ''}")
    for r in partial:
        any_failure = True
        logger.info(f"  {r.q.id}  PARTIAL — {r.judge_reason.splitlines()[-1] if r.judge_reason else ''}")
    for r in abstention:
        if not r.abstained:
            any_failure = True
            logger.info(f"  {r.q.id}  FAILED TO ABSTAIN (answered an out-of-corpus question)")
    if not any_failure:
        logger.info("  (none)")

    n_answerable = len(answerable)
    end_to_end = len(correct) / n_answerable if n_answerable else 0.0

    logger.info(f"\nSUMMARY")
    logger.info(f"  answerable ({n_answerable}):")
    logger.info(f"    answered: {len(answered)}   false-abstention: {len(false_abstentions)}")
    logger.info(f"    of answered -> CORRECT={len(correct)} PARTIAL={len(partial)} INCORRECT={len(incorrect)}")
    logger.info(f"    end-to-end success (answered AND correct): {len(correct)}/{n_answerable} = {end_to_end:.3f}")
    logger.info(f"  abstention ({len(abstention)}): abstained correctly {len(abstained_correctly)}/{len(abstention)}")

    return {
        "model": model,
        "answerable": n_answerable,
        "answered": len(answered),
        "false_abstention": len(false_abstentions),
        "correct": len(correct),
        "partial": len(partial),
        "incorrect": len(incorrect),
        "end_to_end_success": end_to_end,
        "abstention_total": len(abstention),
        "abstained_correctly": len(abstained_correctly),
    }


def persist(summary: dict, results: List[QAResult], config: dict) -> None:
    """Write a run record to eval_runs/answer_correctness/<timestamp>.json (gitignored)."""
    path = eval_run_path("answer_correctness")

    per_question = []
    for r in results:
        # Record the retrieved chunks (source + text) so a run can be diagnosed offline:
        # for a failure, read the actual passages the generator saw and decide whether the
        # needed fact was even present (retrieval problem) or was present but the answer
        # dropped it (generation problem).
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
            "should_abstain": r.q.should_abstain,
            "abstained": r.abstained,
            "verdict": r.verdict,
            "answer": r.answer,
            "expected_answer": r.q.expected_answer,
            "expected_sources": r.q.expected_sources,
            "judge_reason": r.judge_reason,
            "retrieved": retrieved,
        })

    record = {
        "timestamp": path.stem,
        "metric": "answer_correctness",
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
    parser = argparse.ArgumentParser(description="Score end-to-end answer correctness over the seed eval set.")
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N questions (quick check).")
    parser.add_argument("--no-save", action="store_true", help="Don't write a JSON run record.")
    args = parser.parse_args()
    configure_run_logging("evals/answer_correctness")
    run(save=not args.no_save, limit=args.limit)


if __name__ == "__main__":
    main()
