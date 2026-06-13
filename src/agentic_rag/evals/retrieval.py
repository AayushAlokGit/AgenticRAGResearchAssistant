"""Retrieval-recall eval — rung 1: deterministic, no LLM.

Measures whether the naive retriever surfaces the documents each question needs, using
``seed.yaml``'s ``expected_sources`` as ground truth. This is the baseline the whole
change->measure loop grades against (EVALUATION_PRINCIPLES.md, P4 rung 1).

Only questions that declare ``expected_sources`` are scored — recall is undefined for a
question with no target document, so those are left to a later answer-layer eval.

We report recall at SEVERAL cutoffs plus MRR, on purpose: recall@5 over an 11-doc
corpus saturates at 1.0 (no headroom — can't show improvement, only regressions). The
stricter cutoffs and the ranking-sensitive MRR are what actually discriminate.

Per question, over the source docs within the first *k* retrieved chunks:
  match: any  -> hit if AT LEAST ONE expected_source is retrieved
  match: all  -> hit if EVERY expected_source is retrieved (multi-hop; partial = miss)

  recall@k = (questions that hit at k) / (scored questions)
  MRR      = mean of 1/rank, where rank is the position of the first chunk whose source
             is an expected_source (0 if none found). Rewards ranking the right doc high.

Run (needs an ingested store):
    python -m agentic_rag.evals.retrieval
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from agentic_rag.config import load_config, resolve_path
from agentic_rag.evals.dataset import EvalQuestion, load_eval_dataset
from agentic_rag.rag.embeddings import LocalEmbedder
from agentic_rag.rag.vector_store import ChromaVectorStore, Hit

# Recall cutoffs to report. Spans from the discriminating (1) to the saturating (5).
K_VALUES = (1, 3, 5)


@dataclass
class QuestionResult:
    q: EvalQuestion
    retrieved_sources: List[str]       # unique source docs over retrieval depth, best-first
    top_score: float                   # similarity of the #1 hit
    recall_at: Dict[int, bool] = field(default_factory=dict)            # k -> hit
    first_relevant_rank: Optional[int] = None   # 1-based rank of first expected-source chunk
    missing_at_depth: List[str] = field(default_factory=list)           # expected docs never found


# ───────────────────────────── scoring one question ─────────────────────────────

def find_first_relevant_rank(chunk_sources: List[str], expected_sources: List[str]) -> Optional[int]:
    """Return the 1-based rank of the first retrieved chunk whose source is expected.

    Returns None if none of the retrieved chunks came from an expected document.
    """
    for index, source in enumerate(chunk_sources):
        if source in expected_sources:
            return index + 1          # +1 because ranks are 1-based, not 0-based
    return None


def hit_at_k(expected_sources: List[str], match: str, chunk_sources: List[str], k: int) -> bool:
    """Did we retrieve the expected document(s) within the first k chunks?

    match == "all": every expected source must be present (multi-hop; partial = miss).
    match == "any": at least one expected source is enough.
    """
    sources_in_top_k = set(chunk_sources[:k])

    found = []
    for source in expected_sources:
        if source in sources_in_top_k:
            found.append(source)

    if match == "all":
        return len(expected_sources) > 0 and len(found) == len(expected_sources)
    return len(found) > 0


def unique_in_order(items: List[str]) -> List[str]:
    """Drop duplicates while keeping first-seen order (best-first stays best-first)."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def score_question(q: EvalQuestion, hits: List[Hit]) -> QuestionResult:
    # One source filename per retrieved chunk, in rank order (may repeat across chunks).
    chunk_sources = [hit.source for hit in hits]
    unique_sources = unique_in_order(chunk_sources)
    top_score = hits[0].score if hits else 0.0

    first_rank = find_first_relevant_rank(chunk_sources, q.expected_sources)

    recall_at = {}
    for k in K_VALUES:
        recall_at[k] = hit_at_k(q.expected_sources, q.match, chunk_sources, k)

    missing = []
    for source in q.expected_sources:
        if source not in unique_sources:
            missing.append(source)

    return QuestionResult(q, unique_sources, top_score, recall_at, first_rank, missing)


# ───────────────────────────── aggregating + running ─────────────────────────────

def mean(numbers: List[float]) -> float:
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def aggregate(results: List[QuestionResult]) -> tuple[Dict[int, float], float]:
    """Compute recall@k (for each k) and MRR over a list of scored results."""
    recalls = {}
    for k in K_VALUES:
        hit_count = 0
        for r in results:
            if r.recall_at[k]:
                hit_count += 1
        recalls[k] = hit_count / len(results) if results else 0.0

    # Why MRR matters: recall@k is position-blind — the right doc at rank 1 and at rank 5
    # score identically. MRR is rank-sensitive (1/rank), so it catches a retriever that
    # technically returns the right doc but buries it under distractors — a regression
    # recall@5 can't see, yet one that hurts the generator (it reads top hits first).
    #
    # Interpreting the value: MRR is the average of 1/rank, so it reads back as "how high,
    # on average, is the first right doc?" 1.0 = always rank 1 (perfect); 0.5 = rank 2 on
    # average; 0.33 = rank 3; closer to 0 = right doc consistently deep or missing.
    # Example: 3 questions whose first relevant hit lands at ranks [1, 1, 2] give
    # reciprocals [1.0, 1.0, 0.5] -> MRR = 2.5 / 3 = 0.833.
    reciprocal_ranks = []
    for r in results:
        if r.first_relevant_rank is None:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / r.first_relevant_rank)
    mrr = mean(reciprocal_ranks)

    return recalls, mrr


def run(save: bool = True) -> dict:
    config = load_config()
    depth = max(K_VALUES)

    store = ChromaVectorStore(resolve_path(config["vector_store"]["path"]), config["vector_store"]["collection"])
    if store.count() == 0:
        raise SystemExit("Vector store is empty — run `python -m agentic_rag.rag.ingest` first.")
    embedder = LocalEmbedder(config["embedding"]["model"])

    # Recall needs a ground-truth doc to check against, so only score questions that
    # declare expected_sources.
    questions = []
    for q in load_eval_dataset():
        if q.expected_sources:
            questions.append(q)

    results = []
    for q in questions:
        query_vector = embedder.embed_query(q.question)
        hits = store.query(query_vector, depth)
        results.append(score_question(q, hits))

    summary = report(results, config)
    if save:
        persist(summary, results, config)
    return summary


# ───────────────────────────── printing + saving ─────────────────────────────

def report(results: List[QuestionResult], config: dict) -> dict:
    print(f"\n=== Retrieval Recall (depth {max(K_VALUES)}) ===")
    print(f"embedding={config['embedding']['model']}  chunk={config['retrieval']['chunk_size']}/{config['retrieval']['chunk_overlap']}\n")

    print(f"QUESTIONS ({len(results)})   rank = position of first expected-source chunk")
    for r in results:
        flag_parts = []
        for k in K_VALUES:
            mark = "Y" if r.recall_at[k] else "n"
            flag_parts.append(f"@{k}:{mark}")
        flags = " ".join(flag_parts)

        rank = r.first_relevant_rank if r.first_relevant_rank else "-"
        line = f"  {r.q.id:<4} {r.q.type:<9} match:{r.q.match:<3} {flags}  rank={rank!s:<2} top={r.top_score:.3f}"
        if r.missing_at_depth:
            line += f"  MISSING: {', '.join(r.missing_at_depth)}"
        print(line)

    # Overall numbers.
    recalls, mrr = aggregate(results)
    print(f"\nSUMMARY  (n={len(results)} scored)")
    recall_str = "  ".join(f"@{k}={recalls[k]:.3f}" for k in K_VALUES)
    print(f"  recall  {recall_str}   MRR={mrr:.3f}")

    # Same numbers sliced by question type, so you can see WHERE it's weak.
    types = sorted({r.q.type for r in results})
    for t in types:
        subset = []
        for r in results:
            if r.q.type == t:
                subset.append(r)
        sub_recalls, sub_mrr = aggregate(subset)
        sub_recall_str = "  ".join(f"@{k}={sub_recalls[k]:.3f}" for k in K_VALUES)
        print(f"    {t:<10} {sub_recall_str}   MRR={sub_mrr:.3f}   (n={len(subset)})")

    return {
        "recall": recalls,
        "mrr": mrr,
        "scored": len(results),
    }


def persist(summary: dict, results: List[QuestionResult], config: dict) -> None:
    """Write a timestamped JSON to eval_runs/ (gitignored) so baselines are diffable."""
    out_dir = resolve_path("./eval_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    per_question = []
    for r in results:
        recall_at = {}
        for k in K_VALUES:
            recall_at[str(k)] = r.recall_at[k]
        per_question.append({
            "id": r.q.id,
            "type": r.q.type,
            "match": r.q.match,
            "recall_at": recall_at,
            "first_relevant_rank": r.first_relevant_rank,
            "top_score": round(r.top_score, 4),
            "expected": r.q.expected_sources,
            "missing": r.missing_at_depth,
            "retrieved": r.retrieved_sources,
        })

    recall_summary = {}
    for k in K_VALUES:
        recall_summary[str(k)] = summary["recall"][k]

    record = {
        "timestamp": stamp,
        "metric": "retrieval_recall",
        "k_values": list(K_VALUES),
        "config": {
            "embedding": config["embedding"]["model"],
            "chunk_size": config["retrieval"]["chunk_size"],
            "chunk_overlap": config["retrieval"]["chunk_overlap"],
        },
        "summary": {**summary, "recall": recall_summary},
        "per_question": per_question,
    }

    path = out_dir / f"retrieval_recall_{stamp}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\n[saved] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score retrieval recall + MRR over the seed eval set.")
    parser.add_argument("--no-save", action="store_true", help="Don't write a JSON run record.")
    run(save=not parser.parse_args().no_save)


if __name__ == "__main__":
    main()
