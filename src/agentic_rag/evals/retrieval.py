"""Retrieval-recall eval — rung 1: deterministic, no LLM.

Measures whether the naive retriever surfaces the documents each question needs, using
``seed.yaml``'s ``expected_sources`` as ground truth. This is the baseline the whole
change->measure loop grades against (EVALUATION_PRINCIPLES.md, P4 rung 1).

We report recall at SEVERAL cutoffs plus MRR, on purpose: recall@5 over an 11-doc
corpus saturates at 1.0 (no headroom — can't show improvement, only regressions). The
stricter cutoffs and the ranking-sensitive MRR are what actually discriminate between a
better and worse retriever.

Per answerable question, over the unique source docs within the first *k* retrieved chunks:
  match: any  -> hit if AT LEAST ONE expected_source is retrieved
  match: all  -> hit if EVERY expected_source is retrieved (multi-hop; partial = miss)

  recall@k = hits / answerable questions
  MRR      = mean of 1/rank, where rank is the position of the first chunk whose source
             is an expected_source (0 if none in the retrieved depth). Ranking-sensitive:
             rewards putting the right doc higher. (For match:all it only sees the first
             relevant doc — a known MRR limitation; recall@k carries the all-or-nothing part.)

Abstention questions (``should_abstain: true``, no expected_sources) are NOT retrieval
tests — abstaining is a generator/agent decision scored later. Excluded from recall/MRR;
we report their top-1 similarity instead (the signal a future abstain-threshold uses).

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
    recall_at: Dict[int, Optional[bool]] = field(default_factory=dict)  # k -> hit (None=abstention)
    first_relevant_rank: Optional[int] = None   # 1-based rank of first expected-source chunk
    missing_at_depth: List[str] = field(default_factory=list)           # expected not found at all


def _hit_at_k(expected: List[str], match: str, sources_in_order: List[str], k: int) -> bool:
    """Apply any/all recall over the sources of the first k retrieved chunks."""
    window = set(sources_in_order[:k])
    found = [s for s in expected if s in window]
    if match == "all":
        return len(expected) > 0 and len(found) == len(expected)
    return len(found) > 0


def score_question(q: EvalQuestion, hits: List[Hit]) -> QuestionResult:
    chunk_sources = [h.source for h in hits]               # one entry per chunk, ranked
    seen, unique_sources = set(), []
    for s in chunk_sources:
        if s not in seen:
            seen.add(s)
            unique_sources.append(s)
    top_score = hits[0].score if hits else 0.0

    if q.should_abstain:
        return QuestionResult(q, unique_sources, top_score, {k: None for k in K_VALUES})

    # First rank (1-based, over chunks) whose source is an expected source -> drives MRR.
    first_rank = next((i + 1 for i, s in enumerate(chunk_sources) if s in q.expected_sources), None)
    recall_at = {k: _hit_at_k(q.expected_sources, q.match, chunk_sources, k) for k in K_VALUES}
    missing = [s for s in q.expected_sources if s not in seen]
    return QuestionResult(q, unique_sources, top_score, recall_at, first_rank, missing)


def run(save: bool = True) -> dict:
    config = load_config()
    depth = max(K_VALUES)

    store = ChromaVectorStore(resolve_path(config["vector_store"]["path"]), config["vector_store"]["collection"])
    if store.count() == 0:
        raise SystemExit("Vector store is empty — run `python -m agentic_rag.rag.ingest` first.")
    embedder = LocalEmbedder(config["embedding"]["model"])

    questions = load_eval_dataset()
    results = [score_question(q, store.query(embedder.embed_query(q.question), depth)) for q in questions]

    summary = _report(results, config)
    if save:
        _persist(summary, results, config)
    return summary


def _report(results: List[QuestionResult], config: dict) -> dict:
    answerable = [r for r in results if not r.q.should_abstain]
    abstain = [r for r in results if r.q.should_abstain]

    print(f"\n=== Retrieval Recall (depth {max(K_VALUES)}) ===")
    print(f"embedding={config['embedding']['model']}  chunk={config['retrieval']['chunk_size']}/{config['retrieval']['chunk_overlap']}\n")

    print(f"ANSWERABLE ({len(answerable)})   rank = position of first expected-source chunk")
    for r in answerable:
        flags = " ".join(f"@{k}:{'Y' if r.recall_at[k] else 'n'}" for k in K_VALUES)
        rank = r.first_relevant_rank if r.first_relevant_rank else "-"
        line = f"  {r.q.id:<4} {r.q.type:<9} match:{r.q.match:<3} {flags}  rank={rank!s:<2} top={r.top_score:.3f}"
        if r.missing_at_depth:
            line += f"  MISSING: {', '.join(r.missing_at_depth)}"
        print(line)

    print(f"\nABSTENTION ({len(abstain)}) — excluded; top-1 score (lower = easier to refuse):")
    for r in abstain:
        print(f"  {r.q.id:<4} top={r.top_score:.3f}  nearest: {r.retrieved_sources[0] if r.retrieved_sources else '-'}")

    # Overall + by-type recall at each k, and MRR.
    recalls = {k: _mean(1.0 if r.recall_at[k] else 0.0 for r in answerable) for k in K_VALUES}
    mrr = _mean((1.0 / r.first_relevant_rank if r.first_relevant_rank else 0.0) for r in answerable)

    print(f"\nSUMMARY  (n={len(answerable)} answerable)")
    print("  recall  " + "  ".join(f"@{k}={recalls[k]:.3f}" for k in K_VALUES) + f"   MRR={mrr:.3f}")
    types = sorted({r.q.type for r in answerable})
    for t in types:
        sub = [r for r in answerable if r.q.type == t]
        rk = "  ".join(f"@{k}={_mean(1.0 if r.recall_at[k] else 0.0 for r in sub):.3f}" for k in K_VALUES)
        sub_mrr = _mean((1.0 / r.first_relevant_rank if r.first_relevant_rank else 0.0) for r in sub)
        print(f"    {t:<10} {rk}   MRR={sub_mrr:.3f}   (n={len(sub)})")
    if abstain:
        s = [r.top_score for r in abstain]
        print(f"  abstention top-1: min={min(s):.3f} mean={_mean(iter(s)):.3f} max={max(s):.3f}")

    return {
        "recall": recalls,
        "mrr": mrr,
        "answerable": len(answerable),
        "abstention_top_scores": {r.q.id: round(r.top_score, 4) for r in abstain},
    }


def _mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _persist(summary: dict, results: List[QuestionResult], config: dict) -> None:
    """Write a timestamped JSON to eval_runs/ (gitignored) so baselines are diffable."""
    out_dir = resolve_path("./eval_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {
        "timestamp": stamp,
        "metric": "retrieval_recall",
        "k_values": list(K_VALUES),
        "config": {
            "embedding": config["embedding"]["model"],
            "chunk_size": config["retrieval"]["chunk_size"],
            "chunk_overlap": config["retrieval"]["chunk_overlap"],
        },
        "summary": {**summary, "recall": {str(k): v for k, v in summary["recall"].items()}},
        "per_question": [
            {
                "id": r.q.id, "type": r.q.type, "match": r.q.match,
                "recall_at": {str(k): r.recall_at[k] for k in K_VALUES},
                "first_relevant_rank": r.first_relevant_rank,
                "top_score": round(r.top_score, 4),
                "expected": r.q.expected_sources, "missing": r.missing_at_depth,
                "retrieved": r.retrieved_sources,
            }
            for r in results
        ],
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
