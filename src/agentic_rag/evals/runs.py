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


def retrieval_config_snapshot(config: dict) -> dict:
    """The retrieval-relevant knobs, embedded in every run record so a run is self-describing.

    Lets you read any eval JSON and know exactly what pipeline produced it — which mode,
    rerank on/off + model, chunking, top_k — without guessing from the timestamp.
    """
    retrieval = config["retrieval"]
    ingestion = config["ingestion"]
    rerank = retrieval.get("rerank", {})
    rerank_on = rerank.get("enabled", False)
    return {
        "embedding": config["embedding"]["model"],
        "chunk_strategy": ingestion.get("chunking", {}).get("strategy", "fixed"),
        "chunk_size": ingestion["chunk_size"],
        "chunk_overlap": ingestion["chunk_overlap"],
        "mode": retrieval.get("mode", "dense"),
        "top_k": retrieval["top_k"],
        "rerank_enabled": rerank_on,
        "rerank_model": rerank.get("model") if rerank_on else None,
        "rerank_candidate_k": rerank.get("candidate_k") if rerank_on else None,
        "hybrid_rrf_k": retrieval.get("hybrid", {}).get("rrf_k"),
        "hybrid_candidate_k": retrieval.get("hybrid", {}).get("candidate_k"),
    }
