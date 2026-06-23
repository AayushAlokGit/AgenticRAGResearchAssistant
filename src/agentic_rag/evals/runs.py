"""Where eval run records are written.

One convention for every eval: ``eval_runs/<metric>/<timestamp>.json`` (gitignored).
The timestamp is local-time and human-readable (``DD_MM_YYYY_HH_MM_SS``) so the files
sort and read naturally when you're diffing runs by hand.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from agentic_rag.config import resolve_path
from agentic_rag.llm.provider import Usage


def usage_record(controller: Usage, generator: Usage, judge: Usage) -> dict:
    """Shape per-run token usage for the eval JSON, kept in SEPARATE buckets on purpose.

    ``controller`` + ``generator`` are SYSTEM cost — the tokens you'd actually pay in
    production — but split by ROLE because they can be different models at different prices
    (DD-025), and the controller is called many times/question while the generator runs once,
    so summing them hides where the cost lives. ``judge`` is INSTRUMENT cost — the cost of
    *measuring*, with no production counterpart (nobody runs the judge when you ship).
    We deliberately do NOT collapse these: a single "tokens per run" would mislead every
    cost trade-off. Tokens (not dollars) because a count is provider-neutral; dollars need a
    drifting price table.
    """
    return {
        "controller": controller.as_dict(),  # production cost — routing (called N times/Q)
        "generator": generator.as_dict(),    # production cost — final answer (called once/Q)
        "judge": judge.as_dict(),            # measurement cost (no production counterpart)
    }


def eval_run_path(metric: str, dataset_version: str) -> Path:
    """Return a fresh timestamped json path under ``eval_runs/<metric>/v<version>/``.

    The dataset version is a path segment so runs against different eval sets land in
    separate folders — you can't accidentally diff a v2 run against a v3 run.
    """
    label = f"v{dataset_version}" if dataset_version else "vunknown"
    out_dir = resolve_path("./eval_runs") / metric / label
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    return out_dir / f"{stamp}.json"


def agent_config_snapshot(config: dict) -> dict:
    """The agentic-loop knobs, embedded in answer-eval records so a run says whether the
    agent loop was on (and its budget) — the difference between a naive and an agentic A/B.
    """
    agent = config.get("agent", {})
    enabled = bool(agent.get("enabled", False))
    return {
        "enabled": enabled,
        "max_rounds": agent.get("max_rounds") if enabled else None,
        # The action space the run used (A1) — records WHICH tools were available, so an
        # A/B over the tool set is reproducible from the file alone.
        "tools": agent.get("tools") if enabled else None,
    }


def retrieval_config_snapshot(config: dict) -> dict:
    """The retrieval-relevant knobs, embedded in every run record so a run is self-describing.

    Lets you read any eval JSON and know exactly what pipeline produced it — which mode,
    rerank on/off + model, chunking, top_k — without guessing from the timestamp.
    """
    retrieval = config["retrieval"]
    ingestion = config["ingestion"]
    rerank = retrieval.get("rerank", {})
    rerank_on = rerank.get("enabled", False)
    parent = retrieval.get("parent_expansion", {})
    parent_on = parent.get("enabled", False)
    multi_query = retrieval.get("multi_query", {})
    mq_on = multi_query.get("enabled", False)
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
        "parent_expansion": parent_on,
        "parent_window": parent.get("window") if parent_on else None,
        "hybrid_rrf_k": retrieval.get("hybrid", {}).get("rrf_k"),
        "hybrid_candidate_k": retrieval.get("hybrid", {}).get("candidate_k"),
        "multi_query": mq_on,
        "multi_query_n": multi_query.get("n_queries") if mq_on else None,
    }


def context_config_snapshot(config: dict) -> dict:
    """The Module-3 context-engineering knobs, embedded in run records so a run says HOW it
    curated the final answer window (ordering, …) — makes a context-eng A/B reproducible from
    the file alone, the same way the retrieval/agent snapshots already are.
    """
    context = config.get("context", {})
    return {
        "ordering": context.get("ordering", "arrival"),
    }
