"""Build the agent graph + shared deps ONCE at startup, held in app state for every request.

Rebuilding the retriever (810-chunk BM25 index + the cross-encoder) per request would dominate
latency, so app.py constructs one ``AppState`` on boot and each ``/ask`` only calls ``graph.stream``.

Prod knobs come from ``config/default.yaml`` unchanged (spend-cap on, cache on, memory off) except the
store: the deployed backend runs on Postgres+pgvector, so we force ``STORE_PROVIDER=pgvector`` by
default here (override via the env var for a local Chroma run). Episodic memory stays OFF — a single
shared store would leak episodes between anonymous visitors.

Bring-your-own-doc + the compare/scope toggles need graphs built with per-request retrieval knobs.
The expensive singletons (store, embedder, cross-encoder reranker, the LLMs) are cached on ``AppState``
so ``build_scoped_graph`` can assemble a fresh graph cheaply — no cross-encoder reload.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    config: dict
    graph: object
    store: object          # the tools' store handle — also used for /sources, /health, uploads
    max_rounds: int
    model_names: dict      # {"controller": ..., "generator": ...} for /config
    budget: object = None  # DailyBudget or None (the durable $0 kill-switch)
    # Cached singletons so a per-upload / per-request graph rebuilds cheaply (no cross-encoder reload):
    embedder: object = None
    reranker: object = None
    controller_llm: object = None
    generator_llm: object = None
    registry: object = None
    react_prompt: str = ""
    answer_prompt: str = ""
    top_k: int = 5
    answer_char_budget: int = 0
    ordering: str = "arrival"
    spend_cap_tokens: int = 0


def build_app_state() -> AppState:
    from dotenv import load_dotenv

    load_dotenv()  # GOOGLE/GROQ keys, LANGSMITH_*, and DATABASE_URL
    # Force the reranker to load from the baked-in cache, never the network (see rag/rerank.py).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from agentic_rag.agent.graph import build_graph
    from agentic_rag.agent.loop import build_agent_deps
    from agentic_rag.agent.tools import DEFAULT_TOOLS
    from agentic_rag.config import load_config
    from agentic_rag.llm.provider import build_llm
    from agentic_rag.rag.answer import load_prompt
    from agentic_rag.rag.embeddings import build_embedder
    from agentic_rag.rag.retriever import build_retriever

    config = load_config()
    # The deployed backend's reason for being: durable pgvector storage. Default to it, allow override.
    config["vector_store"]["provider"] = os.environ.get("STORE_PROVIDER", "pgvector")
    logger.info("server: vector store provider = %s", config["vector_store"]["provider"])

    agent_cfg = config.get("agent", {})
    # ONE store handle, shared by the retriever AND the corpus-reading tools (list_sources) AND uploads.
    registry, store = build_agent_deps(config, agent_cfg.get("tools", DEFAULT_TOOLS))
    embedder = build_embedder(config)

    # Build the cross-encoder ONCE and reuse it for every per-request retriever. It loads its weights
    # lazily on first rerank(), which the warm-up query below triggers — so the load is paid at boot.
    reranker = None
    rerank_cfg = config["retrieval"].get("rerank", {})
    if rerank_cfg.get("enabled", False):
        from agentic_rag.rag.rerank import CrossEncoderReranker, DEFAULT_MODEL

        reranker = CrossEncoderReranker(rerank_cfg.get("model", DEFAULT_MODEL))

    controller_llm = build_llm(config, role="controller")
    generator_llm = build_llm(config, role="generator")
    react_prompt = load_prompt(config, "agent_react")
    answer_prompt = load_prompt(config, "answer_with_citations")

    top_k = config["retrieval"]["top_k"]
    max_rounds = agent_cfg.get("max_rounds", 5)
    answer_char_budget = agent_cfg.get("answer_char_budget", 0)
    ordering = config.get("context", {}).get("ordering", "arrival")
    spend_cap_tokens = config.get("guardrails", {}).get("spend_cap_tokens", 0)

    # Default retriever/graph: config knobs, scope="both", reusing the shared store/embedder/reranker.
    retriever = build_retriever(config, store=store, embedder=embedder, reranker=reranker)
    graph = build_graph(retriever, controller_llm, generator_llm, registry, store,
                        react_prompt, answer_prompt, top_k, max_rounds, answer_char_budget,
                        ordering, spend_cap_tokens=spend_cap_tokens)

    roles = config.get("llm", {}).get("roles", {})
    model_names = {
        "controller": roles.get("controller", {}).get("model", "?"),
        "generator": roles.get("generator", {}).get("model", "?"),
    }
    logger.info("server: graph built (store holds %d chunks, max_rounds=%d, spend_cap=%d)",
                store.count(), max_rounds, spend_cap_tokens)

    # Warm the lazily-loaded models (cross-encoder reranker + embedder) at boot so the FIRST real
    # request doesn't eat the ~tens-of-seconds cold model-load spike. Best-effort: a failure here
    # must not crash startup — the models would just load lazily on first use, as before.
    t0 = time.perf_counter()
    try:
        retriever.query("warmup", 1)
        logger.info("server: warmed retriever models in %.1fs", time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001 — warm-up is opportunistic
        logger.warning("server: retriever warm-up skipped (%s)", exc)

    # Durable daily token budget (the $0 kill-switch). Needs Postgres; skipped if off or no DB.
    daily_budget = int(os.environ.get(
        "DAILY_TOKEN_BUDGET", config.get("guardrails", {}).get("daily_token_budget", 0)))
    budget = None
    dsn = os.environ.get("DATABASE_URL")
    if daily_budget > 0 and dsn:
        from agentic_rag.server.budget import DailyBudget

        budget = DailyBudget(dsn, daily_budget)
        logger.info("server: daily token budget = %d tokens/day (persisted in Postgres)", daily_budget)
    elif daily_budget > 0:
        logger.warning("server: daily_token_budget set but no DATABASE_URL — cap DISABLED")

    return AppState(config=config, graph=graph, store=store,
                    max_rounds=max_rounds, model_names=model_names, budget=budget,
                    embedder=embedder, reranker=reranker,
                    controller_llm=controller_llm, generator_llm=generator_llm,
                    registry=registry, react_prompt=react_prompt, answer_prompt=answer_prompt,
                    top_k=top_k, answer_char_budget=answer_char_budget, ordering=ordering,
                    spend_cap_tokens=spend_cap_tokens)


def build_scoped_graph(state: AppState, *, mode=None, rerank=None, parent_expansion=None,
                       scope: str = "both", max_rounds=None, top_k=None):
    """Assemble a fresh graph over the SAME store/embedder/reranker singletons but with per-request
    retrieval knobs + scope. Cheap — no cross-encoder reload. Powers the scope toggle and the
    Basic-vs-Full compare (each preset is just a different set of knobs)."""
    from agentic_rag.agent.graph import build_graph
    from agentic_rag.rag.retriever import build_retriever

    retriever = build_retriever(state.config, mode=mode, rerank=rerank,
                                parent_expansion=parent_expansion, scope=scope,
                                store=state.store, embedder=state.embedder, reranker=state.reranker)
    rounds = state.max_rounds if max_rounds is None else max_rounds
    k = state.top_k if top_k is None else top_k
    return build_graph(retriever, state.controller_llm, state.generator_llm, state.registry,
                       state.store, state.react_prompt, state.answer_prompt, k,
                       rounds, state.answer_char_budget, state.ordering,
                       spend_cap_tokens=state.spend_cap_tokens)


def rebuild_default_graph(state: AppState) -> None:
    """Rebuild the default graph so its BM25 index picks up store changes. Call after an upload or
    delete — new docs are dense-searchable immediately, but keyword-searchable only after this."""
    state.graph = build_scoped_graph(state)
