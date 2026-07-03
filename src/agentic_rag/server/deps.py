"""Build the agent graph + shared deps ONCE at startup, held in app state for every request.

Rebuilding the retriever (810-chunk BM25 index + the cross-encoder) per request would dominate
latency, so app.py constructs one ``AppState`` on boot and each ``/ask`` only calls ``graph.stream``.

Prod knobs come from ``config/default.yaml`` unchanged (spend-cap on, cache on, memory off) except the
store: the deployed backend runs on Postgres+pgvector, so we force ``STORE_PROVIDER=pgvector`` by
default here (override via the env var for a local Chroma run). Episodic memory stays OFF — a single
shared store would leak episodes between anonymous visitors.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    config: dict
    graph: object
    store: object          # the tools' store handle — also used for /sources and /health
    max_rounds: int
    model_names: dict      # {"controller": ..., "generator": ...} for /config


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
    from agentic_rag.rag.retriever import build_retriever

    config = load_config()
    # The deployed backend's reason for being: durable pgvector storage. Default to it, allow override.
    config["vector_store"]["provider"] = os.environ.get("STORE_PROVIDER", "pgvector")
    logger.info("server: vector store provider = %s", config["vector_store"]["provider"])

    retriever = build_retriever(config)   # builds its own store handle + rehydrates BM25 from it
    controller_llm = build_llm(config, role="controller")
    generator_llm = build_llm(config, role="generator")
    react_prompt = load_prompt(config, "agent_react")
    answer_prompt = load_prompt(config, "answer_with_citations")

    agent_cfg = config.get("agent", {})
    top_k = config["retrieval"]["top_k"]
    max_rounds = agent_cfg.get("max_rounds", 5)
    answer_char_budget = agent_cfg.get("answer_char_budget", 0)
    ordering = config.get("context", {}).get("ordering", "arrival")
    spend_cap_tokens = config.get("guardrails", {}).get("spend_cap_tokens", 0)

    registry, store = build_agent_deps(config, agent_cfg.get("tools", DEFAULT_TOOLS))
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
    return AppState(config=config, graph=graph, store=store,
                    max_rounds=max_rounds, model_names=model_names)
