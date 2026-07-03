"""FastAPI app: the public demo backend.

Endpoints:
  GET  /health   — liveness + chunk count + today's token spend
  GET  /config   — public-safe knobs for the UI (models, budgets, tools)
  GET  /sources  — the corpus map (source -> chunk count + free-form tags) for the sidebar
  POST /ask      — the main event: streams the agent's trajectory as Server-Sent Events

The graph is built once at startup (deps.build_app_state) and reused. graph.stream is blocking, but
the /ask handler and its event generator are plain sync code: sse-starlette runs a sync generator in a
threadpool (iterate_in_threadpool), so the blocking work never touches the event loop — no manual
thread/queue plumbing needed. The read endpoints are sync too, so their blocking DB calls also run off
the loop (FastAPI runs sync handlers in a threadpool).
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from agentic_rag.server.deps import build_app_state
from agentic_rag.server.schemas import (AskRequest, ConfigResponse, ErrorEvent, HealthResponse,
                                        SourceInfo, SourcesResponse)
from agentic_rag.server.stream import stream_events

logger = logging.getLogger(__name__)

_RESTING_MESSAGE = (
    "The demo has reached today's shared usage budget and is resting to stay free. "
    "Please try again tomorrow — thanks for stopping by!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("server: building app state (graph + retriever + store) ...")
    app.state.deps = build_app_state()
    logger.info("server: ready")
    yield


app = FastAPI(title="Agentic RAG Research Assistant — demo API", lifespan=lifespan)

# CORS: the Next.js frontend calls this from a different origin. Locked to an allowlist (localhost for
# dev, the Vercel origins in prod via FRONTEND_ORIGINS="https://a.com,https://b.com").
_default_origins = "http://localhost:3000,http://127.0.0.1:3000"
_origins = [o.strip() for o in os.environ.get("FRONTEND_ORIGINS", _default_origins).split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"],
                   allow_headers=["*"])


def _sse(event) -> dict:
    """Serialize a schema event into the SSE frame sse-starlette expects."""
    return {"event": event.type, "data": event.model_dump_json()}


@app.get("/health", response_model=HealthResponse)
def health():
    deps = app.state.deps
    used = deps.budget.tokens_today() if deps.budget else 0
    return HealthResponse(status="ok", chunks=deps.store.count(), daily_tokens_used=used)


@app.get("/config", response_model=ConfigResponse)
def config():
    deps = app.state.deps
    cfg = deps.config
    return ConfigResponse(
        max_rounds=deps.max_rounds,
        controller_model=deps.model_names["controller"],
        generator_model=deps.model_names["generator"],
        top_k=cfg["retrieval"]["top_k"],
        tools=cfg.get("agent", {}).get("tools", []),
        spend_cap_tokens=cfg.get("guardrails", {}).get("spend_cap_tokens", 0),
        store_provider=cfg["vector_store"]["provider"],
        daily_token_budget=deps.budget.budget if deps.budget else 0,  # the cap actually enforced
    )


@app.get("/sources", response_model=SourcesResponse)
def sources():
    deps = app.state.deps
    counts: dict = {}
    for chunk in deps.store.all_chunks():
        counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1
    tags = deps.store.load_tags()
    infos = [SourceInfo(source=s, chunks=counts[s], tags=tags.get(s, {})) for s in sorted(counts)]
    return SourcesResponse(sources=infos)


def _ask_stream(graph, question: str, max_rounds: int, budget):
    """Yield SSE frames for one question, then charge the run's tokens to the daily budget.

    A plain sync generator (sse-starlette offloads it to a threadpool). Any failure is surfaced as a
    typed error event rather than a broken stream; the daily-budget charge in `finally` is a no-op on a
    cache hit (0 tokens).
    """
    spent = 0
    try:
        for event in stream_events(graph, question, max_rounds):
            if event.type == "done":
                spent = event.total_tokens
            yield _sse(event)
    except Exception as exc:
        logger.exception("server: /ask stream failed")
        yield _sse(ErrorEvent(message=str(exc)))
    finally:
        if budget is not None:
            budget.add(spent)


@app.post("/ask")
def ask(req: AskRequest):
    deps = app.state.deps
    if deps.budget is not None and deps.budget.is_exhausted():
        return EventSourceResponse(iter([_sse(ErrorEvent(message=_RESTING_MESSAGE))]))
    return EventSourceResponse(_ask_stream(deps.graph, req.question, deps.max_rounds, deps.budget))
