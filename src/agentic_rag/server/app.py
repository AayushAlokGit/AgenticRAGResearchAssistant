"""FastAPI app: the public demo backend.

Endpoints:
  GET  /health   — liveness + chunk count (checks the store is reachable)
  GET  /config   — public-safe knobs for the UI (models, budgets, tools)
  GET  /sources  — the corpus map (source -> chunk count + free-form tags) for the sidebar
  POST /ask      — the main event: streams the agent's trajectory as Server-Sent Events

The graph is built once at startup (deps.build_app_state) and reused. graph.stream is blocking, so
/ask runs it in a worker thread and bridges events to the async SSE response via a queue — the event
loop is never blocked, so concurrent requests stay responsive.
"""
from __future__ import annotations

import asyncio
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    deps = app.state.deps
    return HealthResponse(status="ok", chunks=deps.store.count())


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


async def _sse(graph, question: str, max_rounds: int):
    """Bridge the blocking stream_events generator to async SSE via a worker thread + queue."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    def worker():
        try:
            for event in stream_events(graph, question, max_rounds):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:  # surface any failure to the client as a typed error event
            logger.exception("server: /ask stream failed")
            loop.call_soon_threadsafe(queue.put_nowait, ErrorEvent(message=str(exc)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    task = loop.run_in_executor(None, worker)
    try:
        while True:
            event = await queue.get()
            if event is _DONE:
                break
            yield {"event": event.type, "data": event.model_dump_json()}
    finally:
        await task


@app.post("/ask")
async def ask(req: AskRequest):
    deps = app.state.deps
    return EventSourceResponse(_sse(deps.graph, req.question, deps.max_rounds))
