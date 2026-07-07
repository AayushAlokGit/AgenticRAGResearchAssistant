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
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from agentic_rag.rag.upload import (ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES, display_name, ingest_bytes,
                                    is_upload)
from agentic_rag.server.deps import build_app_state, build_scoped_graph, rebuild_default_graph
from agentic_rag.server.request_context import (RequestContextFilter, RequestContextMiddleware,
                                                current_client_ip, current_request_id)
from agentic_rag.server.schemas import (AskRequest, ConfigResponse, ErrorEvent, HealthResponse,
                                        SourceInfo, SourcesResponse, UploadResponse)
from agentic_rag.server.stream import stream_events

logger = logging.getLogger(__name__)

# Container stdout is the only log surface on HF Spaces, so every request-scoped line carries its
# [req_id] for grep-one-request tracing (RequestContextFilter fills it in; "-" outside a request).
_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(req_id)s] | %(message)s"

# Third-party loggers that would drown our own trace at INFO (mirrors logging_setup._NOISY_LIBRARIES).
_NOISY_LIBRARIES = ["httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers",
                    "transformers", "huggingface_hub", "groq"]


def _configure_server_logging() -> None:
    """One stdout handler for the root logger, req-id-aware. Explicit (not basicConfig) so the format
    is deterministic under uvicorn and every record gets the correlation id via the filter."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.addFilter(RequestContextFilter())  # stamps req_id/client_ip onto every record it formats
    root.addHandler(handler)

    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)

_RESTING_MESSAGE = (
    "The demo has reached today's shared usage budget and is resting to stay free. "
    "Please try again tomorrow — thanks for stopping by!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_server_logging()
    logger.info("server: building app state (graph + retriever + store) ...")
    app.state.deps = build_app_state()
    logger.info("server: ready")
    yield


app = FastAPI(title="Agentic RAG Research Assistant — demo API", lifespan=lifespan)

# Correlation id + client IP for every request (must wrap the app closest to the ASGI layer so it
# covers the streaming /ask body — see request_context for why it's pure ASGI, not BaseHTTPMiddleware).
app.add_middleware(RequestContextMiddleware)

# CORS: the Next.js frontend calls this from a different origin. Locked to an allowlist (localhost for
# dev, the Vercel origins in prod via FRONTEND_ORIGINS="https://a.com,https://b.com").
_default_origins = "http://localhost:3000,http://127.0.0.1:3000"
_origins = [o.strip() for o in os.environ.get("FRONTEND_ORIGINS", _default_origins).split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST", "DELETE"],
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
    ingestion = cfg.get("ingestion", {})
    embedding = cfg.get("embedding", {})
    emb_provider = embedding.get("provider", "local")
    emb_model = (embedding.get("gemini_model", "gemini-embedding-001") if emb_provider == "gemini"
                 else embedding.get("model", ""))
    return ConfigResponse(
        max_rounds=deps.max_rounds,
        controller_model=deps.model_names["controller"],
        generator_model=deps.model_names["generator"],
        top_k=cfg["retrieval"]["top_k"],
        tools=cfg.get("agent", {}).get("tools", []),
        spend_cap_tokens=cfg.get("guardrails", {}).get("spend_cap_tokens", 0),
        store_provider=cfg["vector_store"]["provider"],
        daily_token_budget=deps.budget.budget if deps.budget else 0,  # the cap actually enforced
        chunk_size=ingestion.get("chunk_size", 0),
        chunk_overlap=ingestion.get("chunk_overlap", 0),
        chunking_strategy=ingestion.get("chunking", {}).get("strategy", "fixed"),
        embedding_provider=emb_provider,
        embedding_model=emb_model,
        embedding_dims=embedding.get("dims", 0),
    )


@app.get("/sources", response_model=SourcesResponse)
def sources():
    deps = app.state.deps
    counts: dict = {}
    for chunk in deps.store.all_chunks():
        counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1
    tags = deps.store.load_tags()
    infos = [SourceInfo(source=s, chunks=counts[s], tags=tags.get(s, {}), uploaded=is_upload(s))
             for s in sorted(counts)]
    return SourcesResponse(sources=infos)


@app.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile = File(...)):
    """Bring-your-own-doc: ingest an uploaded md/txt/pdf into the shared store (marked ``upload:``),
    then rebuild the default graph's BM25 so the new doc is keyword-searchable immediately. A sync
    handler (FastAPI runs it in a threadpool), so the blocking extract/embed/index work stays off the
    event loop. Uploaded docs are shared, deletable via /documents, and never touch the seed corpus.
    """
    deps = app.state.deps
    name = file.filename or "document"
    logger.info("▶ /upload received | ip=%s file=%s", current_client_ip(), name)
    if not name.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=415,
                            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    data = file.file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    try:
        result = ingest_bytes(name, data, deps.embedder, deps.store, deps.config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    rebuild_default_graph(deps)   # refresh BM25 so the upload is keyword-searchable, not just dense
    logger.info("◀ /upload done | source=%s chunks=%d total_chunks=%d",
                result["source"], result["chunks"], deps.store.count())
    return UploadResponse(source=result["source"], name=display_name(result["source"]),
                          chunks=result["chunks"], total_chunks=deps.store.count())


@app.delete("/documents/{source:path}")
def delete_document(source: str):
    """Delete one UPLOADED document (seed corpus is protected — only ``upload:``-marked sources)."""
    deps = app.state.deps
    logger.info("▶ /documents DELETE | ip=%s source=%s", current_client_ip(), source)
    if not is_upload(source):
        raise HTTPException(status_code=403,
                            detail="Only uploaded documents can be deleted; the seed corpus is protected.")
    deps.store.delete_by_source(source)
    rebuild_default_graph(deps)
    logger.info("◀ /documents DELETE done | source=%s total_chunks=%d", source, deps.store.count())
    return {"deleted": source, "total_chunks": deps.store.count()}


def _ask_stream(graph, question: str, max_rounds: int, budget, *,
                question_log=None, scope: str = "both", req_id=None):
    """Yield SSE frames for one question, then charge the run's tokens to the daily budget and
    record the question to the durable analytics sink.

    A plain sync generator (sse-starlette offloads it to a threadpool). Any failure is surfaced as a
    typed error event rather than a broken stream; the daily-budget charge in `finally` is a no-op on a
    cache hit (0 tokens). Both `finally` side effects are best-effort — they run AFTER the user has
    their answer, so a sink failure can't break the response.
    """
    spent = 0
    summary = None   # the DoneEvent's run stats, captured for the question log
    try:
        for event in stream_events(graph, question, max_rounds):
            if event.type == "done":
                spent = event.total_tokens
                summary = event
                logger.info("/ask done | rounds=%d exit=%s tokens=%d latency_ms=%d",
                            event.rounds, event.exit_reason, event.total_tokens, event.latency_ms)
            yield _sse(event)
    except Exception as exc:
        logger.exception("/ask stream failed")
        yield _sse(ErrorEvent(message=str(exc)))
    finally:
        if budget is not None:
            budget.add(spent)
        if question_log is not None:
            try:
                question_log.record(
                    question, scope=scope, req_id=req_id,
                    rounds=summary.rounds if summary else None,
                    exit_reason=summary.exit_reason if summary else None,
                    total_tokens=summary.total_tokens if summary else None,
                    latency_ms=summary.latency_ms if summary else None,
                )
            except Exception:
                logger.exception("question log write failed (ignored)")


@app.post("/ask")
def ask(req: AskRequest):
    deps = app.state.deps
    scoped = req.scope != "both" or req.knobs is not None
    logger.info("▶ /ask received | ip=%s scope=%s graph=%s knobs=%s | Q: %.200s",
                current_client_ip(), req.scope, "scoped" if scoped else "default",
                req.knobs.model_dump() if req.knobs else None, req.question)
    # Real-time sink: buzz the phone at the moment of asking (throttled, fire-and-forget). Placed
    # before the budget check on purpose — "someone asked" is true even on a resting demo, and the
    # throttle stops a bot from machine-gunning notifications.
    if deps.notifier is not None:
        deps.notifier.notify_question(req.question, req_id=current_request_id())
    if deps.budget is not None and deps.budget.is_exhausted():
        logger.info("/ask refused — daily token budget exhausted (resting)")
        return EventSourceResponse(iter([_sse(ErrorEvent(message=_RESTING_MESSAGE))]))

    # Default fast path reuses the prebuilt graph; a scope toggle or knob override (Compare / Advanced)
    # builds a per-request graph cheaply over the same store/embedder/reranker singletons.
    graph = deps.graph
    max_rounds = deps.max_rounds
    knobs = req.knobs
    if req.scope != "both" or knobs is not None:
        graph = build_scoped_graph(
            deps,
            mode=knobs.mode if knobs else None,
            rerank=knobs.rerank if knobs else None,
            parent_expansion=knobs.parent_expansion if knobs else None,
            scope=req.scope,
            max_rounds=knobs.max_rounds if knobs else None,
            top_k=knobs.top_k if knobs else None,
        )
        if knobs and knobs.max_rounds:
            max_rounds = knobs.max_rounds
    return EventSourceResponse(_ask_stream(graph, req.question, max_rounds, deps.budget,
                                           question_log=deps.question_log, scope=req.scope,
                                           req_id=current_request_id()))
