"""Typed request/response + SSE-event contract for the demo backend.

These Pydantic models ARE the wire format the Next.js frontend consumes: the ``/ask`` stream emits
one of the ``*Event`` models per step (serialized as the SSE ``data:`` payload), and the frontend's
TypeScript types mirror them one-to-one. Keeping the contract in one file means a field rename here
is a single, greppable source of truth for both sides.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# Input cap (Phase-2 abuse control, cheap to enforce here): reject over-long questions before any
# LLM call. Generous enough for real multi-hop questions, small enough to bound a single request.
MAX_QUESTION_CHARS = 500


class Knobs(BaseModel):
    """Per-request retrieval overrides (the Compare/Advanced toggles). All optional — an unset field
    falls back to the config default. A 'Basic RAG' preset is mode=dense, rerank/parent off,
    max_rounds=1 (single-shot); 'Full System' leaves them at config (hybrid+rerank+parent, agentic)."""
    mode: Optional[Literal["dense", "hybrid"]] = None
    rerank: Optional[bool] = None
    parent_expansion: Optional[bool] = None
    max_rounds: Optional[int] = Field(default=None, ge=1, le=8)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    scope: Literal["both", "uploads", "demo"] = "both"   # retrieval scope toggle
    knobs: Optional[Knobs] = None                          # per-request technique overrides


class ChunkView(BaseModel):
    """One retrieved chunk, shaped for the UI (source panel + trajectory)."""
    source: str
    chunk_index: int
    snippet: str
    score: float
    retrieved_by: Optional[str] = None


class Action(BaseModel):
    tool: str
    args: dict


class ThinkEvent(BaseModel):
    """A controller turn: its reasoning + the actions it chose this round."""
    type: Literal["think"] = "think"
    round: int
    reasoning: str
    actions: List[Action] = Field(default_factory=list)
    controller_tokens: int = 0
    finish: bool = False          # this turn chose to finish (no retrieval)


class EvidenceEvent(BaseModel):
    """The tools round: chunks pulled + whether the round added anything new."""
    type: Literal["evidence"] = "evidence"
    round: int
    chunks: List[ChunkView] = Field(default_factory=list)
    new_count: int = 0
    redundant: bool = False       # round returned hits but nothing new (oscillation signal)


class AnswerEvent(BaseModel):
    """The final cited answer + the exact window the generator saw (powers clickable citations)."""
    type: Literal["answer"] = "answer"
    text: str
    citations: List[str] = Field(default_factory=list)       # filenames cited in the text
    grounded: bool = True
    ungrounded: List[str] = Field(default_factory=list)      # cited-but-not-retrieved (fabricated)
    retrieved: List[ChunkView] = Field(default_factory=list)


class DoneEvent(BaseModel):
    """Run totals for the cost/latency meter."""
    type: Literal["done"] = "done"
    rounds: int = 0
    exit_reason: str = "budget"
    controller_tokens: int = 0
    generator_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    est_cost_usd: float = 0.0


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


# ── plain (non-stream) endpoints ────────────────────────────────────────────────────────
class SourceInfo(BaseModel):
    source: str
    chunks: int
    tags: dict = Field(default_factory=dict)
    uploaded: bool = False       # True for a bring-your-own-doc upload (deletable; not seed corpus)


class SourcesResponse(BaseModel):
    sources: List[SourceInfo]


class UploadResponse(BaseModel):
    """Result of ingesting one uploaded document into the store."""
    source: str                  # the stored source id (upload:<name>)
    name: str                    # display filename (marker stripped)
    chunks: int                  # chunks indexed from this doc
    total_chunks: int            # store size after the upload


class ConfigResponse(BaseModel):
    """Public-safe knobs the UI shows (no secrets)."""
    max_rounds: int
    controller_model: str
    generator_model: str
    top_k: int
    tools: List[str]
    spend_cap_tokens: int
    store_provider: str
    daily_token_budget: int = 0


class HealthResponse(BaseModel):
    status: str
    chunks: int
    daily_tokens_used: int = 0
