"""Translate the LangGraph agent's streamed node updates into the typed SSE events the UI consumes.

This is the heart of the backend: ``graph.stream(..., stream_mode="updates")`` yields one
``{node_name: partial_state_update}`` per node execution, and we map each to a ``schemas`` event —
``think`` for a controller turn, ``evidence`` for a tools round, ``answer`` for the final write, and
a closing ``done`` with run totals. The mapping reuses the SAME partial updates the graph's reducers
consume, so the trajectory the user watches is exactly the loop's real trajectory, not a reconstruction.

The generator is SYNCHRONOUS (graph.stream is blocking — LLM calls, the CPU reranker); app.py bridges
it to the async SSE response through a worker thread so it never blocks the event loop.
"""
from __future__ import annotations

import time
from typing import Iterator

from agentic_rag.harness.guardrails import extract_citations
from agentic_rag.server.schemas import (Action, AnswerEvent, ChunkView, DoneEvent, EvidenceEvent,
                                        ThinkEvent)

# Approximate Gemini 2.5 Flash pricing (USD per 1M tokens) for the cost meter. Deliberately a rough
# blended estimate, labelled as such in the UI — the point is order-of-magnitude cost transparency,
# not billing accuracy.
_INPUT_USD_PER_1M = 0.30
_OUTPUT_USD_PER_1M = 2.50

_SNIPPET_CHARS = 400


def _snippet(text: str) -> str:
    text = (text or "").strip()
    return text if len(text) <= _SNIPPET_CHARS else text[:_SNIPPET_CHARS].rstrip() + " ..."


def _chunk_views(hits) -> list:
    return [
        ChunkView(source=h.source, chunk_index=h.chunk_index, snippet=_snippet(h.text),
                  score=round(float(h.score), 4), retrieved_by=h.retrieved_by)
        for h in hits
    ]


def _think_event(update: dict) -> ThinkEvent:
    pending = update.get("pending", [])
    if pending:
        actions = [Action(tool=d.tool.name, args=d.args.model_dump()) for d in pending]
        # dict.fromkeys keeps first-seen order while de-duplicating repeated thoughts across a batch.
        reasoning = " ".join(dict.fromkeys(d.thought for d in pending if d.thought))
        finish = False
    else:
        steps = update.get("steps", [])
        actions = []
        reasoning = steps[0].thought if steps else ""
        finish = update.get("exit_reason") == "finish"
    return ThinkEvent(
        round=update.get("rounds", 0),
        reasoning=reasoning,
        actions=actions,
        controller_tokens=update["controller_usage"].total_tokens,
        finish=finish,
    )


def _evidence_event(update: dict, round_no: int) -> EvidenceEvent:
    # The tools update carries no "rounds" (only the controller advances it), so the caller passes
    # the current round it's tracking — otherwise evidence would render under round 0 in the UI.
    steps = update.get("steps", [])
    return EvidenceEvent(
        round=round_no,
        chunks=_chunk_views(update.get("evidence", [])),
        new_count=sum(s.new_chunks for s in steps),
        redundant="redundant_searches" in update,
    )


def _answer_event(update: dict) -> AnswerEvent:
    text = update.get("answer", "")
    return AnswerEvent(
        text=text,
        citations=sorted(extract_citations(text)),
        grounded=update.get("grounded", True),
        ungrounded=update.get("ungrounded_citations", []),
        retrieved=_chunk_views(update.get("retrieved", [])),
    )


def _est_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return round(prompt_tokens / 1e6 * _INPUT_USD_PER_1M
                 + completion_tokens / 1e6 * _OUTPUT_USD_PER_1M, 6)


def stream_events(graph, question: str, max_rounds: int) -> Iterator[object]:
    """Yield schema events for one question, ending with a DoneEvent. Blocking — run off-loop."""
    from agentic_rag.agent.graph import initial_state

    start = time.perf_counter()
    rounds = 0
    exit_reason = "budget"       # graph default; overwritten by a finish/oscillation/spend_cap stop
    prompt_tokens = 0
    completion_tokens = 0
    controller_total = 0
    generator_total = 0

    for chunk in graph.stream(initial_state(question),
                              config={"recursion_limit": max_rounds * 2 + 5},
                              stream_mode="updates"):
        for node, update in chunk.items():
            if node == "controller":
                usage = update["controller_usage"]
                prompt_tokens += usage.prompt_tokens
                completion_tokens += usage.completion_tokens
                controller_total += usage.total_tokens
                rounds = update.get("rounds", rounds)
                if update.get("exit_reason"):
                    exit_reason = update["exit_reason"]
                yield _think_event(update)
            elif node == "tools":
                if update.get("exit_reason"):
                    exit_reason = update["exit_reason"]
                yield _evidence_event(update, rounds)
            elif node == "answer":
                usage = update["generator_usage"]
                prompt_tokens += usage.prompt_tokens
                completion_tokens += usage.completion_tokens
                generator_total += usage.total_tokens
                yield _answer_event(update)
            # memory_read / memory_write nodes carry no user-facing update (memory is off for public)

    yield DoneEvent(
        rounds=rounds,
        exit_reason=exit_reason,
        controller_tokens=controller_total,
        generator_tokens=generator_total,
        total_tokens=controller_total + generator_total,
        latency_ms=int((time.perf_counter() - start) * 1000),
        est_cost_usd=_est_cost(prompt_tokens, completion_tokens),
    )
