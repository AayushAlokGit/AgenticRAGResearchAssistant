"""Module 3: Context engineering.

Token budgeting, relevance filtering + dedup of retrieved chunks, compression of tool
outputs, compaction of old turns, ordering to fight lost-in-the-middle, and
just-in-time loading - curating a finite context window under a token budget.

Inhabitants: `order_evidence` (the ORDER lever) and `dedup_overlapping_windows` (a COMPRESS/dedup
lever). Token-budgeting (`select_within_budget`) still lives in agent/loop.py for now; compaction
will join this package as it lands. See docs/context/CONTEXT_ENGINEERING.md for the general framing.

NOTE: `dedup_overlapping_windows` is currently UNWIRED from the agent system. It was A/B'd and n=3
it regressed completeness + faithfulness despite being provably lossless (duplicated evidence acts
as implicit salience), so its loop/config references were removed — but the function is KEPT here
for reference / future reuse (e.g. a longer-horizon agent where the dials change). See
DESIGN_DECISIONS.md DD-044 / EXPERIMENTS v2-14.
"""
from __future__ import annotations

import logging
from typing import List

from agentic_rag.rag.vector_store import Hit

logger = logging.getLogger(__name__)


def dedup_overlapping_windows(hits: List[Hit], store, window: int) -> List[Hit]:
    """Merge parent-expansion windows that OVERLAP within a source into one span — lossless Compress.

    UNWIRED (see module docstring): A/B-rejected at n=3, kept for reference. Pass the configured
    parent-expansion `window`; <=0 (expansion off) makes this a no-op.

    Parent-expansion (retriever.py) returns each retrieved child as a ±`window` span of its
    same-source neighbours. Across rounds, two searches often land on NEAR-ADJACENT chunks of the
    same doc, so their spans overlap heavily — different (source, chunk_index) keys, so the
    scratchpad's exact-dedup keeps BOTH, inflating the window with duplicated text for no new info.
    This merges each source's overlapping/adjacent ranges (a standard interval merge) and re-emits
    ONE Hit per span. The result is the exact UNION of chunks parent-expansion already showed, minus
    the duplication — nothing dropped (unlike text-containment dedup, which loses the unique tail of
    a partial overlap). Non-overlapping evidence is untouched, so cost falls only where redundancy is.

    Worked example (window=2, one source `doc`):
        in : Hit(center=5) -> covers chunks [3,4,5,6,7]      # from round 1's search
             Hit(center=6) -> covers chunks [4,5,6,7,8]      # from round 2's search
             (chunks 4-7 are stored TWICE — ~80% duplicated text, two separate Hits)
        out: Hit(span=[3..8]) -> chunks 3,4,5,6,7,8 joined once   # one lossless span, no dup
    """
    if window <= 0 or len(hits) <= 1:
        return list(hits)

    # Group hit positions by source, remembering ARRIVAL position so the output can keep arrival order.
    by_source: dict = {}
    for pos, hit in enumerate(hits):
        by_source.setdefault(hit.source, []).append((pos, hit))

    merged: list = []   # (arrival_key, Hit) — arrival_key = earliest contributing position
    for source, items in by_source.items():
        existing = store.fetch_source_chunks(source)   # {chunk_index: text}, fetched once per source

        # One interval per hit: the chunk range its window covers, tagged with arrival pos + score.
        intervals = [{"lo": hit.chunk_index - window, "hi": hit.chunk_index + window,
                      "pos": pos, "score": hit.score} for pos, hit in items]
        intervals.sort(key=lambda iv: iv["lo"])

        # Merge intervals that overlap OR touch (lo <= prev.hi + 1). Touch is still lossless: the
        # boundary chunk belongs to the second interval, so the union pulls in no UNcovered chunk.
        runs: list = []
        for iv in intervals:
            if runs and iv["lo"] <= runs[-1]["hi"] + 1:
                runs[-1]["hi"] = max(runs[-1]["hi"], iv["hi"])
                runs[-1]["pos"] = min(runs[-1]["pos"], iv["pos"])     # earliest arrival wins the slot
                runs[-1]["score"] = max(runs[-1]["score"], iv["score"])  # best relevance of the span
            else:
                runs.append(dict(iv))

        # Re-emit one Hit per merged span: join the existing chunks in [lo, hi] in document order.
        for run in runs:
            texts = []
            first_index = None
            for index in range(run["lo"], run["hi"] + 1):
                if index in existing:
                    if first_index is None:
                        first_index = index
                    texts.append(existing[index])
            if texts:
                merged.append((run["pos"], Hit(source, first_index, "\n".join(texts), run["score"])))

    merged.sort(key=lambda pair: pair[0])   # stable: order by each span's earliest contributing hit
    return [hit for _, hit in merged]


def order_evidence(hits: List[Hit], strategy: str) -> List[Hit]:
    """Reorder the assembled evidence to fight LOST-IN-THE-MIDDLE before the generator reads it.

    Models attend most to the START and END of a context and skim the middle (Liu et al., 2023).
    Our answer prompt is `CONTEXT ... QUESTION`, so the END of the context is the high-attention
    slot nearest the question. WHERE a chunk sits therefore changes whether its fact gets used —
    ordering is a real lever, not cosmetics. Cost-neutral by construction (same chunks, reshuffled),
    so it moves correctness/faithfulness, never the context-cost meters — those act as a control.

    Strategies (the A/B knob, config `context.ordering`):
      - arrival        : leave as gathered (search-1 hits, then search-2 ...). The baseline; the
                         scratchpad already arrives best-first WITHIN each round.
      - relevance_last : sort by relevance ASCENDING, so the most-relevant chunk is LAST — nearest
                         the question, in the peak-attention slot.
      - u_shaped       : most-relevant chunks at BOTH ends, least-relevant buried in the dead middle
                         (the canonical mitigation, a.k.a. LongContextReorder).

    Relevance = the cross-encoder rerank score carried on each Hit (parent-expansion preserves the
    child's score, see retriever.py). CAVEAT: across a MULTI-HOP scratchpad the scores come from
    DIFFERENT sub-queries, so they're only roughly comparable hop-to-hop — fine for the common
    single-hop case, a known imprecision for the few multi-hop ones.
    """
    if strategy == "arrival" or len(hits) <= 1:
        return list(hits)
    if strategy == "relevance_last":
        return sorted(hits, key=lambda h: h.score)          # ascending -> highest score LAST
    if strategy == "u_shaped":
        return _u_shaped(hits)
    raise ValueError(f"Unknown context.ordering strategy: {strategy!r} "
                     f"(expected arrival | relevance_last | u_shaped)")


def _u_shaped(hits: List[Hit]) -> List[Hit]:
    """Place the most-relevant chunks at the two EDGES and the least-relevant in the MIDDLE.

    Walk the hits best-first; send the best to the END side, the next to the START side, and keep
    alternating. So the top chunk lands at one edge, the 2nd at the other edge, the 3rd just inside
    the first, and so on — the weakest chunks settle in the center where attention is lowest.
    """
    ranked = sorted(hits, key=lambda h: h.score, reverse=True)  # best first
    head: List[Hit] = []   # grows inward from the START
    tail: List[Hit] = []   # grows inward from the END (reversed when joined)
    for rank, hit in enumerate(ranked):
        if rank % 2 == 0:
            tail.append(hit)   # even ranks (incl. the very best) take the END side
        else:
            head.append(hit)   # odd ranks take the START side
    return head + list(reversed(tail))
