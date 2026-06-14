"""Cross-encoder reranking — stage 2 of two-stage retrieval.

Stage 1 (a bi-encoder + BM25 hybrid) casts a wide, cheap net to pull a CANDIDATE POOL.
This module is stage 2: it re-scores that small pool for PRECISION and keeps the best few.

The crucial difference from the first stage:

- A bi-encoder (our `LocalEmbedder`) embeds the query and each chunk SEPARATELY, then
  compares the two vectors. That independence is what makes it indexable and fast (chunk
  vectors are precomputed once) — but it means the model never sees the query and chunk
  TOGETHER, so it scores topical proximity, not "does this passage answer this question."

- A CROSS-encoder feeds the (query, chunk) pair through the model JOINTLY, with full
  attention across both. It outputs a single relevance score that reflects actual
  question-passage match. Far more accurate — but the score depends on the query, so it
  CAN'T be precomputed/indexed. You can only afford it on a handful of candidates.

That asymmetry is the whole point of the two-stage pattern (universal in search/recsys,
not RAG-specific): cheap-and-wide to recall, expensive-and-precise to rank. The trade-off
being bought is latency (one forward pass per candidate) for precision.

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` — a small CPU-friendly cross-encoder trained
on MS MARCO passage ranking. Local + free, like our embedder. `.predict([(q, doc), ...])`
returns a relevance logit per pair (higher = more relevant; unbounded, can be negative).
"""
from __future__ import annotations

import logging
from typing import List

from agentic_rag.rag.vector_store import Hit

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Re-scores candidate hits by true (query, chunk) relevance, best-first.

    Optional RELATIVE relevance gate (`score_margin`): after scoring, keep only hits within
    `score_margin` of the top hit's score, then cap at top_k — so the reranker can return
    FEWER than top_k when the tail is weak, cutting distractors. It's RELATIVE (margin from
    this query's own top) not ABSOLUTE, because cross-encoder logits are uncalibrated and
    model-specific — an absolute cutoff wouldn't transfer across rerankers (same reason RRF
    fuses on ranks, not raw scores). The margin is still an eval-gated, per-model knob.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, score_margin=None):
        self._model_name = model_name
        self._score_margin = score_margin  # None = gate off (always return a full top_k)
        self._model = None  # loaded lazily on first rerank (keeps import cheap)

    def _ensure_model(self):
        if self._model is None:
            # Imported lazily so importing this module doesn't pull in sentence-transformers
            # until a reranker is actually used.
            from sentence_transformers import CrossEncoder

            logger.info("loading cross-encoder reranker: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)

    def rerank(self, query: str, hits: List[Hit], top_k: int) -> List[Hit]:
        """Re-score `hits` against `query` with the cross-encoder, return the top_k."""
        if not hits:
            return []
        self._ensure_model()

        # Build one (query, chunk-text) pair per candidate; the model scores all at once.
        pairs = []
        for hit in hits:
            pairs.append((query, hit.text))
        scores = self._model.predict(pairs)

        # Attach each cross-encoder score to its hit, then sort most-relevant first.
        scored_hits = []
        for hit, score in zip(hits, scores):
            scored_hits.append((float(score), hit))
        scored_hits.sort(key=lambda pair: pair[0], reverse=True)

        # Relative relevance gate: drop hits that fall more than `score_margin` below the top
        # hit. The top hit always survives (its gap to itself is 0), so we never return empty.
        kept = scored_hits
        if self._score_margin is not None and scored_hits:
            top_score = scored_hits[0][0]
            cutoff = top_score - self._score_margin
            kept = []
            for score, hit in scored_hits:
                if score >= cutoff:
                    kept.append((score, hit))
            if len(kept) < len(scored_hits):
                logger.debug("rerank gate: kept %d/%d (top=%.3f, cutoff=%.3f)",
                             len(kept), len(scored_hits), top_score, cutoff)

        # Rebuild the kept hits carrying the cross-encoder score (NOT the old RRF/cosine
        # score) — downstream now ranks by relevance, so the score should reflect that.
        reranked = []
        for score, hit in kept[:top_k]:
            reranked.append(Hit(hit.source, hit.chunk_index, hit.text, score))
        return reranked
