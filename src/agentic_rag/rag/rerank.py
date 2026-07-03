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
    """Re-scores candidate hits by true (query, chunk) relevance, best-first."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._model = None  # loaded lazily on first rerank (keeps import cheap)

    def _ensure_model(self):
        if self._model is None:
            # Imported lazily so importing this module doesn't pull in sentence-transformers
            # until a reranker is actually used.
            from sentence_transformers import CrossEncoder

            logger.info("loading cross-encoder reranker: %s", self._model_name)
            # Load strictly from the local cache. The model is baked into the deploy image, so the
            # container must NEVER reach huggingface.co — sentence-transformers otherwise fires a HEAD
            # request to check for updates on load, and a blip on that call crashed a run. Cold cache
            # (a fresh clone with nothing downloaded) falls back to a one-time online fetch.
            try:
                self._model = CrossEncoder(self._model_name, local_files_only=True)
            except Exception:
                logger.warning("reranker not in local cache — downloading once from the hub")
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

        # Rebuild the kept hits carrying the cross-encoder score (NOT the old RRF/cosine
        # score) — downstream now ranks by relevance, so the score should reflect that.
        reranked = []
        for score, hit in scored_hits[:top_k]:
            reranked.append(Hit(hit.source, hit.chunk_index, hit.text, score))
        return reranked
