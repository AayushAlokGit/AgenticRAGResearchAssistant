"""Retrievers: dense (vector), sparse (BM25), and hybrid (fused).

A retriever turns a query string into a ranked list of ``Hit`` chunks. Flavours:

- ``DenseRetriever``  — embed the query, nearest-neighbour search in ChromaDB (semantic).
- ``HybridRetriever`` — run dense AND sparse BM25 (``bm25.py``), fuse with Reciprocal
  Rank Fusion (RRF).

Why hybrid: dense embeddings match MEANING but can rank an exact rare term (an acronym, a
model name like ``all-MiniLM-L6-v2``) below a semantically-similar-but-wrong chunk. BM25
matches exact TOKENS and nails those. Fusing the two ranked lists gets the best of both.

``build_retriever(config, mode)`` picks dense or hybrid from config so the eval can A/B
them against a fixed baseline.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from agentic_rag.config import resolve_path
from agentic_rag.rag.bm25 import BM25Index
from agentic_rag.rag.embeddings import LocalEmbedder
from agentic_rag.rag.vector_store import ChromaVectorStore, Hit

logger = logging.getLogger(__name__)


class DenseRetriever:
    name = "dense"

    def __init__(self, embedder: LocalEmbedder, store: ChromaVectorStore):
        self.embedder = embedder
        self.store = store

    def query(self, text: str, k: int) -> List[Hit]:
        query_vector = self.embedder.embed_query(text)
        return self.store.query(query_vector, k)


class HybridRetriever:
    name = "hybrid"

    def __init__(self, dense: DenseRetriever, bm25: BM25Index, rrf_k: int, candidate_k: int):
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k            # RRF damping constant
        self.candidate_k = candidate_k  # how many to pull from EACH retriever before fusing

    def query(self, text: str, k: int) -> List[Hit]:
        dense_hits = self.dense.query(text, self.candidate_k)
        sparse_hits = self.bm25.query(text, self.candidate_k)
        return reciprocal_rank_fusion(dense_hits, sparse_hits, self.rrf_k, k)


class RerankRetriever:
    """Stage-2 wrapper: pull a candidate pool from any base retriever, then cross-encode.

    Decorator over dense OR hybrid — reranking is orthogonal to how candidates were found.
    Asks the base for `candidate_k` chunks (a wide net), then the cross-encoder re-scores
    that pool for precision and returns the top k.
    """

    def __init__(self, base, reranker, candidate_k: int):
        self.base = base
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.name = f"{base.name}+rerank"

    def query(self, text: str, k: int) -> List[Hit]:
        candidates = self.base.query(text, self.candidate_k)
        return self.reranker.rerank(text, candidates, k)


def reciprocal_rank_fusion(dense_hits: List[Hit], sparse_hits: List[Hit], rrf_k: int, top_k: int) -> List[Hit]:
    """Fuse two ranked lists by Reciprocal Rank Fusion.

    Each list contributes ``1 / (rrf_k + rank)`` to a chunk's score, where ``rank`` is the
    chunk's 1-based position in that list. A chunk near the top of BOTH lists wins big.

    RRF uses only RANKS, not raw scores, so it sidesteps the fact that cosine similarities
    (~0.6) and BM25 scores (~5) live on totally different scales — no normalization needed.

    ``rrf_k`` (default 60) sets how steep the reward for top ranks is. Small k -> being #1
    in one list dominates (1/rank: rank1=1.0 vs rank2=0.5, a huge gap). Large k flattens
    the curve (k=60: rank1=1/61≈0.0164 vs rank2≈0.0161, a gentle gap), so appearing near
    the top of BOTH lists matters more than being #1 in just one. 60 is the canonical default.

    Example (k=60), dense=[A,B,C], sparse=[B,D,A]:
        B = 1/62 + 1/61 = 0.0325   (rank2 dense + rank1 sparse)  -> winner
        A = 1/61 + 1/63 = 0.0323   (rank1 dense + rank3 sparse)
        D = 1/62        = 0.0161   (sparse only)
        C = 1/63        = 0.0159   (dense only)
    B beats A despite A being #1 in dense, because B ranks high in BOTH — RRF rewards agreement.
    """
    fused_score = {}
    hit_by_key = {}

    for rank, hit in enumerate(dense_hits):
        key = (hit.source, hit.chunk_index)
        contribution = 1.0 / (rrf_k + rank + 1)   # rank + 1 makes the position 1-based
        fused_score[key] = fused_score.get(key, 0.0) + contribution
        hit_by_key[key] = hit

    for rank, hit in enumerate(sparse_hits):
        key = (hit.source, hit.chunk_index)
        contribution = 1.0 / (rrf_k + rank + 1)
        fused_score[key] = fused_score.get(key, 0.0) + contribution
        if key not in hit_by_key:
            hit_by_key[key] = hit

    ranked_keys = sorted(fused_score, key=lambda key: fused_score[key], reverse=True)

    fused_hits = []
    for key in ranked_keys[:top_k]:
        original = hit_by_key[key]
        fused_hits.append(Hit(
            source=original.source,
            chunk_index=original.chunk_index,
            text=original.text,
            score=fused_score[key],   # NOTE: this is an RRF score, not a cosine similarity
        ))
    return fused_hits


def build_base_retriever(config: dict, mode: str, dense: DenseRetriever, store: ChromaVectorStore):
    """Build the stage-1 retriever (dense or hybrid) — the candidate generator."""
    if mode == "dense":
        return dense

    if mode == "hybrid":
        hybrid_cfg = config["retrieval"].get("hybrid", {})
        rrf_k = hybrid_cfg.get("rrf_k", 60)
        candidate_k = hybrid_cfg.get("candidate_k", 20)
        chunks = store.all_chunks()
        bm25 = BM25Index(chunks)
        logger.debug("BM25 index built over %d chunks (rrf_k=%s, candidate_k=%s)", len(chunks), rrf_k, candidate_k)
        return HybridRetriever(dense, bm25, rrf_k=rrf_k, candidate_k=candidate_k)

    raise ValueError(f"Unknown retrieval mode: {mode!r} (expected 'dense' or 'hybrid')")


def build_retriever(config: dict, mode: Optional[str] = None, rerank: Optional[bool] = None):
    """Construct the retriever: stage-1 base (``mode``), optionally wrapped with reranking.

    ``mode``   — "dense" | "hybrid" (falls back to config ``retrieval.mode``).
    ``rerank`` — True/False to force the cross-encoder stage on/off; None uses
                 config ``retrieval.rerank.enabled``. Lets the eval A/B rerank cleanly.
    """
    retrieval_cfg = config["retrieval"]
    if mode is None:
        mode = retrieval_cfg.get("mode", "dense")

    store = ChromaVectorStore(resolve_path(config["vector_store"]["path"]), config["vector_store"]["collection"])
    if store.count() == 0:
        raise SystemExit("Vector store is empty — run `python -m agentic_rag.rag.ingest` first.")
    embedder = LocalEmbedder(config["embedding"]["model"])
    dense = DenseRetriever(embedder, store)

    base = build_base_retriever(config, mode, dense, store)

    rerank_cfg = retrieval_cfg.get("rerank", {})
    if rerank is None:
        rerank = rerank_cfg.get("enabled", False)

    logger.info("building retriever: mode=%s, rerank=%s, store holds %d chunks", mode, rerank, store.count())
    if not rerank:
        return base

    # Stage 2: wrap the base in a cross-encoder reranker (imported here so the dependency
    # only loads when reranking is actually on).
    from agentic_rag.rag.rerank import CrossEncoderReranker, DEFAULT_MODEL

    reranker = CrossEncoderReranker(rerank_cfg.get("model", DEFAULT_MODEL),
                                    score_margin=rerank_cfg.get("score_margin"))
    candidate_k = rerank_cfg.get("candidate_k", 20)
    logger.debug("reranking on: candidate_k=%s -> top_k", candidate_k)
    return RerankRetriever(base, reranker, candidate_k)
