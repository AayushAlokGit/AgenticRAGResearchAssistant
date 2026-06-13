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

from typing import List, Optional

from agentic_rag.config import resolve_path
from agentic_rag.rag.bm25 import BM25Index
from agentic_rag.rag.embeddings import LocalEmbedder
from agentic_rag.rag.vector_store import ChromaVectorStore, Hit


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


def build_retriever(config: dict, mode: Optional[str] = None):
    """Construct the retriever named by ``mode`` (falls back to config ``retrieval.mode``)."""
    retrieval_cfg = config["retrieval"]
    if mode is None:
        mode = retrieval_cfg.get("mode", "dense")

    store = ChromaVectorStore(resolve_path(config["vector_store"]["path"]), config["vector_store"]["collection"])
    if store.count() == 0:
        raise SystemExit("Vector store is empty — run `python -m agentic_rag.rag.ingest` first.")
    embedder = LocalEmbedder(config["embedding"]["model"])
    dense = DenseRetriever(embedder, store)

    if mode == "dense":
        return dense

    if mode == "hybrid":
        hybrid_cfg = retrieval_cfg.get("hybrid", {})
        rrf_k = hybrid_cfg.get("rrf_k", 60)
        candidate_k = hybrid_cfg.get("candidate_k", 20)
        bm25 = BM25Index(store.all_chunks())
        return HybridRetriever(dense, bm25, rrf_k=rrf_k, candidate_k=candidate_k)

    raise ValueError(f"Unknown retrieval mode: {mode!r} (expected 'dense' or 'hybrid')")
