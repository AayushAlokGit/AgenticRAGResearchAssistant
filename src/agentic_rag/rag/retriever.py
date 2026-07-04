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

import json
import logging
import re
from typing import List, Optional

from agentic_rag.config import resolve_path
from agentic_rag.rag.bm25 import BM25Index
from agentic_rag.rag.embeddings import build_embedder
from agentic_rag.rag.vector_store import UPLOAD_PREFIX, ChromaVectorStore, Hit, build_store

logger = logging.getLogger(__name__)


class ScopedStore:
    """A read-only view over a store restricted to one retrieval scope: 'both' | 'uploads' | 'demo'.

    'both' is a pass-through; 'uploads'/'demo' filter on the ``upload:`` source prefix so the UI scope
    toggle can search the seed corpus, the user's own uploads, or everything. Only the three methods
    the retriever stack actually calls are proxied — ``query`` (dense), ``all_chunks`` (BM25 build),
    ``fetch_source_chunks`` (parent-expansion). A chunk's source is entirely one scope, so the
    parent-fetch needs no filtering; the dense filter is pushed down into the store's ``query`` so the
    ANN search itself is scoped (over-fetch-then-filter would lose recall for a minority scope).
    """

    def __init__(self, store, scope: str = "both"):
        self.store = store
        self.scope = scope

    def _in_scope(self, source: str) -> bool:
        if self.scope == "uploads":
            return source.startswith(UPLOAD_PREFIX)
        if self.scope == "demo":
            return not source.startswith(UPLOAD_PREFIX)
        return True

    def query(self, embedding, top_k):
        return self.store.query(embedding, top_k, source_scope=self.scope)

    def all_chunks(self):
        return [c for c in self.store.all_chunks() if self._in_scope(c["source"])]

    def fetch_source_chunks(self, source):
        return self.store.fetch_source_chunks(source)

    def count(self):
        return self.store.count() if self.scope == "both" else len(self.all_chunks())


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
        return reciprocal_rank_fusion([dense_hits, sparse_hits], self.rrf_k, k)


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


def reciprocal_rank_fusion(hit_lists: List[List[Hit]], rrf_k: int, top_k: int) -> List[Hit]:
    """Fuse N ranked lists by Reciprocal Rank Fusion.

    Each list contributes ``1 / (rrf_k + rank)`` to a chunk's score, where ``rank`` is the
    chunk's 1-based position in that list. A chunk near the top of MANY lists wins big. Used
    for two lists (dense + BM25 in hybrid) AND for N lists (one per query variant in RAG-Fusion).

    RRF uses only RANKS, not raw scores, so it sidesteps the fact that cosine similarities
    (~0.6) and BM25 scores (~5) live on totally different scales — no normalization needed.
    The same rank-only property is what lets us fuse N query-variant lists: their scores aren't
    comparable, but their ranks are.

    ``rrf_k`` (default 60) sets how steep the reward for top ranks is. Small k -> being #1
    in one list dominates (1/rank: rank1=1.0 vs rank2=0.5, a huge gap). Large k flattens
    the curve (k=60: rank1=1/61≈0.0164 vs rank2≈0.0161, a gentle gap), so appearing near
    the top of MANY lists matters more than being #1 in just one. 60 is the canonical default.

    Example (k=60, two lists), dense=[A,B,C], sparse=[B,D,A]:
        B = 1/62 + 1/61 = 0.0325   (rank2 dense + rank1 sparse)  -> winner
        A = 1/61 + 1/63 = 0.0323   (rank1 dense + rank3 sparse)
        D = 1/62        = 0.0161   (sparse only)
        C = 1/63        = 0.0159   (dense only)
    B beats A despite A being #1 in dense, because B ranks high in BOTH — RRF rewards agreement.
    """
    fused_score = {}
    hit_by_key = {}

    for hits in hit_lists:
        for rank, hit in enumerate(hits):
            key = (hit.source, hit.chunk_index)
            contribution = 1.0 / (rrf_k + rank + 1)   # rank + 1 makes the position 1-based
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


class ParentExpansionRetriever:
    """Small-to-big: retrieve precise child chunks, then return each with its contiguous
    neighbours (same source, ±window) merged into one 'parent' window for the generator.

    Adds CONTIGUOUS, same-location context (the rest of a split table/section) WITHOUT
    pulling in separately-ranked chunks from elsewhere — so it raises coverage without
    raising the distractor count (the failure mode a bigger top_k hit). Wraps whatever it's
    given (typically the reranked retriever): retrieve+rank children, then expand.

    Children arrive ranked, so we emit parents in that order and skip any chunk already
    pulled into an earlier (better-ranked) parent — that single `seen` set replaces explicit
    window-merging. Neighbour texts are just joined; consecutive chunks share their overlap
    seam, so a parent has minor (~overlap) duplicated text — harmless, not worth de-duping.
    """

    def __init__(self, base, store: ChromaVectorStore, window: int):
        self.base = base
        self.store = store
        self.window = window
        self.name = f"{base.name}+parent{window}"

    def query(self, text: str, k: int) -> List[Hit]:
        child_hits = self.base.query(text, k)
        source_chunks = {}   # cache: source -> {index: text}, fetched once per source
        seen = set()         # (source, index) already inside an emitted parent
        parents: List[Hit] = []
        for hit in child_hits:
            if (hit.source, hit.chunk_index) in seen:
                continue     # already absorbed by an earlier (better-ranked) parent
            if hit.source not in source_chunks:
                source_chunks[hit.source] = self.store.fetch_source_chunks(hit.source)
            chunks = source_chunks[hit.source]

            # This chunk + the neighbours that actually exist, in document order. A missing
            # index (out of range) simply isn't in `chunks`, so no clamping is needed.
            texts = []
            for index in range(hit.chunk_index - self.window, hit.chunk_index + self.window + 1):
                if index in chunks:
                    texts.append(chunks[index])
                    seen.add((hit.source, index))
            parents.append(Hit(hit.source, hit.chunk_index, "\n".join(texts), hit.score))
        return parents


_QUERIES_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_queries(raw: str, cap: int) -> List[str]:
    """Parse the reformulator's JSON ({"queries": [...]}) into a capped, stripped list.

    Lenient + FAIL-EMPTY: anything unparseable returns [] (the caller then retrieves with the
    original query alone, i.e. behaves like the base retriever) — a broken instrument degrades
    gracefully, never crashes retrieval.
    """
    match = _QUERIES_JSON_RE.search(raw or "")
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    raw_queries = data.get("queries", [])
    if not isinstance(raw_queries, list):
        return []
    cleaned = [str(q).strip() for q in raw_queries if str(q).strip()]
    return cleaned[:cap]


class MultiQueryRetriever:
    """RAG-Fusion: expand the query into several facet-diverse variants, retrieve each through
    the base, and RRF-fuse the pooled results — recovering passages a single phrasing misses.

    Sits OUTSIDE the base candidate generator (hybrid) and INSIDE rerank: it produces a fused
    candidate pool, which the cross-encoder then re-scores for precision. Motivation (DD-034):
    the retrievability diagnostic showed the docs our failures need ARE retrievable (often #1)
    for a well-aimed, doc-style query, but NOT for the controller's actual phrasing — and that
    enumeration questions fail STRUCTURALLY for any single query (only per-facet sub-queries
    surface each component). Casting several doc-style facet queries and fusing recovers them.

    The generated variants are CACHED per query string: the agent re-issues the same query
    across rounds, and reformulation is an LLM call we don't want to repeat.
    """

    def __init__(self, base, llm, prompt: str, n_queries: int, rrf_k: int):
        self.base = base
        self.llm = llm                 # the reformulation model (cheap; routing-tier)
        self.prompt = prompt           # prompts/query_reformulate.md
        self.n_queries = n_queries     # max variants to GENERATE (original is added on top)
        self.rrf_k = rrf_k
        self._cache: dict = {}         # query text -> generated variants
        self.name = f"{base.name}+mq{n_queries}"

    def query(self, text: str, k: int) -> List[Hit]:
        # The original query is always included — on its own it already ranked the missed docs
        # ~#5 (diagnostic); the variants are there to lift them into the top of the fused pool.
        queries = self._dedup([text] + self._variants(text))
        hit_lists = [self.base.query(q, k) for q in queries]
        return reciprocal_rank_fusion(hit_lists, self.rrf_k, k)

    def _variants(self, text: str) -> List[str]:
        if text in self._cache:
            return self._cache[text]
        variants: List[str] = []
        try:
            completion = self.llm.complete([
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": f"QUERY:\n{text}"},
            ])
            variants = parse_queries(completion.text or "", self.n_queries)
        except Exception as exc:  # a reformulation failure must NOT break retrieval
            logger.warning("multi-query generation failed (%s) — using original query only", exc)
        self._cache[text] = variants
        logger.info("multi-query: %r -> %d variant(s): %s", text[:60], len(variants), variants)
        return variants

    @staticmethod
    def _dedup(queries: List[str]) -> List[str]:
        """Drop case-insensitive duplicates, keep first occurrence (the original stays first)."""
        seen = set()
        unique = []
        for q in queries:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(q)
        return unique


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


def build_retriever(config: dict, mode: Optional[str] = None, rerank: Optional[bool] = None, *,
                    store=None, embedder=None, reranker=None,
                    parent_expansion: Optional[bool] = None, scope: str = "both"):
    """Construct the retriever: stage-1 base (``mode``), optionally wrapped with reranking.

    ``mode``   — "dense" | "hybrid" (falls back to config ``retrieval.mode``).
    ``rerank`` — True/False to force the cross-encoder stage on/off; None uses
                 config ``retrieval.rerank.enabled``. Lets the eval A/B rerank cleanly.

    Injected ``store`` / ``embedder`` / ``reranker`` are REUSED instead of rebuilt — this is what
    lets the server build a fresh retriever per upload / per request without reloading the ~22s
    cross-encoder. ``parent_expansion`` (True/False) overrides config; ``scope`` ('both'|'uploads'|
    'demo') restricts retrieval to the seed corpus, the user's uploads, or everything (the UI toggle).
    """
    retrieval_cfg = config["retrieval"]
    if mode is None:
        mode = retrieval_cfg.get("mode", "dense")

    if store is None:
        store = build_store(config)
        if store.count() == 0:
            raise SystemExit("Vector store is empty — run `python -m agentic_rag.rag.ingest` first.")
    if embedder is None:
        embedder = build_embedder(config)

    # Everything below retrieves through the scoped view; scope="both" is a transparent pass-through.
    scoped = ScopedStore(store, scope)
    dense = DenseRetriever(embedder, scoped)

    base = build_base_retriever(config, mode, dense, scoped)

    # Stage 1.5 (optional): RAG-Fusion. Expand each query into facet-diverse variants and fuse
    # their results BEFORE reranking, so the cross-encoder re-scores a richer pool (DD-034).
    mq_cfg = retrieval_cfg.get("multi_query", {})
    mq_on = mq_cfg.get("enabled", False)
    if mq_on:
        from agentic_rag.llm.provider import build_llm  # lazy: only when multi_query is on

        reformulator = build_llm(config, role="controller")  # routing-tier; cheap
        prompt = (resolve_path(config["prompts"]["dir"]) / "query_reformulate.md").read_text(encoding="utf-8")
        n_queries = mq_cfg.get("n_queries", 4)
        rrf_k = retrieval_cfg.get("hybrid", {}).get("rrf_k", 60)
        base = MultiQueryRetriever(base, reformulator, prompt, n_queries, rrf_k)

    rerank_cfg = retrieval_cfg.get("rerank", {})
    if rerank is None:
        rerank = rerank_cfg.get("enabled", False)

    pe_cfg = retrieval_cfg.get("parent_expansion", {})
    parent_on = pe_cfg.get("enabled", False) if parent_expansion is None else parent_expansion
    logger.info("building retriever: mode=%s, multi_query=%s, rerank=%s, parent_expansion=%s, scope=%s, store holds %d chunks",
                mode, mq_on, rerank, parent_on, scope, scoped.count())

    if rerank:
        # Stage 2: wrap the base in a cross-encoder reranker. Reuse an injected reranker if given
        # (the server passes the singleton so this doesn't reload the ~22s cross-encoder each call);
        # otherwise build one here (imported lazily so the dependency only loads when rerank is on).
        if reranker is None:
            from agentic_rag.rag.rerank import CrossEncoderReranker, DEFAULT_MODEL

            reranker = CrossEncoderReranker(rerank_cfg.get("model", DEFAULT_MODEL))
        candidate_k = rerank_cfg.get("candidate_k", 20)
        logger.debug("reranking on: candidate_k=%s -> top_k", candidate_k)
        base = RerankRetriever(base, reranker, candidate_k)

    if parent_on:
        # Stage 3 (optional): expand each retrieved child to its contiguous neighbours.
        base = ParentExpansionRetriever(base, scoped, pe_cfg.get("window", 1))

    return base
