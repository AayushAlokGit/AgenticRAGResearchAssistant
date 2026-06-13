"""Module 1: RAG substrate.

Ingestion (structure-aware + parent/child chunking, metadata), hybrid retrieval
(BM25 + dense), reranking, query transformation (decomposition, multi-query, HyDE),
incremental re-indexing, citations, and a "not enough info" path.

Built naive first (basic chunk + single vector search), then upgraded technique by
technique, each earning its place against the eval set.

Status: ingestion implemented — `ingest.py` (pipeline), `chunking.py` (fixed-size char
chunks), `embeddings.py` (local all-MiniLM-L6-v2), `vector_store.py` (ChromaDB),
`tracker.py` (content-hash idempotency). Run with `python -m agentic_rag.rag.ingest`.
Retrieval: `retriever.py` (dense / hybrid + RRF fusion) and `bm25.py` (hand-rolled
sparse) — hybrid is the eval-gated default (DD-009). Reranking + query transformation
still to come — see ProjectIdea.md (module 1).
"""
