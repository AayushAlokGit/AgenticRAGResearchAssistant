"""Module 1: RAG substrate.

Ingestion (structure-aware + parent/child chunking, metadata), hybrid retrieval
(BM25 + dense), reranking, query transformation (decomposition, multi-query, HyDE),
incremental re-indexing, citations, and a "not enough info" path.

Built naive first (basic chunk + single vector search), then upgraded technique by
technique, each earning its place against the eval set.

Status: not yet implemented - see ProjectIdea.md (module 1).
"""
