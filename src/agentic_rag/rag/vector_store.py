"""ChromaDB-backed vector store (naive baseline).

A thin wrapper over a single persistent ChromaDB collection. We pass our OWN embeddings
in (computed by ``LocalEmbedder``) rather than letting Chroma embed for us — that keeps
the embedder swappable and the retrieval math explicit. The collection uses cosine
distance, which pairs with the normalized embeddings the embedder produces.

The interface is deliberately small — ``add`` / ``delete_by_source`` / ``query`` /
``count`` / ``reset`` — i.e. exactly what ingestion and the retrieval-recall meter need,
so a different backend could replace it behind the same methods later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import chromadb
from chromadb.config import Settings

_COLLECTION_METADATA = {"hnsw:space": "cosine"}  # cosine distance = 1 − cosine similarity


@dataclass
class Hit:
    """One retrieved chunk."""

    source: str        # corpus filename the chunk came from — what evals match on
    chunk_index: int
    text: str
    score: float       # cosine similarity in [−1, 1]; higher = closer
    # PROVENANCE (set by the agent loop, not retrieval): which action first surfaced this chunk
    # — the search query, or a tool call for non-search tools. None on the naive single-shot path
    # (the query is trivially the question). Lets a run record show WHICH reformulation found a
    # second-hop doc. Stamped on first occurrence only (dedup keeps the earliest retriever).
    retrieved_by: Optional[str] = None


class ChromaVectorStore:
    def __init__(self, persist_directory, collection_name: str):
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata=_COLLECTION_METADATA
        )

    def add(self, source: str, chunks: List[str], embeddings: List[List[float]]) -> None:
        """Insert (or overwrite) all chunks for one source document.

        IDs are deterministic (``<source>::<index>``) so re-adding the same file
        overwrites its chunks instead of duplicating them — idempotency at the storage
        layer, independent of the tracker. (Assumes a flat corpus where filename is
        unique; subfolders would need the relative path here.)
        """
        if not chunks:
            return
        ids = [f"{source}::{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
        self.collection.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)

    def delete_by_source(self, source: str) -> None:
        """Remove every chunk belonging to one source document (on change or delete)."""
        self.collection.delete(where={"source": source})

    def query(self, embedding: List[float], top_k: int) -> List[Hit]:
        """Return the ``top_k`` nearest chunks to ``embedding``, best first."""
        res = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        # ChromaDB nests results one level per query; we issued one query.
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        hits: List[Hit] = []
        for i in range(len(ids)):
            meta = metas[i] or {}
            hits.append(
                Hit(
                    source=meta.get("source", ""),
                    chunk_index=meta.get("chunk_index", -1),
                    text=docs[i] if i < len(docs) else "",
                    score=1.0 - dists[i] if i < len(dists) else 0.0,
                )
            )
        return hits

    def count(self) -> int:
        return self.collection.count()

    def all_chunks(self) -> List[dict]:
        """Return every stored chunk as ``{source, chunk_index, text}``.

        Used to build the sparse (BM25) index, which needs the full chunk corpus in
        memory. Cheap at our scale (hundreds of chunks); would need batching at millions.
        """
        got = self.collection.get(include=["documents", "metadatas"])
        ids = got.get("ids", [])
        documents = got.get("documents", [])
        metadatas = got.get("metadatas", [])

        chunks = []
        for i in range(len(ids)):
            meta = metadatas[i] or {}
            chunks.append({
                "source": meta.get("source", ""),
                "chunk_index": meta.get("chunk_index", -1),
                "text": documents[i] if i < len(documents) else "",
            })
        return chunks

    def fetch_source_chunks(self, source: str) -> dict:
        """Return ``{chunk_index: text}`` for every chunk of one source document.

        Used by parent-expansion to pull a retrieved chunk's contiguous neighbours. A
        document has at most a few dozen chunks, so fetching them all and slicing in memory
        is cheap (and avoids guessing which neighbour ids exist).
        """
        got = self.collection.get(where={"source": source}, include=["documents", "metadatas"])
        documents = got.get("documents", [])
        metadatas = got.get("metadatas", [])

        by_index = {}
        for i in range(len(documents)):
            meta = metadatas[i] or {}
            by_index[meta.get("chunk_index", -1)] = documents[i]
        return by_index

    def reset(self) -> None:
        """Drop and recreate the collection (full wipe)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata=_COLLECTION_METADATA
        )
