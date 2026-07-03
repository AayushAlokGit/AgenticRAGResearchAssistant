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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings

_COLLECTION_METADATA = {"hnsw:space": "cosine"}  # cosine distance = 1 − cosine similarity


def _rows(documents, metadatas):
    """Zip Chroma's parallel documents/metadatas lists into ``(i, text, meta)`` triples.

    Chroma returns each field as its own list; a missing/short field comes back as ``[]`` or
    ``None``. Driving off ``metadatas`` and index-guarding ``documents`` tolerates that without
    the three read paths (query / all_chunks / fetch_source_chunks) each re-rolling the guard.
    """
    documents = documents or []
    metadatas = metadatas or []
    for i, meta in enumerate(metadatas):
        text = documents[i] if i < len(documents) else ""
        yield i, text, (meta or {})


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
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        hits: List[Hit] = []
        for i, text, meta in _rows(docs, metas):
            score = 1.0 - dists[i] if i < len(dists) else 0.0
            hits.append(Hit(source=meta.get("source", ""),
                            chunk_index=meta.get("chunk_index", -1),
                            text=text, score=score))
        return hits

    def count(self) -> int:
        return self.collection.count()

    def all_chunks(self) -> List[dict]:
        """Return every stored chunk as ``{source, chunk_index, text}``.

        Used to build the sparse (BM25) index, which needs the full chunk corpus in
        memory. Cheap at our scale (hundreds of chunks); would need batching at millions.
        """
        got = self.collection.get(include=["documents", "metadatas"])
        chunks = []
        for _, text, meta in _rows(got.get("documents", []), got.get("metadatas", [])):
            chunks.append({
                "source": meta.get("source", ""),
                "chunk_index": meta.get("chunk_index", -1),
                "text": text,
            })
        return chunks

    def fetch_source_chunks(self, source: str) -> dict:
        """Return ``{chunk_index: text}`` for every chunk of one source document.

        Used by parent-expansion to pull a retrieved chunk's contiguous neighbours. A
        document has at most a few dozen chunks, so fetching them all and slicing in memory
        is cheap (and avoids guessing which neighbour ids exist).
        """
        got = self.collection.get(where={"source": source}, include=["documents", "metadatas"])
        by_index = {}
        for _, text, meta in _rows(got.get("documents", []), got.get("metadatas", [])):
            by_index[meta.get("chunk_index", -1)] = text
        return by_index

    def reset(self) -> None:
        """Drop and recreate the collection (full wipe)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata=_COLLECTION_METADATA
        )

    # ── per-doc free-form tags (rag/tagging.py) ─────────────────────────────────────────────
    # The store owns tag persistence so it follows the backend: local Chroma keeps a JSON file
    # beside the vectors; the pgvector backend keeps a table. Same two methods either way.
    def _tags_path(self) -> Path:
        return Path(self.persist_directory) / "doc_tags.json"

    def load_tags(self) -> dict:
        """Return {source: {doc_type, topics, entities}} — empty if the corpus hasn't been tagged."""
        path = self._tags_path()
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def save_tags(self, tags: dict) -> None:
        """Persist the full tag map (caller passes the complete desired state)."""
        self._tags_path().write_text(
            json.dumps(tags, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def build_store(config: dict):
    """Pick the vector store from config: ``vector_store.provider`` = chromadb | pgvector.

    One factory so the construction sites don't each branch — the same "shape it so a swap is
    cheap" move as ``build_embedder`` and the LLM layer. Local dev keeps ChromaDB (a directory
    on disk); the deployed backend points this at Postgres+pgvector, whose durable storage
    survives an ephemeral host. Both back ends expose the identical method surface, so the
    retriever/ingest/tagging code above the store is untouched by the swap.
    """
    from agentic_rag.config import resolve_path

    vs = config["vector_store"]
    provider = vs.get("provider", "chromadb")
    if provider in ("pgvector", "postgres"):
        # Lazy import: psycopg/pgvector only load when the Postgres backend is actually selected,
        # so local Chroma runs don't pay for a dependency they don't use.
        from agentic_rag.rag.pgvector_store import PgVectorStore

        return PgVectorStore(vs["collection"], dims=config["embedding"].get("dims", 768))
    if provider in ("chromadb", "chroma"):
        return ChromaVectorStore(resolve_path(vs["path"]), vs["collection"])
    raise ValueError(
        f"Unknown vector_store.provider {provider!r} (expected 'chromadb' or 'pgvector')"
    )
