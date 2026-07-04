"""Postgres + pgvector vector store — the deployed-backend counterpart to ``ChromaVectorStore``.

Same method surface as the Chroma store (``add`` / ``delete_by_source`` / ``query`` / ``count`` /
``all_chunks`` / ``fetch_source_chunks`` / ``reset``), so the retriever, ingest, and tagging code
above it is identical whichever back end ``build_store`` picks. The point of moving here for the
hosted demo is **durability**: the app runs on an ephemeral host whose filesystem can vanish, so all
persistent state (the corpus vectors here; tags and the spend counter later) lives in a managed
Postgres that outlives any single container.

Design notes:
- **Cosine, matching the rest of the pipeline.** Embeddings arrive L2-normalized (both the local and
  Gemini embedders do this), and we search with pgvector's ``<=>`` cosine-distance operator, then
  report ``score = 1 - distance`` — the same [-1, 1] cosine similarity ``ChromaVectorStore`` returns.
- **Connection pool, not one shared connection.** A psycopg connection isn't safe for concurrent use,
  and the FastAPI server will handle overlapping requests. ``ConnectionPool(min_size=0)`` also lets
  Neon scale to zero when idle (no connections held open) — the free-tier $0 story.
- **Deterministic ids** (``<source>::<index>``) give storage-layer idempotency: re-adding a document
  overwrites its chunks instead of duplicating them, exactly like the Chroma store's upsert.
"""
from __future__ import annotations

import logging
import os
import re
from typing import List

from pgvector import Vector
from psycopg import sql

from agentic_rag.rag.vector_store import Hit

logger = logging.getLogger(__name__)

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_table(name: str) -> str:
    """Guard the config-supplied collection name before it becomes a SQL identifier.

    The name comes from our own config (``vector_store.collection``), not user input, but validating
    it keeps the dynamic-identifier SQL honest and fails loudly on a typo rather than mis-quoting.
    """
    if not _IDENT.match(name):
        raise ValueError(f"Illegal collection/table name {name!r} (need ^[A-Za-z_][A-Za-z0-9_]*$)")
    return name


class PgVectorStore:
    def __init__(self, collection_name: str, dims: int = 768, dsn: str | None = None):
        self.collection = _safe_table(collection_name)
        self.dims = dims
        if dsn is None:
            from dotenv import load_dotenv

            load_dotenv()
            dsn = os.environ["DATABASE_URL"]
        self.dsn = dsn
        self._pool = None  # opened + schema-ensured on first use (lazy, like the embedders)

    # ── connection plumbing ────────────────────────────────────────────────────────────────
    @property
    def pool(self):
        if self._pool is None:
            from pgvector.psycopg import register_vector
            from psycopg_pool import ConnectionPool

            def _configure(conn):
                # pgvector's type adapters must be registered per connection so Python lists bind to
                # the ``vector`` column and rows come back as numpy arrays.
                register_vector(conn)

            self._pool = ConnectionPool(
                self.dsn, min_size=0, max_size=4, open=True,
                kwargs={"autocommit": True}, configure=_configure,
            )
            self._ensure_schema()
            logger.info("pgvector store ready (table=%s, dims=%d)", self.collection, self.dims)
        return self._pool

    def _table(self) -> sql.Identifier:
        return sql.Identifier(self.collection)

    def _ensure_schema(self) -> None:
        """Create the chunks table + indexes if absent. Idempotent — safe on every boot."""
        table = self._table()
        with self._pool.connection() as conn:
            conn.execute(sql.SQL("create extension if not exists vector"))
            conn.execute(sql.SQL(
                "create table if not exists {t} ("
                " id text primary key,"
                " source text not null,"
                " chunk_index integer not null,"
                " text text not null,"
                " embedding vector({d}) not null)"
            ).format(t=table, d=sql.Literal(self.dims)))
            conn.execute(sql.SQL(
                "create index if not exists {i} on {t} (source)"
            ).format(i=sql.Identifier(f"{self.collection}_source_idx"), t=table))
            # HNSW ANN index for cosine — matches Chroma's HNSW cosine space. Trivial to build at our
            # scale; the planner may still seq-scan a few hundred rows, which is exact and instant.
            conn.execute(sql.SQL(
                "create index if not exists {i} on {t} using hnsw (embedding vector_cosine_ops)"
            ).format(i=sql.Identifier(f"{self.collection}_embedding_idx"), t=table))

    # ── the ChromaVectorStore method surface ───────────────────────────────────────────────
    def add(self, source: str, chunks: List[str], embeddings: List[List[float]]) -> None:
        """Insert (or overwrite) all chunks for one source document.

        Deterministic ids (``<source>::<index>``) mean re-adding a file overwrites its chunks — the
        same storage-layer idempotency the Chroma upsert gives.
        """
        if not chunks:
            return
        # Wrap each embedding in pgvector's ``Vector`` so it binds to the ``vector`` column; a bare
        # Python list would otherwise be adapted as a Postgres array and rejected.
        rows = [
            (f"{source}::{i}", source, i, chunks[i], Vector(embeddings[i]))
            for i in range(len(chunks))
        ]
        stmt = sql.SQL(
            "insert into {t} (id, source, chunk_index, text, embedding)"
            " values (%s, %s, %s, %s, %s)"
            " on conflict (id) do update set"
            " source = excluded.source, chunk_index = excluded.chunk_index,"
            " text = excluded.text, embedding = excluded.embedding"
        ).format(t=self._table())
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(stmt, rows)

    def delete_by_source(self, source: str) -> None:
        """Remove every chunk belonging to one source document (on change or delete)."""
        stmt = sql.SQL("delete from {t} where source = %s").format(t=self._table())
        with self.pool.connection() as conn:
            conn.execute(stmt, (source,))

    def query(self, embedding: List[float], top_k: int, source_scope: str = "both") -> List[Hit]:
        """Return the ``top_k`` nearest chunks to ``embedding``, best first.

        ``<=>`` is cosine distance; ``1 - distance`` recovers the cosine similarity in [-1, 1] that
        the rest of the pipeline (and ``ChromaVectorStore``) speaks.

        ``source_scope`` restricts the search to the retrieval scope: ``"both"`` (no filter),
        ``"uploads"`` (only ``upload:``-prefixed docs) or ``"demo"`` (only the seed corpus). The
        filter is pushed into SQL so the ANN search itself is scoped — over-fetch-then-filter would
        lose recall when one scope is a small minority of the table.
        """
        if source_scope == "uploads":
            where, pattern = sql.SQL("where source like %s "), "upload:%"
        elif source_scope == "demo":
            where, pattern = sql.SQL("where source not like %s "), "upload:%"
        else:
            where, pattern = sql.SQL(""), None
        stmt = sql.SQL(
            "select source, chunk_index, text, 1 - (embedding <=> %s) as score"
            " from {t} {w} order by embedding <=> %s limit %s"
        ).format(t=self._table(), w=where)
        vec = Vector(embedding)
        params = (vec, vec, top_k) if pattern is None else (vec, pattern, vec, top_k)
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(stmt, params)
                rows = cur.fetchall()
        return [
            Hit(source=r[0], chunk_index=r[1], text=r[2], score=float(r[3]))
            for r in rows
        ]

    def count(self) -> int:
        stmt = sql.SQL("select count(*) from {t}").format(t=self._table())
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(stmt)
                return int(cur.fetchone()[0])

    def all_chunks(self) -> List[dict]:
        """Return every stored chunk as ``{source, chunk_index, text}`` (for building the BM25 index).

        Ordered for determinism; cheap at our scale (hundreds of chunks), would need streaming at
        millions.
        """
        stmt = sql.SQL(
            "select source, chunk_index, text from {t} order by source, chunk_index"
        ).format(t=self._table())
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(stmt)
                rows = cur.fetchall()
        return [{"source": r[0], "chunk_index": r[1], "text": r[2]} for r in rows]

    def fetch_source_chunks(self, source: str) -> dict:
        """Return ``{chunk_index: text}`` for every chunk of one source (used by parent-expansion)."""
        stmt = sql.SQL(
            "select chunk_index, text from {t} where source = %s"
        ).format(t=self._table())
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(stmt, (source,))
                rows = cur.fetchall()
        return {r[0]: r[1] for r in rows}

    def reset(self) -> None:
        """Drop all rows (full wipe), keeping the table + indexes."""
        stmt = sql.SQL("truncate table {t}").format(t=self._table())
        with self.pool.connection() as conn:
            conn.execute(stmt)

    # ── per-doc free-form tags (rag/tagging.py) ─────────────────────────────────────────────
    # Durable counterpart to the Chroma store's doc_tags.json: a `<collection>_tags` table so the
    # tags survive the ephemeral host, exactly the reason we moved storage to Postgres.
    def _tags_table(self) -> sql.Identifier:
        return sql.Identifier(f"{self.collection}_tags")

    def _ensure_tags_table(self) -> None:
        stmt = sql.SQL(
            "create table if not exists {t} (source text primary key, tags jsonb not null)"
        ).format(t=self._tags_table())
        with self.pool.connection() as conn:
            conn.execute(stmt)

    def load_tags(self) -> dict:
        """Return {source: {doc_type, topics, entities}} — empty if the corpus hasn't been tagged."""
        self._ensure_tags_table()
        stmt = sql.SQL("select source, tags from {t}").format(t=self._tags_table())
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(stmt)
                return {r[0]: r[1] for r in cur.fetchall()}

    def save_tags(self, tags: dict) -> None:
        """Persist the full tag map (caller passes the complete desired state) — replace-all."""
        from psycopg.types.json import Jsonb

        self._ensure_tags_table()
        tbl = self._tags_table()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("truncate table {t}").format(t=tbl))
                cur.executemany(
                    sql.SQL("insert into {t} (source, tags) values (%s, %s)").format(t=tbl),
                    [(source, Jsonb(tag)) for source, tag in tags.items()],
                )
