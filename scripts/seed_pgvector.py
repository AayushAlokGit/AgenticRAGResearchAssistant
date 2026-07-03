"""One-time migration: copy the corpus vectors from the local ChromaDB store into Postgres+pgvector.

Why this exists: the deployed backend stores vectors in pgvector (durable — it survives the ephemeral
host the demo runs on). The embeddings already exist in the local Chroma store, so we COPY them
verbatim rather than re-embedding: zero Gemini spend to fill the DB. General lesson — migrate a
computed artifact, don't recompute it.

Idempotent: TRUNCATEs the pgvector table and reloads, so re-running after a corpus change is safe.

Usage:
    python -m scripts.seed_pgvector
"""
from __future__ import annotations

from collections import defaultdict

from agentic_rag.config import load_config, resolve_path
from agentic_rag.rag.pgvector_store import PgVectorStore
from agentic_rag.rag.vector_store import ChromaVectorStore


def _export_by_source(chroma: ChromaVectorStore) -> dict:
    """Pull every chunk + its embedding out of Chroma, grouped by source, sorted by chunk_index.

    Sorting matters: ``PgVectorStore.add`` derives each chunk's id/index from its LIST POSITION, so
    the list must be in chunk_index order for position == chunk_index to hold. We assert contiguity
    (0..n-1 per source) so a broken assumption fails loudly instead of silently renumbering chunks.
    """
    got = chroma.collection.get(include=["embeddings", "documents", "metadatas"])
    embeddings = got.get("embeddings")
    if embeddings is None:
        embeddings = []
    documents = got.get("documents") or []
    metadatas = got.get("metadatas") or []

    by_source = defaultdict(list)
    for i, meta in enumerate(metadatas):
        source = meta.get("source", "")
        idx = meta.get("chunk_index", -1)
        text = documents[i] if i < len(documents) else ""
        emb = [float(x) for x in embeddings[i]]
        by_source[source].append((idx, text, emb))

    for source, rows in by_source.items():
        rows.sort(key=lambda t: t[0])
        indices = [idx for idx, _, _ in rows]
        if indices != list(range(len(indices))):
            raise ValueError(
                f"source {source!r} has non-contiguous chunk_index {indices} — "
                "position-based re-add would renumber chunks"
            )
    return by_source


def main():
    config = load_config()
    vs = config["vector_store"]
    chroma = ChromaVectorStore(resolve_path(vs["path"]), vs["collection"])
    src_count = chroma.count()
    print(f"chroma source store: {src_count} chunks")

    by_source = _export_by_source(chroma)
    print(f"grouped into {len(by_source)} source documents")

    dims = config["embedding"].get("dims", 768)
    pg = PgVectorStore(vs["collection"], dims=dims)
    pg.reset()
    for source, rows in by_source.items():
        texts = [text for _, text, _ in rows]
        embs = [emb for _, _, emb in rows]
        pg.add(source, texts, embs)

    dst_count = pg.count()
    print(f"pgvector store now: {dst_count} chunks")
    assert dst_count == src_count, f"COUNT MISMATCH: chroma {src_count} != pg {dst_count}"
    print("counts match [OK]")

    # Token-free correctness check: query pgvector with a chunk's OWN migrated embedding. If the
    # vectors landed intact and cosine search works, that chunk must come back as the top hit at
    # similarity ~1.0. No new embedding call, so no API spend.
    probe_source = next(iter(by_source))
    probe_idx, _, probe_emb = by_source[probe_source][0]
    top = pg.query(probe_emb, top_k=1)[0]
    print(f"self-retrieval probe: {probe_source}#{probe_idx} -> "
          f"top={top.source}#{top.chunk_index} score={top.score:.4f}")
    assert top.source == probe_source and top.chunk_index == probe_idx and top.score > 0.999, \
        "self-retrieval failed — migrated vectors do not match"
    print("self-retrieval parity [OK]")

    # Migrate the per-doc free-form tags too (same principle: copy, don't re-run the LLM tagger).
    tags = chroma.load_tags()
    pg.save_tags(tags)
    reloaded = pg.load_tags()
    print(f"tags migrated: {len(tags)} docs -> pg holds {len(reloaded)}")
    assert len(reloaded) == len(tags), f"TAG COUNT MISMATCH: chroma {len(tags)} != pg {len(reloaded)}"
    print("tags parity [OK]")


if __name__ == "__main__":
    main()
