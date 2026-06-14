"""Naive ingestion pipeline: corpus Markdown -> chunks -> embeddings -> ChromaDB.

Flow, one document at a time::

    discover *.md  ->  [tracker: changed?]  ->  chunk  ->  embed  ->  store.upsert  ->  mark

Idempotent *and* incrementally re-indexing — it handles all three cases that
"incremental re-indexing" actually means, not just the easy one:

  - NEW file      -> embed + add
  - CHANGED file  -> delete old chunks, re-embed, re-add   (content hash detects this)
  - DELETED file  -> purge its chunks, drop it from the tracker
  - UNCHANGED     -> skip (no embed, no write)

Run from anywhere::

    python -m agentic_rag.rag.ingest            # incremental
    python -m agentic_rag.rag.ingest --force    # wipe store + tracker, rebuild

This is the NAIVE baseline (see CLAUDE.md build approach): fixed-size character chunks,
one local embedder, a single flat vector collection. Every smarter technique
(structure-aware chunking, hybrid retrieval, reranking) is a later change that must
beat this baseline on the eval set before it stays.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from agentic_rag.config import load_config, resolve_path
from agentic_rag.logging_setup import configure_run_logging
from agentic_rag.rag.chunking import chunk_document
from agentic_rag.rag.embeddings import LocalEmbedder
from agentic_rag.rag.tracker import DocumentProcessingTracker
from agentic_rag.rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# Files in the corpus that are scaffolding, not corpus content (the eval set is written
# against the 11 real docs; the README explains the folder).
SKIP_FILENAMES = {"README.md"}


def discover_markdown(corpus_root: Path) -> List[Path]:
    """All Markdown files under the corpus minus scaffolding, sorted for a deterministic
    processing order (so run-to-run logs diff cleanly)."""
    return [p for p in sorted(corpus_root.rglob("*.md")) if p.name not in SKIP_FILENAMES]


def ingest(force: bool = False) -> None:
    config = load_config()
    corpus_root = resolve_path(config["corpus"]["root"])
    ingestion_cfg = config["ingestion"]
    chunk_size = ingestion_cfg["chunk_size"]
    overlap = ingestion_cfg["chunk_overlap"]
    strategy = ingestion_cfg.get("chunking", {}).get("strategy", "fixed")
    vs_cfg = config["vector_store"]

    store = ChromaVectorStore(resolve_path(vs_cfg["path"]), vs_cfg["collection"])
    tracker = DocumentProcessingTracker(resolve_path(config["ingestion"]["tracking_file"]))
    embedder = LocalEmbedder(config["embedding"]["model"])

    logger.info(f"[ingest] corpus = {corpus_root}  chunking={strategy} ({chunk_size}/{overlap})")
    if force:
        logger.info("[ingest] --force: wiping collection + tracker")
        store.reset()
        tracker.clear()

    discovered = discover_markdown(corpus_root)
    discovered_keys = {str(p) for p in discovered}
    logger.info(f"[ingest] discovered {len(discovered)} markdown file(s)")

    # --- Deletions: files tracked on a previous run but gone from disk now ---
    removed = 0
    for key in tracker.tracked_keys():
        if key not in discovered_keys:
            store.delete_by_source(Path(key).name)
            tracker.remove(key)
            removed += 1
            logger.info(f"   - deleted: {Path(key).name} (purged from store)")

    # --- New / changed / unchanged ---
    added = changed = skipped = total_chunks = 0
    for path in discovered:
        source = path.name
        if not force and tracker.is_processed(path):
            skipped += 1
            continue

        is_update = tracker.is_known(path)
        # A changed file: clear its old chunks first, so a shorter new version doesn't
        # leave stale trailing chunks behind (upsert overwrites by id but won't delete
        # chunk indexes that no longer exist).
        if is_update:
            store.delete_by_source(source)

        chunks = chunk_document(path.read_text(encoding="utf-8"), chunk_size, overlap, strategy)
        store.add(source, chunks, embedder.embed(chunks))
        tracker.mark_processed(path, chunk_count=len(chunks))
        total_chunks += len(chunks)

        if is_update:
            changed += 1
            logger.info(f"   ~ changed: {source} ({len(chunks)} chunks)")
        else:
            added += 1
            logger.info(f"   + added:   {source} ({len(chunks)} chunks)")

    tracker.save()

    logger.info("\n[ingest] done.")
    logger.info(f"   added={added} changed={changed} skipped={skipped} deleted={removed}")
    logger.info(f"   chunks written this run = {total_chunks}")
    logger.info(f"   collection now holds {store.count()} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the corpus into the vector store.")
    parser.add_argument(
        "--force", action="store_true",
        help="Wipe the collection + tracker and rebuild from scratch.",
    )
    args = parser.parse_args()
    configure_run_logging("ingestion")
    ingest(force=args.force)


if __name__ == "__main__":
    main()
