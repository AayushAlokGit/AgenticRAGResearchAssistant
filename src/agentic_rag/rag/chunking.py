"""Naive fixed-size character chunking with overlap.

The simplest thing that works: slide a fixed-size window over the raw text with a
fixed overlap, so a fact straddling a boundary still lives intact in at least one
chunk. Each cut is snapped back to the nearest whitespace so we never slice a word in
half.

**Why characters, not tokens?** It's dependency-free and predictable. The cost: a size
in characters only loosely maps to model tokens (~4 chars/token for English). At
``chunk_size=800`` that's ~200 tokens — comfortably under all-MiniLM-L6-v2's 256-token
limit, so the embedder won't silently truncate a chunk. Token-aware chunking is a
later, eval-gated upgrade.

**What we are deliberately NOT doing (yet):** structure-aware chunking that splits on
Markdown headings and keeps parent/child relationships. That's a known-better technique
(module 1), held back so the eval set can later prove it earns its place over this
baseline.
"""
from __future__ import annotations

from typing import List

# How far back from a hard cut we'll look for whitespace to snap to. Small, so we never
# wander far from the target chunk size.
_SNAP_WINDOW = 100


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split ``text`` into overlapping, roughly ``chunk_size``-character chunks.

    Args:
        text: raw document text.
        chunk_size: target max characters per chunk.
        overlap: characters each chunk shares with the previous one.

    Returns:
        List of non-empty chunk strings (``[]`` for empty input).
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    chunks: List[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # Snap the cut back to the nearest whitespace (space or newline) so a word/line
        # isn't split — but only search a small window so we stay near the target size.
        if end < n:
            window_start = max(start + 1, end - _SNAP_WINDOW)
            cut = max(text.rfind(" ", window_start, end), text.rfind("\n", window_start, end))
            if cut > start:
                end = cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap  # step forward, keeping `overlap` chars of context
    return chunks
