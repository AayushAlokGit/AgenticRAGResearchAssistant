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

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# How far back from a hard cut we'll look for whitespace to snap to. Small, so we never
# wander far from the target chunk size.
_SNAP_WINDOW = 100

# Separator hierarchy for recursive splitting, highest-level boundary first. Format-AGNOSTIC
# (paragraph blank-line -> line -> sentence -> word -> character) — these exist in any plain
# text, so the splitter never parses markdown/HTML/PDF structure. "" is the terminal: a hard
# character cut when nothing better remains. (LangChain's RecursiveCharacterTextSplitter uses
# the same idea; hand-rolled here to keep the substrate dependency-free and transparent.)
_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


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


# ───────────────────────────── recursive splitting ─────────────────────────────
#
# The format-agnostic upgrade from fixed-size (see docs/rag/CHUNKING.md). Two phases:
#   1. SPLIT the text into atomic units no larger than chunk_size, preferring the highest
#      natural boundary (paragraph > line > sentence > word > char) — so we only fall back
#      to cutting a word when nothing larger fits.
#   2. MERGE consecutive units greedily into chunks near chunk_size, carrying an `overlap`
#      tail of real content from each chunk into the next (boundary insurance).

def _choose_separator(text: str, separators: List[str]) -> Tuple[str, List[str]]:
    """Pick the highest-priority separator that occurs in `text`.

    Returns (separator, lower_priority_separators_still_to_try). The terminal "" separator
    means "no natural boundary left — hard-cut by character".
    """
    for index, separator in enumerate(separators):
        if separator == "":
            return "", []
        if separator in text:
            return separator, separators[index + 1:]
    return "", []


def _hard_split(text: str, chunk_size: int) -> List[str]:
    """Last resort: cut every `chunk_size` characters (used when no separator remains)."""
    pieces = []
    for start in range(0, len(text), chunk_size):
        pieces.append(text[start:start + chunk_size])
    return pieces


def _split_to_units(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    """Recursively break `text` into pieces each <= chunk_size, best boundary first."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separator, remaining = _choose_separator(text, separators)
    if separator == "":
        return _hard_split(text, chunk_size)

    units = []
    for part in text.split(separator):
        if not part.strip():
            continue
        if len(part) <= chunk_size:
            units.append(part)
        else:
            # Still too big — recurse with the next-lower separator (or hard-split).
            units.extend(_split_to_units(part, chunk_size, remaining))
    return units


def _carry_overlap(units: List[str], overlap: int) -> Tuple[List[str], int]:
    """Take trailing units totalling up to `overlap` chars to seed the next chunk."""
    if overlap <= 0:
        return [], 0
    carried = []
    length = 0
    for unit in reversed(units):
        if carried and length + len(unit) + 1 > overlap:
            break
        carried.insert(0, unit)
        length += len(unit) + 1
    return carried, length


def _merge_units(units: List[str], chunk_size: int, overlap: int) -> List[str]:
    """Greedily combine units into chunks near chunk_size, with an overlap tail between them."""
    chunks = []
    current: List[str] = []
    current_len = 0
    for unit in units:
        # +1 accounts for the space we join units with.
        if current and current_len + len(unit) + 1 > chunk_size:
            chunks.append(" ".join(current))
            current, current_len = _carry_overlap(current, overlap)
        current.append(unit)
        current_len += len(unit) + (1 if current_len else 0)
    if current:
        chunks.append(" ".join(current))
    return chunks


def recursive_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Recursive splitting: respect natural text boundaries, then merge to ~chunk_size.

    Format-agnostic (operates on universal paragraph/sentence/word boundaries, never on
    document markup). Same chunk_size/overlap units as fixed-size, so the two are an A/B.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    units = _split_to_units(text, chunk_size, _RECURSIVE_SEPARATORS)
    return _merge_units(units, chunk_size, overlap)


# ───────────────────────────── strategy dispatch ─────────────────────────────

def chunk_document(text: str, chunk_size: int, overlap: int, strategy: str = "fixed") -> List[str]:
    """Chunk `text` with the named strategy: "fixed" (baseline) or "recursive"."""
    if strategy == "fixed":
        return chunk_text(text, chunk_size, overlap)
    if strategy == "recursive":
        return recursive_chunk(text, chunk_size, overlap)
    raise ValueError(f"Unknown chunking strategy: {strategy!r} (expected 'fixed' or 'recursive')")
