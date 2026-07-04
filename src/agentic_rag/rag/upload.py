"""Ingest an UPLOADED document (raw bytes) into the shared vector store, on the fly.

Bring-your-own-doc: a user uploads md/txt/pdf; we extract text -> chunk -> embed -> persist to the
SAME store as the seed corpus, marked with an ``upload:`` source prefix. The prefix is load-bearing:
the manage UI lists/deletes only ``upload:``-marked docs (so the seed corpus can't be nuked), and the
retrieval scope toggle ("demo corpus / my documents / both") filters on it.

This reuses the corpus ingest core verbatim (``chunk_document`` + ``embedder.embed`` + ``store.add``);
the only genuinely new bit is turning arbitrary uploaded BYTES into text — the corpus path reads .md
off disk, an uploaded file may be a .pdf. Chunking stays format-agnostic on purpose.
"""
from __future__ import annotations

import io

from agentic_rag.rag.chunking import chunk_document
from agentic_rag.rag.vector_store import UPLOAD_PREFIX  # marker, isolation, delete-protection in one

# Public-demo abuse limits (also re-checked at the HTTP layer).
MAX_UPLOAD_BYTES = 2 * 1024 * 1024   # 2 MB — bounds one request's embed cost + store growth
MAX_UPLOAD_CHUNKS = 200              # cap chunks per doc so a huge upload can't blow up cost/store
ALLOWED_EXTENSIONS = (".md", ".txt", ".pdf")


def is_upload(source: str) -> bool:
    """True if a store `source` belongs to an uploaded doc (vs the seed corpus)."""
    return source.startswith(UPLOAD_PREFIX)


def display_name(source: str) -> str:
    """The user-facing filename for an uploaded source (strip the marker)."""
    return source[len(UPLOAD_PREFIX):] if is_upload(source) else source


def source_name(filename: str) -> str:
    """The store `source` for an uploaded file: the prefix marker + a sanitized base name."""
    base = (filename or "document").rsplit("/", 1)[-1].rsplit("\\", 1)[-1].strip()
    return f"{UPLOAD_PREFIX}{base or 'document'}"


def extract_text(filename: str, data: bytes) -> str:
    """Turn uploaded bytes into plain text. .md/.txt decode directly; .pdf via pypdf.

    Everything downstream (chunking, embedding) treats the result as raw text, so all this layer
    owes is *getting to text*. This is the extraction seam, so it also owns FAILING CLEANLY: it only
    ever raises ``ValueError`` (which the /upload handler maps to a 400), never lets a library-
    specific error escape as a 500. PDF extraction quality varies with how the PDF was produced —
    that's inherent to PDFs, not something we can fix here.
    """
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader  # lazy: only imported when a PDF is actually uploaded

        # H1 — a corrupt / encrypted / not-really-a-PDF upload makes pypdf raise (PdfReadError and
        # friends). Catch broadly at this untrusted-input boundary and re-raise as a user-facing
        # ValueError so it becomes a clean 400, not an unhandled 500.
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [(page.extract_text() or "") for page in reader.pages]
        except Exception as exc:
            raise ValueError("Could not read this PDF — it may be corrupt, encrypted, "
                             "or not a valid PDF file.") from exc
        text = "\n\n".join(pages).strip()
        # H2 — a scanned / image-only PDF parses fine but yields no selectable text. Name the likely
        # cause so the user knows why (and that OCR, which we don't do, is what it would need).
        if not text:
            raise ValueError("This PDF has no selectable text — it looks scanned or image-only, "
                             "so there is nothing to index. Try a text-based PDF.")
        return text
    # .md / .txt (and any allowed text file) — decode as UTF-8, tolerate stray bytes.
    return data.decode("utf-8", errors="replace").strip()


def ingest_bytes(filename: str, data: bytes, embedder, store, config: dict) -> dict:
    """Extract -> chunk -> embed -> persist one uploaded doc into the shared store.

    Overwrites any previous upload of the same name (deterministic ids at the storage layer).
    Returns ``{"source": ..., "chunks": n}``. Raises ``ValueError`` if no text could be extracted.
    """
    text = extract_text(filename, data)
    if not text:
        raise ValueError("Could not extract any text from the uploaded file.")

    ingestion_cfg = config["ingestion"]
    chunk_size = ingestion_cfg["chunk_size"]
    overlap = ingestion_cfg["chunk_overlap"]
    strategy = ingestion_cfg.get("chunking", {}).get("strategy", "fixed")

    chunks = chunk_document(text, chunk_size, overlap, strategy)[:MAX_UPLOAD_CHUNKS]
    source = source_name(filename)
    store.delete_by_source(source)                 # replace-in-place if the same name is re-uploaded
    store.add(source, chunks, embedder.embed(chunks))
    return {"source": source, "chunks": len(chunks)}
