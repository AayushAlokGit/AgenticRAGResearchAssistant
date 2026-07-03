"""Document-level FREE-FORM metadata tagging (supersedes the closed-schema induction of DD-036).

Each document gets ONE independent LLM call that extracts free-form metadata (doc_type, topics,
entities) from its opening excerpt. There is NO global schema induction — so tagging is
INCREMENTAL and scales to arbitrary / user-uploaded corpora: a new doc is tagged on its own in
O(1), with no whole-corpus pass. (The old closed-schema induction fed ALL doc snippets into a
single call — an O(N_docs) prompt that hit the context ceiling and degraded on large corpora.)

An OPEN vocabulary is safe here because retrieval uses the tags SOFTLY: `list_sources` surfaces
each doc's tags to the agent, which JUDGES relevance (threshold-free — the DD-045 pattern). There
is no hard metadata WHERE-filter, so the query vocabulary need not match the tag vocabulary
exactly (a query for "vector store" still benefits from a doc tagged "vector database").

Persisted by the active store backend (``store.save_tags`` / ``store.load_tags``) as
{source: {doc_type, topics: [...], entities: [...]}} — a JSON file for local Chroma, a table for
pgvector.

Run:
    python -m agentic_rag.rag.tagging          # (re)tag the whole corpus, one call per doc
"""
from __future__ import annotations

import json
import logging
import re
import sys
from typing import Dict, List

from agentic_rag.config import load_config
from agentic_rag.rag.answer import load_prompt
from agentic_rag.rag.vector_store import ChromaVectorStore, build_store

logger = logging.getLogger(__name__)

TAG_SNIPPET_CHARS = 800   # opening excerpt the model sees per doc (enough to place it, bounded prompt)
_MAX_LIST = 6             # cap topics/entities so the surfaced tag map stays compact
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def doc_snippets(store: ChromaVectorStore) -> Dict[str, str]:
    """One opening excerpt per source document, rebuilt from the stored chunks."""
    by_source: Dict[str, Dict[int, str]] = {}
    for chunk in store.all_chunks():
        by_source.setdefault(chunk["source"], {})[chunk["chunk_index"]] = chunk["text"]
    snippets = {}
    for source, chunks in by_source.items():
        head = chunks.get(0) or next(iter(chunks.values()), "")
        snippets[source] = head[:TAG_SNIPPET_CHARS]
    return snippets


def _clean_list(value, limit: int = _MAX_LIST) -> List[str]:
    """Normalize a raw topics/entities value into a short, deduped list of lowercase phrases."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        phrase = str(item).strip().lower()
        if phrase and phrase not in out:
            out.append(phrase)
    return out[:limit]


def parse_doc_tag(raw: str) -> dict:
    """Parse one document's free-form tag JSON into {doc_type, topics, entities} ({} if unparseable)."""
    match = _JSON_OBJECT_RE.search(raw or "")
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        "doc_type": str(data.get("doc_type", "")).strip().lower(),
        "topics": _clean_list(data.get("topics")),
        "entities": _clean_list(data.get("entities")),
    }


def extract_tags(snippet: str, llm, prompt: str) -> dict:
    """One independent LLM call -> free-form tags for a SINGLE document (the incremental unit —
    this is what an ingest hook / a per-upload call would invoke for one new doc)."""
    completion = llm.complete([
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"DOCUMENT EXCERPT:\n{snippet}"},
    ])
    return parse_doc_tag(completion.text or "")


def tag_corpus(store: ChromaVectorStore, llm, prompt: str) -> Dict[str, dict]:
    """Tag every document, one independent call each — per-doc, so it scales and is order-free."""
    snippets = doc_snippets(store)
    tags: Dict[str, dict] = {}
    for i, (source, snippet) in enumerate(snippets.items(), start=1):
        tags[source] = extract_tags(snippet, llm, prompt)
        logger.info("tagging %d/%d: %s -> type=%s", i, len(snippets), source,
                    tags[source].get("doc_type", "?"))
    return tags


# ───────────────────────────── persistence ─────────────────────────────
# Tag persistence lives ON the store now (store.load_tags / store.save_tags), so it follows the
# active backend: a JSON file for local Chroma, a table for pgvector. See rag/vector_store.py and
# rag/pgvector_store.py.


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    from agentic_rag.llm.provider import build_llm
    from agentic_rag.logging_setup import configure_run_logging

    configure_run_logging("rag/tagging")
    config = load_config()
    store = build_store(config)
    llm = build_llm(config, role="controller")   # cheap routing-tier model
    prompt = load_prompt(config, "tag_document")

    tags = tag_corpus(store, llm, prompt)
    store.save_tags(tags)
    print(f"[tagging] tagged {len(tags)} document(s) -> {type(store).__name__}")


if __name__ == "__main__":
    main()
