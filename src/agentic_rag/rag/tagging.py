"""Document-level metadata tagging for metadata-filtered retrieval (DD-036).

WHY doc-level + closed-schema (not per-chunk free-form): metadata filtering only works if the
tags written at INGEST and the tags extracted from the QUERY are drawn from the SAME small set —
otherwise a hard filter silently drops the answer (an empty tag intersection). So we CLASSIFY
into a fixed vocabulary, not GENERATE open-vocabulary tags. And the discriminating axes are
DOCUMENT properties (every chunk of `resolver.md` is the same), so we tag once per doc in a
single batched call (corpus-wide context keeps labels consistent), then stamp the doc's tags
onto all its chunks.

WHY the schema is INDUCED, not hard-coded: a fixed `claims|rag` schema is brittle — it's wrong
for any other corpus. There is no universal tag SET (a legal corpus and a codebase share no
categories); what's universal is the PROCESS. So step 1 derives a closed schema FROM the corpus
(`induce_schema`), step 2 classifies every doc into it. Point this at a new corpus and it
re-derives. Every axis carries an `other` value: a rising `other` rate is the signal the schema
has gone stale and should be re-induced (fits the evolving-corpus goal).

Both the induced SCHEMA and the per-doc TAGS are persisted under ``<vector_store>/`` so the
query-time filter step uses the exact same schema.

SCALING TODO: induction currently feeds ALL docs (fine for a small corpus). For a large corpus,
sample for DIVERSITY first (embed -> cluster -> sample per cluster, or stratify by folder) so
rare categories aren't drowned out; tagging itself already scales (doc-by-doc against the schema).

Run:
    python -m agentic_rag.rag.tagging        # induce schema + tag the whole corpus
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict

from agentic_rag.config import load_config, resolve_path
from agentic_rag.rag.answer import load_prompt
from agentic_rag.rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# How much of each document the LLM sees. The opening of a doc is plenty to place it on coarse
# axes, and it keeps the prompt bounded.
TAG_SNIPPET_CHARS = 500

# Documents classified per LLM call. The OUTPUT (one {axis:value} object per doc) scales with the
# batch size, so a whole-corpus call overruns max_tokens and the JSON truncates — classify in
# small batches so each response stays well under the ceiling (this is also what lets tagging
# scale to a large corpus). Induction stays a single call (its output is schema-sized, not
# corpus-sized).
TAG_BATCH_SIZE = 8

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


def _corpus_blob(snippets: Dict[str, str]) -> str:
    return "\n\n".join(f"### {source}\n{text}" for source, text in snippets.items())


# ───────────────────────────── step 1: induce the schema ─────────────────────────────

def parse_schema(raw: str) -> Dict[str, dict]:
    """Parse the induced schema JSON into {axis: {description, values: {value: meaning}}}.

    Normalizes axis/value names to lowercase tokens (so tagging and query-filtering compare
    against a clean, consistent vocabulary). Drops malformed axes rather than guessing.
    """
    match = _JSON_OBJECT_RE.search(raw or "")
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    schema: Dict[str, dict] = {}
    for axis, spec in data.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("values"), dict):
            continue
        values = {str(v).strip().lower(): str(meaning)
                  for v, meaning in spec["values"].items() if str(v).strip()}
        if values:
            schema[str(axis).strip().lower()] = {
                "description": str(spec.get("description", "")),
                "values": values,
            }
    return schema


def induce_schema(store: ChromaVectorStore, llm, prompt: str) -> Dict[str, dict]:
    """Derive a closed classification schema FROM the corpus (one batched call over all docs)."""
    blob = _corpus_blob(doc_snippets(store))
    completion = llm.complete([
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"DOCUMENTS:\n{blob}"},
    ])
    return parse_schema(completion.text or "")


def render_schema(schema: Dict[str, dict]) -> str:
    """Human-readable schema for injecting into the classify / query-filter prompts."""
    lines = []
    for axis, spec in schema.items():
        lines.append(f"- {axis}: {spec.get('description', '')}")
        for value, meaning in spec["values"].items():
            lines.append(f"    {value} = {meaning}")
    return "\n".join(lines)


# ───────────────────────────── step 2: classify each doc ─────────────────────────────

def coerce_tag(tag: dict, schema: Dict[str, dict]) -> dict:
    """Force a raw tag into the schema: one value per axis, from that axis's allowed set.
    Unknown/missing -> `other` if the axis has it, else the axis's first value. Keeps the index
    vocabulary clean so the query-side filter can rely on exact matches."""
    out = {}
    for axis, spec in schema.items():
        allowed = list(spec["values"].keys())
        value = str(tag.get(axis, "")).strip().lower()
        if value not in allowed:
            value = "other" if "other" in allowed else allowed[0]
        out[axis] = value
    return out


def parse_tags(raw: str, known_sources: set, schema: Dict[str, dict]) -> Dict[str, dict]:
    match = _JSON_OBJECT_RE.search(raw or "")
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    tags = {}
    for source, tag in data.items():
        if source in known_sources and isinstance(tag, dict):
            tags[source] = coerce_tag(tag, schema)
    return tags


def generate_doc_tags(store: ChromaVectorStore, llm, prompt: str,
                      schema: Dict[str, dict]) -> Dict[str, dict]:
    """Classify every document into the (induced) schema, in batches of TAG_BATCH_SIZE docs."""
    snippets = doc_snippets(store)
    sources = list(snippets)
    schema_str = render_schema(schema)
    tags: Dict[str, dict] = {}
    for start in range(0, len(sources), TAG_BATCH_SIZE):
        batch = sources[start:start + TAG_BATCH_SIZE]
        blob = _corpus_blob({s: snippets[s] for s in batch})
        completion = llm.complete([
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"SCHEMA:\n{schema_str}\n\nFILES:\n{blob}"},
        ])
        tags.update(parse_tags(completion.text or "", set(batch), schema))
    for source in set(sources) - set(tags):  # never leave a doc untagged
        logger.warning("tagging: %s got no tag — defaulting to 'other'", source)
        tags[source] = coerce_tag({}, schema)
    return tags


# ───────────────────────────── persistence ─────────────────────────────

def schema_path(config: dict) -> Path:
    return resolve_path(config["vector_store"]["path"]) / "tag_schema.json"


def tags_path(config: dict) -> Path:
    return resolve_path(config["vector_store"]["path"]) / "doc_tags.json"


def load_schema(config: dict) -> Dict[str, dict]:
    path = schema_path(config)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_tags(config: dict) -> Dict[str, dict]:
    path = tags_path(config)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    from agentic_rag.llm.provider import build_llm
    from agentic_rag.logging_setup import configure_run_logging

    configure_run_logging("rag/tagging")
    config = load_config()
    store = ChromaVectorStore(resolve_path(config["vector_store"]["path"]),
                              config["vector_store"]["collection"])
    llm = build_llm(config, role="controller")  # cheap routing-tier model

    # Step 1: induce the schema from the corpus.
    schema = induce_schema(store, llm, load_prompt(config, "induce_schema"))
    if not schema:
        raise SystemExit("schema induction returned nothing parseable — aborting.")
    schema_path(config).write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")

    # Step 2: classify every doc into it.
    tags = generate_doc_tags(store, llm, load_prompt(config, "tag_document"), schema)
    tags_path(config).write_text(json.dumps(tags, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"INDUCED SCHEMA ({len(schema)} axis/axes) -> {schema_path(config)}")
    print(render_schema(schema))
    print(f"\nTAGGED {len(tags)} document(s) -> {tags_path(config)}\n")
    axes = list(schema)
    for source in sorted(tags):
        cells = "  ".join(f"{axis}={tags[source][axis]}" for axis in axes)
        print(f"  {source:50s} {cells}")


if __name__ == "__main__":
    main()
