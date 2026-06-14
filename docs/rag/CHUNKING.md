# Chunking Strategies

> A portable design surface. Chunking decisions apply to **any system that retrieves spans
> of text to ground a model** — RAG, long-document QA, semantic search, code search. This
> project's fixed-size baseline (DD-006) and its planned upgrades are the worked example in
> the last section. See also `VECTOR_DB_INTERNALS.md` (what the chunk vector is),
> `RERANKING.md` (the stage that reorders chunks), and `../evals/EVALUATION_PRINCIPLES.md`
> (chunking is eval-gated like everything else).

Chunking looks trivial — "just split the text" — and is in fact one of the highest-leverage
and most uncorrectable decisions in the pipeline. **The chunk is the atomic unit every later
stage operates on** (embedding, BM25, reranking, generation). A fact split across a bad
boundary can never be reassembled by a better embedder, reranker, or prompt — so chunking
mistakes cap the ceiling of the whole system. That's why it's usually the *fix-first* lever.

## 1. Why chunking exists (the forcing functions)

You don't chunk for style — four hard constraints force it:

1. **Embedder input limit** — embedding models have a max input (e.g. MiniLM ~256–512
   tokens); feed more and it's silently truncated.
2. **Retrieval precision** — a single vector for a whole document *averages away* specifics;
   a query about one sentence drowns in the rest. Smaller units match more sharply.
3. **Generation budget** — you can't stuff whole documents into the LLM prompt.
4. **Signal-to-noise** — a focused chunk gives the generator the relevant passage without
   surrounding distractor text.

## 2. The central tension (every strategy navigates this)

> **Small chunks → precise retrieval, fragmented context** (you find the exact sentence but
> lose the surrounding info that makes it usable).
> **Large chunks → rich context, diluted embeddings & noisier retrieval** (the vector is a
> blurry average; matching gets less discriminating).

Every technique below is a different answer to *"how do I get precision AND context at once?"*

## 3. Format-specific vs format-agnostic structure (the architectural decision)

Before the strategies, the decision that shapes which ones you can use. "Structure-aware
chunking" bundles two very different things:

- **Format-*specific* structure** — markdown headings, HTML tags, code ASTs, PDF layout
  blocks. Needs a *different parser per format*, so it does **not** generalize to arbitrary
  uploads (plain text, scanned PDFs, docx, …). Powerful when you control the corpus format;
  a scaling liability when you don't.
- **Format-*agnostic* structure** — paragraphs (blank lines), sentences, words. These exist
  in **any plain text**, regardless of source. Splitting on these is not format-coupled.

**The key insight: separate extraction from chunking.** Format-specificity is unavoidable —
you *must* turn a PDF/docx/HTML into text somehow — but that belongs in an **extraction
layer**, not the chunker:

```
[arbitrary file] ─► EXTRACTION (format-SPECIFIC, unavoidable) ─► normalized plain text ─► CHUNKING (format-AGNOSTIC) ─► chunks
 PDF/docx/html/txt   pypdf, Unstructured.io, OCR, …               paragraphs/sentences      recursive / semantic
```

Quarantine all format knowledge in extraction, and the chunker sees only plain text — so one
chunking strategy serves every document type. A reinforcing reality: PDF/OCR extraction often
*destroys* layout structure anyway, so for arbitrary uploads you frequently can't rely on
clean structure even if you wanted to — which makes format-agnostic chunking the *robust*
default, not a compromise. **"Structure-independent" should mean "no format-specific
parsing," NOT "blind fixed-size cutting"** — you still respect the universal paragraph/
sentence boundaries that survive in any text.

## 4. The strategies (naive → sophisticated)

### Family A — Size-based (structure-blind)

- **A1. Fixed-size** — cut every N characters/tokens. Trivial, predictable, fast; but
  structure-blind (cuts mid-sentence, mid-table). **Character vs token** matters: embedders
  count *tokens*, so token-based sizing honors the real limit; character-based gives uneven
  token counts.
- **A2. Sliding window / overlap** — adjacent chunks share M tokens, so a fact on a boundary
  still appears whole in one window. Cheap boundary insurance; cost is duplication.

### Family B — Structure-aware (respect the document's own boundaries)

- **B1. Recursive splitting** *(the industry default for arbitrary ingestion)* — split on a
  universal separator **hierarchy**: paragraph (`\n\n`) → line (`\n`) → sentence → word →
  char, taking the largest natural break that fits the size cap. **Format-AGNOSTIC** (works
  on any plain text) yet boundary-respecting — the sweet spot between A1 and format coupling.
- **B2. Format-/document-aware** — split on markdown headings, HTML tags, code functions
  (AST), PDF blocks. Keeps tables/lists/sections intact (best structure fidelity) but is
  **format-SPECIFIC** (a parser per format) and yields variable sizes. Great for a
  controlled corpus; doesn't scale to "ingest anything."

### Family C — Meaning-based (format-agnostic)

- **C1. Semantic chunking** — embed each sentence, cut where consecutive-sentence similarity
  drops (a topic shift). Boundaries follow meaning, not layout. Expensive (embed every
  sentence) and needs a breakpoint threshold; good for prose without markup.
- **C2. Proposition / atomic chunking** — an LLM rewrites the doc into standalone atomic
  facts. Maximally precise retrieval; costly (LLM per doc), can strip context or distort.

### Family D — Decouple retrieval unit ≠ generation unit (the most powerful idea)

Attacks the §2 tension head-on: retrieve with small units, generate with large ones. Both are
**format-agnostic** (about unit decoupling, not document format).

- **D1. Parent–child / small-to-big** — index small **child** chunks (precise retrieval); on
  a hit, hand the larger **parent** section to the LLM (rich context). Best-of-both; the
  highest-leverage fix for multi-fact / coverage gaps. Cost: parent↔child bookkeeping.
- **D2. Sentence-window** — retrieve at sentence granularity, expand to neighboring sentences
  before generation. Pinpoint retrieval + local context; less suited to facts spread far apart.

### Family E — Context-preserving embeddings (newer; fix "chunk embedded in isolation")

- **E1. Contextual Retrieval** *(Anthropic)* — prepend a short LLM-generated context blurb to
  each chunk before embedding ("from the ChromaDB architecture doc, search-types section…"),
  so a lonely chunk knows its place. Strong accuracy gains; an LLM call per chunk at index
  time (cacheable, one-time).
- **E2. Late chunking** *(Jina)* — embed the *whole document* with a long-context embedder,
  then pool token embeddings into chunks *after*, so each chunk vector carries whole-doc
  context. Needs a long-context embedder.
- **E3. Metadata enrichment** *(cross-cutting)* — attach source, heading-path, page to each
  chunk for filtering and for telling the generator where a fact came from.

## 5. The dials you tune

| Dial | Effect |
|---|---|
| **Chunk size** | the master precision↔context knob |
| **Overlap** | boundary insurance vs duplication |
| **Granularity** | sentence / paragraph / section |
| **Separator hierarchy** | where natural breaks are allowed (recursive) |
| **Retrieval unit vs generation unit** | decouple them (Family D) |
| **top_k / candidate_k** | how many chunks reach the generator |

**Coupling to remember:** chunk size and `top_k` move together — halve the chunk size and you
usually need to *raise* `top_k` to keep the same total context reaching the LLM.

## 6. How to choose (decision factors)

There is **no universal best** — it depends on:

- **Corpus control** — fixed, known format (e.g. your own markdown) can afford B2; arbitrary
  uploads demand format-agnostic (B1/C1/D1) behind an extraction layer.
- **Document type** — structured docs → B2 (if you control format) or B1; flowing prose → C1
  or B1.
- **Query type** — "list all / enumeration" questions need whole enumerations in one unit
  (favor larger / parent-child); pinpoint fact lookups favor small precise chunks.
- **Embedder context window** — caps max chunk size.
- **Generation budget** — caps chunk_size × top_k.

## 7. Transferable principles

1. **Chunk size trades retrieval precision against context completeness** — the dial under
   everything.
2. **Respect natural boundaries** — never split a coherent unit if you can avoid it.
3. **Separate extraction (format-specific) from chunking (format-agnostic)** — so one chunker
   serves all document types and format knowledge stays quarantined.
4. **Decouple the retrieval unit from the generation unit** when you can (small-to-big) — the
   highest-leverage structural idea.
5. **Preserve context** — via overlap, parent expansion, contextual headers, or late chunking.
6. **Match chunking to both document type AND query type.**
7. **It's eval-gated** — no closed form; change it and re-measure. It's *fix-first* only
   because its mistakes are the ones nothing downstream can repair.

## 8. In this project (the worked instance)

We are at **A1 + A2**: fixed-size 800-char windows / 100-char overlap with a whitespace snap
(DD-006) — deliberately the naive baseline. The eval exposed its cost: even with
document-level recall@5 = 1.0 after reranking, several answers came back PARTIAL because a
single fact was **fragmented across chunk boundaries** — the q25 per-search-type backing
table and the q27 embedding-model list were split, so the top-5 carried part of the
enumeration, not all of it (passage/fact-level coverage < document-level recall).

**Design stance for this build: stay format-agnostic.** The corpus will eventually be the
user's own arbitrary uploads (text, PDFs, …), so format-specific structure parsing (B2) is a
scaling liability. The chosen upgrade path:

- **B1 (recursive, token-based) + A2 (overlap)** — the format-independent baseline upgrade
  from A1; respects paragraph/sentence boundaries that exist in any extracted text.
- **D1 (parent–child)** — next lever if multi-fact coverage still lags (q22-style): retrieve
  a precise child, hand the surrounding parent region to the generator.
- All format-specificity is to live in a future **extraction layer**, never in the chunker.

Honest trade-off accepted: format-agnostic recursive splitting respects paragraph/sentence
breaks but won't *guarantee* an oversized table or long list stays whole the way a markdown
splitter would. Overlap and especially parent-child buy back most of that integrity without
the format coupling — and like every change here, it only ships if it beats the current
chunker on the eval set.
