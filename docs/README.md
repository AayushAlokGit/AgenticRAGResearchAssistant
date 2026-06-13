# docs/

The project's own authored documentation — written by hand, **tracked in git**
(unlike the frozen seed corpus under `corpus/`, which is gitignored).

These docs do double duty:

1. **Human-facing** — explain the ideas behind the build (e.g.
   `evals/EVALUATION_PRINCIPLES.md`, the portable model for evaluating compound AI
   systems). Docs are organized by topic into subfolders (e.g. `evals/`).
2. **Corpus material** — per CLAUDE.md, the real corpus is meant to be *the user's own
   documentation*. This folder is the seed of that: as it grows, it becomes content the
   RAG system retrieves over.

## Not yet ingested — and that's deliberate

The eval set (`evals/datasets/seed.yaml`) is written against the **frozen 11-doc
snapshot** in `corpus/`. Adding these docs to the *live* corpus would change the
retrieval space and silently move the eval baseline — a textbook unmeasured change.

So `docs/` is **not** wired into ingestion yet. Folding it into the corpus is a future,
eval-gated step: re-snapshot, extend the eval set to cover the new docs, re-baseline,
*then* keep it. Until then this is documentation only.
