# Design Decisions

A running log of deliberate choices made on this project. One entry per decision.
Keep entries short: what we decided, why, and what we rejected. Newest at the top.

Format per entry:
- **Decision** — the choice, stated plainly.
- **Why** — the reasoning / what problem it solves.
- **Rejected** — alternatives considered and why not.
- **Status** — accepted / superseded (link the superseding entry).

---

## DD-001 — Evals are a first-class, build-early concern

- **Date:** 2026-06-11
- **Decision:** A minimal eval set is built alongside the naive end-to-end version
  (not deferred to "module 2 or later"). Start with ~10 hand-written questions over
  the seed corpus, each tagged with the document/chunk that *should* be retrieved,
  plus a couple of out-of-corpus "I don't know" questions that test correct abstention.
  Retrieval is scored separately from answer quality.
- **Why:**
  - RAG/agent failures are *silent* — output stays fluent and plausible even when wrong,
    so regressions don't surface the way a failing test or crash does. Evals make
    silent failure loud.
  - The project's core loop is change → measure → keep/revert. The "measure" step is
    meaningless without evals, so every from-scratch choice (chunk size, top-k, prompt
    shape) would otherwise be a blind guess.
  - Separating retrieval eval from answer eval localizes failure across the eventual
    multi-stage pipeline (query transform → retrieve → rerank → assemble → generate).
  - Establishes a baseline *before* the fancy machinery, so later we can answer "how
    much did all this actually buy us over the dumb version?"
- **Rejected:**
  - *Full eval suite on day one (Ragas + LLM-judge + nDCG + CI):* too heavy to start;
    add incrementally once the cheap version is already catching regressions.
  - *Evals last / after modules 1–5:* loses the baseline and makes per-stage debugging
    a vibes exercise; retrofitting "correct" onto an over-fit system is expensive.
- **Status:** accepted.
