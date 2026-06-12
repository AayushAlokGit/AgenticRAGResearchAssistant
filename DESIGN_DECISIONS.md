# Design Decisions

A running log of deliberate choices made on this project. One entry per decision.
Keep entries short: what we decided, why, and what we rejected. Newest at the top.

Format per entry:
- **Decision** — the choice, stated plainly.
- **Why** — the reasoning / what problem it solves.
- **Rejected** — alternatives considered and why not.
- **Status** — accepted / superseded (link the superseding entry).

---

## DD-003 — Gemini-only for now; embeddings stay local

- **Date:** 2026-06-11
- **Decision:** Use Google Gemini (`gemini-2.5-flash`) as the sole LLM for the agent loop
  and LLM-as-judge, because it is the only provider key currently available. **Embeddings
  stay local** (`all-MiniLM-L6-v2`), not Gemini. The LLM layer is still built
  router-shaped (`provider_order` is a list) so adding tiers later is config-only.
- **Why:**
  - Gemini is the only key on hand; Claude-as-primary (previously assumed) isn't usable yet.
  - Local embeddings need no key/quota and work offline, so they fully satisfy the
    "only a Gemini key" constraint while **preserving** scarce Gemini quota for agent
    reasoning. Ingestion embeds every chunk and the corpus is re-indexed repeatedly while
    tuning against evals — embeddings are where free-tier quota burns fastest.
  - Keeping the layer router-shaped means the eventual provider-fallback router (still a
    learning goal) is a config change, not a rewrite, once more keys exist.
- **Rejected:**
  - *Gemini embeddings too ("Gemini everywhere" literally):* spends limited quota on the
    one workload that doesn't need a cloud model; one-line flip in config if desired later.
  - *Claude/Anthropic as primary (prior intent):* no key available right now.
- **Status:** accepted (revisit when additional provider keys exist).
- **Supersedes:** the "Claude is the intended primary tier" note previously in CLAUDE.md.

---

## DD-002 — Installable `src/` package; config / prompts / corpus / evals versioned

- **Date:** 2026-06-11
- **Decision:** Use a `src/` layout with an installable package (`agentic_rag`) via
  `pyproject.toml` + editable install (`pip install -e .`) — no `sys.path` manipulation.
  Dependencies live in `pyproject.toml` as the single source of truth (no
  `requirements.txt`). Top-level directories map to project concerns: `config/`,
  `prompts/`, `corpus/`, `evals/`, `src/agentic_rag/{llm,rag,agent,context,memory,harness}`,
  `tests/`. The package subdirs mirror the five modules so the structure is self-documenting.
- **Why:**
  - Installable packages give imports that work regardless of the working directory —
    avoids the `sys.path.insert(...)` hack the prior MCP server relied on.
  - One dependency source (pyproject) can't drift the way pyproject + requirements.txt can.
  - Directory taxonomy that mirrors the five modules keeps the repo legible as it grows.
  - Versioning `config/`, `prompts/`, `corpus/` is what makes single changes A/B-testable
    against the eval set (DD-001 + ProjectIdea.md note #2).
  - `tests/` (code correctness) is kept separate from `evals/` (system quality) on purpose.
- **Rejected:**
  - *Flat layout + `sys.path` insertion:* couples imports to cwd, not installable.
  - *`requirements.txt` as the dependency source:* duplicate source of truth alongside
    `pyproject.toml`.
- **Status:** accepted.

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
