# Design Decisions

Short log of deliberate choices. One line each, newest first.

- **DD-005** (2026-06-13) — Seed corpus = curated 11-doc subset of `DocumentationRetrievalMCPServer/docs` (copied/frozen into `corpus/`, not referenced live), chosen for cross-document multi-hop + an Azure-vs-ChromaDB distractor tension; remaining ~29 docs reserved to exercise incremental re-indexing later. Seed eval set `evals/datasets/seed.yaml` v1 = 12 questions (9 answerable + 3 abstention); schema adds `match` (any/all) and `should_abstain` for unambiguous recall scoring.
- **DD-004** (2026-06-13) — Groq-only for the LLM (`llama-3.3-70b-versatile` for agent/judge); fast LPU inference suits the eval loop. Embeddings **stay local** (`all-MiniLM-L6-v2`) because Groq has no embeddings endpoint (text-gen + Whisper only) — local is also the low-hardware-friendly choice. Layer stays router-shaped. Supersedes DD-003's Gemini choice.
- **DD-003** (2026-06-11) — Gemini-only for now (`gemini-2.5-flash` for agent/judge); embeddings stay local (`all-MiniLM-L6-v2`) to save quota. Layer kept router-shaped. Supersedes "Claude as primary."
- **DD-002** (2026-06-11) — Installable `src/` package via `pyproject.toml`; deps single-sourced there (no requirements.txt); `config/`/`prompts/`/`corpus/`/`evals/` versioned.
- **DD-001** (2026-06-11) — Evals are built early, alongside the naive baseline (not deferred); retrieval scored separately from answers.
