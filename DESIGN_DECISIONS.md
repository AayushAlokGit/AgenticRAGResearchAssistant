# Design Decisions

Short log of deliberate choices. One line each, newest first.

- **DD-003** (2026-06-11) — Gemini-only for now (`gemini-2.5-flash` for agent/judge); embeddings stay local (`all-MiniLM-L6-v2`) to save quota. Layer kept router-shaped. Supersedes "Claude as primary."
- **DD-002** (2026-06-11) — Installable `src/` package via `pyproject.toml`; deps single-sourced there (no requirements.txt); `config/`/`prompts/`/`corpus/`/`evals/` versioned.
- **DD-001** (2026-06-11) — Evals are built early, alongside the naive baseline (not deferred); retrieval scored separately from answers.
