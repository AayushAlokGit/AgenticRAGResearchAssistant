# Agentic RAG Research Assistant

A from-scratch, learning-focused build of a production-grade agentic RAG system.
The goal is understanding, not just shipping — see `CLAUDE.md` for the working
agreement and `ProjectIdea.md` for the full spec.

## Orientation

| File | Purpose |
| --- | --- |
| `ProjectIdea.md` | Full project spec — the 5 modules, eval-driven build approach. |
| `CLAUDE.md` | Working agreement; encodes the learning-first goal. |
| `DESIGN_DECISIONS.md` | ADR-style log of deliberate choices. Read this to know *why* things are shaped the way they are. |

## Layout

```
config/      # versioned configuration (naive-baseline knobs, tuned against evals)
prompts/     # versioned prompts (A/B tested against evals)
corpus/      # seed corpus until the user's own documentation exists
evals/       # eval datasets + harness — a first-class, build-early concern (DD-001)
src/agentic_rag/
  llm/       # provider interface + fallback router (harness)
  rag/       # RAG substrate: ingestion, chunking, retrieval (module 1)
  agent/     # agentic loop (module 2)
  context/   # context engineering (module 3)
  memory/    # working + long-term memory (module 4)
  harness/   # orchestration loop, tools, guardrails, tracing (module 5)
tests/
```

## Setup (when implementation starts)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .          # editable install; clean imports, no sys.path hacks
copy .env.example .env    # then fill in provider keys
```

Status: **scaffolding only — no implementation yet.**
