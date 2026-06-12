# Agentic RAG Research Assistant

## What this project is

A from-scratch, learning-focused build of a production-grade agentic RAG system that
answers complex, multi-hop questions over an evolving corpus and gets smarter across
sessions. Full project spec is in `ProjectIdea.md`.

The corpus is **the user's own documentation** (not yet written). Until it exists, we
seed with a small real document set (e.g. the `DocumentationRetrievalMCPServer/docs`
folder) so the agent loop and evals have something to retrieve against.

## The actual goal (read this before helping)

**The point is for the user to LEARN and become an engineer who can push agentic
systems to production — not to ship something agentic by vibes.** Optimize every
interaction for the user's understanding, not for task completion speed.

Concretely, this changes how you should work here:

- **Teach, don't just do.** Explain the *why* behind a technique before/while
  implementing it. Name the trade-off being made. If there's a standard term for what
  we're doing (HyDE, reranking, parent/child chunking, context compaction), use it and
  define it.
- **Prefer hand-rolled over framework magic.** The user is deliberately building the
  agent loop, context assembly, and harness by hand *before* reaching for LangGraph or
  similar, to understand what those frameworks abstract. Do not introduce a heavy
  framework to "save time" unless asked — it defeats the purpose.
- **Make the user make the decisions.** When there's a real design fork, surface it
  with the trade-offs and a recommendation, rather than silently picking. Record the
  outcome (see Design decisions below).
- **No unmeasured changes.** This system fails silently — fluent wrong answers look
  like fluent right answers. Treat the eval set as the source of truth. Tie changes to
  a measurable effect where one exists; if we can't measure it yet, say so explicitly.
- **Call out when something is a learning shortcut vs production-grade.** Be honest
  about what's a naive baseline and what would need hardening for real use.

## Build approach

Ship a deliberately naive end-to-end version first, then attack the five modules in
order, **re-running the eval set after each change so every technique earns its place.**
The change → measure → keep-or-revert loop is the real skill being practiced.

Five modules (see `ProjectIdea.md` for detail), built in order:
1. RAG substrate — ingestion/chunking, hybrid retrieval, reranking, query transformation, incremental re-indexing, citations, "not enough info" path.
2. Agentic layer — retrieve→reason→retrieve loops with budgets and stop conditions.
3. Context engineering — token budgeting, dedup, compression, compaction, ordering.
4. Memory — working scratchpad + long-term memory across sessions.
5. Harness — typed tool schemas, orchestration loop, retries/fallbacks, guardrails, caching, cost/latency tracking.

### Rough starting sequence
1. Repo skeleton — versioned prompts, configs, and corpus from the start.
2. Thin single-provider LLM interface, shaped so the provider-fallback router slots in later.
3. Naive RAG substrate — basic chunk + single vector search over the seed corpus.
4. Naive single-tool agent loop — retrieve → answer with citations.
5. Minimal 10-question eval set + baseline score to beat.
6. Then iterate: expand the router, then work modules 1→5 behind the eval gate.

## Conventions

- **Design decisions** are tracked in `DESIGN_DECISIONS.md` — short ADR-style entries.
  When a real architectural fork is resolved, add an entry there.
- **LLM layer:** a provider-fallback router is a learning goal (primary → secondary →
  local Ollama). Build it *router-shaped but single-tier first*; expand once the naive
  loop + evals exist, so the router can be proven not to degrade quality. Claude is the
  intended primary tier for agent/judge roles (strongest tool-use + JSON reliability).
- **Versioning:** keep prompts, configs, and corpus versioned so single changes can be
  A/B tested against the eval set.

## Related code (not a dependency)

`C:\Users\aayus\Desktop\DocumentationRetrievalMCPServer` — the user's existing
ChromaDB + local-embeddings RAG MCP server. This project is built **from scratch** and
does NOT depend on it, but its patterns (structure-aware chunking, idempotent
processing, a vector-search interface abstraction) are worth learning from and
re-deriving with the upgrades this project's plan calls for.

## Environment

- Platform: Windows. Shell is PowerShell (use PowerShell syntax).
- Not yet a git repository.
