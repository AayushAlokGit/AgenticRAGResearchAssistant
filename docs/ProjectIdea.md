# Agentic RAG Research Assistant

A learning-focused, end-to-end project to build a production-grade agent that answers complex, multi-hop questions over a large, evolving corpus — and gets smarter across sessions.

## Goal

Build one system that forces me to learn, in depth:
- **Agentic systems** — planning, tool use, multi-hop reasoning, self-correction
- **Context engineering** — curating a finite context window under a token budget
- **Memory management** — working + long-term memory that persists across sessions
- **Harness engineering** — the scaffolding: tools, orchestration loop, retries, guardrails, evals, tracing
- **Production-grade RAG** — hybrid retrieval, reranking, query transformation, incremental indexing, and real evaluation

## What it does

Given a high-level question (e.g. *"Compare the top 5 project-management tools and recommend one for a 10-person startup"*), the agent plans the task, decides when to retrieve, runs agentic multi-hop search over its corpus (+ web), cross-checks sources, and produces a cited answer. It remembers what it learns and my preferences for future sessions. Every step is traced and scored against an eval set.

**Corpus to start with:** _[pick one: my own codebase / a technical-docs set / a research field]_

## Architecture (modules, build in order)

1. **RAG substrate** — ingestion (structure-aware + parent/child chunking, metadata), hybrid retrieval (BM25 + dense embeddings), reranker, query transformation (decomposition, multi-query, HyDE), incremental re-indexing, citations, and a "not enough info" path.
2. **Agentic layer** — agent decides whether/what to retrieve, reformulates queries, does retrieve→reason→retrieve loops, with budgets and clear stop conditions.
3. **Context engineering** — token budgeting, relevance filtering + dedup of retrieved chunks, compression of tool outputs, compaction of old turns, ordering to fight lost-in-the-middle, just-in-time loading.
4. **Memory** — working scratchpad (within task) + long-term episodic / semantic / procedural memory across sessions, with write/read/consolidation/forgetting logic. Memory retrieval is itself a small RAG problem.
5. **Harness** — typed tool schemas with validation, orchestration loop, structured-output parsing, retries/fallbacks, guardrails (budget + step limits, confirmation gates), caching, and cost/latency tracking.

## Evaluation (build early — by module 2, not last)

- **Retrieval eval:** recall@k, MRR, nDCG
- **End-to-end eval:** faithfulness / groundedness, answer relevance, cost, step count
- A small, versioned test set run as regression on every prompt/config change (Ragas + a custom LLM-as-judge).

## Suggested free-tier stack ($0 to build)

- **LLM:** Google AI Studio (Gemini 2.5 Flash) as primary; Groq for speed; **Ollama** local as fallback
- **Embeddings:** Gemini Embedding (or local Qwen3/BGE)
- **Reranker:** local `ms-marco-MiniLM-L-6-v2` (CPU) to start; Cohere Rerank free tier optional
- **Vector DB:** local Chroma/Qdrant, or Pinecone free tier (2GB)
- **Tracing:** Langfuse (self-hosted or free cloud)
- **Pattern:** provider-fallback router (primary free tier → secondary → local Ollama) — also good harness practice

## Build approach

Ship a deliberately naive end-to-end version first (basic chunking + single vector search + simple agent loop + ~10 eval questions). Then attack modules 1→5 in order, **re-running the eval set after each change** so every technique has to earn its place. The change → measure → keep-or-revert loop is the real skill underneath all five topics.

## First milestone

Ingest the corpus, store embeddings in a vector DB, wire a single-tool agent loop that retrieves + answers with citations, set up Langfuse tracing, and create the initial 10-question eval set with a baseline score to beat.

---

### Notes for Claude Code
- Keep prompts, configs, and corpus versioned so single changes can be A/B tested.
- Add the provider-fallback layer and tracing early — later modules depend on being able to measure.