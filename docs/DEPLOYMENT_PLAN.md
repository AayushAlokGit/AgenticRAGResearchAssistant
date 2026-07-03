# Deployment Plan — Agentic RAG → live public demo

Goal: a hosted, public, **$0** demo that shows the engineering (live agent reasoning, grounded
citations, cost transparency) — not just another chatbot. Ship phase by phase; each phase is
independently shippable.

## Architecture

```
Next.js frontend (Vercel, free)  ──POST /ask (SSE stream)──▶  FastAPI backend (HF Spaces, free)
                                 ◀──── trajectory events ────
                                                                  │ SQL / pgvector
                                                    Postgres + pgvector (Neon, free)
```

- **Frontend** (Vercel): chat + example questions, live trajectory view, clickable citations,
  cost/latency meter, naive⇄agentic toggle, methodology page. Never sees API keys.
- **Backend** (HF Spaces Docker): LangGraph `graph.stream()`, Gemini embeddings (cloud), local
  cross-encoder reranker + BM25 + parent-expansion, provider fallback Gemini→Groq, durable daily
  token cap.
- **DB** (Neon Postgres+pgvector): the single durable store — see below.

## Storage (the key decision)

The host is ephemeral, so **nothing important lives on its filesystem** — all mutable state goes to
Postgres and is rehydrated at boot.

| State | Home |
|---|---|
| Corpus chunks + embeddings | Postgres (pgvector) |
| Documents + free-form tags | Postgres (JSONB) — was `doc_tags.json` on disk |
| Episodic memory | Postgres (table exists; OFF for public MVP) |
| Daily token-spend counter | Postgres (`daily_spend`) — durable, can't be restart-farmed |
| BM25 index | In-process, rebuilt at boot from PG text |
| Reranker model (92 MB) | Baked into image (+ `HF_HUB_OFFLINE`) |

- **Seed once, don't re-embed:** a migration script copies the existing Chroma vectors + text + tags
  into pgvector — **zero Gemini spend** to fill the DB.
- **`build_store(config)` factory:** `chroma` for local dev, `pgvector` for prod. Retriever/reranker
  code above the store is untouched.
- **Local retrieval compute retained:** reranker, BM25, chunking, parent-expansion all stay local;
  only the *storage substrate* and the embedder are cloud.

## Cost control (minimal for launch)

- **Durable daily token cap** — `daily_spend` row in PG; once the day's Gemini budget is hit, `/ask`
  returns "demo is resting, back tomorrow". Survives restarts → **this is what guarantees $0.**
- **Per-question spend-cap** (already built, DD-052 — wire into the graph).
- **Input caps** — max question length, reject empty/gibberish before any LLM call.
- **Deferred** (revisit if abuse shows up): per-IP rate limit, concurrency cap, Turnstile, trace
  sampling. Also: **do not attach billing to the Google key** — free-tier quota only.

## Work by area

**Backend (`server/`)**
- `app.py` — `POST /ask` (SSE), `GET /sources`, `GET /health`, `GET /config`.
- `stream.py` — wrap `graph.stream(..., stream_mode="updates")`, map each node to a typed SSE event:
  `think` (controller reasoning + actions), `search`/`evidence` (queries + chunks + new-vs-redundant),
  `answer` (cited text + grounded flag), `done` (rounds/tokens/latency/cost).
- `schemas.py` — Pydantic event models. `deps.py` — build graph once at startup (connect PG, rebuild
  BM25, load reranker), reuse per request.

**Existing-code changes**
- `rag/vector_store.py` — add `build_store` + pgvector backend.
- `rag/tagging.py` + `ingest.py` — tags to PG instead of `doc_tags.json`.
- `agent/graph.py` — wire the spend-cap into the graph path.
- `rag/rerank.py` — bake `HF_HUB_OFFLINE=1` so the reranker loads from cache, never the network
  (today's eval crash). **Required correctness fix.**
- `config/` — prod profile (`DATABASE_URL`, `store.provider: pgvector`, `memory.enabled: false`).
- New deps: `fastapi`, `uvicorn`, `sse-starlette`, `psycopg` + `pgvector`.

**Frontend (Next.js + Tailwind + shadcn/ui on Vercel)**
- Chat + example-question chips.
- **Live trajectory timeline** — each round as a card (reasoning, searches, chunks w/ source+score,
  "redundant round" marker). The differentiator: shows it's an agent, not a lookup.
- Streaming answer with **clickable citation chips** → slide-over showing the exact chunk the
  generator saw (proves grounded).
- **Cost/latency meter** (rounds/tokens/¢). Honest **"not enough info"** path.
- **Naive⇄agentic toggle** (same Q, single-shot vs full loop, side by side).
- **Methodology page** — architecture, eval discipline, links to `DESIGN_DECISIONS.md` / docs.
- Polish: dark mode, mobile, error toasts, About + LinkedIn.

**Deploy/ops**
- Monorepo: `frontend/`, `src/` (package), `server/`. Root README = public landing.
- Neon: enable `vector`, run schema + seed. Vercel: root `frontend/`, `NEXT_PUBLIC_BACKEND_URL`.
- HF Spaces: Docker, secrets `GOOGLE_API_KEY` / `GROQ_API_KEY` / `LANGSMITH_*` / `DATABASE_URL`.
- CORS locked to Vercel origins. `.env` gitignored (+ `.env.example`). `/health` checks PG.

**Presentation**
- README landing (pitch, live link, trajectory GIF, diagram, "what I learned").
- Recorded GIF of a multi-hop question for the LinkedIn post.

## Phases (ship in order)

1. **Backend + Postgres** — provision Neon, `build_store` pgvector + seed script, tags→PG, `server/`
   FastAPI + SSE, wire spend-cap, HF-offline reranker fix, prod config. Verify locally (curl).
2. **Durable daily cap** — persisted kill-switch + per-question cap + input caps. *Gate before public URL.*
3. **Package + deploy backend** — Dockerfile, bake reranker, HF Spaces, secrets, `/health` green.
4. **Frontend MVP** — chat + trajectory + citations + cost meter. **First shareable demo.**
5. **Pop features** — naive⇄agentic toggle, methodology page, eval scoreboard, polish, GIF.
6. **Stretch** — token-streaming answer, per-session memory, bring-your-own-corpus (needs the
   deferred abuse caps), shareable permalinks.

## Decisions

- Storage = Postgres+pgvector, single durable store.
- DB = **Neon** (auto-wakes from idle; Supabase free pauses after ~7 days needing manual restore).
- BM25 = **in-process** (corpus is small; retains local dep, eval semantics unchanged).
- Backend host = **HF Spaces** (16 GB free RAM fits torch; Render's 512 MB would OOM).
- Abuse hardening = minimal (durable daily cap only); heavier layers deferred.
- Doc tags = in PG, not on the ephemeral filesystem.
- **Open:** launch scope — ship Phases 1–4 + methodology first, layer 5–6 after? *(Rec: yes.)*

## Provenance
- Embeddings→Gemini + free-form tagging: commit `ae3cf65`, re-baselined quality-neutral in **DD-055**.
- Graph streams via `graph.stream()`, parity proven **DD-053**; guardrails built **DD-052**.
- Storage spine set by the 2026-07-03 plan review.
