# Deployment Plan — Agentic RAG Research Assistant → live public demo

Status: **DRAFT for review (rev 2 — storage spine set to Postgres+pgvector per review).** Turns the
learning project into a hosted, public, $0 demo that shows the engineering (live agent reasoning,
grounded citations, cost transparency) — not just another chatbot. Work through it phase by phase;
each phase is independently shippable.

---

## 0. Guiding constraints (these shape every decision)

1. **Public = untrusted traffic on YOUR API keys.** The dominant cost risk is a stranger or bot
   running up your Gemini quota. For launch we keep the abuse defense **deliberately minimal** — a
   single **daily token cap** — but that cap must be **durable** (see §2/§B), because the host is
   ephemeral and an in-memory counter resets on every restart.
2. **Everything free ($0), guaranteed — not just "cheap".** The guarantee comes from *cap-to-degrade,
   not cap-to-charge*: **do not attach billing to the Google key** (free-tier quota only). Under
   heavy load the demo *stops answering* ("resting, back tomorrow") — it never bills you. The DB is a
   **free-tier Postgres** (Neon/Supabase). No paid infrastructure anywhere.
3. **Retain local retrieval compute.** The **cross-encoder reranker** (torch), **BM25**, **chunking**,
   and **parent-expansion** stay local/in-process, baked into the backend image. Only two things go
   cloud: the **embedder** (Gemini, already done, behind `build_embedder`) and the **storage
   substrate** (Postgres+pgvector, behind a new `build_store` — see §2). Local dev still runs the
   local embedder + Chroma via those factories, so nothing local is *lost*, only made swappable.
4. **The host is ephemeral — so nothing important may live on its filesystem.** All mutable state
   (corpus vectors, doc tags, episodic memory, spend counter) lives in Postgres and is rehydrated at
   boot. This is the review's key decision and it drives §2.
5. **The demo's job is to SHOW the engineering.** The differentiator vs. a plain chatbot is that
   people can *watch the agent reason, retrieve, cite, and cost money in real time* — and read the
   methodology. The UI exposes the machinery (§D).

---

## 1. Target architecture

```
┌─────────────────────────┐         HTTPS / SSE          ┌──────────────────────────────┐
│  Next.js frontend        │  ───── POST /ask (stream) ──▶ │  FastAPI backend             │
│  (Vercel Hobby, free)    │  ◀──── SSE trajectory ─────   │  (HF Spaces free CPU, Docker)│
│                          │                               │                              │
│  • chat + example Qs     │  ───── GET /sources ───────▶  │  • LangGraph graph.stream()  │
│  • live trajectory view  │                               │  • Gemini embeddings (cloud) │
│  • clickable citations   │                               │  • local cross-encoder (CPU) │
│  • cost/latency meter     │                               │  • local BM25 + parent-expand│
│  • naive⇄agentic toggle   │                               │  • daily token cap (durable) │
│  • methodology page       │                               │  • provider fallback G→Groq  │
└─────────────────────────┘                               └───────────────┬──────────────┘
      env: NEXT_PUBLIC_BACKEND_URL                                         │ SQL / pgvector
                                                            ┌──────────────▼──────────────┐
   backend secrets (HF Space):                              │  Postgres + pgvector         │
   GOOGLE_API_KEY, GROQ_API_KEY,                            │  (Neon/Supabase free tier)   │
   LANGSMITH_*, DATABASE_URL                                │  • chunks + embeddings        │
                                                            │  • documents + tags (JSONB)   │
                                                            │  • episodic memory (v2)       │
                                                            │  • daily_spend counter        │
                                                            └──────────────────────────────┘
```

Two deploy targets, one monorepo, one managed DB. Frontend never sees API keys or the DB; all model
and storage access goes through the backend.

---

## 2. Storage & persistence — Postgres + pgvector as the single system-of-record

**General principle (transfers beyond RAG): on an ephemeral host, split state into "must survive a
restart" vs "cheap to rebuild at boot."** The first group needs a durable store; the second you
recompute or reload in-process. Here that split is:

| State | Must survive restart? | Where it lives | Notes |
|---|---|---|---|
| Corpus chunks + embeddings | **Yes** | **Postgres (pgvector)** | dense search via `<=>` cosine |
| Documents + free-form tags | **Yes** | **Postgres** (JSONB column) | was `doc_tags.json` on disk → moved to DB (review pt. 3) |
| Episodic memory | **Yes (when on)** | **Postgres** | table exists; feature OFF for public MVP (shared-store leak) |
| Daily token-spend counter | **Yes** | **Postgres** (`daily_spend`) | durable cap survives restart (review pt. 2) |
| BM25 index | No — rebuild at boot | **In-process** (rank_bm25) | loaded from PG chunk text on startup — retains the local dep |
| Reranker model (92 MB) | No — read-only artifact | **Baked into image** (+ `HF_HUB_OFFLINE`) | a model, not state; no reason to DB it |
| LLM cache (sqlite) | No — disposable | Ephemeral (re-warms) | fine to lose |

**Why Postgres+pgvector and not "bake a read-only Chroma store into the image"** (the rev-1 plan):
the review chose durability + consolidation on purpose. One managed DB gives every writable feature
(tags updated incrementally, episodic memory, a spend cap that can't be restart-farmed, and later
bring-your-own-corpus) a home that outlives the ephemeral host — instead of each needing its own
bolt-on. Trade-off paid: a network hop per vector query (adds latency vs in-process Chroma) and a
`build_store` adapter to write. Worth it here because *persistence* is the driving requirement.

**We do NOT re-embed to populate Postgres.** The embeddings already exist in the local Chroma store.
A one-time **seed/migration script copies vectors + text + metadata + tags straight into pgvector** —
no Gemini tokens spent to fill the DB. (General lesson: migrate computed artifacts, don't recompute.)

**`build_store(config)` factory (mirrors `build_embedder`).** New in Phase 1, not deferred:
- `provider: chroma` → local Chroma (dev / local evals — retains the local path).
- `provider: pgvector` → Postgres for prod.
Same `add / query / list_sources / delete` interface the retriever already calls, so the retriever,
reranker, and parent-expansion code above the store are untouched.

**Free-tier Postgres — DECIDED: Neon.** The deciding factor is *idle behavior* for an intermittent
public demo: **Supabase free pauses the whole project after ~7 days idle and needs a manual un-pause**
(a visitor could hit a dead DB until you notice) — whereas **Neon scales to zero but auto-wakes on the
next connection** (sub-second to a few seconds), so idle is truly $0 *and* self-healing. Neon is also
pure Postgres with native pgvector + cheap branching. Supabase's batteries (auth/storage/REST) aren't
needed for the MVP, and `build_store` keeps that door open if BYO-corpus later wants them.

**BM25 substrate — DECIDED: rank_bm25 in-process** (rebuild the index at boot from PG chunk text)
rather than Postgres full-text search. Retains the local dep, keeps hybrid-retrieval semantics
byte-identical to what the evals measured, and 810 chunks in memory is nothing. Revisit only if the
corpus grows large enough that boot-time indexing hurts.

---

## Workstream A — Backend service (`server/` package + FastAPI)

**New code**
- `server/app.py` — FastAPI app: `POST /ask` (SSE stream), `GET /sources` (corpus + tags for the
  sidebar, read from PG), `GET /health`, `GET /config` (public-safe knobs: max_rounds, model names).
- `server/stream.py` — **the trajectory streaming adapter (heart of the backend).** Wrap
  `graph.stream(initial_state(q), stream_mode="updates")` and translate each node's partial update
  into a typed SSE event:

  | graph node fires | SSE event | payload |
  |---|---|---|
  | `controller` | `think` | round's reasoning + chosen actions + rounds-used + controller tokens |
  | `tools` | `search` / `evidence` | per-query observation, chunks retrieved (source, snippet, score), new-vs-redundant count |
  | `answer` | `answer` | final cited text, retrieved window, citation list, `grounded` flag |
  | stream end | `done` | totals: rounds, tokens, latency, exit_reason, est. cost |

- `server/schemas.py` — Pydantic models for each event (typed contract the frontend consumes).
- `server/deps.py` — **build the graph once at startup**: connect to PG, rehydrate the BM25 index from
  chunk text, load the reranker from the baked cache, build retriever + both LLMs + registry + store;
  hold in app state. Per-request only calls `.stream()`. Avoids rebuilding the retriever per request.

**Changes to existing code**
- `rag/vector_store.py`: add the **`build_store(config)` factory + a pgvector backend** (§2). Phase 1,
  not a v2 seam. The pgvector backend implements the same interface the retriever calls today.
- `rag/tagging.py` + `rag/ingest.py`: **move doc tags from `doc_tags.json` on disk into Postgres**
  (JSONB column on a `documents` table). Incremental tag updates during ingest write to PG;
  `list_sources` reads from PG. (Review pt. 3 — survives the ephemeral filesystem.)
- `agent/graph.py`: **wire the spend-cap guardrail into the graph path.** It's built
  (`harness/guardrails.spend_cap_tripped`, DD-052) but deferred out of the graph (DD-053). The
  deployed app *is* the graph, so the per-question cap must live in `route_after_tools` /
  `controller_node` — check `controller_usage.total_tokens` vs `spend_cap_tokens`, set
  `exit_reason="spend_cap"`, route to `answer`.
- `rag/rerank.py` (or process entry): **bake in `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`.**
  Today's eval crash proved the reranker tries to reach huggingface.co at load and dies on a network
  blip. In a container it MUST load from the baked-in cache and never touch the network. **Required
  correctness change, not just hygiene.**
- `config/`: add a **prod profile** (env-driven: `DATABASE_URL`, `store.provider: pgvector`, keys,
  `memory.enabled: false`, spend-cap on, cache on) via `AGENTIC_RAG_CONFIG` or `config/prod.yaml`.

**New deps:** `fastapi`, `uvicorn[standard]`, `sse-starlette`, plus a PG client with pgvector support
(`psycopg[binary]` + `pgvector`, or SQLAlchemy if preferred).

---

## Workstream B — Cost control (deliberately minimal for launch)

Per review: **just the durable daily token cap for now.** Heavier abuse hardening is deferred, not
designed away.

**At launch (Phase 2 — required before public URL):**
1. **Global daily token cap, persisted to Postgres.** A `daily_spend(day, tokens)` row; each request
   reads today's total, and once the day's Gemini budget is hit, `/ask` returns "demo is resting,
   back tomorrow" instead of spending. **Durable = survives host restarts** (an in-memory counter
   would reset and be restart-farmed). *This is what makes $0 a guarantee.*
2. **Spend-cap per question** (already built — wire into graph, §A): bounds a single runaway question.
3. **Input caps**: max question length (~500 chars); reject empty/gibberish before any LLM call. Near-
   free to add, stops the dumbest abuse.

**Deferred (note in plan, revisit post-launch — review pt. 2, "something to think on"):**
per-IP rate limiting (slowapi), concurrency semaphore, Cloudflare Turnstile bot-friction, LangSmith
trace sampling. Add if real traffic/abuse shows up. The persisted daily cap is the backstop that keeps
$0 true even without them.

---

## Workstream C — Data & model packaging

- **Seed Postgres once** (not baked into the image): run the migration script (§2) to copy the
  prebuilt Chroma vectors + text + metadata + tags into pgvector. **No re-embedding, no Gemini spend.**
  Re-run only when the corpus changes. The corpus vectors now live in the DB, so the image doesn't
  carry them.
- **Bake the reranker model** (`ms-marco-MiniLM-L-6-v2`, 92 MB) + `HF_HUB_OFFLINE` into the image
  (it's a read-only artifact, not state). Image still needs **torch** (~1.5–2 GB image); HF Spaces
  handles it (cold start ~30–60 s, plus PG connect + BM25 rebuild at boot).
- **Episodic memory OFF for public** at launch (one shared store would leak between anonymous
  visitors). The **table exists in PG**, so turning it on later (per-session isolation) is a config
  flip, not a migration — this is exactly the consolidation the review asked for.
- **Dockerfile** for the Space: python 3.12 base, install from `pyproject`, copy reranker cache + any
  local config, `uvicorn server.app:app`. `DATABASE_URL` + keys come from Space secrets at runtime.

---

## Workstream D — Frontend UI (Next.js on Vercel) — the "make it pop" workstream

Stack: **Next.js (App Router) + TypeScript + Tailwind + shadcn/ui**, on Vercel. Consumes the backend SSE.

**Core screen — the "watch it think" chat**
- **Chat input** + a row of **example-question chips** (multi-hop questions that show the agent
  looping). Steers usage + demos strength.
- **Live agent trajectory timeline (the differentiator):** as SSE events arrive, render each round as
  a card — the controller's *reasoning*, the *searches* fired, the *chunks* pulled (source + snippet +
  score), with a "redundant round" marker when the oscillation guard notes no new evidence. Tells the
  "it's an agent, not a lookup" story visually.
- **Streaming answer** with **inline citation chips** `[filename]`. MVP: answer appears when ready;
  polish: typewriter token streaming (needs the streaming generate path, §F/stretch).
- **Clickable citations → source panel:** click `[filename]` → slide-over showing the actual chunk(s)
  the generator saw + doc name + context. Proves grounded, not hallucinated — big for credibility.
- **Cost & latency meter:** live rounds / tokens / latency / estimated ¢ per question. Transparency
  reads as engineering maturity; most demos hide it.
- **"Not enough information" path** rendered honestly when the agent abstains — shows the failure mode
  is handled.

**Features that sell the story (reputation-boosters)**
- **Naive RAG ⇄ Agentic toggle:** same question, single-shot retrieval vs. full agent loop, side by
  side. Makes the *value of the agent loop* visible — ties to the "every technique earned its place on
  the eval" narrative.
- **Methodology / "How it works" page:** architecture diagram (incl. the pgvector store), the six
  harness rings, the eval discipline (change→measure→keep), links to `DESIGN_DECISIONS.md` + docs.
  Turns your rigor into the pitch — the page that makes a hiring manager stop scrolling.
- **Eval scoreboard page (optional, strong):** show measured numbers (e2e / faithfulness /
  partial-credit) — "measured, not vibes."
- **Polish:** dark mode, responsive/mobile, loading skeletons, error toasts (rate-limit / "resting"),
  "About" with your LinkedIn.

**Stretch (v2)**
- **Bring-your-own-corpus:** upload → per-session ingest → ask against it. The `build_store` +
  Postgres spine now makes this natural (per-session namespace/table in the same DB), but it's still
  the **highest cost/abuse-risk feature** (arbitrary uploads = arbitrary embedding spend). Needs tight
  caps (size limit, ephemeral TTL, per-session isolation) — and this is where the deferred §B hardening
  comes back. Ship the fixed-corpus demo first.
- **Shareable answer permalinks** (store a run in PG, share a URL) — nice for people reposting your demo.

---

## Workstream E — Deployment & ops

- **Repo → monorepo:** `frontend/` (Next.js), keep `src/` (Python package), add `server/`. Root
  `README` becomes the public landing.
- **Managed Postgres:** provision Neon/Supabase (§2), enable the `vector` extension, run the schema +
  seed script. Connection string → `DATABASE_URL`.
- **Vercel:** connect repo, root `frontend/`, env `NEXT_PUBLIC_BACKEND_URL`. Auto-deploys on push;
  preview URLs per PR.
- **HF Spaces (Docker Space):** backend deploy; secrets `GOOGLE_API_KEY`, `GROQ_API_KEY`,
  `LANGSMITH_*`, **`DATABASE_URL`** as **Space secrets** (never in repo). Space `README.md` carries the
  Space config header.
- **CORS:** backend allows the Vercel production + preview origins only.
- **Secrets hygiene:** `.env` stays gitignored; add `.env.example` (incl. `DATABASE_URL`); never bake
  keys or the DB string into the image.
- **Health/uptime:** `/health` (checks PG connectivity too); optional free uptime pinger (mind
  free-tier sleep rules).
- **Observability:** structured request logs + (deferred) LangSmith sampled tracing + the existing
  cost/latency meters surfaced per request.

---

## Workstream F — Public presentation assets

- **Rewrite root `README`** as a landing: one-line pitch, live demo link, a GIF of the trajectory view,
  the architecture diagram, "what I learned" (links to `RETROSPECTIVE.md` + DDs).
- **A recorded GIF/short video** of a multi-hop question streaming through the agent — attach to the
  LinkedIn follow-up post.
- **The methodology page (§D)** doubles as portfolio evidence.

---

## Phasing (ship in this order — each phase independently shippable)

- **Phase 1 — Backend MVP + Postgres store:** provision free-tier PG, `build_store` pgvector backend +
  seed script (migrate vectors, no re-embed), move tags into PG, `server/` FastAPI + SSE adapter, wire
  spend-cap into graph, HF-offline reranker fix, prod config. Verify locally end-to-end against PG (curl).
- **Phase 2 — Durable daily cap:** persisted `daily_spend` kill-switch + per-question spend-cap + input
  caps. **Gate before any public URL.**
- **Phase 3 — Packaging + deploy backend:** Dockerfile, bake reranker, push to HF Spaces, secrets
  (incl. `DATABASE_URL`), `/health` green against the managed DB.
- **Phase 4 — Frontend MVP on Vercel:** chat + live trajectory + clickable citations + cost meter,
  wired to the deployed backend. **First shareable demo.**
- **Phase 5 — Pop features:** naive⇄agentic toggle, methodology page, eval scoreboard, polish (dark
  mode, mobile, GIF).
- **Phase 6 — Stretch:** token-streaming answer, per-session episodic memory (table's already there),
  bring-your-own-corpus (behind the deferred §B caps), shareable permalinks.

---

## Open decisions

1. **Feature scope for launch** — ship Phase 1–4 (solid, shareable) + methodology page first, layer
   5–6 after? *(Rec: yes — momentum beats polish.)* — **still open, your call.**

**Resolved on review (2026-07-03):**
- **Storage** = Postgres+pgvector as the single durable system-of-record (was: bake read-only, no DB).
- **Postgres provider** = **Neon** — auto-wakes from scale-to-zero, so an idle demo self-heals;
  Supabase free pauses after ~7 days idle needing a manual un-pause (dead-demo risk).
- **BM25 substrate** = **rank_bm25 in-process** (rehydrate at boot) — retains the local dep, eval
  semantics unchanged. Corpus is small.
- **Backend host** = **HF Spaces** — free tier gives 16 GB RAM (fits torch + reranker) where Render's
  512 MB free web service would likely OOM on model load; also on-brand for an ML demo.
- **Abuse hardening** = minimal for launch — just a *durable* daily token cap; heavier layers deferred.
- **Doc tags** = stored in PG, not on the ephemeral filesystem.

---

## Notes / provenance
- Embeddings swapped to Gemini + free-form tagging folded into ingest: commit `ae3cf65`, re-baselined
  quality-neutral in **DD-055** (cold n=1: e2e .619 / pc .810 / faith .952).
- The agent graph (`agent/graph.py`) already streams via `graph.stream()`; parity with the hand-rolled
  loop proven in **DD-053**. Guardrails (spend-cap + citation grounding) built in **DD-052**.
- Storage spine (Postgres+pgvector, tags-in-DB, durable cap) set by the 2026-07-03 plan review.
- Queued A/B candidate (independent of deploy): **contextual retrieval** (per-chunk augmentation,
  ~25× ingest cost) vs. the DD-055 baseline.
