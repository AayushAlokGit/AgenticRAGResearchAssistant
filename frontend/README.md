# Frontend — Agentic RAG Research Assistant

The Next.js (App Router) frontend for the live demo. It streams the agent's trajectory from the
FastAPI backend over Server-Sent Events and renders it as an inspectable timeline: each controller
turn's reasoning, the searches it runs, the exact chunks it retrieved (click a citation to see the
chunk the model was given), the grounded answer, and a live token/cost meter.

**Live:** [agentic-rag-research-assistant-green.vercel.app](https://agentic-rag-research-assistant-green.vercel.app)
· Project overview + the engineering story: [`../README.md`](../README.md).

## Features

- **Ask** — multi-hop question → streamed reason→retrieve→re-retrieve trajectory + cited answer.
- **Documents tab** — upload your own `md`/`txt`/`pdf`, indexed live; delete anytime. A scope toggle
  (demo corpus / your uploads / both) controls what a query searches.
- **Compare** — run the same question under *Basic RAG* vs the *Full System* side by side.

## Local development

```bash
npm install
npm run dev          # http://localhost:3000
```

The backend base URL comes from `NEXT_PUBLIC_BACKEND_URL`:

- `.env.local` → `http://127.0.0.1:8000` for a local backend (run `uvicorn agentic_rag.server.app:app
  --port 8000` from the repo root). Note the backend's CORS allowlist defaults to `localhost:3000`, so
  run the frontend on port 3000.
- `.env.production` → the deployed Hugging Face Space URL (baked into the Vercel build).

## Key files

- `app/page.tsx` — the whole single-page app: tabs, hero, streaming trajectory, compare view, upload/manage.
- `app/methodology/page.tsx` — the "how it works" page.
- `lib/api.ts` — backend client; the interesting part is `askStream`, which parses the SSE framing by hand.
- `lib/types.ts` — wire types mirroring the backend's Pydantic schemas (`agentic_rag/server/schemas.py`).
