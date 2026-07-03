import Link from "next/link";

export const metadata = {
  title: "How it works — Agentic RAG Research Assistant",
};

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mt-8">
      <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-zinc-400">{title}</h2>
      <div className="space-y-3 text-[15px] leading-7 text-zinc-400">{children}</div>
    </section>
  );
}

function Step({ n, name, children }: { n: number; name: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-3 rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-zinc-800 font-mono text-xs text-zinc-400">
        {n}
      </span>
      <div>
        <div className="text-sm font-medium text-zinc-200">{name}</div>
        <div className="mt-0.5 text-sm text-zinc-400">{children}</div>
      </div>
    </div>
  );
}

export default function Methodology() {
  return (
    <div className="mx-auto w-full max-w-3xl px-4 pb-16">
      <header className="sticky top-0 z-20 -mx-4 flex items-center gap-3 border-b border-zinc-800 bg-zinc-950/80 px-4 py-3 backdrop-blur">
        <Link href="/" className="text-xs text-zinc-400 hover:text-zinc-100">
          ← Back to the demo
        </Link>
      </header>

      <h1 className="mt-8 text-2xl font-semibold text-zinc-100">How it works</h1>
      <p className="mt-2 text-[15px] leading-7 text-zinc-400">
        This is a from-scratch, hand-built agentic RAG system — the agent loop, context assembly, and
        harness are written by hand rather than pulled from a framework, so every technique is
        deliberate and measurable. It answers multi-hop questions over an evolving document corpus and
        shows its work: the reasoning, the searches, the exact chunks it cited, and what it cost.
      </p>

      <Section title="The retrieve → reason → retrieve loop">
        <p>
          Rather than a single lookup, a controller model decides — each round — what to search for
          next, reads the evidence gathered so far, and either searches again or stops and answers.
          Budgets and guardrails keep it honest: a round limit, an oscillation guard that halts a loop
          adding nothing new, and a token spend-cap circuit breaker.
        </p>
      </Section>

      <Section title="The retrieval pipeline (each earned its place on the eval)">
        <div className="space-y-2">
          <Step n={1} name="Hybrid retrieval">
            Dense vector search (semantic) fused with BM25 (keyword) via Reciprocal Rank Fusion — one
            recalls paraphrases, the other exact terms.
          </Step>
          <Step n={2} name="Cross-encoder reranking">
            A second-stage model re-scores the candidate pool by true (query, passage) relevance —
            expensive but precise, affordable because it only runs on a handful of candidates.
          </Step>
          <Step n={3} name="Parent expansion">
            Each surviving chunk is expanded into its neighbours so split tables and sections are
            restored without pulling in distractors.
          </Step>
          <Step n={4} name="Free-form document tags">
            Every document is tagged (type, topics, entities) at ingest; the agent reads the corpus map
            to answer “which documents are about X” without hoping similarity search surfaces them all.
          </Step>
        </div>
      </Section>

      <Section title="Grounding & honesty">
        <p>
          Answers cite their sources as <span className="font-mono text-sky-300">[filename]</span>, and
          a deterministic guard checks every citation against what was actually retrieved — a citation
          naming a document the loop never saw is flagged. When the evidence isn&apos;t there, the agent
          is built to say so rather than fabricate.
        </p>
      </Section>

      <Section title="Measured, not vibes">
        <p>
          This system fails silently — a fluent wrong answer looks like a fluent right one — so the eval
          set is the source of truth. Every change runs the loop: answer-correctness (partial-credit),
          faithfulness, and trajectory metrics (rounds, redundant work, cost). A technique is kept only
          if it earns its place; several plausible ideas were measured and reverted.
        </p>
      </Section>

      <Section title="Stack">
        <p>
          Next.js frontend (Vercel) · FastAPI backend streaming the trajectory over Server-Sent Events ·
          a LangGraph agent graph reusing the entire hand-built substrate · Gemini for the controller /
          generator / embeddings · a local cross-encoder reranker + BM25 · Postgres + pgvector as the
          durable store.
        </p>
      </Section>
    </div>
  );
}
