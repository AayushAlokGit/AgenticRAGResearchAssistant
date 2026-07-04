"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { askStream, deleteDoc, getConfig, getSources, uploadDoc } from "@/lib/api";
import type {
  AnswerEvent,
  ChunkView,
  ConfigInfo,
  DoneEvent,
  EvidenceEvent,
  Knobs,
  Scope,
  SourceInfo,
  StreamEvent,
  ThinkEvent,
} from "@/lib/types";

// ── Retrieval-pipeline knobs — the project's first-class idea: toggle a technique, watch behavior
// change. These are RETRIEVAL-TIME (live per request). Index-time knobs (chunk size, embedding dims)
// are baked at ingest and shown read-only via <IndexSettings>.
type KnobConfig = {
  hybrid: boolean; // hybrid (dense + BM25) vs dense-only
  rerank: boolean; // cross-encoder rerank
  parent: boolean; // parent-child expansion
  agentic: boolean; // multi-round agentic loop vs single-shot
  topK: number; // chunks fed to the generator
};

const FULL_CONFIG: KnobConfig = { hybrid: true, rerank: true, parent: true, agentic: true, topK: 5 };

// The eval knob-ladder (README's headline table) as snap-to presets — dense → … → full.
const LADDER: { label: string; config: KnobConfig }[] = [
  { label: "Naive", config: { hybrid: false, rerank: false, parent: false, agentic: false, topK: 5 } },
  { label: "+ Hybrid", config: { hybrid: true, rerank: false, parent: false, agentic: false, topK: 5 } },
  { label: "+ Rerank", config: { hybrid: true, rerank: true, parent: false, agentic: false, topK: 5 } },
  { label: "+ Parent-child", config: { hybrid: true, rerank: true, parent: true, agentic: false, topK: 5 } },
  { label: "Full (agentic)", config: FULL_CONFIG },
];

const KNOB_TOGGLES: { key: "hybrid" | "rerank" | "parent" | "agentic"; label: string; desc: string }[] = [
  { key: "hybrid", label: "Hybrid retrieval", desc: "dense + BM25 keyword, fused with RRF" },
  { key: "rerank", label: "Cross-encoder rerank", desc: "re-score candidates for precision" },
  { key: "parent", label: "Parent-child expansion", desc: "add each hit's neighbouring chunks (±2 window)" },
  { key: "agentic", label: "Agentic loop", desc: "multi-round reason→retrieve (vs single-shot)" },
];

function toKnobs(k: KnobConfig): Knobs {
  return {
    mode: k.hybrid ? "hybrid" : "dense",
    rerank: k.rerank,
    parent_expansion: k.parent,
    max_rounds: k.agentic ? undefined : 1, // single-shot when the agentic loop is off
    top_k: k.topK,
  };
}

function knobChips(k: KnobConfig): string[] {
  const chips = [k.hybrid ? "hybrid" : "dense"];
  if (k.rerank) chips.push("rerank");
  if (k.parent) chips.push("parent-child");
  chips.push(k.agentic ? "agentic" : "single-shot");
  chips.push(`k=${k.topK}`);
  return chips;
}

function sameConfig(a: KnobConfig, b: KnobConfig): boolean {
  return (
    a.hybrid === b.hybrid && a.rerank === b.rerank && a.parent === b.parent &&
    a.agentic === b.agentic && a.topK === b.topK
  );
}

function configName(c: KnobConfig): string {
  return LADDER.find((p) => sameConfig(p.config, c))?.label ?? "Custom config";
}

const EXAMPLES = [
  "What embedding model does the system use and why is it kept local?",
  "How does hybrid retrieval combine BM25 and vector search?",
  "Which documents describe the ChromaDB architecture, and what do they cover?",
  "What are the trade-offs between the Azure and local ChromaDB approaches?",
];

const REPO_URL = "https://github.com/AayushAlokGit/AgenticRAGResearchAssistant";

const SCOPE_LABELS: Record<Scope, string> = {
  both: "All",
  uploads: "Uploaded",
  demo: "Demo corpus",
};

// Uploaded sources are stored as `upload:<filename>`; strip the marker for display.
function docName(source: string): string {
  return source.startsWith("upload:") ? source.slice("upload:".length) : source;
}

// Inline GitHub mark so we don't pull an icon dependency for one glyph.
function GitHubIcon({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" aria-hidden className={className} fill="currentColor">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

function Footer() {
  return (
    <footer className="mt-auto flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-zinc-800 py-4 text-xs text-zinc-600">
      <span>Built with Claude</span>
      <span className="text-zinc-700">·</span>
      <a
        href={REPO_URL}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-1 hover:text-zinc-300"
      >
        <GitHubIcon className="h-3 w-3" />
        Source
      </a>
      <span className="text-zinc-700">·</span>
      <Link href="/methodology" className="hover:text-zinc-300">
        How it works
      </Link>
      <span className="ml-auto">Aayush Alok</span>
    </footer>
  );
}

function Hero() {
  return (
    <section className="pt-8 pb-6">
      <div className="mb-3 inline-flex items-center gap-1.5 rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1 font-mono text-[11px] text-zinc-400">
        <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-emerald-400" />
        live demo
      </div>
      <h1 className="text-3xl font-semibold leading-tight tracking-tight text-zinc-100 sm:text-4xl">
        An agentic RAG system that shows its work.
      </h1>
      <p className="mt-3 max-w-2xl text-[15px] leading-7 text-zinc-400">
        Ask a multi-hop question about the corpus. Watch the agent{" "}
        <span className="text-zinc-200">reason → retrieve → re-retrieve</span>, cite its sources, and
        count every token in real time.
      </p>
      <div className="mt-4 flex flex-wrap items-center gap-3 text-xs">
        <a
          href={REPO_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-zinc-200 transition-colors hover:border-zinc-500 hover:bg-zinc-800"
        >
          <GitHubIcon className="h-3.5 w-3.5" />
          View the source
        </a>
        <Link
          href="/methodology"
          className="inline-flex items-center gap-1.5 rounded-lg border border-transparent px-1 py-1.5 text-zinc-400 transition-colors hover:text-zinc-100"
        >
          How it works →
        </Link>
      </div>
    </section>
  );
}

type RoundData = { round: number; think?: ThinkEvent; evidence?: EvidenceEvent };

function upsertRound(
  rounds: RoundData[],
  n: number,
  patch: (r: RoundData) => RoundData,
): RoundData[] {
  const idx = rounds.findIndex((r) => r.round === n);
  if (idx === -1) return [...rounds, patch({ round: n })];
  const next = [...rounds];
  next[idx] = patch(next[idx]);
  return next;
}

const EXIT_STYLES: Record<string, string> = {
  finish: "text-emerald-400 border-emerald-500/30 bg-emerald-500/10",
  budget: "text-amber-400 border-amber-500/30 bg-amber-500/10",
  oscillation: "text-amber-400 border-amber-500/30 bg-amber-500/10",
  spend_cap: "text-red-400 border-red-500/30 bg-red-500/10",
};

// ── inline answer rendering: **bold** + [file.ext] citation chips ─────────────────────────
function renderBold(text: string, key: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={`${key}-b${i}`} className="font-semibold text-zinc-100">
        {part.slice(2, -2)}
      </strong>
    ) : (
      <span key={`${key}-s${i}`}>{part}</span>
    ),
  );
}

function renderInline(
  text: string,
  onCite: (file: string) => void,
  key: string,
): ReactNode[] {
  const nodes: ReactNode[] = [];
  const citationRe = /\[([^\][]+?\.[A-Za-z0-9]+)\]/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = citationRe.exec(text)) !== null) {
    if (m.index > last) nodes.push(...renderBold(text.slice(last, m.index), `${key}-t${i}`));
    const file = m[1];
    nodes.push(
      <button
        key={`${key}-c${i}`}
        onClick={() => onCite(file)}
        title={file}
        className="mx-0.5 inline-flex items-center rounded bg-sky-500/15 px-1.5 py-0.5 align-baseline font-mono text-[11px] text-sky-300 ring-1 ring-sky-500/30 transition-colors hover:bg-sky-500/25"
      >
        {file.replace(/\.[^.]+$/, "")}
      </button>,
    );
    last = m.index + m[0].length;
    i++;
  }
  if (last < text.length) nodes.push(...renderBold(text.slice(last), `${key}-t${i}`));
  return nodes;
}

function AnswerBody({ text, onCite }: { text: string; onCite: (f: string) => void }) {
  const lines = text.split("\n");
  return (
    <div className="space-y-1.5 text-[15px] leading-7 text-zinc-300">
      {lines.map((line, i) => {
        const trimmed = line.trim();
        if (!trimmed) return <div key={i} className="h-2" />;
        const isBullet = /^[*\-•]\s+/.test(trimmed);
        const content = isBullet ? trimmed.replace(/^[*\-•]\s+/, "") : line;
        return (
          <div key={i} className={isBullet ? "flex gap-2 pl-1" : ""}>
            {isBullet && <span className="mt-0.5 text-zinc-600">•</span>}
            <p>{renderInline(content, onCite, `l${i}`)}</p>
          </div>
        );
      })}
    </div>
  );
}

// ── small building blocks ─────────────────────────────────────────────────────────────────
function Spinner({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden className={`h-3.5 w-3.5 animate-spin ${className}`} fill="none">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
      <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

function Badge({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded border border-zinc-700/70 bg-zinc-900 px-2 py-0.5 font-mono text-[11px] text-zinc-400 ${className}`}
    >
      {children}
    </span>
  );
}

function RoundCard({
  data,
  active,
  onOpenSource,
}: {
  data: RoundData;
  active: boolean;
  onOpenSource: (s: string) => void;
}) {
  const { think, evidence } = data;
  return (
    <div
      className={`rounded-lg border bg-zinc-900/40 ${active ? "border-sky-500/40" : "border-zinc-800"}`}
    >
      <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-2">
        <span
          className={`text-xs font-medium ${think?.finish ? "text-emerald-300" : "text-zinc-400"}`}
        >
          {think?.finish ? "decided to answer" : "reason → retrieve"}
        </span>
        {think && (
          <Badge className="ml-auto">{think.controller_tokens} ctrl tok</Badge>
        )}
      </div>

      {think && (
        <div className="space-y-2 px-3 py-2.5">
          <p className="text-sm italic leading-6 text-zinc-400">{think.reasoning}</p>
          {think.actions.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {think.actions.map((a, i) => (
                <span
                  key={i}
                  className="inline-flex max-w-full items-start gap-1.5 rounded-md border border-zinc-700 bg-zinc-800/60 px-2 py-1 text-xs"
                >
                  <span className="shrink-0 font-mono text-sky-400">{a.tool}</span>
                  {typeof a.args.query === "string" && (
                    <span className="min-w-0 break-words text-zinc-400">“{a.args.query as string}”</span>
                  )}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {evidence && (
        <div className="border-t border-zinc-800/70 px-3 py-2.5">
          <div className="mb-2 flex items-center gap-2 text-xs">
            <span className="text-zinc-500">retrieved</span>
            <Badge>{evidence.chunks.length} chunks</Badge>
            <Badge className="text-emerald-400">+{evidence.new_count} new</Badge>
            {evidence.redundant && (
              <Badge className="border-amber-500/40 text-amber-400">redundant round</Badge>
            )}
          </div>
          <div className="space-y-1">
            {evidence.chunks.map((c, i) => (
              <button
                key={i}
                onClick={() => onOpenSource(c.source)}
                className="flex w-full items-center gap-2 rounded px-2 py-1 text-left transition-colors hover:bg-zinc-800/60"
              >
                <span className="font-mono text-[11px] text-zinc-500">
                  {c.score.toFixed(2)}
                </span>
                <span className="truncate text-xs text-zinc-300">
                  {c.source}
                  <span className="text-zinc-600"> #{c.chunk_index}</span>
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function SourceDrawer({
  source,
  chunks,
  tags,
  onClose,
}: {
  source: string;
  chunks: ChunkView[];
  tags?: SourceInfo["tags"];
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <div className="flex-1 bg-black/50" onClick={onClose} />
      <aside className="flex h-full w-full max-w-md flex-col border-l border-zinc-800 bg-zinc-950 shadow-2xl">
        <div className="flex items-start justify-between gap-3 border-b border-zinc-800 p-4">
          <div>
            <div className="text-[11px] uppercase tracking-wide text-zinc-500">source document</div>
            <div className="mt-0.5 break-all font-mono text-sm text-zinc-200">{source}</div>
          </div>
          <button
            onClick={onClose}
            className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        {tags && (tags.doc_type || tags.topics?.length) && (
          <div className="flex flex-wrap gap-1.5 border-b border-zinc-800 p-3">
            {tags.doc_type && <Badge className="text-violet-300">{tags.doc_type}</Badge>}
            {tags.topics?.map((t) => (
              <Badge key={t}>{t}</Badge>
            ))}
          </div>
        )}
        <div className="flex-1 space-y-3 overflow-y-auto p-4">
          <div className="text-xs text-zinc-500">
            {chunks.length} chunk(s) the generator saw from this document
          </div>
          {chunks.map((c, i) => (
            <div key={i} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
              <div className="mb-1.5 flex items-center gap-2 font-mono text-[11px] text-zinc-500">
                <span>#{c.chunk_index}</span>
                <span>score {c.score.toFixed(3)}</span>
              </div>
              <pre className="whitespace-pre-wrap break-words font-mono text-[12px] leading-5 text-zinc-400">
                {c.snippet}
              </pre>
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
}

// ── one run's live state (trajectory + answer + cost) — reused for single ask AND each compare column
type RunState = {
  rounds: RoundData[];
  answer: AnswerEvent | null;
  done: DoneEvent | null;
  error: string | null;
  running: boolean;
};

const EMPTY_RUN: RunState = { rounds: [], answer: null, done: null, error: null, running: false };

function RunView({
  run,
  onOpenSource,
  title,
  knobs,
}: {
  run: RunState;
  onOpenSource: (s: string) => void;
  title?: string;
  knobs?: KnobConfig; // the retrieval knobs this run used — shown as chips so every run is labelled
}) {
  const { rounds, answer, done, error, running } = run;
  return (
    <div>
      {(title || knobs) && (
        <div className="mb-3 border-b border-zinc-800 pb-2">
          {title && <h3 className="text-sm font-semibold text-zinc-100">{title}</h3>}
          {knobs && (
            <div className="mt-1 flex flex-wrap gap-1">
              {knobChips(knobs).map((c) => (
                <span
                  key={c}
                  className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400"
                >
                  {c}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="mb-3 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* trajectory */}
      {rounds.length > 0 && (
        <section className="mb-4">
          <div className="mb-2 flex items-center gap-2">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Agent trajectory</h2>
            {running && <span className="text-xs text-sky-400">streaming…</span>}
          </div>
          <div>
            {[...rounds]
              .sort((a, b) => a.round - b.round)
              .map((r, i, all) => {
                const isLast = i === all.length - 1;
                const isActive = running && isLast;
                const isFinish = !!r.think?.finish;
                return (
                  <div key={r.round} className="flex gap-3 pb-3 last:pb-0">
                    <div className="flex flex-col items-center">
                      <span
                        className={`relative flex h-6 w-6 shrink-0 items-center justify-center rounded-full border font-mono text-[11px] ${
                          isFinish
                            ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-300"
                            : isActive
                              ? "border-sky-500/50 bg-sky-500/15 text-sky-300"
                              : "border-zinc-700 bg-zinc-800 text-zinc-400"
                        }`}
                      >
                        {isActive && (
                          <span className="absolute inset-0 animate-ping rounded-full bg-sky-500/30" />
                        )}
                        <span className="relative">{r.round}</span>
                      </span>
                      {!isLast && <span className="w-px flex-1 bg-zinc-800" />}
                    </div>
                    <div className="min-w-0 flex-1">
                      <RoundCard data={r} active={isActive} onOpenSource={onOpenSource} />
                    </div>
                  </div>
                );
              })}
          </div>
        </section>
      )}

      {/* answer */}
      {answer && (
        <section className="mb-4">
          <div className="mb-2 flex items-center gap-2">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Answer</h2>
            {answer.grounded ? (
              <Badge className="text-emerald-400">grounded ✓</Badge>
            ) : (
              <Badge className="border-red-500/40 text-red-400">ungrounded citations</Badge>
            )}
          </div>
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-4">
            <AnswerBody text={answer.text} onCite={onOpenSource} />
            {!answer.grounded && answer.ungrounded.length > 0 && (
              <p className="mt-3 text-xs text-red-400">
                cited but not retrieved: {answer.ungrounded.join(", ")}
              </p>
            )}
          </div>
        </section>
      )}

      {/* couldn't-answer empty state */}
      {done && !answer && !error && (
        <section className="mb-4">
          <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-200">
            The agent stopped before it could answer
            {done.exit_reason !== "finish" && <> (hit its {done.exit_reason.replace("_", " ")} limit)</>}.
            Try rephrasing, or ask something the corpus actually covers.
          </div>
        </section>
      )}

      {/* cost meter */}
      {done && (
        <section className="flex flex-wrap items-center gap-1.5">
          {done.total_tokens === 0 && (
            <Badge className="border-sky-500/40 text-sky-300">cached ✓ · served free</Badge>
          )}
          <Badge className={EXIT_STYLES[done.exit_reason] ?? ""}>exit: {done.exit_reason}</Badge>
          <Badge>{done.rounds} rounds</Badge>
          <Badge>{done.total_tokens} tok</Badge>
          <Badge>{(done.latency_ms / 1000).toFixed(1)}s</Badge>
          <Badge className="text-amber-300">≈ ${done.est_cost_usd.toFixed(4)}</Badge>
        </section>
      )}
    </div>
  );
}

// The first-class knob panel: toggle retrieval techniques + top_k, snap to eval-ladder presets.
function KnobPanel({
  config,
  onChange,
}: {
  config: KnobConfig;
  onChange: (c: KnobConfig) => void;
}) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-zinc-400">
          Retrieval pipeline
        </span>
        <label className="flex items-center gap-1.5 text-[11px] text-zinc-500">
          top_k
          <select
            value={config.topK}
            onChange={(e) => onChange({ ...config, topK: Number(e.target.value) })}
            className="rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-xs text-zinc-200 outline-none"
          >
            {[3, 5, 8].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {KNOB_TOGGLES.map((t) => {
          const on = config[t.key];
          return (
            <button
              key={t.key}
              type="button"
              onClick={() => onChange({ ...config, [t.key]: !on })}
              title={t.desc}
              className={`rounded-md border px-2.5 py-1 text-xs transition-colors ${
                on
                  ? "border-sky-500/50 bg-sky-500/15 text-sky-200"
                  : "border-zinc-700 bg-zinc-900 text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {on ? "✓ " : ""}
              {t.label}
            </button>
          );
        })}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[11px]">
        <span className="text-zinc-600">presets:</span>
        {LADDER.map((p) => (
          <button
            key={p.label}
            type="button"
            onClick={() => onChange(p.config)}
            className={`rounded border px-2 py-0.5 transition-colors ${
              sameConfig(p.config, config)
                ? "border-zinc-600 bg-zinc-800 text-zinc-200"
                : "border-zinc-800 text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>
      {/* legend: a one-line definition of each knob */}
      <dl className="mt-2 grid gap-x-5 gap-y-1 border-t border-zinc-800 pt-2 text-[11px] text-zinc-500 sm:grid-cols-2">
        {KNOB_TOGGLES.map((t) => (
          <div key={t.key}>
            <span className="text-zinc-400">{t.label}</span> — {t.desc}
          </div>
        ))}
        <div>
          <span className="text-zinc-400">top_k</span> — how many retrieved chunks are fed to the generator
        </div>
      </dl>
    </div>
  );
}

// Read-only readout of the index-time knobs (baked at ingest; the honest "these exist but re-index to change").
function IndexSettings({ config }: { config: ConfigInfo }) {
  if (!config.chunk_size) return null;
  return (
    <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[11px] text-zinc-500">
      <span className="text-zinc-600">index-time (set at ingest):</span>
      <Badge>
        chunk {config.chunk_size}/{config.chunk_overlap} · {config.chunking_strategy}
      </Badge>
      <Badge>
        {config.embedding_model} · {config.embedding_dims}d
      </Badge>
      <span className="text-zinc-600">— re-index to change</span>
    </div>
  );
}

// ── page ───────────────────────────────────────────────────────────────────────────────────
export default function Home() {
  const [config, setConfig] = useState<ConfigInfo | null>(null);
  const [sources, setSources] = useState<SourceInfo[]>([]);
  const [question, setQuestion] = useState("");
  const [running, setRunning] = useState(false);
  const [rounds, setRounds] = useState<RoundData[]>([]);
  const [answer, setAnswer] = useState<AnswerEvent | null>(null);
  const [done, setDone] = useState<DoneEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedSource, setSelectedSource] = useState<string | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [view, setView] = useState<"ask" | "documents">("ask");
  const [scope, setScope] = useState<Scope>("both");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [compareRuns, setCompareRuns] = useState<[RunState, RunState] | null>(null);
  const [knobConfig, setKnobConfig] = useState<KnobConfig>(FULL_CONFIG); // the active retrieval config
  const [singleKnobs, setSingleKnobs] = useState<KnobConfig | null>(null); // snapshot shown on the single run
  const [compareB, setCompareB] = useState<KnobConfig>(LADDER[0].config); // the config to compare against (default Naive)
  const [compareConfigs, setCompareConfigs] = useState<[KnobConfig, KnobConfig] | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const compareAbortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // The demo backend runs on a free Hugging Face Space that sleeps when idle and takes ~30–60s to
  // wake, so the first /config call can fail a few times before it answers. Retry a handful of
  // times, then give up and let the user retry manually.
  const connect = useCallback(async () => {
    setStatus("loading");
    for (let attempt = 0; attempt < 15; attempt++) {
      try {
        setConfig(await getConfig(AbortSignal.timeout(8000)));
        getSources().then(setSources).catch(() => {});
        setStatus("ready");
        return;
      } catch {
        await new Promise((resolve) => setTimeout(resolve, 4000));
      }
    }
    setStatus("error");
  }, []);

  useEffect(() => {
    connect();
  }, [connect]);

  const run = useCallback(async (q: string) => {
    const query = q.trim();
    if (!query || running || status !== "ready") return;
    setCompareRuns(null); // leave compare mode when running a single ask
    setSingleKnobs(knobConfig); // snapshot the config this run uses, for the RunView label
    setRunning(true);
    setRounds([]);
    setAnswer(null);
    setDone(null);
    setError(null);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await askStream(
        query,
        (e) => {
          if (e.type === "think") setRounds((prev) => upsertRound(prev, e.round, (r) => ({ ...r, think: e })));
          else if (e.type === "evidence") setRounds((prev) => upsertRound(prev, e.round, (r) => ({ ...r, evidence: e })));
          else if (e.type === "answer") setAnswer(e);
          else if (e.type === "done") setDone(e);
          else if (e.type === "error") setError(e.message);
        },
        controller.signal,
        { scope, knobs: toKnobs(knobConfig) },
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") setError(String(err));
    } finally {
      setRunning(false);
    }
  }, [running, status, scope, knobConfig]);

  // Run the SAME question under two knob configs in parallel — the core "toggle a knob, watch behavior
  // change" proof. Config A = the active knob panel; config B = the compare-against selection.
  const runCompare = useCallback(async (q: string) => {
    const query = q.trim();
    if (!query || running || status !== "ready") return;
    const configs: [KnobConfig, KnobConfig] = [knobConfig, compareB];
    setRounds([]);
    setAnswer(null);
    setDone(null);
    setError(null);
    setCompareConfigs(configs);
    setCompareRuns([{ ...EMPTY_RUN, running: true }, { ...EMPTY_RUN, running: true }]);
    const controller = new AbortController();
    compareAbortRef.current = controller;

    const update = (idx: 0 | 1, fn: (rs: RunState) => RunState) =>
      setCompareRuns((prev) => {
        if (!prev) return prev;
        const next: [RunState, RunState] = [prev[0], prev[1]];
        next[idx] = fn(next[idx]);
        return next;
      });

    const onEvent = (idx: 0 | 1) => (e: StreamEvent) =>
      update(idx, (rs) => {
        if (e.type === "think") return { ...rs, rounds: upsertRound(rs.rounds, e.round, (r) => ({ ...r, think: e })) };
        if (e.type === "evidence") return { ...rs, rounds: upsertRound(rs.rounds, e.round, (r) => ({ ...r, evidence: e })) };
        if (e.type === "answer") return { ...rs, answer: e };
        if (e.type === "done") return { ...rs, done: e, running: false };
        if (e.type === "error") return { ...rs, error: e.message, running: false };
        return rs;
      });

    const stream = (idx: 0 | 1) =>
      askStream(query, onEvent(idx), controller.signal, { scope, knobs: toKnobs(configs[idx]) })
        .catch((err) => {
          if ((err as Error).name !== "AbortError")
            update(idx, (rs) => ({ ...rs, error: String(err) }));
        })
        .finally(() => update(idx, (rs) => ({ ...rs, running: false })));

    await Promise.all([stream(0), stream(1)]);
  }, [running, status, scope, knobConfig, compareB]);

  const compareRunning = !!compareRuns && compareRuns.some((r) => r.running);

  const refreshSources = useCallback(() => {
    getSources().then(setSources).catch(() => {});
  }, []);

  const handleUpload = useCallback(async (file: File) => {
    setUploadError(null);
    setUploading(true);
    try {
      await uploadDoc(file);
      refreshSources();
      setScope("uploads"); // focus the just-uploaded doc so the next question queries it
    } catch (err) {
      setUploadError((err as Error).message);
    } finally {
      setUploading(false);
    }
  }, [refreshSources]);

  const handleDelete = useCallback(async (source: string) => {
    try {
      await deleteDoc(source);
      const remaining = await getSources();
      setSources(remaining);
      if (!remaining.some((s) => s.uploaded)) setScope("both"); // nothing left to scope to
    } catch {
      // leave the doc listed if the delete failed
    }
  }, []);

  const uploadedDocs = sources.filter((s) => s.uploaded);
  const totalChunks = sources.reduce((n, s) => n + s.chunks, 0);

  // Pool every chunk we've seen (evidence + final window) so the drawer can show a source's chunks.
  const chunksForSource = (src: string): ChunkView[] => {
    const map = new Map<number, ChunkView>();
    for (const r of rounds) for (const c of r.evidence?.chunks ?? []) if (c.source === src) map.set(c.chunk_index, c);
    for (const c of answer?.retrieved ?? []) if (c.source === src) map.set(c.chunk_index, c);
    return [...map.values()].sort((a, b) => a.chunk_index - b.chunk_index);
  };

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-4xl flex-col px-4">
      {/* tab bar — keeps document upload/management off the Ask (query + trajectory) view */}
      <nav className="flex items-center gap-1 border-b border-zinc-800 pt-4">
        {(["ask", "documents"] as const).map((v) => (
          <button
            key={v}
            type="button"
            onClick={() => setView(v)}
            className={`relative px-3 py-2 text-sm transition-colors ${
              view === v ? "text-zinc-100" : "text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {v === "ask" ? "Ask" : "Documents"}
            {v === "documents" && uploadedDocs.length > 0 && (
              <span className="ml-1.5 rounded-full bg-zinc-700 px-1.5 py-0.5 font-mono text-[10px] text-zinc-300">
                {uploadedDocs.length}
              </span>
            )}
            {view === v && <span className="absolute inset-x-2 -bottom-px h-0.5 rounded bg-sky-500" />}
          </button>
        ))}
      </nav>

      {/* backend connection status (both tabs) */}
      {status === "loading" && (
        <div className="mt-3 mb-1 flex items-center gap-2 text-xs text-zinc-500">
          <Spinner /> Connecting to the backend… the free demo server can take ~30–60s to wake.
        </div>
      )}
      {status === "error" && (
        <div className="mt-3 mb-1 flex items-center justify-between gap-3 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">
          <span>Backend isn&apos;t responding. It may still be starting up.</span>
          <button
            onClick={connect}
            className="shrink-0 rounded-md border border-red-500/40 bg-red-500/10 px-3 py-1 text-xs text-red-200 hover:bg-red-500/20"
          >
            Retry
          </button>
        </div>
      )}

      {/* DOCUMENTS TAB — upload + manage, isolated from the trajectory UI */}
      {view === "documents" && (
        <section className="py-6">
          <h2 className="text-sm font-semibold text-zinc-100">Your documents</h2>
          <p className="mt-1 max-w-2xl text-sm text-zinc-400">
            Add your own documents to the corpus, then ask questions about them on the Ask tab.
            Uploads are shared and can be deleted by anyone.
          </p>
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading || status !== "ready"}
              className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-200 transition-colors hover:border-zinc-500 hover:bg-zinc-800 disabled:opacity-50"
            >
              {uploading ? (
                <>
                  <Spinner /> indexing…
                </>
              ) : (
                <>+ Add your document</>
              )}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".md,.txt,.pdf"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleUpload(f);
                e.target.value = ""; // allow re-selecting the same file
              }}
            />
            <span className="text-[11px] text-zinc-600">md · txt · pdf, up to 2 MB</span>
          </div>

          {uploadError && (
            <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-1.5 text-xs text-red-300">
              {uploadError}
            </div>
          )}

          <div className="mt-4">
            {uploadedDocs.length === 0 ? (
              <div className="rounded-lg border border-dashed border-zinc-800 px-4 py-10 text-center text-sm text-zinc-600">
                No uploaded documents yet.
              </div>
            ) : (
              <div className="divide-y divide-zinc-800/70 overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900/40">
                {uploadedDocs.map((d) => (
                  <div key={d.source} className="flex items-center gap-2 px-3 py-2 hover:bg-zinc-800/40">
                    <span className="min-w-0 flex-1 truncate text-sm text-zinc-300">
                      {docName(d.source)}
                      <span className="text-zinc-600"> · {d.chunks} chunk{d.chunks === 1 ? "" : "s"}</span>
                    </span>
                    <button
                      type="button"
                      onClick={() => handleDelete(d.source)}
                      className="rounded px-2 py-0.5 text-xs text-zinc-500 hover:bg-red-500/10 hover:text-red-400"
                      aria-label={`Delete ${docName(d.source)}`}
                    >
                      Delete
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>
      )}

      {/* ASK TAB — query, trajectory, answer, cost */}
      {view === "ask" && (
        <>
          {/* hero (idle only — collapses once a query runs) */}
          {!running && rounds.length === 0 && !answer && <Hero />}

          {/* config strip + read-only index-time settings */}
          {config && (
            <div className="py-3">
              <div className="flex flex-wrap items-center gap-1.5">
                <Badge className="text-emerald-400">
                  <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-emerald-400" /> live
                </Badge>
                <Badge>model: {config.controller_model}</Badge>
                <Badge>store: {config.store_provider}</Badge>
                <Badge>tools: {config.tools.join(", ")}</Badge>
                {totalChunks > 0 && <Badge>{sources.length} docs · {totalChunks} chunks</Badge>}
              </div>
              <IndexSettings config={config} />
            </div>
          )}

          {/* KNOB PANEL — the first-class idea: tune the retrieval pipeline, watch behavior change */}
          {status === "ready" && (
            <div className="mb-3">
              <KnobPanel config={knobConfig} onChange={setKnobConfig} />
            </div>
          )}

          {/* scope toggle — controls what a query searches (appears once a doc is uploaded) */}
          {status === "ready" && uploadedDocs.length > 0 && (
            <div className="mb-2 flex items-center gap-2 text-xs">
              <span className="text-zinc-500">Searching:</span>
              <div className="inline-flex rounded-lg border border-zinc-800 bg-zinc-900 p-0.5">
                {(["both", "uploads", "demo"] as Scope[]).map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => setScope(s)}
                    className={`rounded-md px-2.5 py-1 transition-colors ${
                      scope === s ? "bg-zinc-700 text-zinc-100" : "text-zinc-400 hover:text-zinc-200"
                    }`}
                  >
                    {SCOPE_LABELS[s]}
                  </button>
                ))}
              </div>
            </div>
          )}

      {/* input */}
      <div className="mt-1">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            run(question);
          }}
          className="flex gap-2"
        >
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a multi-hop question about the corpus…"
            maxLength={500}
            className="flex-1 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2.5 text-sm text-zinc-100 placeholder-zinc-600 outline-none focus:border-sky-600"
          />
          {running ? (
            <button
              type="button"
              onClick={() => abortRef.current?.abort()}
              className="rounded-lg border border-zinc-700 bg-zinc-800 px-4 text-sm text-zinc-300 hover:bg-zinc-700"
            >
              Stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={!question.trim() || status !== "ready"}
              title={status !== "ready" ? "Waiting for the backend to be ready…" : undefined}
              className="rounded-lg bg-sky-600 px-5 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-40"
            >
              Ask
            </button>
          )}
        </form>

        {/* example chips (idle only) */}
        {!running && rounds.length === 0 && !answer && (
          <div className="mt-3 flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                onClick={() => {
                  setQuestion(ex);
                  run(ex);
                }}
                className="rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1.5 text-left text-xs text-zinc-400 transition-colors hover:border-zinc-600 hover:text-zinc-200"
              >
                {ex}
              </button>
            ))}
          </div>
        )}

        {/* Compare: run the same question under the active config (A) vs a chosen config (B) */}
        {!running && (
          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
            {compareRunning ? (
              <button
                type="button"
                onClick={() => compareAbortRef.current?.abort()}
                className="rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-zinc-300 hover:bg-zinc-700"
              >
                Stop comparison
              </button>
            ) : (
              <>
                <button
                  type="button"
                  onClick={() => runCompare(question)}
                  disabled={!question.trim() || status !== "ready"}
                  title="Run the same question under your current knobs vs the chosen config, side by side"
                  className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-zinc-300 transition-colors hover:border-sky-600 hover:text-zinc-100 disabled:opacity-40"
                >
                  ⚡ Compare current knobs
                </button>
                <span className="text-zinc-500">vs</span>
                <select
                  value={configName(compareB)}
                  onChange={(e) =>
                    setCompareB(LADDER.find((p) => p.label === e.target.value)?.config ?? LADDER[0].config)
                  }
                  className="rounded-lg border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-zinc-300 outline-none"
                >
                  {LADDER.map((p) => (
                    <option key={p.label} value={p.label}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </>
            )}
          </div>
        )}
      </div>

      {/* results: single run, or the two-config compare (side by side, each labelled with its knobs) */}
      {compareRuns && compareConfigs ? (
        <div className="mb-10 mt-6">
          <p className="mb-3 text-xs text-zinc-500">
            Same question, two knob configs — watch how the retrieval pipeline changes the trajectory and answer.
          </p>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            <RunView
              run={compareRuns[0]}
              onOpenSource={setSelectedSource}
              title={configName(compareConfigs[0])}
              knobs={compareConfigs[0]}
            />
            <RunView
              run={compareRuns[1]}
              onOpenSource={setSelectedSource}
              title={configName(compareConfigs[1])}
              knobs={compareConfigs[1]}
            />
          </div>
        </div>
      ) : (
        <div className="mb-10 mt-6">
          <RunView
            run={{ rounds, answer, done, error, running }}
            onOpenSource={setSelectedSource}
            knobs={rounds.length > 0 || done ? singleKnobs ?? undefined : undefined}
          />
          {answer && (
            <p className="mt-2 text-xs text-zinc-600">
              Tip: click a citation to see the exact chunk the model was given.
            </p>
          )}
        </div>
      )}
        </>
      )}

      <Footer />

      {selectedSource && (
        <SourceDrawer
          source={selectedSource}
          chunks={chunksForSource(selectedSource)}
          tags={sources.find((s) => s.source === selectedSource)?.tags}
          onClose={() => setSelectedSource(null)}
        />
      )}
    </div>
  );
}
