"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { askStream, getConfig, getSources } from "@/lib/api";
import type {
  AnswerEvent,
  ChunkView,
  ConfigInfo,
  DoneEvent,
  EvidenceEvent,
  SourceInfo,
  ThinkEvent,
} from "@/lib/types";

const EXAMPLES = [
  "What embedding model does the system use and why is it kept local?",
  "How does hybrid retrieval combine BM25 and vector search?",
  "Which documents describe the ChromaDB architecture, and what do they cover?",
  "What are the trade-offs between the Azure and local ChromaDB approaches?",
];

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
  onOpenSource,
}: {
  data: RoundData;
  onOpenSource: (s: string) => void;
}) {
  const { think, evidence } = data;
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40">
      <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-2">
        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-zinc-800 font-mono text-[11px] text-zinc-400">
          {data.round}
        </span>
        <span className="text-xs font-medium text-zinc-400">
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
                  className="inline-flex items-center gap-1.5 rounded-md border border-zinc-700 bg-zinc-800/60 px-2 py-1 text-xs"
                >
                  <span className="font-mono text-sky-400">{a.tool}</span>
                  {typeof a.args.query === "string" && (
                    <span className="text-zinc-400">“{a.args.query as string}”</span>
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
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    getConfig().then(setConfig).catch(() => setError("Backend not reachable — is it running on :8000?"));
    getSources().then(setSources).catch(() => {});
  }, []);

  const run = useCallback(async (q: string) => {
    const query = q.trim();
    if (!query || running) return;
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
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") setError(String(err));
    } finally {
      setRunning(false);
    }
  }, [running]);

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
      {/* header */}
      <header className="sticky top-0 z-20 -mx-4 flex items-center gap-3 border-b border-zinc-800 bg-zinc-950/80 px-4 py-3 backdrop-blur">
        <div className="flex-1">
          <h1 className="text-sm font-semibold text-zinc-100">Agentic RAG Research Assistant</h1>
          <p className="text-xs text-zinc-500">retrieve → reason → retrieve, with citations you can inspect</p>
        </div>
        <Link href="/methodology" className="text-xs text-zinc-400 hover:text-zinc-100">
          How it works →
        </Link>
      </header>

      {/* config strip */}
      {config && (
        <div className="flex flex-wrap items-center gap-1.5 py-3">
          <Badge className="text-emerald-400">
            <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-emerald-400" /> live
          </Badge>
          <Badge>model: {config.controller_model}</Badge>
          <Badge>store: {config.store_provider}</Badge>
          <Badge>max_rounds: {config.max_rounds}</Badge>
          <Badge>top_k: {config.top_k}</Badge>
          <Badge>tools: {config.tools.join(", ")}</Badge>
          {totalChunks > 0 && <Badge>{sources.length} docs · {totalChunks} chunks</Badge>}
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
              disabled={!question.trim()}
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
      </div>

      {error && (
        <div className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* trajectory */}
      {rounds.length > 0 && (
        <section className="mt-6">
          <div className="mb-2 flex items-center gap-2">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
              Agent trajectory
            </h2>
            {running && <span className="text-xs text-sky-400">streaming…</span>}
          </div>
          <div className="space-y-2">
            {rounds
              .sort((a, b) => a.round - b.round)
              .map((r) => (
                <RoundCard key={r.round} data={r} onOpenSource={setSelectedSource} />
              ))}
          </div>
        </section>
      )}

      {/* answer */}
      {answer && (
        <section className="mt-6">
          <div className="mb-2 flex items-center gap-2">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Answer</h2>
            {answer.grounded ? (
              <Badge className="text-emerald-400">grounded ✓</Badge>
            ) : (
              <Badge className="border-red-500/40 text-red-400">ungrounded citations</Badge>
            )}
          </div>
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-4">
            <AnswerBody text={answer.text} onCite={setSelectedSource} />
            {!answer.grounded && answer.ungrounded.length > 0 && (
              <p className="mt-3 text-xs text-red-400">
                cited but not retrieved: {answer.ungrounded.join(", ")}
              </p>
            )}
          </div>
          <p className="mt-2 text-xs text-zinc-600">
            Tip: click a citation to see the exact chunk the model was given.
          </p>
        </section>
      )}

      {/* cost meter */}
      {done && (
        <section className="mt-4 mb-10 flex flex-wrap items-center gap-1.5">
          {done.total_tokens === 0 && (
            <Badge className="border-sky-500/40 text-sky-300">cached ✓ · served free</Badge>
          )}
          <Badge className={EXIT_STYLES[done.exit_reason] ?? ""}>exit: {done.exit_reason}</Badge>
          <Badge>{done.rounds} rounds</Badge>
          <Badge>ctrl {done.controller_tokens} tok</Badge>
          <Badge>gen {done.generator_tokens} tok</Badge>
          <Badge>{done.total_tokens} total tok</Badge>
          <Badge>{(done.latency_ms / 1000).toFixed(1)}s</Badge>
          <Badge className="text-amber-300">≈ ${done.est_cost_usd.toFixed(4)}</Badge>
        </section>
      )}

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
