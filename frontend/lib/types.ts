// Wire types — a one-to-one mirror of the backend's Pydantic schemas (agentic_rag/server/schemas.py).
// The SSE `/ask` stream emits one of the *Event shapes per step; keep these in sync with that file.

export type Action = { tool: string; args: Record<string, unknown> };

export type ChunkView = {
  source: string;
  chunk_index: number;
  snippet: string;
  score: number;
  retrieved_by?: string | null;
};

export type ThinkEvent = {
  type: "think";
  round: number;
  reasoning: string;
  actions: Action[];
  controller_tokens: number;
  finish: boolean;
};

export type EvidenceEvent = {
  type: "evidence";
  round: number;
  chunks: ChunkView[];
  new_count: number;
  redundant: boolean;
};

export type AnswerEvent = {
  type: "answer";
  text: string;
  citations: string[];
  grounded: boolean;
  ungrounded: string[];
  retrieved: ChunkView[];
};

export type DoneEvent = {
  type: "done";
  rounds: number;
  exit_reason: string;
  controller_tokens: number;
  generator_tokens: number;
  total_tokens: number;
  latency_ms: number;
  est_cost_usd: number;
};

export type ErrorEvent = { type: "error"; message: string };

export type StreamEvent =
  | ThinkEvent
  | EvidenceEvent
  | AnswerEvent
  | DoneEvent
  | ErrorEvent;

export type DocTags = { doc_type?: string; topics?: string[]; entities?: string[] };
export type SourceInfo = { source: string; chunks: number; tags: DocTags; uploaded: boolean };

// Retrieval scope toggle + per-request technique overrides (bring-your-own-doc + compare/knobs).
export type Scope = "both" | "uploads" | "demo";
export type Knobs = {
  mode?: "dense" | "hybrid";
  rerank?: boolean;
  parent_expansion?: boolean;
  max_rounds?: number;
  top_k?: number;
};
export type UploadResponse = {
  source: string;
  name: string;
  chunks: number;
  total_chunks: number;
};
export type ConfigInfo = {
  max_rounds: number;
  controller_model: string;
  generator_model: string;
  top_k: number;
  tools: string[];
  spend_cap_tokens: number;
  store_provider: string;
  daily_token_budget?: number;
  // Index-time knobs (baked at ingest; read-only in the UI).
  chunk_size?: number;
  chunk_overlap?: number;
  chunking_strategy?: string;
  embedding_provider?: string;
  embedding_model?: string;
  embedding_dims?: number;
};
