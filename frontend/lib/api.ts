// Backend client. The interesting part is askStream: the agent trajectory arrives as Server-Sent
// Events, but EventSource only does GET and we POST the question, so we read the fetch body as a
// stream and parse the SSE framing (blank-line-delimited `data:` blocks) by hand.

import { logEvent, newRequestId } from "./log";
import { ConfigInfo, Knobs, Scope, SourceInfo, StreamEvent, UploadResponse } from "./types";

const BASE = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:8000";

export async function getConfig(signal?: AbortSignal): Promise<ConfigInfo> {
  const r = await fetch(`${BASE}/config`, { signal });
  if (!r.ok) throw new Error(`/config ${r.status}`);
  return r.json();
}

export async function getSources(): Promise<SourceInfo[]> {
  const r = await fetch(`${BASE}/sources`);
  if (!r.ok) throw new Error(`/sources ${r.status}`);
  return (await r.json()).sources;
}

// Bring-your-own-doc: multipart upload. Surfaces the backend's HTTPException `detail` (type/size
// errors) so the UI can show a useful message instead of a bare status code.
export async function uploadDoc(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const r = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  if (!r.ok) {
    let detail = `Upload failed (${r.status})`;
    try {
      const body = await r.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // non-JSON error body — keep the status-code message
    }
    throw new Error(detail);
  }
  return r.json();
}

export async function deleteDoc(source: string): Promise<void> {
  const r = await fetch(`${BASE}/documents/${encodeURIComponent(source)}`, { method: "DELETE" });
  if (!r.ok) throw new Error(`Delete failed (${r.status})`);
}

export async function askStream(
  question: string,
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
  opts?: { scope?: Scope; knobs?: Knobs },
): Promise<void> {
  const reqId = newRequestId();
  const body: Record<string, unknown> = { question };
  if (opts?.scope) body.scope = opts.scope;
  if (opts?.knobs) body.knobs = opts.knobs;
  logEvent(reqId, "▶ /ask", { question, scope: opts?.scope ?? "both", knobs: opts?.knobs });
  const resp = await fetch(`${BASE}/ask`, {
    method: "POST",
    // X-Request-Id lets the backend adopt this id, so browser console + HF Space logs share it.
    headers: { "Content-Type": "application/json", "X-Request-Id": reqId },
    body: JSON.stringify(body),
    signal,
  });
  if (!resp.ok || !resp.body) {
    logEvent(reqId, "✗ http error", { status: resp.status });
    onEvent({ type: "error", message: `Backend returned ${resp.status}` });
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE events are separated by a blank line. sse-starlette uses CRLF, so match both \r\n\r\n
    // and \n\n; keep the trailing partial event in the buffer for the next chunk.
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      const data = block
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("");
      if (!data) continue; // keep-alive ping comments (": ping") carry no data line
      try {
        const event = JSON.parse(data) as StreamEvent;
        logEvent(reqId, event.type, summarizeEvent(event));
        onEvent(event);
      } catch {
        // ignore a malformed frame rather than kill the stream
      }
    }
  }
}

// Compact one-line summary per SSE event for the console trace — the shape of each step, not its full
// payload (answer text / chunk bodies are omitted; the UI renders those).
function summarizeEvent(e: StreamEvent): Record<string, unknown> {
  switch (e.type) {
    case "think":
      return { round: e.round, actions: e.actions.map((a) => a.tool), finish: e.finish,
               controller_tokens: e.controller_tokens };
    case "evidence":
      return { round: e.round, chunks: e.chunks.length, new: e.new_count, redundant: e.redundant };
    case "answer":
      return { chars: e.text.length, citations: e.citations, grounded: e.grounded };
    case "done":
      return { rounds: e.rounds, exit: e.exit_reason, total_tokens: e.total_tokens,
               latency_ms: e.latency_ms };
    case "error":
      return { message: e.message };
  }
}
