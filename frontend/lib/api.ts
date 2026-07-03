// Backend client. The interesting part is askStream: the agent trajectory arrives as Server-Sent
// Events, but EventSource only does GET and we POST the question, so we read the fetch body as a
// stream and parse the SSE framing (blank-line-delimited `data:` blocks) by hand.

import { ConfigInfo, SourceInfo, StreamEvent } from "./types";

const BASE = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:8000";

export async function getConfig(): Promise<ConfigInfo> {
  const r = await fetch(`${BASE}/config`);
  if (!r.ok) throw new Error(`/config ${r.status}`);
  return r.json();
}

export async function getSources(): Promise<SourceInfo[]> {
  const r = await fetch(`${BASE}/sources`);
  if (!r.ok) throw new Error(`/sources ${r.status}`);
  return (await r.json()).sources;
}

export async function askStream(
  question: string,
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const resp = await fetch(`${BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!resp.ok || !resp.body) {
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
        onEvent(JSON.parse(data) as StreamEvent);
      } catch {
        // ignore a malformed frame rather than kill the stream
      }
    }
  }
}
