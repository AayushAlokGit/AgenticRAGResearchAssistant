// Lightweight browser-console tracing for one /ask request. Each request gets a short id that is ALSO
// sent to the backend as X-Request-Id, so the SAME id ties this console to the Hugging Face Space logs
// (grep `req=<id>` there). Filter the browser console by "[arag]" to isolate these lines.

const PREFIX = "[arag]";

export function newRequestId(): string {
  // 8 hex chars, matching the backend's minted-id length. crypto.randomUUID where available.
  try {
    return crypto.randomUUID().replace(/-/g, "").slice(0, 8);
  } catch {
    return Math.random().toString(16).slice(2, 10);
  }
}

export function logEvent(reqId: string, phase: string, data?: unknown): void {
  if (data === undefined) console.info(`${PREFIX} ${reqId} ${phase}`);
  else console.info(`${PREFIX} ${reqId} ${phase}`, data);
}
