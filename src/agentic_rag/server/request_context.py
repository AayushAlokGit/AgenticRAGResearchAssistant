"""Per-request correlation: a request id + client IP that follow one request through every log line.

The deployed backend answers a question by streaming a LangGraph trajectory across many log calls
(controller -> tools -> answer), deep inside code that has no handle on the HTTP request. A
``ContextVar`` set once per request — plus a logging filter that stamps it onto every record — lets a
single ``grep req=ab12cd34`` in the HF Space logs replay one request end to end, even if two visitors
overlap.

Implemented as a *pure ASGI* middleware (not ``BaseHTTPMiddleware``) on purpose: BaseHTTPMiddleware
runs the endpoint in a separate task, which famously drops ContextVars set before ``call_next``. A
pure ASGI middleware wraps ``await self.app(...)`` in the SAME task, so the value is still set while
the streaming response body runs — and Starlette copies that task context into the threadpool workers
that execute our blocking graph, so the graph's own log lines inherit the id too.
"""
from __future__ import annotations

import contextvars
import logging
import uuid

_req_id: contextvars.ContextVar[str] = contextvars.ContextVar("req_id", default="-")
_client_ip: contextvars.ContextVar[str] = contextvars.ContextVar("client_ip", default="-")


def current_request_id() -> str:
    return _req_id.get()


def current_client_ip() -> str:
    return _client_ip.get()


class RequestContextFilter(logging.Filter):
    """Stamp the current req id / client IP onto every record so the formatter can print them."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.req_id = _req_id.get()
        record.client_ip = _client_ip.get()
        return True


def _client_ip_from(headers: dict, scope) -> str:
    """Real visitor IP. HF Spaces sits behind a proxy, so scope["client"] is the proxy — the original
    client is the FIRST hop of X-Forwarded-For (proxies append, so index 0 is the closest to the user).
    Fall back to X-Real-IP, then the direct peer."""
    xff = headers.get(b"x-forwarded-for")
    if xff:
        return xff.decode("latin-1").split(",")[0].strip()
    real = headers.get(b"x-real-ip")
    if real:
        return real.decode("latin-1").strip()
    client = scope.get("client")
    return client[0] if client else "-"


class RequestContextMiddleware:
    """Assign (or adopt) a request id + capture the client IP for the duration of each HTTP request,
    and echo the id back in the ``X-Request-Id`` response header so the frontend can log the same id."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        incoming = headers.get(b"x-request-id")
        # Adopt the frontend's id if it sent one (ties the browser console to these logs); else mint one.
        req_id = incoming.decode("latin-1")[:64] if incoming else uuid.uuid4().hex[:8]
        ip = _client_ip_from(headers, scope)

        id_token = _req_id.set(req_id)
        ip_token = _client_ip.set(ip)

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                message["headers"].append((b"x-request-id", req_id.encode("latin-1")))
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            _req_id.reset(id_token)
            _client_ip.reset(ip_token)
