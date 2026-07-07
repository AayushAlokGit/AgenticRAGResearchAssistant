"""Push notifier — a real-time sink that pings your phone when someone asks a question (ntfy.sh).

The real-time twin of ``QuestionLog``: instead of persisting the event for later, it fires a single
HTTP POST to an ntfy.sh topic, which the ntfy mobile app turns into a push notification. Same
instrumentation discipline as the log — fire-and-forget on a daemon thread so it adds ZERO latency to
the request, and wrapped so a failed push can never touch the response.

Two guards keep it from becoming a nuisance:
  - THROTTLE: at most one push per ``min_interval_s``, so a burst (or a bot) can't machine-gun your
    phone. Throttling drops only the PUSH — ``QuestionLog`` still records every question, so no data
    is lost, only the buzzing is rate-limited.
  - dependency-free: stdlib ``urllib``, no new package (matches the lightweight ethos).

Enabled by env: ``NTFY_TOPIC`` (an unguessable topic name you subscribe to in the app). Optional:
``NTFY_SERVER`` (default https://ntfy.sh) and ``NTFY_MIN_INTERVAL_SECONDS`` (default 60).
"""
from __future__ import annotations

import logging
import threading
import time
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

_TIMEOUT_S = 5.0
_MAX_BODY_CHARS = 500   # questions are already capped at 500 upstream; belt-and-braces


class PushNotifier:
    def __init__(self, topic: str, server: str = "https://ntfy.sh", min_interval_s: float = 60.0):
        self.url = f"{server.rstrip('/')}/{topic}"
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last_sent = 0.0   # monotonic time of the last push we let through (0 = never)

    def notify_question(self, question: str, req_id: Optional[str] = None) -> None:
        """Throttled, fire-and-forget push that someone asked a question. Returns immediately."""
        if not self._allow():
            logger.info("push notification throttled (min_interval=%.0fs) — skipped", self.min_interval_s)
            return
        body = (question or "").strip()[:_MAX_BODY_CHARS] or "(empty question)"
        if req_id:
            body = f"{body}\n\nreq {req_id}"
        # Daemon thread: the POST (network I/O, up to _TIMEOUT_S) never blocks the request path.
        threading.Thread(target=self._send, args=("New question - Agentic RAG", body),
                         daemon=True).start()

    def _allow(self) -> bool:
        """True if we're past the cooldown; stamps the send time. Atomic under the lock, so a
        concurrent burst lets exactly one push through per window."""
        with self._lock:
            now = time.monotonic()
            if now - self._last_sent < self.min_interval_s:
                return False
            self._last_sent = now
            return True

    def _send(self, title: str, body: str) -> None:
        """Blocking ntfy POST — runs on the daemon thread. Best-effort: any failure is logged and
        swallowed so instrumentation can't break anything. Title stays ASCII (HTTP headers are
        latin-1); the question rides in the UTF-8 body where any character is safe."""
        try:
            request = urllib.request.Request(
                self.url, data=body.encode("utf-8"), method="POST",
                headers={"Title": title, "Tags": "speech_balloon"})
            urllib.request.urlopen(request, timeout=_TIMEOUT_S).close()
        except Exception:
            logger.exception("push notification failed (ignored)")
