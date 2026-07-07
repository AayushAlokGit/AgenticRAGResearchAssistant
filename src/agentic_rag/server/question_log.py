"""Durable question log — a record of every question the demo answers (analytics sink).

The twin of ``DailyBudget`` (server/budget.py): a small, lazily-opened Postgres-backed sink
written ONCE per answered question, at the same ``/ask`` seam. Durable in Neon so it survives the
ephemeral HF host — the same reason the vector store and the daily-spend counter live in Postgres.

Records METADATA only (question text + run stats), by choice: no answer body, no client IP. It is a
best-effort sink — the caller wraps ``record`` so a logging failure can never break a response the
user has already received (instrumentation must not degrade the primary flow).
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class QuestionLog:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool = None  # opened + schema-ensured on first use (lazy, like DailyBudget)

    @property
    def pool(self):
        if self._pool is None:
            from psycopg_pool import ConnectionPool

            self._pool = ConnectionPool(self._dsn, min_size=0, max_size=2, open=True,
                                        kwargs={"autocommit": True})
            with self._pool.connection() as conn:
                conn.execute(
                    "create table if not exists questions ("
                    " id bigserial primary key,"
                    " asked_at timestamptz not null default now(),"  # server-side clock (see budget.py)
                    " req_id text,"                                  # ties a row to the stdout logs
                    " scope text,"
                    " question text not null,"
                    " rounds integer,"
                    " exit_reason text,"
                    " total_tokens integer,"
                    " latency_ms integer)")
            logger.info("question log ready (table=questions)")
        return self._pool

    def record(self, question: str, *, scope: str = "both", req_id: Optional[str] = None,
               rounds: Optional[int] = None, exit_reason: Optional[str] = None,
               total_tokens: Optional[int] = None, latency_ms: Optional[int] = None) -> None:
        """Persist one answered question (metadata only). Best-effort: the caller catches, so a
        write failure is logged and swallowed rather than surfaced to the user."""
        with self.pool.connection() as conn:
            conn.execute(
                "insert into questions"
                " (req_id, scope, question, rounds, exit_reason, total_tokens, latency_ms)"
                " values (%s, %s, %s, %s, %s, %s, %s)",
                (req_id, scope, question, rounds, exit_reason, total_tokens, latency_ms))

    def recent(self, limit: int = 50) -> list:
        """Return the most recent questions, newest first, as plain dicts (for the /questions read
        endpoint). ``asked_at`` is serialized to ISO-8601 so it survives JSON."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "select asked_at, req_id, scope, question, rounds, exit_reason,"
                    " total_tokens, latency_ms from questions order by asked_at desc limit %s",
                    (limit,))
                rows = cur.fetchall()
        return [
            {"asked_at": r[0].isoformat(), "req_id": r[1], "scope": r[2], "question": r[3],
             "rounds": r[4], "exit_reason": r[5], "total_tokens": r[6], "latency_ms": r[7]}
            for r in rows
        ]

    def count(self) -> int:
        """Total questions recorded (all time)."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("select count(*) from questions")
                return int(cur.fetchone()[0])
