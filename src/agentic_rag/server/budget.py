"""Durable daily token budget — the demo's $0 kill-switch (harness abuse ring, deployed backend).

A public demo spends YOUR Gemini quota. This caps total tokens spent PER DAY across all visitors and,
crucially, PERSISTS the counter in Postgres — so restarting the ephemeral host can't reset it and let a
bot restart-farm the quota (an in-memory counter would). When the day's budget is reached, /ask stops
running the agent and returns a friendly "resting" message instead of spending. That is what turns
"free-tier" into a $0 *guarantee*: cap-to-degrade, not cap-to-charge.

Notes:
- `current_date` is evaluated SERVER-SIDE by Postgres, so the day boundary doesn't trust a container
  clock. (Rolls over at UTC midnight on Neon's default timezone.)
- Cache hits spend 0 tokens, so they never count here — they stretch how many distinct questions the
  budget buys before the demo rests.
- The check-before / add-after has a benign race under concurrency (a slight overshoot), which is fine:
  this is a runaway circuit-breaker, not exact accounting. The increment itself is atomic.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class DailyBudget:
    def __init__(self, dsn: str, budget_tokens: int):
        self.budget = budget_tokens  # 0 = off (no cap)
        self._dsn = dsn
        self._pool = None

    @property
    def pool(self):
        if self._pool is None:
            from psycopg_pool import ConnectionPool

            self._pool = ConnectionPool(self._dsn, min_size=0, max_size=2, open=True,
                                        kwargs={"autocommit": True})
            with self._pool.connection() as conn:
                conn.execute(
                    "create table if not exists daily_token_spend ("
                    "day date primary key, tokens bigint not null default 0)")
            logger.info("daily budget ready (cap=%d tokens/day)", self.budget)
        return self._pool

    def tokens_today(self) -> int:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("select tokens from daily_token_spend where day = current_date")
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def is_exhausted(self) -> bool:
        if self.budget <= 0:
            return False
        return self.tokens_today() >= self.budget

    def add(self, tokens: int) -> None:
        """Atomically add a run's token spend to today's total (upsert-increment)."""
        if tokens <= 0:
            return
        with self.pool.connection() as conn:
            conn.execute(
                "insert into daily_token_spend (day, tokens) values (current_date, %s) "
                "on conflict (day) do update set tokens = daily_token_spend.tokens + excluded.tokens",
                (tokens,))
