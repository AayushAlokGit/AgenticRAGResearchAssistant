"""LLM response caching — memoize deterministic complete() calls (Module 5, harness).

Caching is plain MEMOIZATION: skip the work on a repeated call by storing input -> output.
For an LLM call that reuse is only LOSSLESS when the call is deterministic (temperature 0),
so the cached answer equals what a fresh call would return; we therefore treat the cache as
safe only in that regime (the eval runs at temp 0). At temp > 0 a cache would freeze one
random sample and serve it forever — a diversity-for-cost trade, not a free win.

The KEY is the identity of the computation: provider + model + temperature + max_tokens +
the exact messages. Because the messages are hashed VERBATIM — and the controller's messages
already contain the question, history, and gathered evidence rendered into text — editing a
prompt or changing the evidence yields a different key and a clean MISS. Invalidation is
automatic: if it's in the prompt, it's in the key; you never bust the cache by hand.

The win is mostly CROSS-RUN. At temp 0 an eval re-run replays the whole deterministic
trajectory from cache — near-free and instant — which is what makes the change -> measure loop fast.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

from agentic_rag.llm.provider import Completion, Message, Usage

logger = logging.getLogger(__name__)


def cache_key(provider_name: str, model: str, temperature: float, max_tokens: int,
              messages: List[Message]) -> str:
    """Content-addressed key: sha256 over everything that determines the output.

    Miss an input that affects the output -> false hits (a wrong cached answer served);
    include an irrelevant nonce -> never hits. So the key is EXACTLY the generation inputs.
    """
    payload = {
        "provider": provider_name,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    # sort_keys makes the encoding stable across dict orderings; the messages LIST order is
    # preserved (order is semantically meaningful — it's the conversation).
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class SqliteCache:
    """On-disk backend: a sqlite table key -> (text, total_tokens). Persists ACROSS runs, so an
    eval re-run at temp 0 replays from cache — the real win.

    sqlite (not the JSON precedent) on purpose: it appends a row per call and is written by
    concurrent eval runs. JSON would rewrite the whole file each put and two runs would clobber
    each other; sqlite does row-level writes, and WAL + a busy timeout let concurrent runs
    read/write the same file safely. Content-addressed keys make one shared file safe to reuse.
    """

    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the connection may be touched from the thread that built the
        # router vs. the one that calls it; our access is serial, so this is safe.
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")     # readers don't block the writer
        self._conn.execute("PRAGMA busy_timeout=30000")   # wait, don't raise, on a locked db
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "key TEXT PRIMARY KEY, text TEXT NOT NULL, total_tokens INTEGER NOT NULL)")
        self._conn.commit()

    def get(self, key: str) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT text, total_tokens FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return {"text": row[0], "total_tokens": row[1]}

    def put(self, key: str, text: str, total_tokens: int) -> None:
        # INSERT OR REPLACE: a re-store of the same key (same inputs -> same output) is a no-op
        # overwrite, so a race between two runs writing the same entry is harmless.
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, text, total_tokens) VALUES (?, ?, ?)",
            (key, text, total_tokens))
        self._conn.commit()


class CachedProvider:
    """Decorator that memoizes a wrapped provider's complete(messages).

    The decorator PATTERN: add a cross-cutting behavior (caching) without touching the wrapped
    provider or the router. Because each provider knows its own name/model/params, the key is
    correct per tier automatically — the router can wrap every tier and fallback still works
    (a Groq answer can never be served as if it were Gemini).

    Caches on SUCCESS only: an exception from the inner provider propagates untouched, so the
    existing per-provider retries and router fallback are unaffected.
    """

    def __init__(self, inner, cache):
        self.inner = inner
        self.cache = cache
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0

    @property
    def name(self) -> str:
        return self.inner.name

    def complete(self, messages: List[Message]) -> Completion:
        key = cache_key(self.inner.name, self.inner.model, self.inner.temperature,
                        self.inner.max_tokens, messages)
        start = time.perf_counter()
        record = self.cache.get(key)
        if record is not None:
            lookup_s = time.perf_counter() - start
            self.hits += 1
            self.tokens_saved += record["total_tokens"]
            logger.debug("cache HIT (%s/%s): saved %d tokens in %.4fs",
                         self.inner.name, self.inner.model, record["total_tokens"], lookup_s)
            # A hit is no API round-trip: report ZERO tokens so the run's cost reflects real
            # spend (tokens_saved tracks the would-have-cost). latency_s is the real lookup time
            # (~0), so a fully-cached re-run honestly shows near-zero wall-clock, not a fake 0.
            return Completion(text=record["text"], usage=Usage(latency_s=lookup_s))
        self.misses += 1
        completion = self.inner.complete(messages)
        self.cache.put(key, completion.text, completion.usage.total_tokens)
        return completion
