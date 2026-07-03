"""Episodic memory store — a "soft cache" of past answered questions (Module 4, slice 1).

Records each answered question as a {question -> answer} episode so a later equivalent question
can be answered by recall instead of redoing the work. Reading is a small RAG problem: embed
the incoming question, find the most-similar past episode.

Deliberately a JSON file + brute-force cosine, kept SEPARATE from the corpus — at a handful of
episodes an O(n) scan is trivial and a vector index would be value-less here. Episode embeddings
are unit-normalized (LocalEmbedder returns them so), so cosine similarity is a plain dot product.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EpisodicStore:
    """A persistent list of past-question episodes with cosine recall over the question text.

    Each episode is a dict: {question, answer, meta, embedding, written_at}. The embedding keys
    on the QUESTION (recall matches questions, not answers). Persisted as one JSON file.
    """

    def __init__(self, path, embedder):
        self.path = Path(path)
        self.embedder = embedder
        self.episodes: List[dict] = []
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────────────────
    def _load(self) -> None:
        """Read episodes from disk (empty if the file doesn't exist yet)."""
        if self.path.exists():
            self.episodes = json.loads(self.path.read_text(encoding="utf-8"))
            logger.debug("episodic memory: loaded %d episode(s) from %s", len(self.episodes), self.path)
        else:
            self.episodes = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.episodes, indent=2, ensure_ascii=False), encoding="utf-8")

    def reload(self) -> None:
        """Drop in-memory state and re-read from disk (simulates a process restart)."""
        self.episodes = []
        self._load()

    def clear(self) -> None:
        """Wipe the store (memory + file). Used to start each ON/OFF eval run from empty."""
        self.episodes = []
        if self.path.exists():
            self.path.unlink()

    # ── write ────────────────────────────────────────────────────────────────────────────
    def write(self, question: str, answer: str, meta: Optional[dict] = None) -> None:
        """Record one answered question as an episode (embeds the QUESTION for later recall)."""
        episode = {
            "question": question,
            "answer": answer,
            "meta": meta or {},
            "embedding": self.embedder.embed_query(question),
            "written_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.episodes.append(episode)
        self._save()
        logger.debug("episodic memory: wrote episode (now %d) for q=%.60s", len(self.episodes), question)

    # ── read ─────────────────────────────────────────────────────────────────────────────
    def read(self, question: str, k: int = 1) -> List[Tuple[dict, float]]:
        """Return the top-k most-similar past episodes as (episode, similarity), best first."""
        if not self.episodes:
            return []
        query_vector = np.array(self.embedder.embed_query(question))
        scored: List[Tuple[dict, float]] = []
        for episode in self.episodes:
            similarity = float(np.dot(query_vector, np.array(episode["embedding"])))
            scored.append((episode, similarity))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def __len__(self) -> int:
        return len(self.episodes)
