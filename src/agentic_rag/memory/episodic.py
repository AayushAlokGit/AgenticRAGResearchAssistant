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
            "kind": "episode",      # raw, written verbatim (vs "synthesis", produced by consolidate)
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

    def recall(self, question: str, threshold: float) -> Optional[Tuple[dict, float]]:
        """Best episode IF it clears the similarity threshold, else None."""
        top = self.read(question, k=1)
        if not top:
            return None
        episode, similarity = top[0]
        if similarity >= threshold:
            return episode, similarity
        return None

    # ── consolidate (episodic → semantic) ──────────────────────────────────────────────────
    def consolidate(self, llm, prompt: str) -> int:
        """Merge related RAW episodes into synthesis records via one LLM call. Returns # created.

        WHY this is LLM-driven, not a cosine clustering threshold: different FACETS of one subject
        are only ~0.4 similar to each other in question-embedding space, no more than two facets of
        DIFFERENT subjects are — so no scalar cutoff separates "same subject" from "different
        subject" (measured). The LLM judges subject identity; similarity is not used here at all.
        Consistent with the threshold-free recall design (DD-045).

        APPEND, not replace: the synthesis is added alongside its source episodes, NOT in place of
        them. Replacing was measured to REGRESS narrow-paraphrase recall (a buried facet question
        fell 0.945→0.244 against the broad synthesis); keeping both lets the unchanged top-k recall
        route each query for free — narrow→its facet, broad→the synthesis. Consolidation is a
        COVERAGE op; shrinking the store is FORGET's job (a separate operation, deferred). The
        synthesis keys (embeds) on its canonical QUESTION, so it is recalled by the same
        question→question path slice 1 validated.

        ONE STORE, discriminated by `kind`, IDEMPOTENT: consolidation reads only raw
        `kind == "episode"` records that have NOT already been consolidated, and on merging marks
        each source episode `meta.consolidated = True` (it stays in the store for narrow recall, it
        is just never re-fed to the merger). So re-running on an unchanged store is a no-op — no
        synthesis-of-synthesis, and no duplicate synthesis from the same sources.

        Scaling caveat (learning shortcut, not production): every eligible raw episode goes into one
        call. Fine at a handful; a real system would pre-cluster by embedding then LLM-merge per cluster.
        """
        raw = [ep for ep in self.episodes
               if ep.get("kind", "episode") == "episode" and not ep.get("meta", {}).get("consolidated")]
        if len(raw) < 2:
            return 0

        listing = self._render_for_consolidation(raw)
        completion = llm.complete([
            {"role": "system", "content": prompt},
            {"role": "user", "content": listing},
        ])
        groups = _parse_group_array(completion.text)
        if not groups:
            return 0

        merged_positions = set()     # positions into `raw` that ended up in a synthesis
        syntheses: List[dict] = []
        for group in groups:
            member_ids = group.get("member_ids") or []
            # member_ids are the 1-based [n] labels from the listing of eligible RAW episodes.
            positions = [n - 1 for n in member_ids if isinstance(n, int) and 1 <= n <= len(raw)]
            if len(positions) < 2:
                continue             # a "group" of one merges nothing — ignore it
            source_questions = [raw[p]["question"] for p in positions]
            syntheses.append(self._make_synthesis(group, source_questions))
            merged_positions.update(positions)

        if not syntheses:
            return 0

        # Mark the merged sources so a future consolidate won't re-merge them (raw[p] is the same
        # dict object as in self.episodes, so this updates the stored record in place).
        for position in merged_positions:
            raw[position].setdefault("meta", {})["consolidated"] = True
        self.episodes.extend(syntheses)   # APPEND — source episodes are kept (for narrow recall)
        self._save()
        logger.info("episodic memory: consolidated into %d synthesis record(s) (now %d total: %d raw + %d synthesis)",
                    len(syntheses), len(self.episodes), len(raw), len(self.episodes) - len(raw))
        return len(syntheses)

    def _render_for_consolidation(self, raw: List[dict]) -> str:
        """Number the raw episodes [1..n] as the consolidation prompt expects."""
        lines = []
        for i, episode in enumerate(raw, start=1):
            lines.append(f"[{i}] Q: {episode['question']}\n    A: {episode['answer']}")
        return "\n".join(lines)

    def _make_synthesis(self, group: dict, source_questions: List[str]) -> dict:
        """Build one synthesis episode from an LLM merge group (keyed on its canonical question)."""
        question = group["question"]
        return {
            "kind": "synthesis",    # distilled by consolidate; excluded from future consolidation input
            "question": question,
            "answer": group["answer"],
            "meta": {"consolidated_from": source_questions},
            "embedding": self.embedder.embed_query(question),
            "written_at": datetime.now().isoformat(timespec="seconds"),
        }

    def __len__(self) -> int:
        return len(self.episodes)


def _parse_group_array(raw: str) -> list:
    """Pull the JSON array of synthesis groups out of the LLM reply (tolerates ``` fences/prose).

    The prompt asks for a bare JSON array; we slice from the first '[' to the last ']' and decode.
    Returns [] on anything unparseable (a bad consolidation must be a no-op, never a crash)."""
    if not raw:
        return []
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        value = json.loads(raw[start:end + 1])
    except (ValueError, TypeError):
        return []
    return value if isinstance(value, list) else []
