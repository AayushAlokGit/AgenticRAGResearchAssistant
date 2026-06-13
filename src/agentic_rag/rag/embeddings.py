"""Local embedding model wrapper (all-MiniLM-L6-v2 by default).

A thin wrapper over ``sentence-transformers`` so the rest of the pipeline depends on a
small interface (``embed`` / ``embed_query``) rather than the library directly — the
same "shape it so a swap is cheap later" move used for the LLM layer.

Embeddings are local by necessity: Groq has no embeddings endpoint (DD-004), and a
22M-param CPU model is the lightweight choice for limited hardware. The model (~80MB)
downloads once on first use and is cached by sentence-transformers under ``~/.cache``.
Loading is lazy, so merely importing this module stays cheap.
"""
from __future__ import annotations

from typing import List


class LocalEmbedder:
    """Encodes text into unit-normalized vectors with a local SentenceTransformer."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None  # loaded on first use

    @property
    def model(self):
        if self._model is None:
            # Imported lazily so the heavy dependency (torch) stays out of the import
            # path until embeddings are actually needed.
            from sentence_transformers import SentenceTransformer

            print(f"[embedder] loading {self.model_name} (first run downloads ~80MB) ...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents.

        Returns unit-normalized vectors so that a dot product equals cosine similarity,
        matching the vector store's cosine space.
        """
        if not texts:
            return []
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self.embed([text])[0]
