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

import logging
from typing import List

logger = logging.getLogger(__name__)


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

            logger.info("loading embedding model %s (first run downloads ~80MB) ...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.debug("embedding model loaded")
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


def _l2_normalize(vec: List[float]) -> List[float]:
    """Unit-normalize so a dot product equals cosine similarity (matches the store's cosine space
    and LocalEmbedder). Also required when truncating an embedding via output_dimensionality."""
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]


class GeminiEmbedder:
    """Encodes text via the Gemini embeddings API (google-genai) — same interface as LocalEmbedder.

    Uses TASK TYPES (the RAG-correct move): documents are embedded as RETRIEVAL_DOCUMENT and queries
    as RETRIEVAL_QUERY, which Google tunes differently for asymmetric search. Vectors are L2-normalized
    to match the store's cosine space. No torch — a network call per batch instead of a local model.
    """

    _BATCH = 100  # texts per request (stay under the API's per-call limit)

    def __init__(self, model_name: str, dims: int = 768):
        self.model_name = model_name          # e.g. "text-embedding-004"
        self.dims = dims
        self._client = None                   # constructed on first use (lazy, like LocalEmbedder)

    @property
    def client(self):
        if self._client is None:
            import os

            from dotenv import load_dotenv
            from google import genai

            load_dotenv()
            self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
            logger.info("Gemini embedder ready (model=%s, dims=%d)", self.model_name, self.dims)
        return self._client

    def _embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        from google.genai import types

        out: List[List[float]] = []
        for start in range(0, len(texts), self._BATCH):
            batch = texts[start:start + self._BATCH]
            resp = self.client.models.embed_content(
                model=self.model_name, contents=batch,
                config=types.EmbedContentConfig(task_type=task_type,
                                                output_dimensionality=self.dims))
            out.extend(_l2_normalize(list(e.values)) for e in resp.embeddings)
        return out

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents (RETRIEVAL_DOCUMENT)."""
        if not texts:
            return []
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query (RETRIEVAL_QUERY — the asymmetric counterpart to the docs)."""
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def build_embedder(config: dict):
    """Pick the embedder from config: `embedding.provider` = local | gemini. One place so the 5
    construction sites don't each branch — the same 'shape it so a swap is cheap' move as the LLM layer."""
    emb_cfg = config["embedding"]
    provider = emb_cfg.get("provider", "local")
    if provider == "gemini":
        return GeminiEmbedder(emb_cfg["gemini_model"], dims=emb_cfg.get("dims", 768))
    return LocalEmbedder(emb_cfg["model"])
