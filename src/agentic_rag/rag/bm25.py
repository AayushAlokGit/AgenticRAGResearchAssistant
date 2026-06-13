"""BM25 sparse (lexical) retrieval — via the ``rank_bm25`` library.

BM25 ("Best Matching 25") is the classic keyword ranking function from information
retrieval. Unlike dense embeddings, which match MEANING, BM25 matches exact TOKENS — so
it shines on rare/specific terms (acronyms, model names, identifiers) that an embedder
might blur. We use the well-tested ``rank_bm25`` library rather than hand-rolling the
formula; the comments below explain what it's doing internally so it isn't a black box.

How ``rank_bm25``'s ``BM25Okapi`` scores a chunk d for a query q — a sum over query terms t:

    score(d, q) = Σ_t  IDF(t) · ( f(t,d) · (k1 + 1) )
                        ─────────────────────────────────────────────
                        ( f(t,d) + k1 · (1 − b + b · |d| / avgdl) )

where
    f(t,d)  = how many times term t appears in chunk d (term frequency)
    IDF(t)  = inverse document frequency — rarer terms across the corpus weigh more
    |d|     = length of chunk d in tokens, avgdl = average chunk length
    k1=1.5  = term-frequency saturation (the 10th occurrence adds less than the 1st)
    b=0.75  = length normalization (a long chunk can't score high just by being long)

Two library-specific internals worth knowing:
  - BM25Okapi's IDF can go NEGATIVE for terms that appear in more than half the corpus.
    rank_bm25 floors those at ``epsilon · average_idf`` (epsilon=0.25 by default) so very
    common words don't actively subtract from a chunk's score.
  - The index precomputes all term/document frequencies, IDFs, and avgdl up front in the
    constructor, so each ``get_scores`` call is just a fast scoring pass over the corpus.

Tokenization is ours (lowercase, split on non-alphanumeric); rank_bm25 takes pre-tokenized
input. No stemming / stopword removal yet — those are later, eval-gated upgrades.
"""
from __future__ import annotations

import re
from typing import List

from rank_bm25 import BM25Okapi

from agentic_rag.rag.vector_store import Hit


def tokenize(text: str) -> List[str]:
    """Lowercase and split into alphanumeric tokens. (No stemming / stopwords yet.)"""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, chunks: List[dict]):
        # chunks: list of {source, chunk_index, text}
        self.chunks = chunks

        tokenized_corpus = []
        for chunk in chunks:
            tokenized_corpus.append(tokenize(chunk["text"]))

        # Building the index here precomputes term/doc frequencies, IDFs and avgdl, so
        # every later query is just a scoring pass. Uses BM25Okapi defaults (k1=1.5, b=0.75).
        self.bm25 = BM25Okapi(tokenized_corpus)

    def query(self, text: str, k: int) -> List[Hit]:
        query_tokens = tokenize(text)

        # get_scores returns one BM25 score per chunk, aligned to the corpus order we
        # passed in (index i here == self.chunks[i]).
        scores = self.bm25.get_scores(query_tokens)

        # Keep only chunks the query actually touched, then take the top k.
        scored = []
        for i in range(len(scores)):
            if scores[i] > 0.0:
                scored.append((i, float(scores[i])))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        hits = []
        for i, score in scored[:k]:
            chunk = self.chunks[i]
            hits.append(Hit(
                source=chunk["source"],
                chunk_index=chunk["chunk_index"],
                text=chunk["text"],
                score=score,
            ))
        return hits
