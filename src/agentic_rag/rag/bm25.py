"""BM25 sparse (lexical) retrieval — hand-rolled.

BM25 ("Best Matching 25") is the classic keyword ranking function from information
retrieval. Unlike dense embeddings, which match MEANING, BM25 matches exact TOKENS — so
it shines on rare/specific terms (acronyms, model names, identifiers) that an embedder
might blur. We hand-roll it (rather than pull in rank_bm25) because it's a foundational
algorithm worth understanding, like the chunker and tracker we wrote by hand.

The score of a chunk d for a query q is a sum over the query's terms t:

    score(d, q) = Σ_t  IDF(t) · ( f(t,d) · (k1 + 1) )
                        ─────────────────────────────────────────────
                        ( f(t,d) + k1 · (1 − b + b · |d| / avgdl) )

where
    f(t,d)  = how many times term t appears in chunk d (term frequency)
    IDF(t)  = inverse document frequency — rarer terms across the corpus weigh more
    |d|     = length of chunk d in tokens
    avgdl   = average chunk length
    k1, b   = tuning knobs: k1 controls term-frequency saturation (~1.5),
              b controls length normalization (0..1, ~0.75).

Two intuitions baked into the formula:
  - IDF: a term in every chunk (e.g. "the") tells you nothing; a rare term is informative.
  - Saturation + length-norm: the 10th occurrence of a word adds less than the 1st, and a
    long chunk isn't allowed to score high just by being long.

Naive on purpose: tokenization is lowercase + split on non-alphanumeric, with no stemming
or stopword removal (those are later, eval-gated upgrades).
"""
from __future__ import annotations

import math
import re
from typing import List

from agentic_rag.rag.vector_store import Hit


def tokenize(text: str) -> List[str]:
    """Lowercase and split into alphanumeric tokens. (No stemming / stopwords yet.)"""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, chunks: List[dict], k1: float = 1.5, b: float = 0.75):
        # chunks: list of {source, chunk_index, text}
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        # Tokenize every chunk once, and record its length in tokens.
        self.doc_tokens = []
        self.doc_length = []
        for chunk in chunks:
            tokens = tokenize(chunk["text"])
            self.doc_tokens.append(tokens)
            self.doc_length.append(len(tokens))

        self.num_docs = len(chunks)
        if self.num_docs > 0:
            self.avg_length = sum(self.doc_length) / self.num_docs
        else:
            self.avg_length = 0.0

        # Term frequency per chunk: {term: count}.
        self.term_freqs = []
        for tokens in self.doc_tokens:
            counts = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self.term_freqs.append(counts)

        # Document frequency: in how many chunks does each term appear at least once.
        self.doc_freq = {}
        for counts in self.term_freqs:
            for term in counts:
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

    def idf(self, term: str) -> float:
        """Inverse document frequency with the standard BM25 smoothing (+0.5), and a +1
        inside the log so the value stays non-negative even for very common terms."""
        n_with_term = self.doc_freq.get(term, 0)
        return math.log((self.num_docs - n_with_term + 0.5) / (n_with_term + 0.5) + 1.0)

    def query(self, text: str, k: int) -> List[Hit]:
        query_terms = tokenize(text)

        # Precompute IDF once per DISTINCT query term (iterating unique terms also avoids
        # double-counting a term that appears twice in the question).
        idf_by_term = {}
        for term in set(query_terms):
            idf_by_term[term] = self.idf(term)

        # Score every chunk against the query.
        scored = []
        for i in range(self.num_docs):
            term_freq = self.term_freqs[i]
            length = self.doc_length[i]

            score = 0.0
            for term in idf_by_term:
                if term not in term_freq:
                    continue
                f = term_freq[term]
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                score += idf_by_term[term] * numerator / denominator

            if score > 0.0:
                scored.append((i, score))

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
