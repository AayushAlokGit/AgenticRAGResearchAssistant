# Reranking & Two-Stage Retrieval

> A portable pattern. It applies to **any system that must pick a few best items out of
> many** — web search, recommendations, ad ranking, RAG retrieval. This project's
> cross-encoder reranker (DD-014) is the worked example in the last section.
> See also `VECTOR_DB_INTERNALS.md` (the bi-encoder/vector side) and
> `../evals/EVALUATION_PRINCIPLES.md` (how we proved this earned its place).

The trap this doc exists to prevent: assuming one retrieval model can be both **fast
enough to scan everything** and **accurate enough to rank the winners**. It can't — those
pull in opposite directions. The resolution is to use *two* models in sequence.

## 1. The tension: recall-at-scale vs precision

Every "find the best matches" system faces the same conflict:

- To search **millions** of items in milliseconds, scoring has to be *cheap and
  precomputable* — you can't run a heavy model against every item at query time.
- To rank the **top handful** correctly, scoring has to be *accurate* — which means an
  expensive model that looks hard at the query and the item together.

You cannot have both in one pass. So you **stage** it:

1. **Retrieve (stage 1) — recall-oriented.** A cheap, scalable scorer casts a wide net and
   returns a *candidate pool* (e.g. top-20–100). Its only job: get the right items
   *somewhere* in the pool. It will mis-*order* them — that's fine.
2. **Rerank (stage 2) — precision-oriented.** An expensive, accurate scorer re-scores only
   that small pool and picks the true top-k.

Cheap-and-wide narrows the field; expensive-and-precise cleans up the ordering. This is
**the** canonical structure of large-scale ranking, not a RAG trick.

## 2. The two architectures (the heart of it)

The stages differ because they use two fundamentally different ways to score a
(query, item) pair. The example here is text, but the shapes generalize to any sentence-
pair / query-item task (entailment, duplicate detection, reward models).

**Bi-encoder (stage 1) — encode each side ALONE, compare vectors.**
```
query ─► [model] ─► vec_q ┐
                          ├─► cosine(vec_q, vec_d) → score
item  ─► [model] ─► vec_d ┘     (two separate passes)
```
The query never sees the item during encoding. You compress each item into one fixed
vector *before knowing the query* — which is exactly what makes it **indexable**:
precompute every item vector once, store it, and at query time do cheap vector math. The
price: that vector is a **lossy, query-blind summary**. A detail the query cares about may
have been averaged away during pooling.

**Cross-encoder (stage 2) — encode the pair TOGETHER, score jointly.**
```
[CLS] query [SEP] item [SEP] ─► [model] ─► pooled vector ─► linear head ─► ONE number
                                (one pass over BOTH together)
```
Query and item are concatenated into one sequence. Now, inside every self-attention layer,
**every query token can attend to every item token and vice versa** — the model can form
"does *this phrase* in the question line up with *that phrase* in the passage?" as an
internal computation. That cross-attention is the entire reason it's more accurate.

**The catch that forces two stages:** the cross-encoder's representation depends on the
query, so there is **nothing to precompute or index** — you must run a fresh forward pass
for *every* (query, item) pair, *at query time*. Affordable on 20 candidates; impossible on
20 million. Hence: bi-encoder to recall, cross-encoder to rank.

## 3. What the reranker score actually is

A cross-encoder ends in a **head with a single output** (`num_labels=1`) that a plain
embedding model does not have. For each (query, item) pair, inference:

1. tokenizes the pair as `[CLS] query [SEP] item [SEP]` (with a max length — long items get
   **truncated**, so the reranker only sees the head of an over-long chunk);
2. runs one joint transformer forward pass (the cross-attention step);
3. pools (e.g. the `[CLS]` hidden state) and passes it through the linear head;
4. emits **one scalar logit** — higher = more relevant. It is *not* a calibrated
   probability: unbounded, often negative. Ranking only needs the *relative* order, so raw
   logits are fine (a negative top score can still be rank 1).

**Where "relevance" comes from:** the head's weights were **fine-tuned on a labeled ranking
dataset** (e.g. MS MARCO — real queries each paired with human-judged relevant/irrelevant
passages). So the number encodes a *learned, supervised* notion of "does this item answer
this query" — strictly more informative than the geometric cosine distance stage 1 uses.
That's why a reranker routinely fixes "right document retrieved, wrong passage ranked top."

**Important observation — it scores RELEVANCE, not ANSWERABILITY or CORRECTNESS.** What the
model learned is *"is this passage a good match for this query"* — strong topical/semantic
alignment between question and passage. That correlates heavily with "contains the answer"
(which is exactly why it surfaces fact-bearing chunks), but it is **not** a fact-checker: it
does not verify that the passage's claims are *true*, nor that the eventual answer is
*correct* or *grounded*. A confidently-written but wrong passage can still score high. Those
are different jobs handled by different machinery — *answer correctness* (vs a reference) and
*faithfulness* (grounded in the retrieved context) are scored by the answer-layer evals, not
by the reranker (see `../evals/ANSWER_QUALITY.md`). Keep the boundary clear: **the reranker's
job is to put the most relevant passages in front of the generator; judging whether the
resulting answer is right is a separate stage.**

## 4. One backbone, different head + training

A reranker is often the **same transformer family** as the embedder, specialized two ways:
**(a)** a different *head* (single-score vs vector output) and **(b)** a different
*fine-tuning task* (relevance ranking vs general similarity). This is a deeply transferable
idea — the same base model becomes a chat model, a code model, a reward model, or a
reranker depending on the head and the objective. "Reranker" is a *category* of model, with
local options (`ms-marco-MiniLM`, `BGE-reranker`) and hosted APIs (Cohere Rerank, Voyage).
They share one conceptual interface — **give me (query, candidates) → re-scored candidates**
— so they're swappable behind a thin wrapper.

## 5. The same pattern outside RAG

- **Web search:** an inverted index / BM25 recalls thousands of pages; a learning-to-rank
  model reorders the first page.
- **Recommendations:** a cheap "candidate generation" model pulls a few hundred items from
  millions; a heavy "ranking" model scores those few hundred per user.
- **Ads / feed:** retrieval → ranking → (often) a final re-ranking for business rules.

Different domains, identical skeleton: **cheap recall → expensive precision.**

## 6. Costs, knobs, and failure modes

- **`candidate_k`** — how many stage 1 hands stage 2. Bigger = more chances to recover a
  buried item, but linearly more forward passes (latency). The reranker can only promote
  what stage 1 *recalled* — if the right item isn't in the pool, no reranker can save it.
  So `candidate_k` trades latency for ceiling.
- **Latency** — one model pass *per candidate*, at query time, on the live path. The whole
  reason you don't just rerank everything. In production: cap `candidate_k`, batch, use a
  small/distilled reranker, or a hosted rerank API.
- **Truncation** — long items get cut to the model's max length; the fact may sit past the
  cutoff. Chunk size and reranker max length interact.
- **Metric saturation** — once reranking lifts recall@k to ~1.0, that metric stops
  discriminating; lean on rank-sensitive metrics (MRR/nDCG) and the downstream answer eval.

## 7. Applying this to a new system (the transferable rule)

1. **Are you picking a few best out of many, with a quality-vs-scale tension?** If yes, you
   want two stages — don't try to make one model do both.
2. **Stage 1 = recall + speed.** Optimize for "the right item is *in the pool*," cheaply.
   Measure it with recall@candidate_k. A miss here is unrecoverable downstream.
3. **Stage 2 = precision.** A joint/cross model (or a learned ranker) reorders the pool.
   Measure the *ordering* (MRR/nDCG), not just set membership.
4. **Tune `candidate_k` as the latency↔quality dial**, and watch for truncation.
5. **Keep the reranker behind a thin interface** so local vs hosted is one config change.

## 8. In this project (the worked instance)

`rag/rerank.py` (`CrossEncoderReranker`) + `RerankRetriever` (a wrapper in `retriever.py`
that composes over dense *or* hybrid). Stage 1 = our hybrid (dense `all-MiniLM-L6-v2` +
BM25, fused by RRF) returns `candidate_k=20`; stage 2 = `cross-encoder/ms-marco-MiniLM-L-6-v2`
re-scores those 20 down to `top_k=5`.

Eval-gated A/B vs hybrid alone (DD-014) won every cutoff:

| metric | hybrid | hybrid + rerank |
|---|---|---|
| recall@1 | 0.724 | 0.759 |
| recall@3 | 0.793 | 0.931 |
| recall@5 | 0.897 | **1.000** |
| MRR | 0.903 | 0.948 |

End-to-end (answer-correctness), reranking **eliminated all three false abstentions**
(q01/q25/q26): the fact-bearing chunk was in the candidate pool but ranked below top-5;
the cross-encoder lifted it into the top-5, so the generator could finally answer. That is
the chunk-level fix document-level recall couldn't even see — the precise problem stage 2
exists to solve.
