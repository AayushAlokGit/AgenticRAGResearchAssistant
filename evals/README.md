# Evals

A first-class, build-early concern — see `DESIGN_DECISIONS.md` DD-001 for *why* this
exists before the fancy machinery.

The job of this directory: make "did that change help?" an answerable question, and
turn this system's **silent failures** (fluent wrong answers that look like fluent right
answers) into loud ones.

> New here? Read `../docs/evals/EVALUATION_PRINCIPLES.md` first — it's the **portable mental model**
> (decompose, one-dataset-many-meters, the three metric families, the scoring-cost
> ladder) written to transfer to *any* compound AI system, with this project as the
> worked example. This README is just how that model is staged for *this* build.

## The eval set shape (start small)

`datasets/seed.yaml` — ~10 hand-written questions over the seed corpus. For each:

- `question` — the natural-language query.
- `expected_sources` — which document(s) *should* be retrieved to answer it.
  This alone gives **retrieval recall** with no LLM judge — just a set-membership check.
- `match` — `any` (retrieving one listed source counts; use when a fact is duplicated
  across docs) or `all` (every listed source required; for true multi-hop, so partial
  retrieval scores as a partial failure). Disambiguates how recall is computed.
- `should_abstain` — `true` marks an out-of-corpus question (`expected_sources: []`);
  success means the system answers "not enough info" rather than hallucinating.
- `expected_answer` (optional) — a reference answer, for later answer-quality scoring.

Include **2–3 out-of-corpus questions** whose answer is *not* in the corpus. These test
whether the system correctly **abstains** ("not enough info") instead of hallucinating —
most people forget these, and they are gold.

## Scoring, added in layers

1. **Retrieval** (built): recall@k via `expected_sources` membership + MRR. Cheap, no LLM.
   Code: `src/agentic_rag/evals/retrieval.py` — run `python -m agentic_rag.evals.retrieval`.
2. **Answer quality** (later): faithfulness / groundedness + answer relevance via
   LLM-as-judge, then Ragas.
3. **System cost** (later): step count, tokens, latency per question.

We report recall at **several** cutoffs (@1/@3/@5) plus **MRR**, not a single recall@5,
because recall@5 over an 11-doc corpus saturates at 1.0 — a metric pinned at the ceiling
can't show improvement, only regressions. The stricter cutoffs and ranking-sensitive MRR
are what discriminate. Each run writes a JSON to `eval_runs/` (gitignored) for diffing.

Run the set before/after every meaningful change and keep the score deltas. The
change -> measure -> keep-or-revert loop is the real skill this project is practicing.

Status: **`datasets/seed.yaml` v2 (32 questions: 29 answerable + 3 abstention) over the
frozen 11-doc corpus, and the retrieval scorer, both exist.** v2 hardened the set
(q13–q32) so recall@5 no longer saturates. Baselines over the 29 recall-eligible
questions (MiniLM, 800/100 char chunks):

| mode | recall@1 | recall@3 | recall@5 | MRR |
|------|----------|----------|----------|-----|
| dense (v2 baseline) | 0.621 | 0.759 | 0.862 | 0.843 |
| **hybrid (current, DD-009)** | **0.759** | **0.862** | 0.862 | **0.931** |

Hybrid (dense + BM25 + RRF) won the A/B and is now the default. Remaining failures to
tune against next (reranking): q05, q25 (slipped out of top-5 under hybrid), and
multi-hop q26/q29. Answer-quality + cost layers not built yet.
