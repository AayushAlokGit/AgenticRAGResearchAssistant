# Evals

A first-class, build-early concern — see `DESIGN_DECISIONS.md` DD-001 for *why* this
exists before the fancy machinery.

The job of this directory: make "did that change help?" an answerable question, and
turn this system's **silent failures** (fluent wrong answers that look like fluent right
answers) into loud ones.

## The eval set shape (start small)

`datasets/seed.yaml` — ~10 hand-written questions over the seed corpus. For each:

- `question` — the natural-language query.
- `expected_sources` — which document(s)/chunk(s) *should* be retrieved to answer it.
  This alone gives **retrieval recall** with no LLM judge — just a set-membership check.
- `expected_answer` (optional) — a reference answer, for later answer-quality scoring.

Include **2–3 out-of-corpus questions** whose answer is *not* in the corpus. These test
whether the system correctly **abstains** ("not enough info") instead of hallucinating —
most people forget these, and they are gold.

## Scoring, added in layers

1. **Retrieval** (now): recall@k via `expected_sources` membership. Cheap, no LLM.
2. **Answer quality** (later): faithfulness / groundedness + answer relevance via
   LLM-as-judge, then Ragas.
3. **System cost** (later): step count, tokens, latency per question.

Run the set before/after every meaningful change and keep the score deltas. The
change -> measure -> keep-or-revert loop is the real skill this project is practicing.

Status: **harness not built yet; this README defines the target shape.**
