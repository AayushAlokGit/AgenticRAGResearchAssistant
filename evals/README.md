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

1. **Retrieval** (now): recall@k via `expected_sources` membership. Cheap, no LLM.
2. **Answer quality** (later): faithfulness / groundedness + answer relevance via
   LLM-as-judge, then Ragas.
3. **System cost** (later): step count, tokens, latency per question.

Run the set before/after every meaningful change and keep the score deltas. The
change -> measure -> keep-or-revert loop is the real skill this project is practicing.

Status: **`datasets/seed.yaml` v1 exists (12 questions: 9 answerable + 3 abstention) over
the frozen 11-doc corpus snapshot. The scoring harness that runs it is not built yet** —
the dataset is the ground truth that harness will consume.
