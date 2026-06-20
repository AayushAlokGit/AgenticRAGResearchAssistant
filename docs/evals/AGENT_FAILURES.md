# Agent failure analysis — adopted baseline

Source: 6 baseline runs (`eval_runs/agentic/v1/*_21_*.json`), config = parent OFF + completeness
prompt, four-tool, `max_rounds:5`. Outcome mean **0.76** (11–14/16). All local, no new spend.

> **STATUS (2026-06-20, after DD-029 — stop-condition redesign):** F1 (oscillation) is **fixed** —
> a02/a05/a10 now robust 3/3, `oscillation` exits → 0. The patience change briefly regressed
> abstention (a08), closed by an abstention-guard prompt bullet (a08/a09/a13 back to 3/3). New
> baseline **0.917** (14–15/16). **Still open:** **a16** (F5 synthesis-incompleteness) and **a07**
> (F6 judge-borderline filename survey) — the next levers. a17 (F4) currently passing but judge-noisy.

## Headline signal: deliberate `finish` beats getting cut off

| Exit reason | Correct | Read |
|---|---|---|
| `finish` (agent chose to stop) | 33/40 = **82%** | healthy |
| `budget` (hit max_rounds) | 13/16 = 81% | ok but wasteful |
| `oscillation` (repeated-search breaker) | 27/40 = **68%** | **the leak** |

40 of 96 question-runs end in oscillation. The agent often *varies a search* instead of
calling `finish`, trips the breaker, and answers from whatever partial evidence it has.

## Failure list (the questions that lose, ranked by robustness of failure)

| ID | Capability | Rate | Mechanism | Evidence |
|---|---|---|---|---|
| **a17** | single_shot | 0/6 | **F4 completeness over-padding.** Right doc, right sense (homonym never confused) — completeness self-check imports an adjacent off-topic example (date-disambiguation); judge calls it drift. | `wrongSense=False` all runs; padding appears only in INCORRECT runs; identical answer flips verdict (judge variance). |
| **a05** | tool_selection | 1/6 | **F2 expand_document not selected.** Needs whole CHUNKING.md §4 (A–E taxonomy); plain search returns partial chunks. | The *only* correct run is the one that called `expand_document`+`finish`; other 5 oscillate on `search:2` → "omitting several families." |
| **a16** | decomposition | 2/6 | **F5 synthesis incompleteness.** Both required docs ARE retrieved, but answer covers only one half (mean-pool/L2 *or* HNSW/cosine); burns to `budget`. | Both docs present every run; INCORRECT runs exit `budget` w/ 5 searches; judge says partial. |
| **a02** | single_shot | 3/6 | **F1 + F5.** Four-item list; INCORRECT runs exit `oscillation` with partial list. | Every INCORRECT a02 = oscillation exit; CORRECT = clean finish / fuller search. |
| **a10** | decomposition | 3/6 | **F1 + F3 second-hop under-retrieval.** Stops before retrieving `resolver.md`. | INCORRECT runs have only DESIGN+normalizer, exit oscillation; CORRECT runs have all 5 docs incl `resolver.md`, exit finish. |
| **a07** | tool_selection | 4/6 | **F6 judge-borderline filename survey.** `list_sources` path fires correctly; answer is a file list, minor differences flip judge. | All runs exit finish with the right path; verdict noise on near-identical file lists. Eval-side, low priority. |

(Chronic-but-passing context: a06/a12 share a05's shape — they need `expand_document` and happen
to pass; a05 is where the missing tool-call actually costs the answer.)

## Cross-cutting mechanisms (what to actually tune)

- **F1 — Oscillation instead of deliberate stop.** The biggest lever: the agent gets *cut off*
  rather than *deciding* it's done. Whatever forces oscillation (no reformulation budget? breaker
  too eager? controller not calling `finish`?) is dragging a02/a05/a10 down. Inspect the loop's
  oscillation/redundant-search logic and the controller's stop reasoning.
- **F2 — `expand_document` under-selected.** Single-doc whole-coverage questions (a05) need it; the
  controller rarely reaches for it. Tool-affordance / prompt-description issue, not retrieval.
- **F3 — Second-hop under-retrieval.** Multi-hop questions stop before the 2nd required doc lands
  (a10 missing `resolver.md`). Overlaps F1 (it stops too early).
- **F4 — Completeness over-padding.** The completeness prompt taxes single-shot "what + why"
  questions (a17) by inviting adjacent material. Net win overall, but cost is concentrated and real.
- **F5 — Synthesis incompleteness.** Even with both docs present, multi-part answers come out
  lopsided (a16, a02). This is what completeness was meant to fix; it half-works.

## Data-quality note

One a16 run has `verdict=None` (judge returned empty) — harness should not silently record a null
verdict. Minor robustness gap to patch.

## Candidate levers (to decide after review — not yet actioned)

1. Fix F1: examine + tune the oscillation breaker / `finish` stop-condition (likely the highest-ROI,
   touches a02/a05/a10 at once).
2. Fix F2: strengthen `expand_document`'s tool description / when-to-use cue.
3. Fix F4: precision-guard clause on the completeness prompt (recover a17 without losing F5 gains).
4. On-demand `expand_around_chunk` coverage tool (the standing next-step) targets F3/F5 surgically.
5. Patch the null-verdict harness gap.
