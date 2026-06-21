# Experiments Log

Terse record of eval-gated A/Bs: the change, the result, keep/revert, the one-line lesson.
Full rationale lives in `DESIGN_DECISIONS.md` (DD refs). Newest last.

**Meters:** retrieval recall@{1,3,5}+MRR (deterministic, free) · answer-correctness end-to-end
(LLM-judge vs reference, non-deterministic — ~1-question run-to-run noise; look for movement, not deltas)
· faithfulness = SUPPORTED/answered (LLM-judge, **reference-free** — grounding not correctness) + a
deterministic cited-source-in-context check (catches citation hallucination, free).

| # | Experiment | Result | Verdict | Lesson |
|---|---|---|---|---|
| 1 | Dense → **hybrid** (dense+BM25, RRF) | recall@1 .621→.724, MRR .843→.903 | ✅ kept (DD-009) | BM25 catches exact-term/findability misses dense embeddings rank low. |
| 2 | + **cross-encoder reranking** | recall@5 .897→**1.0**, MRR→.948; killed 3 false abstentions | ✅ kept, on (DD-014) | Two-stage: cheap recall → precise rerank fixes "right doc, wrong chunk". |
| 3 | + **completeness prompt** rule | end-to-end .655→.724 (CORRECT 19→21) | ✅ kept (DD-015) | Multi-part partials → correct; can't make model use facts it lacks. |
| 4 | Fixed → **recursive chunking** (same chunk_size) | multi_hop@5 **1.0→.333**, end-to-end .724→.655, +3 false abst | ❌ reverted (DD-017) | Smaller chunks (304 vs 262) spread docs thin → worse coverage; **chunk_size & strategy are coupled** — can't A/B one alone. |
| 5 | **top_k 5 → 8** | fixed .724→.655, recursive .655; recall@10 confirmed recoverable | ❌ reverted (DD-017) | More chunks = more **distractors**; k=5 already well-tuned. Constraint flips coverage(k=5)→precision(k=8). |
| 6 | + reranker **relevance gate** (margin=4) | .690→.655 (INCORRECT 1→0 but +1 false abst) | ❌ reverted (DD-019) | A needed multi-hop chunk is *legitimately low-relevance* → looks like a distractor; gate can't separate. *Relevance ≠ answerhood.* |
| 7 | + **parent-child expansion** (neighbour auto-merge, window=1) | answer .690→**.759** (CORRECT 20→22), **reproduced ×2** both arms; recall unchanged | ✅ kept, on (DD-020) | Contiguous same-source context reassembles split tables/lists WITHOUT adding distractors — the win top_k couldn't get. (Same-source only; cross-doc gaps remain.) |
| 8 | + **agentic loop** (hand-rolled ReAct retrieve→reason→retrieve, max_rounds=3) | correctness .750→**.800** (answered 36→40), faithfulness .944→**.975**, **false-abstentions 4→0** | ✅ kept, on (DD-022) | The loop *retries* (reformulate + search again) instead of abstaining → recovers answerable Qs; both gains trace to the 4 recovered answers, which came in grounded. Cost: 1 multi-hop **conflation** (q26 mixed ChromaDB/OpenAI dims, INCORRECT *and* unfaithful — both meters flagged it) + ~3–4× latency. *Single A/B — direction is structural, magnitude (±1–2/40) is within noise.* |
| 9 | + **controller curation** (LLM picks `keep` chunks from snippets; bigger 700-char snippets) | correctness .800→**.750** (CORRECT 32→30, INCORRECT 1→**2**), faithfulness ~flat but **citation-halluc 0→3** | ❌ reverted (DD-023) | Snippet-based LLM curation **over-prunes** — kept 1-of-9 repeatedly, *starving* the generator; it dropped answer-bearing chunks (the 3 citation-hallucs = answer cited a chunk the curator deleted), and on q26 it kept the *wrong* source. Letting the model prune its own evidence trades a transparent failure for a **silent, confident** one. |
| 10 | + **round-robin-by-source** selection (replace arrival-order budget trim) | correctness .800→**.725** (CORRECT 32→29, PARTIAL 7→**10**, INCORRECT 1→**0**), faithfulness .975→**1.000**, citation-halluc 0 | ❌ reverted (DD-023) | A real **trade-off, not a clean loss**: fairness-across-sources *killed every wrong answer* (perfect grounding) but *capped same-source slots* → more PARTIALs (lost completeness, e.g. q26 INCORRECT→PARTIAL). Fights parent-expansion (same-source). Lost the *end-to-end-correct* gate, which rewards completeness. **Surfaced the real lever:** the 16k char budget is a leftover Groq-6K-wall constraint; on Gemini's ~1M context it's discarding evidence (bound 18×) for no survival reason → test removing it. |
| 11 | **remove answer char budget** (16000→0, keep ALL chunks, arrival order) | correctness .800→**.725** (CORRECT 32→29, PARTIAL 7→**10**, INCORRECT 1→**0**), faithfulness .975→**1.000** | ❌ reverted (DD-023) | Hypothesis ("keeping all evidence fixes the partials") was **wrong** — more evidence *diluted* answers (covered fewer required parts) even as grounding went perfect. Landed on the *same* .725 as round-robin from the opposite side. The 16k trim, though born as a Groq-wall hack, now incidentally acts as a **quality filter** (keeps early/on-topic chunks, drops late second-hop accretion). All three evidence-set interventions (curation, round-robin, budget-off) lost to the plain arrival-order trim. |
| 12 | **stronger embedder** MiniLM-L6 (384d) → mpnet-base-v2 (768d), at the C4 champion | recall@5 **.95→.95** (identical), MRR .975→.975; correctness **.773→.775** (flat, within noise) | ❌ reverted (DD-026) | At hybrid+rerank the dense embedder is **masked** — BM25 + the cross-encoder recover the same chunks no matter which embedder ranked them first; 5× CPU/embed for zero gain. The remaining bottleneck is *synthesis* (the 7–8 PARTIALs, multi-source completeness), not dense recall — so a better embedder can't reach it. *Relevance ≠ answerhood*, fourth time. Run at gen `gemini-2.5-flash` / judge `gpt-oss-20b`, single run vs the 3-repeat C4 anchor. |

**Champion config:** hybrid + rerank + parent-expansion (window=1) + **agentic loop (max_rounds=3)**, fixed chunking 800/100, top_k=5.

**v3 matched anchor for the agentic A/B** (gen `gemini-2.5-flash-lite`, judge `gemini-2.5-pro` — both arms share this judge so bias cancels): naive correctness **30/40 = .750** · faithfulness **34/36 = .944**. *Same judge on both arms is the whole point — the absolute number moved when the judge changed (the retired `gpt-oss`-judged 8b anchor read .775/.850), but the A/B Δ is only trustworthy within one judge.* Earlier retired-anchor cross-tab payoff still holds: correctness (vs reference) and faithfulness (vs context) are orthogonal — q26 here is the textbook case where *both* fire on one hallucination.

**Recurring themes:** (a) measure everything — "obviously better" techniques (recursive, bigger top_k, relevance gate) all *lost*; (b) document-level recall overstates fact-level answerability; (c) relevance (embedder, reranker, gate) ≠ answerhood/truth; (d) uncalibrated scores aren't portable (→ RRF uses ranks; gate uses relative margin).

**Open levers (untested):** **answer-context ordering** as its own A/B (arrival vs round-robin-interleaved vs grouped-by-source — lost-in-the-middle; note budget stays ON at 16k now, so this reorders *within* the kept set); thinking-model controller (separate `controller` role); variance re-run of the agentic A/B; parent window=2; cross-*document* completeness (q27-style); raising max_rounds (currently 3); **query transformation** (multi-query/RAG-Fusion, HyDE, decomposition) — aimed at the multi-source PARTIALs the embedder swap couldn't reach; **stronger reranker** (`bge-reranker-large`).

---

# Agentic v2 lineage — eval set `evals/datasets/agentic.yaml` (25 Qs)

A **separate lineage** from the table above (that one is the RAG substrate on the 40-Q seed set under retired judges). This one stress-tests the multi-tool agent **loop** on a harder 25-Q set with capability slices (single_shot_efficiency, decomposition, tool_selection, grounded_stopping, adaptive_recovery). Models: controller + generator `gemini-2.5-flash`, **judge `gemini-2.5-flash-lite`**. Scoring: correctness = mean partial-credit (CORRECT 1 / PARTIALLY_CORRECT 0.5 / INCORRECT 0; correct-abstention 1, false-abstention 0) over all 25; faithfulness = SUPPORTED / answered (reference-free).

## ⭐ B1 BASELINE — 2026-06-20 (n=3) — the bar every future agentic change must beat

| Metric | Runs | **Mean** | Run-to-run spread = **noise floor** |
|---|---|---|---|
| **Correctness** | 0.60 / 0.58 / 0.60 | **0.593** | **±0.02** (tight) |
| **Faithfulness** | 0.762 / 0.842 / 0.833 | **0.812** | ±0.08 (loose) |
| e2e success | 0.33 / 0.29 / 0.33 | 0.317 | ±0.05 |
| avg rounds | 3.88 / 3.72 / 3.84 | 3.81 | — |

**Config (B1):** lean action space `[search, list_sources, finish]` (DD-031) · `max_rounds=5` · `answer_char_budget=0` (no trim) · oscillation patience=2 · top_k=5 · hybrid+rerank+parent-expansion OFF-as-base for the loop. **Detectability:** a real correctness change must clear **≥~0.04**, faithfulness **≥~0.08**. *Judge-CHOICE* noise is far larger (±0.32, llama-4-scout→gemini-flash-lite on identical answers) but is a one-time cost now the judge is fixed — see DD-033.

**Per-question robustness (3 runs) — the map of what to fix vs what's jitter:**
- **SOLID 3/3 (9):** a01 a11 · a09 a13 a18 a19 a20 · a22 a26 — leave alone.
- **DEAD 0/3 (3):** **a14** (decomp) · **a15** (single-shot) · **a23** (grounded-stopping) — reliably broken, highest-value targets.
- **Consistent PARTIAL=0.5 every run (7):** **a02 a03 a10 a05 a07 a24 a25** — the *faithful-but-incomplete* signature (answer drops a sub-part), reliable signal.
- **FLAKY, flips run-to-run (6):** a17 a16 a06 a12 a08 a21 — **noise; do NOT optimize against these** (we over-read them all session).
- Weakest capabilities: **decomposition** and **tool_selection**, dominated by consistent-partials, not total failures.

## Closed lever — in-loop coverage / decomposition (DD-032, DD-033)

| Attempt | Mechanism | Result vs B1 (under trustworthy judge) | Verdict |
|---|---|---|---|
| A | Blind plan-first decomposition (upfront sub-questions) | flat correctness, mixed per-Q | ❌ (DD-032) |
| B | Completeness gate on top of A | gate vetoed 0/25 finishes — inert | ❌ removed (DD-032) |
| C | Adaptive in-loop `open` list (controller self-reports from snippets) | corr 0.55 ≈ B1; signal proven noisy (open>0 scored *higher*) | ❌ (DD-033) |
| D | Full-text coverage gate (re-check completeness on full evidence before finish) | corr 0.58 ≈ B1 0.56, faith ~0.81 both — no gain | ❌ removed, code excised (DD-033) |

**Lesson (the session's big one):** all four failed to beat plain B1. D first *looked* like +0.16 corr — that was a **judge artifact** (the llama judge degenerated to all-PARTIAL faith=0.000 and was generous on correctness; the *same answers* scored 0.840 by llama vs 0.520 by gemini-flash-lite). **A fluent wrong *measurement* looks like a real result** — our noise floor (judge-choice ±0.32, run-to-run ±0.12 across answer-sets) dwarfed every effect we chased. Reliable coverage assessment needs full evidence text (the generator's view), which the cheap routing controller structurally lacks — and even given it (D), the gain didn't survive a trustworthy gauge.

**Next lever (diagnosis-gated):** classify the 10 reliable failures (3 DEAD + 7 PARTIAL) as *evidence-gathered-but-dropped* (→ generator/synthesis lever: pro-generator A/B per DD-025, or answer prompt) vs *evidence-never-retrieved* (→ retrieval lever). Don't build before that split is known.

## Levers tried since the baseline

| # | Change | Result | Verdict | Lesson |
|---|---|---|---|---|
| v2-1 | **Answer-prompt abstention guard** (don't abstain when the answer is in context) | a23 false-abstention 0→1.0; headline flat within ±0.02; SOLID abstentions held | ✅ kept (commit 4f8bc15) | Fixed a real defect invisible to the headline; the why/how clause bundled with it was reverted (caused a10 over-elaboration). |
| v2-2 | **Multi-query / RAG-Fusion** (LLM facet-variants → RRF, retrieval-layer) | smoke-falsified: flash emits question-paraphrases, not corpus-grounded facet queries; missed docs still miss | ❌ not run, code off (DD-034) | **Query reformulation can't decompose into facets it's never seen** — fixes phrasing on known concepts, can't *discover* corpus structure. smoke→fix→re-smoke saved the A/B spend. |
| v2-3 | **Failure audit** (free) | ~half the 8 failures were the flash-lite judge under-grading (a10 misread, a15/a14/a24 over-strict); true correctness ~**0.70** not 0.59 | — (DD-035) | The cheap judge reads **biased-low**, not just noisy. Audit the gauge before concluding the system is stuck. |
| v2-4 | **Pro-ceiling probe** (controller+gen flash→**pro**, judge flash-lite→**flash**, 1 run) | correctness **0.700**, faith 0.826, **0 INCORRECT** | — (DD-035) | Pro **fixes synthesis** (a03/a10/a14/a16 ↑ — a model ceiling) but **not breadth** (a02/a05/a07/a24/a25 still 0.50 — architectural). |
| v2-5 | **Metadata tagging** — induced closed schema, tagged `list_sources` map (cross-doc enumeration) | a07 0.5→1.0; a25 classification *more accurate than the gold*; probe reproduced gold sets exactly | ✅ kept (DD-036) | Enumeration is `SELECT WHERE`, not search — classify ONCE at ingest with corpus-wide context; closed vocab + doc-level (per-chunk free-form drifts). |
| v2-6 | **Judge recalibration** — flash-lite→**flash** + redesigned prompt; same answers | bracketed truth: **0.62 (low) / scout 0.92 (overshoot) / flash 0.72 (calibrated)**; guardrail holds (a02/a16 stay PARTIAL) | ✅ kept (DD-037) | The grader, not the system, moved the headline. Fix golds AND judge separately; never loosen a gold to pass. Dual metrics now recorded (strict vs partial-credit). |
| v2-7 | **Parent-expansion re-enabled** (window=1) — A/B under calibrated judge | strict e2e **0.381→0.524**, partial-credit 0.74→0.80, **faith 0.81→0.95**; 3 PARTIAL→CORRECT, 0 INCORRECT (a02/a05) | ✅ kept (DD-038) | Within-doc coverage fix; auto-expansion beats a discretionary tool the agent won't invoke. Fuller context → less fabrication (the faith jump). n=1; confirmed below. |
| v2-8 | **window=2** parent-expansion (n=3) + **controller empty-finish guard** | window=2: e2e **0.571→0.714** (n=3, +0.143), pc_overall 0.82→0.88, faith 0.984→0.952; guard kills a17's zero-search→naive degradation | ✅ **champion** (DD-039) | window=2 reaches the *spread* within-doc cases (a06); small faith dip accepted. Guard = a harness *invariant* (gather evidence before answering), not a prompt nudge. |
| v2-9 | **Conflict-surfacing answer-prompt** (a24) | a24 PARTIAL in all 3 full runs, **never surfaced the Azure variant** | ❌ reverted (DD-039) | The generator won't recognise two arch docs as one conflicting server (pro failed a24 too). Dropped to keep the prompt lean; a24 stays an open conflict-detection gap. |

**Metric note:** numbers above are **partial-credit** unless tagged "e2e"; the JSON records both `end_to_end_success` (strict) and `correctness_partial_credit_{answerable,overall}`. The gap between strict and partial-credit is the completeness problem (0 INCORRECT, many PARTIAL).

**Where it stands (calibrated judge, n=3):** **champion = window=2 parent-expansion + controller guard → e2e 0.714 / pc_overall 0.880 / faith 0.952** (from the pre-fix baseline e2e 0.38 / pc 0.74 / faith 0.81). Cross-doc enumeration fixed by tagging (DD-036); within-doc coverage by parent-expansion ±2 (DD-038/039); judge recalibrated (DD-037). **Remaining real gaps:** a21 (compound 3-doc enumeration), a17/a05 (multi-part / within-doc, hardest), a24 (conflict-surfacing — needs a non-prompt approach). Candidate next levers: document-level small-to-big (auto whole-doc for high-relevance docs, a12/a06-class), or accept the current plateau and move to a new module.
