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

**Champion config:** hybrid + rerank + parent-expansion (window=1) + **agentic loop (max_rounds=3)**, fixed chunking 800/100, top_k=5.

**v3 matched anchor for the agentic A/B** (gen `gemini-2.5-flash-lite`, judge `gemini-2.5-pro` — both arms share this judge so bias cancels): naive correctness **30/40 = .750** · faithfulness **34/36 = .944**. *Same judge on both arms is the whole point — the absolute number moved when the judge changed (the retired `gpt-oss`-judged 8b anchor read .775/.850), but the A/B Δ is only trustworthy within one judge.* Earlier retired-anchor cross-tab payoff still holds: correctness (vs reference) and faithfulness (vs context) are orthogonal — q26 here is the textbook case where *both* fire on one hallucination.

**Recurring themes:** (a) measure everything — "obviously better" techniques (recursive, bigger top_k, relevance gate) all *lost*; (b) document-level recall overstates fact-level answerability; (c) relevance (embedder, reranker, gate) ≠ answerhood/truth; (d) uncalibrated scores aren't portable (→ RRF uses ranks; gate uses relative margin).

**Open levers (untested):** variance re-run of the agentic A/B (pin down the ±1–2/40 noise band); the q26 conflation (guard multi-hop fact cross-wiring); parent window=2; cross-*document* completeness (q27-style — parent-expansion is same-source only); stronger embedder; raising max_rounds (currently 3).
