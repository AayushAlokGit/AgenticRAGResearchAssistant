# Retrospective — what building this taught

> The goal of this project was never the artifact. It was to become an engineer who can
> push agentic systems to production *on purpose* — knowing which technique to reach for,
> what trade-off it makes, and how to prove it earned its place. This doc is the synthesis:
> each module mapped to the **general principle** it taught, with this RAG build as one
> worked instance. Read it as the transferable core; the RAG specifics are the dialect.

The system was built in the deliberate order the plan prescribed — a naive end-to-end
version first, then five modules (RAG substrate → agentic layer → context engineering →
memory → harness), then a LangGraph capstone — **re-running the eval after every change so
each technique had to earn its keep.** That loop, `change → measure → keep-or-revert`, is
the real skill the whole thing was practice for. What follows is what the loop taught.

---

## The one meta-principle: the eval must match the layer you're changing

If you take a single thing from this project, take this. It surfaced independently in three
modules and is the spine of everything else:

- **Context engineering** changes *what's in the window* — invisible to a correctness score,
  so it needed a **cost axis** (evidence chunks/Q, prompt tokens/Q) to see dilution/dedup wins.
- **Memory** is a *cross-task* capability — invisible to independent one-shot questions by
  construction, so it needed a **sequenced eval** (questions run in order, state persisting).
- **The harness** changes *reliability/cost/latency*, not the answer — a cached response is
  byte-identical to an uncached one — so it's measured on an **operational axis**, not e2e.

Three different layers, same lesson: **an instrument pointed at the wrong axis reports "no
change" and you conclude wrongly.** Before you measure a change, ask *what axis does this
even move?* — and if your eval can't see that axis, the first work is to build the axis, not
to run the change. This is the antidote to the failure mode that makes these systems
dangerous: **fluent wrong answers look exactly like fluent right answers.** Measurement is
not optional; it is the discipline that replaces vibes.

---

## Module by module — the principle, then the instance

### 1. RAG substrate — *retrieval is a data-modeling problem before it's a similarity problem*
The naive "embed the query, take top-k" baseline fails on a whole class of questions, and the
reason is general: **enumeration ("list all X that…") is a `SELECT WHERE`, not a nearest-
neighbor lookup.** Similarity search finds things *like* the query; it can't discover the
*structure* of a corpus. The fix was to **classify documents at ingest with corpus-wide
context** (a closed tag schema induced from the whole corpus) so retrieval could filter, not
just rank. Transferable takeaway: when retrieval underperforms, ask whether the question is
really a *filter/aggregate* in disguise — and if so, move the work to **ingest-time
structure**, because query-time reformulation can't recover information the index never
captured. (Corollary that held all through: keep chunking **format-agnostic** — structure-
extraction is a separate layer from chunking, so the system scales to arbitrary uploads.)

### 2. Agentic layer — *every lever has an inverse risk; find it before you ship it*
Turning retrieval into a retrieve→reason→retrieve loop introduced the module's deepest
lesson, learned twice from opposite directions:
- Every **"gather more" lever** (whole-doc expansion, a verify-and-revise pass) has an
  inverse **precision** risk — it dilutes, breaks abstentions, over-includes on filters.
- Every **"stop earlier" lever** (yield-ratio cutoffs, repeat-query guards) has an inverse
  **completeness + faithfulness** cost — the "wasted" extra rounds were quietly accreting
  coverage and grounding.
Both directions netted flat-or-negative once measured at `n=3`. The general principle:
**a change that helps one metric almost always charges another; your job is to find the
charge before the eval does.** Two techniques *did* earn their place, and both are
transferable: **automatic coverage beats a discretionary tool** (the agent won't reliably
invoke an optional tool — bake the behavior in), and **action batching** (fan out independent
sub-queries in one round). And the sharpest craft lesson: **enforce invariants structurally,
not with prompt nudges** — "never answer from zero evidence" became a guard in the loop
(seed a search), not a plea in the prompt the controller ignores.

### 3. Context engineering — *read the dials before collecting tricks*
Both applicable levers here — **ordering** (lost-in-the-middle) and **dedup** (lossless
window-merge) — measured **null or negative**, and that *is* the lesson. Context engineering
is not a bag of tricks to apply; it's a set of dials (horizon, tool count, agent count) that
tell you *whether a lever even applies*. At ~8 evidence chunks, lost-in-the-middle is too
weak to exploit. And the counter-intuitive finding worth carrying: **even provably-lossless
compression cost quality** — dropping duplicate evidence removed *repetition-as-salience*
(repeated facts implicitly signal importance). Transferable: **don't apply a technique
because it's advanced; apply it because your dials are turned up enough for it to bite** — and
respect that "obviously free" optimizations can have hidden signal cost.

### 4. Memory — *state that outlives the window, with forgetting as a first-class op*
Memory is the **Persist lever from context engineering, promoted to a discipline.** The
transferable skeleton: four operations — **write / read / forget / consolidate** — where
*read is just RAG pointed at your own past*, and **forget is the one everyone skips** (the
op that separates a memory system from a landfill). The build validated an episodic "soft
cache" and produced two durable, general findings: **(1) a scalar similarity threshold can't
gate recall** — a *distractor* out-scored a genuine paraphrase, so the right design lets the
*agent judge* a recalled item rather than a cutoff decide; and **(2) compositional reuse
isn't a retrieval problem** — recalling a past *fact* to extend a reasoning chain is not the
same as recognizing a re-asked *question*, and no amount of retrieval cleverness bridges that
gap (it's the episodic↔semantic boundary). Also, memory recall behaves as a **stop-signal**,
so it inherits Module 2's inverse-risk: cheaper but occasionally less complete.

### 5. Harness — *the layer that turns a model call into a service*
The harness is everything that makes the call itself **dependable**, orthogonal to how good
the answer is. It's a closed set of six concerns — **Interface, Orchestration, Reliability,
Efficiency, Safety, Observability** — and the generating insight is: **a model call is
unreliable I/O, not a function call.** That reframe re-derives most of the harness (retries,
fallback, caching, metering, validation). Concrete transferable wins: **cache at the
`messages` boundary and hash it verbatim → invalidation is automatic** (everything the model
saw is in the key); a content-addressed cache doubles as a **determinism detector** (it
surfaced retriever non-determinism it didn't cause). And **guardrails come in three rings**
(input / action / output) with two rules that generalize past RAG: **denominate a guard in
its true unit** (cap *spend in tokens*, not the round-count proxy) and **observe before you
enforce** (flag-and-measure a new guard before it's allowed to change behavior).

### Capstone — *build it by hand, then let the framework take the parts with no learning left*
Re-expressing the hand-rolled loop as a **LangGraph** graph — and proving it earns the
**same eval verdicts** (24/25 identical) — closed the loop. The payoff wasn't the migration;
it was that having built every piece by hand, each framework abstraction mapped to a twin you
already understood: the `for`-loop → a **cycle** of nodes; mutable `scratchpad.add` →
**reducers** merging a typed State; `if/break` → **conditional edges**; the `max_rounds`
budget → your router plus a `recursion_limit` **backstop**; scattered logs → **LangSmith**
spans for free. Two meta-lessons: **the framework abstracts the *control flow* and nothing
else** — the entire hand-rolled substrate (provider, cache, retriever, tools, prompts) was
reused unchanged — and **use the abstraction for commodity, hand-roll the core**: we
hand-built the agent loop (learning) but did *not* hand-roll a tracer (no learning in span
plumbing — that's what OpenTelemetry/LangSmith are for). The parity A/B also re-taught a
measurement subtlety: the trajectory differences between the two implementations were
**retriever cross-process non-determinism, not the code** — and the cache, by replaying
identical trajectories, is the cleaner instrument for isolating an implementation change.

---

## The recurring shapes (patterns that showed up everywhere)

- **The inverse-risk law.** Add-more trades precision; remove trades completeness; recall
  trades a stop-signal. Name the charge before shipping the lever.
- **Structural over persuasive.** Invariants belong in the loop (a guard), not in the prompt
  (a nudge the model ignores). Same for coverage (automatic > a discretionary tool).
- **Dial-reading, not trick-collecting.** Whether a technique applies is set by your system's
  shape (horizon, tools, agents, recurrence, autonomy, scale, stakes) — not by how advanced
  it is. Most levers here correctly measured *null* because the dials were low.
- **Silent axes.** The wins that matter are often invisible to your headline metric — cost,
  latency, reliability, cross-session reuse. Build the axis that sees them.
- **Read at mechanism level.** `n=1` is jumpy; the headline lies; the per-question / per-
  trajectory story is where the truth (and the noise) is.

---

## How to apply this to a *new* system (the anti-overfit checklist)

1. **What axis does this change even move?** If your eval can't see it, build that axis first
   (cost / sequenced / operational). Never trust "no change" from a blind instrument.
2. **What's the inverse risk?** Every lever charges some other metric — find it before the
   eval does, and measure the metric it charges, not just the one it helps.
3. **Are my dials even turned up for this?** Match the technique to the system's shape
   (horizon, tool count, autonomy, scale, recurrence, stakes) — not to its sophistication.
4. **Can this be an invariant instead of a request?** Prefer a structural guarantee in the
   loop over a prompt asking nicely.
5. **Is this the core, or commodity?** Hand-roll what teaches you (or is genuinely novel);
   use trusted abstractions for solved plumbing (tracing, retries, vector math).
6. **Did I change one thing and re-measure?** The whole craft is the `change → measure →
   keep-or-revert` loop. Everything else is commentary.

---

*See also, for the general-first treatment of each layer:
`docs/evals/EVALUATION_PRINCIPLES.md`, `docs/context/CONTEXT_ENGINEERING.md`,
`docs/memory/MEMORY_ENGINEERING.md`, `docs/harness/HARNESS_ENGINEERING.md`, and
`DESIGN_DECISIONS.md` for the change-by-change record this retrospective distills.*
