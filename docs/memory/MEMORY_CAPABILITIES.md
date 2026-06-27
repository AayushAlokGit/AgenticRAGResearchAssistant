# Capabilities of a Memory-Augmented AI System (and how to evaluate them)

> A portable reference. RAG is the worked example here, but the capability map and the
> eval shapes apply to **any AI system that carries state across time** — a coding agent
> that learns a repo, a support bot that remembers tickets, a personal assistant, a
> long-horizon planner. Learn Part 1 as the checklist; treat Part 2 as one dialect of
> it; use Part 3 when you start a different system.

This doc exists because of one rule from `docs/evals/AGENT_EVAL_SETS.md`: **a capability
with no eval is one you fly blind on.** Memory adds a whole *second axis* of capabilities
on top of the agent's (decomposition, tool-selection, stopping…), and each fails
silently and independently. You cannot test what you have not named. So before writing a
single memory eval case, name the capabilities — then every case traces to one, and a
gap in coverage becomes visible instead of invisible.

Three methodological points govern *all* memory evaluation, regardless of system:

- **M1. Memory is only visible on a *sequenced* eval.** Its entire value is cross-task,
  so an eval of independent one-shot inputs cannot detect it by construction — each input
  starts blank. You must run inputs **in order, with state persisting**, and often across
  a simulated **session boundary** or **world change**. (See `MEMORY_ENGINEERING.md` P6.)
  "The eval must match the capability's time-horizon."
- **M2. Three meters, specialized for memory.** Reuse the metric families from
  `EVALUATION_PRINCIPLES.md` P3, pointed at memory: **cost** (did recall make recurring
  work cheaper — rounds/tokens?), **outcome** (did it stay correct — memory must not
  degrade?), **precision** (did recall fire only when it should — false-recall rate?).
- **M3. Memory's failure mode is two-sided, like a classifier.** It must *fire when a
  prior helps* (recall) **and** *stay quiet when it doesn't* (precision). An eval that
  only tests "does recall help?" will bless a cache that wrecks every unrelated query.
  Always pair positive cases (repeats) with negative cases (distractors, novel, stale) —
  the memory analogue of `EVALUATION_PRINCIPLES.md` P5 (null-case tests are first-class).

---

## Part 1 — The capability map (general)

Each capability is framed as "can the system…?", with the failure mode if absent, the
memory operation it exercises (write / read / forget / consolidate — see
`MEMORY_ENGINEERING.md` P2), a non-RAG example, and **the eval shape that catches it.**
The eval shapes reuse a small set of *roles* (seed = writes an episode; repeat =
paraphrase that should recall; distractor = near-duplicate with a different answer; novel
= unrelated) — the roles are the reusable building blocks of any memory eval.

### A. Episodic recall & reuse — the "soft cache"
- **Can it** reuse a prior result for a recurring/equivalent task, to answer cheaper,
  faster, and more consistently?
- **Fails by:** redoing all the work every time; identical questions give different
  answers across runs.
- **Op:** read. **Non-RAG:** a senior engineer recognizing "I've solved this class of
  bug before" instead of debugging from scratch.
- **Eval shape:** *seed → later repeat (paraphrase).* Measure **cost** (rounds/tokens)
  ON vs OFF at held-equal **outcome**. The paraphrase (not verbatim) is essential — it
  tests *semantic* recall, not a lexical cache.

### B. Recall precision — selective retrieval
- **Can it** fire recall *only* when a prior genuinely applies, and stay quiet / pick the
  *right* episode otherwise?
- **Fails by:** an over-eager cache serving the wrong prior and *corrupting an answer that
  would have been fine* — the single most common memory regression.
- **Op:** read (gating/threshold). **Non-RAG:** an assistant not answering "what's my
  flight?" with last month's itinerary.
- **Eval shape:** *seed → near-duplicate **distractor** (≈ one-token surface diff,
  different answer)* and *unrelated **novel** items.* Measure **false-recall rate**
  (target zero). Hardest variant: two near-identical-question episodes in the store at
  once — recall must return the *right* one, not the most-recent or most-surface-similar.

### C. Cross-session persistence
- **Can it** use in session B what it learned in session A — does memory outlive the
  process? (This is the *definition* of memory.)
- **Fails by:** "amnesia at restart" — looks like memory within a session, forgets
  everything on reboot (state lived only in RAM / the context window).
- **Op:** durable write + read. **Non-RAG:** a chatbot that remembers you between visits
  vs. one that greets you fresh each time.
- **Eval shape:** *seed (phase 1) → **simulated restart** → repeat (phase 2).* The cost
  win and recall must survive the boundary. Realize the restart by re-instantiating the
  store from disk (or a fresh subprocess) between phases.

### D. Recency / staleness handling — forgetting & invalidation
- **Can it** prefer *fresh* truth when the world changes, and invalidate or refresh
  outdated memories?
- **Fails by:** confidently serving an out-of-date cached answer after the source changed
  — the **most dangerous** memory bug, because it's a *faithfulness* failure that looks
  fluent.
- **Op:** forget / refresh + read-with-freshness. **Non-RAG:** a code assistant acting on
  a file's old contents after you edited it; a support bot quoting a discontinued policy.
- **Eval shape:** *seed an answer → **change the underlying source** (re-ingest a new
  value) → re-ask.* The system must serve the **new** answer, not the stale recall.
  Requires a controlled, mutable fixture (you cannot test staleness without a change).

### E. Knowledge accumulation — temporal multi-hop & consolidation
- **Can it** reuse an earlier-established conclusion as a building block, so a later
  question needs fewer hops — does it *get smarter* over a session?
- **Fails by:** every question restarting from raw sources; no compounding; the system is
  no faster on question 50 than question 1.
- **Op:** read + consolidate (episodic → semantic). **Non-RAG:** a researcher who builds
  on yesterday's finding instead of re-deriving it.
- **Eval shape:** *a **dependent chain** where Qₙ builds on the answer to Qₙ₋₁.* Measure
  **hop/round reduction** on the dependent questions ON vs OFF, at equal outcome.

### F. Conversational continuity — reference resolution
- **Can it** resolve follow-ups that depend on prior turns ("…and how is *that*
  configured?") against recent memory?
- **Fails by:** follow-ups being meaningless without the user restating full context.
- **Op:** working + episodic read. **Non-RAG:** any chat assistant honoring "make it
  shorter" without re-stating the whole task.
- **Eval shape:** *an in-session chain of **anaphoric** follow-ups.* Score whether the
  referent is resolved correctly; the negative case is a dangling reference that should
  trigger a clarification, not a guess.

### G. Memory-grounded faithfulness
- **Can it** keep a recall from fabricating or overriding evidence — abstain correctly
  even under a tempting prior, and keep provenance (recalled vs. freshly retrieved)?
- **Fails by:** a confident wrong memory beating a correct "I don't know."
- **Op:** read + guardrail. **Non-RAG:** an agent not "remembering" an API that never
  existed because a prior hallucination got cached.
- **Eval shape:** *a recall that is tempting but wrong/absent → require abstention or
  fresh verification.* Often folded into B and D as their faithfulness check.

### H. Feedback incorporation
- **Can it** store a user correction and not repeat the mistake?
- **Fails by:** repeating corrected errors forever (the correction died with the session).
- **Op:** write-on-feedback (procedural). **Non-RAG:** an assistant that, once told you
  prefer metric units, stops using imperial.
- **Eval shape:** *answer → inject a correction → re-ask.* Measure mistake-repeat rate.
  (Needs a feedback signal in the loop — often the *last* capability a system gains.)

**How to read the map.** A–B are table stakes (the soft cache and its safety). C is the
definition. D is the one most systems ship without and most regret. E–H are the
"gets genuinely smarter" frontier. **Which you owe an eval for is set by your system's
dials** (recurrence, continuity, feedback — `MEMORY_ENGINEERING.md` P4): a single-shot
tool with no returning users owes only A–B; a returning-user product owes C–D; a system
that learns owes E–H.

---

## Part 2 — This project as a worked instance

Our dials are low (one-shot eval origin, single user, no feedback), so this project
builds the map as a *learning* exercise and *manufactures* the recurrence in a sequenced
eval. The capabilities map onto a planned sequence of **sessions** (each session = one
ordered, state-persisting run, scored ON vs OFF):

| Capability | Session | Status |
|---|---|---|
| **A, B** (recall reuse + precision) | `memory_session.yaml` — seed/repeat/distractor/novel | ✅ built (slice 1) |
| **C** (cross-session persistence) | Session 2 — seed → restart → repeat | ⬜ planned |
| **D** (staleness / forgetting) | Session 3 — seed → corpus change → re-ask | ⬜ planned (arc step 2) |
| **E** (knowledge accumulation) | Session 4 — dependent chain | ⬜ planned (arc step 3) |
| **G** (faithfulness) | folded into Session 1 (m11) and Session 3 | 🟡 partial |
| **F, H** (continuity, feedback) | deliberate headroom — low dials | ❌ noted, unbuilt |

This mirrors the build arc (`project-arc` in memory): persistence underpins slice 1,
forgetting is arc step 2, consolidation is arc step 3 — so each session *pre-stages* the
eval for the step that implements it. The three meters (M2) are the same `context_cost`
+ correctness + a new recall-precision axis already used by the agentic harness.

---

## Part 3 — Applying this to a *new* system (the anti-overfit checklist)

When you design memory evals for something that isn't this project, don't reach for
"seed/repeat/distractor over a doc corpus." Reach for the questions that generated them:

1. **Read your dials.** Recurrence, continuity, feedback (`MEMORY_ENGINEERING.md` P4).
   They tell you *which* capabilities (A–H) you actually owe an eval for. Don't write a
   staleness eval for a system whose world never changes.
2. **For each owed capability, instantiate its eval shape** (Part 1). Each is a *sequence*
   with positive and negative roles — never a bag of independent inputs (M1, M3).
3. **Always pair fire-cases with quiet-cases.** Every repeat needs a distractor; every
   "use the memory" needs a "don't use the stale/irrelevant memory." A one-sided memory
   eval is the classic trap (M3).
4. **Decide your boundary mechanisms early.** Persistence needs a *restart* simulation;
   staleness needs a *controlled world-change* fixture. These are harness decisions, not
   dataset decisions — design them before authoring cases.
5. **Point the three meters at it** (M2): cost (cheaper on recurrence?), outcome (still
   correct?), precision (quiet when it should be?). A memory change that improves cost
   while quietly dropping precision is a regression — and invisible without all three.

If you can answer those for a new system, you've transferred the skill. The roles and
metric names will follow from the domain.

---

*See also: `docs/memory/MEMORY_ENGINEERING.md` (the build-side companion — ops, types,
dials, failure modes), `docs/evals/EVALUATION_PRINCIPLES.md` (the three metric families
and null-case discipline this specializes), `docs/evals/AGENT_EVAL_SETS.md` (how to
construct an agent eval set — the non-memory capability axis), and
`evals/datasets/memory_session.yaml` (session 1, the worked dataset).*
