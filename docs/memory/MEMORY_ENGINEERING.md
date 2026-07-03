# Memory Engineering

> A portable mental model. RAG is the worked example here, but the principles
> apply to **any agent meant to improve across time** — a coding agent that learns a
> repo's conventions, a customer-support bot that remembers your tickets, a personal
> assistant that knows your preferences, a long-horizon research agent. Learn Part 1
> as the skill; treat Part 2 as one dialect of it; use Part 3 when you start a
> different system.

This doc exists because of two questions worth taking seriously, asked from opposite
sides: *is memory just "save things to a database and look them up later" — too
trivial to study?* and *is it just an open-ended pile of caches, summaries, profiles,
vector stores, and "learnings" — too varied to have a spine?* The honest answer to
both: it's a **structured umbrella**, and it's the *same* umbrella as context
engineering seen from the other side. Context engineering asks "what goes in the
window **this** turn?"; one of its seven levers is **Persist** — *what survives across
turns and sessions?* Memory engineering is that single lever opened up into a
discipline. Under the sprawl of named techniques (episodic caches, semantic
distillation, user profiles, scratchpads, decay policies, reflection…) sits a small
fixed skeleton: a single shift that defines the field (P1), a *closed* set of four
operations (P2), four kinds of thing worth remembering (P3), three dials that decide
*whether you need any of it* (P4), one failure mode that is the mirror-image of
context-dilution (P5), and one hard rule about measuring it (P6). So it's neither
trivial nor chaotic — and, like context engineering, **which parts you reach for is
dictated by your system's shape, not by how advanced you want to be.**

---

## Part 1 — The general principles

### P1. Memory is state that outlives the context window — it turns a stateless function into a stateful agent.

A model has no state between calls except what you put in the window *this* call.
Without memory, an agent is a **stateless function**: `answer = f(question, corpus)`.
Ask the same thing twice and you get two independent runs that share nothing. The
system has no past — every user is a stranger, every question is the first it has ever
seen.

Memory adds a term that persists: `answer = f(question, corpus, accumulated_state)`.
That single change — a slot that carries forward across calls — is the whole of it.
Everything else is just *what you choose to put in that state* and *how long it lives.*

This is the exact boundary with context engineering. **Context = what's in the window
right now. Memory = what persists outside the window and feeds back into it later.**
Memory is the **Persist** lever from `CONTEXT_ENGINEERING.md` (P3), promoted to a
discipline because deciding *what to keep, distil, and discard across time* turns out
to be a whole problem of its own.

*Non-RAG anchor:* the difference between a junior support agent who solves every
ticket from scratch and a senior who thinks "I've seen this class of bug before."
Same brain, same docs — the senior just kept the *episodes* the junior threw away.

### P2. The work decomposes into four universal operations.

Everything anyone does under "memory," for any agent, is one of these:

| Operation | The question it answers | Typical techniques |
|---|---|---|
| **Write** | What do I record, and in what form? | log Q→A episodes, extract facts, save a profile field, append a scratchpad note |
| **Read** | Given the current situation, what do I recall? | embed the query → retrieve relevant memories (a small RAG problem); threshold-gating |
| **Consolidate** | How do many records become durable knowledge? | summarize episodes → semantic facts; reflection; merge duplicates |
| **Forget** | What do I drop so memory stays useful? | decay, capacity caps, eviction (recency / frequency / relevance), TTL |

Two things to notice. **Read is a retrieval problem** — it's RAG pointed at your own
past instead of at a corpus, so everything from `EVALUATION_PRINCIPLES.md` and the
Select lever applies (embed, rank, threshold). And **Forget is the operation everyone
skips** — and the one that separates a memory system from a landfill (see P5).

### P3. There are four kinds of thing worth remembering — and each unlocks a different capability.

The *type* of memory isn't bookkeeping; it decides *what your system can become.*

| Type | What it stores | Time horizon | What it unlocks |
|---|---|---|---|
| **Working** | the current task's intermediate state | one task | reasoning across steps (a scratchpad) |
| **Episodic** | records of past events (what was asked/answered) | across tasks/sessions | reuse past answers; consistency; "I've seen this" |
| **Semantic** | distilled, stable facts | long-term | a *model of the domain/user*, not re-derived each time |
| **Procedural** | learned strategies & preferences | long-term | the agent improves *how* it does the task (self-improvement) |

They form a natural ladder of ambition: **working** (we almost always have it) →
**episodic** (cheapest cross-session win: a "soft cache" of past Q→A) → **semantic**
(consolidate episodes into knowledge) → **procedural** (learn the policy itself). A
cross-cutting fifth, **user-model memory** (who's asking, what they prefer), is just
semantic memory keyed to a person — it's what "personalization" means.

The through-line: as you climb the ladder, the system's **unit of operation grows** —
from *one question* → *one session* → *one user* → *an accumulating body of
expertise*. That progression **is** "gets smarter across sessions," made concrete.

### P4. Whether you need memory at all is set by three dials.

Memory is not free, and for many systems it is *correctly absent*. Three dials decide:

1. **Recurrence** — does the past predict the future? If queries/tasks repeat or
   relate over time, episodic/semantic memory pays. If every input is unrelated and
   one-shot, memory is dead weight (you'll only ever pay write cost and never get a
   read hit).
2. **Continuity** — does work span sessions / returning users? High continuity drives
   **Persist** scope (durable store, cross-session identity). A pure single-shot tool
   needs none.
3. **Feedback** — does the environment tell you whether you were right (a correction,
   a thumbs-up, an outcome)? Feedback is what makes **procedural** memory possible:
   no signal to store means nothing to learn.

A system with low recurrence, no continuity, and no feedback *should not have a memory
module* — it would add cost and a new failure mode for nothing. Crank a dial and the
corresponding capability stops being optional: recurring queries → episodic; returning
users → semantic/user-model; rich feedback → procedural.

### P5. Memory's failure mode is the dual of context-dilution — an unbounded memory becomes a landfill.

Context engineering's quality failure (P2 there) is a window *full of the wrong
stuff*. Memory has the mirror image across time: **a store that only ever grows fills
with stale, redundant, and low-value records, and recall gets *worse*, not better.**
More memory is not more capability — past some point it's more noise to retrieve
*through*. This is why **Forget is first-class, not an afterthought** (P2).

There's a second, sharper danger unique to the *read* side: **recall precision.** A
"soft cache" that fires too eagerly will answer a *new* question with a stale episode
and *degrade* an answer that would have been fine. So a memory read has the same
two-sided cost as a classifier: it must **fire when there's a useful prior (recall)**
*and* **stay quiet when there isn't (precision)**. Threshold-gating is the dial, and
it must be tuned against examples that *should not* hit — the memory analogue of
negative/null-case tests (`EVALUATION_PRINCIPLES.md` P5).

### P6. You can only measure memory on a *sequenced* eval — the eval must match the capability's time-horizon.

Memory's entire value is **cross-task**: it shows up only when one interaction
benefits from another. So an eval made of **independent, one-shot questions cannot
detect memory by construction** — each question starts blank, there is no past to
recall. This is the most important measurement rule in the discipline, and the easiest
to get wrong: people bolt a memory module onto a one-shot eval, see no change, and
conclude either "it works" or "it doesn't" — when in fact the instrument is blind.

The fix: build a **sequenced eval** — questions run *in order*, with the store
persisting between them, deliberately seeded so that later items *can* benefit from
earlier ones (repeats, paraphrases, follow-ups) **and** so that some items *should
not* (distractors, to measure precision per P5). Then read the three metric families
(`EVALUATION_PRINCIPLES.md` P3) across the *sequence*: did cost drop on the recurring
items (operational), without correctness falling (outcome), without misfiring on the
distractors (trajectory)?

*Non-RAG examples (to prove the levers transfer):*
- **Coding agent.** Remembers a repo's conventions and your past instructions across
  sessions (semantic/procedural); keeps a running task scratchpad (working); must
  *forget* a file's old contents after you edit it or it'll act on stale memory (forget).
- **Customer-support bot.** Reads back your prior tickets (episodic) and account facts
  (semantic/user-model); consolidates a long history into a short profile (consolidate);
  ages out resolved issues so they don't pollute new ones (forget).
- **Personal assistant.** Learns you prefer terse answers (procedural), remembers your
  team's names (semantic), and — critically — *updates* when you correct it instead of
  repeating the mistake forever (write-on-feedback, the capability memory uniquely unlocks).

Same four operations, four types, three dials — different *skin*. The domain picks the
dialect; the skeleton is constant.

---

## Part 2 — This RAG project as one worked instance

Now map Part 1 onto what we've actually built (and are about to build). The general
idea is on the left; this project's dialect and the design-decision it traces to is on
the right.

**Our dials (P4) — read them honestly.** *Recurrence:* our eval is **independent
one-shot questions** — near-zero natural recurrence. *Continuity:* a single user, a
single corpus, each question fresh — near-zero. *Feedback:* none captured today. By
P4, a system with these dials has **little natural memory headroom** — the same honest
verdict context engineering reached for its dials (`CONTEXT_ENGINEERING.md` Part 2,
P4). So why build Module 4 at all? Because **the project's stated goal is to get
"smarter across sessions"** — memory is the lever that *is* that goal — and the way to
learn the mechanism is to (a) build it on the smallest real surface and (b)
**manufacture recurrence** in a sequenced eval so the capability has something to
prove against. We build to learn the loop, not because the dials are already cranked.

**The four types, mapped to this system:**

| Type | In this project | Status |
|---|---|---|
| **Working** | the agent's per-task **scratchpad** of gathered evidence (`agent/loop.py`) | ✅ already have it |
| **Episodic** | a **soft cache** of past Q→A episodes — save each answered question; on a near-repeat, recall it and let the controller finish in fewer rounds | ✅ built + validated (slice 1, DD-045→049) |
| **Semantic** | **consolidate** clusters of episodes into stable distilled facts | 🔬 built + measured + reverted (DD-050) — machinery correct, broad-recall win didn't clear the noise; reproducible at commit `7d97af5` |
| **Procedural** | learned retrieval/stopping strategies from past trajectories | ⬜ stretch (low feedback dial) |

**The honest-measurement problem, made concrete (P6).** This is the module's headline
lesson and the reason scoping came before code. `evals/datasets/agentic.yaml` is
independent one-shot questions, so it **cannot see memory**. Module 4 therefore
*co-builds a sequenced eval*: reuse the existing questions, append **paraphrase-repeats**
(which inherit the original's gold) and **distractors** (which must *not* trigger
recall, per P5), run them **in order** with the store persisting, and A/B memory
**on vs off** on rounds/tokens (operational) at held-equal correctness (outcome) with
no misfires (trajectory). "The eval must match the capability's time-horizon" stops
being a slogan and becomes the harness.

**The build path (the four operations, smallest surface first):**

1. **Write + Read** — an `EpisodicStore` (JSON file + brute-force cosine over the
   local MiniLM embedder; *not* ChromaDB — memory is kept **separate** from the
   corpus), auto-write at task end, auto-read at task start, threshold-gated injection
   into the controller. (Auto, not a discretionary tool — the DD-031 lesson that the
   agent won't reliably invoke an optional tool.) Targets a **cost** win on repeats.
2. **Forget** — capacity bound + eviction, plus distractors in the eval so the recall
   threshold is forced to stay quiet when there's no useful prior (P5).
3. **Consolidate** — distil related episodes into semantic facts (P3's richest op).
4. *(stretch)* **Procedural / feedback** — likely a mechanism demo here, since the
   feedback dial is low.

**Link back to context engineering.** In `CONTEXT_ENGINEERING.md`'s lever table,
**Persist** was marked ❌ "Module 4 (memory)." This module *is* opening that lever —
and `session-checkpoint.md` (the file that lets a new session resume this project) is
itself a worked example of episodic→semantic **consolidation**: a running log of
sessions distilled into a stable, retrievable state.

---

## Part 3 — Applying this to a *new* system (the anti-overfit checklist)

When you start something that isn't this project, don't reach for "an episodic JSON
cache with cosine recall." Reach for the questions that *generated* that choice:

1. **Does the past predict the future here?** If inputs recur or relate over time,
   memory can pay; if every input is unrelated and one-shot, *don't build it* — you'll
   pay write cost for read hits that never come. (P4 recurrence)
2. **What's worth remembering — events, facts, or skills?** Episodic vs semantic vs
   procedural. Start at the cheapest rung the dials justify (usually episodic), earn
   the higher ones. (P3)
3. **What are my four operations — and specifically, what's my *forget* policy?**
   A write+read system with no forget is a landfill in waiting. Decide eviction before
   the store grows, not after. (P2, P5)
4. **Can over-eager recall *hurt*?** A memory read that fires on the wrong situation
   degrades an answer that was fine. Tune the threshold against cases that *should not*
   hit, just like a classifier's false-positive rate. (P5)
5. **Is my eval sequenced to the capability's horizon?** Memory is invisible to a
   one-shot eval. If you can't run examples *in order with state persisting*, you
   cannot measure memory — wire that before you trust any number. (P6)

If you can answer those five for a new system, you've transferred the skill. The
specific techniques (JSON caches, vector stores, reflection loops, profiles) will
follow from which dial is turned up.

---

*See also: `docs/context/CONTEXT_ENGINEERING.md` (memory is its **Persist** lever; the
two docs are the same umbrella from opposite sides), `docs/evals/EVALUATION_PRINCIPLES.md`
(P3 metric families and P5 null-case tests, both used by the sequenced eval),
`docs/ProjectIdea.md` (Module 4 spec), and `DESIGN_DECISIONS.md` (the change→measure→
keep-or-revert record as the slices land).*
