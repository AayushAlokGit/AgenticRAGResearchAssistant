# How to Evaluate Compound AI Systems

> A portable mental model. RAG is the worked example here, but the principles
> apply to **any multi-stage AI system with a controller** — a voice assistant
> (ASR → intent → dialog), a robotics stack (perception → planning → control), a
> fraud pipeline, or any LLM agent. Learn Part 1 as the skill; treat Part 2 as one
> dialect of it; use Part 3 when you start a different system.

This doc exists because evaluation is the part of building these systems that is
**most transferable and least taught**. Models, frameworks, and prompts churn; the
discipline of "did that change actually help, and which part did it help?" does not.

---

## Part 1 — The general principles

### P1. Decompose into independently-failing parts; measure each in isolation *and* end-to-end.

A compound system is a chain of stages, and the stages fail **independently**. A
flawless early stage feeding a broken later stage still produces a broken product.
If you only measure the final output, you learn *that* it failed but not *where* —
so you can't target a fix.

So you build **two kinds of eval**:
- **Component (unit) evals** isolate one stage by feeding it known-good input and
  scoring only its output. They pinpoint blame.
- **End-to-end (system) evals** run the whole thing and score the final result.
  They measure the product the user actually gets.

You need both. Component evals without end-to-end let each part look great while the
whole disappoints; end-to-end without component evals tell you something is wrong but
not what.

*Non-RAG example:* In ASR → intent → dialog, you score word-error-rate on the ASR
alone (feed it audio, check the transcript), intent-accuracy on the parser alone
(feed it *gold* transcripts so ASR errors don't contaminate it), and task-success
end-to-end. If end-to-end drops, the component scores tell you whether the mic
choked or the parser did.

### P2. One dataset, many meters.

Because the stages share an input, you do **not** build a separate eval set per
component. You build **one set of examples** and read **different metrics off the
same run**. A single example simultaneously yields a retrieval score, an
answer-quality score, and a cost number.

This is the single most common beginner mistake: imagining a "retrieval dataset" and
a separate "agent dataset." There's one dataset; there are several meters pointed at
it.

### P3. Three metric families for anything with a controller.

The moment a system *decides what to do* (loop again? call a tool? stop? refuse?),
one output number is not enough. Score three axes — this taxonomy comes straight from
reinforcement-learning / agent evaluation and applies to any sequential
decision-maker:

| Family | Question it answers | Examples |
|---|---|---|
| **Outcome** | Was the final result correct? | answer correctness, task success |
| **Process / trajectory** | Did it take *good actions* to get there? | did it choose the right tool, take the needed steps, avoid the wrong path, refuse when it should |
| **Operational** | What did it *cost*? | steps, tokens, latency, dollars |

Outcome alone hides a system that's right by luck or right at 10× the cost. A change
that adds 2% accuracy but triples cost is usually a bad trade — and invisible unless
you measure the operational axis.

*Non-RAG example:* A robot arm that reaches the goal (outcome ✓) by a flailing,
collision-prone path (trajectory ✗) burning twice the energy (operational ✗) is not
a good policy, and only the three-axis view says so.

### P4. Climb a scoring-cost ladder: cheap & deterministic before expensive & subjective.

Scoring methods form a ladder by cost and reliability. Build the **cheapest meter
that gives real signal first**, get a baseline on the board, and only add costlier
meters when you've earned the need:

1. **Reference-based, deterministic** — compare output to a known answer with code
   (set membership, exact/fuzzy match, numeric error). No model, instant, perfectly
   repeatable. *(RAG: recall@k via `expected_sources`.)*
2. **Reference-based, semantic** — compare to a reference answer but allow
   paraphrase (embedding similarity, or a judge with the gold answer in hand).
3. **Reference-free, judge-based** — an LLM (or human) rates a property that has no
   single right answer: is the answer faithful to its sources? helpful? safe?
   Powerful but slow, costs money, and is itself a model that can be wrong — so it
   needs its own validation.

Don't start at rung 3. A reference-free LLM-judge is seductive because it scores
anything, but it's the *least* trustworthy and *most* expensive rung.

### P5. Negative / null-case tests are first-class, not afterthoughts.

Every real system must sometimes do **nothing**: return no match, refuse, say "I
don't know." Beginners test only the happy path, so the system learns to always
produce a confident answer — and confidently hallucinates on inputs it should have
declined. Deliberately seed examples whose correct behavior is **abstention / empty /
refuse**, and score them. (This is the same instinct as caring about false-positive
rate, not just accuracy, in a classifier.)

### P6. Make silent failure loud; treat the eval set as the regression suite.

These systems fail *silently*: a fluent wrong answer looks exactly like a fluent
right one. There's no stack trace. The eval set's whole job is to convert silent
failure into a number that drops. Once you have it, every change runs the loop:

> **change → run evals → keep or revert.**

That loop — not any individual technique — is the actual engineering skill. It's
ordinary software-test discipline, adapted to systems whose output is
non-deterministic and subjectively scored.

---

## Part 2 — RAG agents as one worked instance

Now map Part 1 onto this project. The general principle is on the left; the
RAG-specific *dialect* is on the right.

**The three surfaces (P1 decomposition).** "The RAG system" is really two
independently-failing stages, and the agent is a third:

| Surface | Does | Fails by | Metric (RAG dialect) | Needs an LLM? |
|---|---|---|---|---|
| **Retriever** | query → chunks | misses the right doc; buries it under distractors | `recall@k`, `precision@k`, MRR/nDCG (P4 rung 1) | No |
| **Generator** | chunks → answer | hallucinates, ignores context, miscites, won't abstain | faithfulness, answer-relevance (P4 rung 3) | Yes (judge) |
| **Agent / controller** | decides: retrieve again? stop? refuse? | loops forever, stops early, fabricates, blows budget | trajectory checks + step/token/latency/cost (P3) | Partly |

**One dataset, many meters (P2).** `evals/datasets/seed.yaml` is the *single*
question set. Question **q05** alone scores all three surfaces in one run:
- retriever — did both `expected_sources` come back? (`match: all`)
- generator — does the answer correctly state "384 → 3072" and stay faithful?
- agent — did it actually perform *two* retrieval hops, then stop?

**The seed.yaml fields are just P3's three families wearing RAG names:**
- `expected_sources` + `match` → **outcome/component** signal for the retriever.
- `expected_answer` + `should_abstain` → **outcome** signal for the generator.
- `type` (`factual` / `multi_hop` / `abstention`) → a **trajectory** label: it
  encodes what the *controller* should have done.
- (recorded per run, not in the file) step count, tokens, latency → **operational**.

**Where the agent is actually being tested (P3 trajectory + P5 null-case):**
- **q10–q12 (abstention)** are primarily *controller* evals. They test the decision
  to **refuse** when retrieval is empty/weak. A naive RAG with no agent will
  cheerfully invent a Groq-embeddings config (q10). This is P5 made concrete.
- **q05 (multi-hop)** tests whether the agent chose to retrieve a **second time**
  instead of answering from hop one. `match: all` makes partial retrieval a partial
  failure *by design* — pressure on the loop logic.
- **q07 (distractor-sensitive)** tests whether it got pulled toward the legacy Azure
  doc that's deliberately left in the corpus as noise.

**The scoring-cost ladder, scheduled (P4 + P6).** `evals/README.md` already
sequences the meters: retrieval recall **now** (rung 1, no LLM — the first baseline
to beat), faithfulness/relevance via LLM-judge **later** (rung 3), cost/step metrics
**later** (operational). That ordering *is* P4.

---

## Part 3 — Applying this to a *new* system (the anti-overfit checklist)

When you start something that isn't this project, don't reach for "recall@k and
faithfulness." Reach for the questions that *generated* those:

1. **What are the stages, and which fail independently?** Draw the pipeline. Each
   arrow is a place to put a component eval. (P1)
2. **What's the one set of representative inputs?** Build it once; plan to read
   several meters off it, not one dataset per stage. (P2)
3. **Is there a controller / decision-making step?** If yes, you owe three axes:
   outcome, trajectory, operational — not just outcome. (P3)
4. **For each thing you want to score, what's the cheapest rung that gives signal?**
   Deterministic check? Reference answer? Or genuinely judge-only? Start low. (P4)
5. **What should the system refuse / return-empty on?** Seed those cases on purpose.
   (P5)
6. **Can the system fail silently?** If wrong output looks like right output, the
   eval set is your only smoke alarm — wire the change→measure→revert loop before
   you start tuning. (P6)

If you can answer those six for a new system, you've transferred the skill. The
metric names will follow from the domain.

---

*See also: `evals/README.md` (how this project's harness is staged) and
`DESIGN_DECISIONS.md` DD-001 (why evals are built early here).*
