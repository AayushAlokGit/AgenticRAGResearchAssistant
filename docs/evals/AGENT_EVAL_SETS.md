# How to Structure an Eval Set for Agents

> The companion to `EVALUATION_PRINCIPLES.md`. That doc answers **what to measure
> and why** (outcome / trajectory / operational, the scoring-cost ladder, null-case
> tests). This doc answers the next question: **how do you actually construct the
> dataset** so that those meters have something to read? It is about the *examples* —
> what to put in the set, how to write them so they genuinely exercise an agent, and
> what to assert about each one.
>
> General-first, as always: Part 1 is the transferable skill, Part 2 maps it onto
> this project's `agentic.yaml`, Part 3 is a checklist for a brand-new agent system.
> RAG is the worked example; the principles apply to any tool-using agent — a coding
> agent, a travel-booking agent, a customer-support bot that can look things up.

---

## The one-sentence thesis

**An eval set for a plain model grades its *answers*; an eval set for an agent must
also provoke and grade its *decisions* — which means the examples themselves have to
be engineered to require those decisions.** A dataset that doesn't *force* tool use,
multi-step reasoning, or restraint cannot tell you whether your agent is good at them.

---

## Part 1 — The general principles

### A1. Coverage must *require* the capability ("test coverage" for agents).

This is the principle everything else serves. Restating the rule from the principles
doc in dataset terms:

> **A metric can only move if the dataset gives it room to move.** Every capability
> you want to measure needs at least one example that *cannot be solved without it*.

This is the agent-eval analog of code coverage: a test suite that never executes a
branch tells you nothing about that branch. If your agent has an `escalate_to_human`
tool but no example where escalating is the only correct move, your "tool-selection
accuracy" is measuring a question you never asked — and an A/B that adds, removes, or
improves that tool will read **flat**, not because the change was neutral but because
the dataset was *blind* to it. A flat result on a blind test is the most expensive
kind of wrong: it looks like evidence and isn't.

The practical test for any candidate example: **"Could a single-shot, no-tools
baseline get this right by luck?"** If yes, it doesn't exercise the agent — it's an
outcome example wearing an agent costume. Keep some of those (see A5), but know which
is which.

*Non-RAG example:* For a coding agent, "fix the failing test" only exercises
multi-step behavior if the fix requires *reading* a second file the error message
doesn't name. If the fix is obvious from the stack trace, you're testing the model,
not the agent.

### A2. Organize the set by *capability*, not by topic.

A plain QA set is organized by subject matter (embeddings questions, chunking
questions, …). An **agent** set is organized by the *behavior under test*. Each
example should target one agent capability so that, when a score moves, you know which
muscle moved. Topic is incidental; capability is the index.

Here is a general capability taxonomy for tool-using agents. It is the spine of the
dataset — aim to fill every row, because a gap is a capability you are flying blind on:

| Capability | The example forces the agent to… | Generic failure it catches |
|---|---|---|
| **Single-shot efficiency** | answer a trivial query in **one** action and stop | over-thinking: calling tools / looping when one shot suffices |
| **Decomposition & composition** | break a task into sub-steps and chain tool calls | answering from the first partial result; never taking hop 2 |
| **Tool selection** | pick a *specific, non-default* tool that is uniquely right | defaulting to the one familiar tool for everything |
| **Adaptive recovery** | notice a bad/empty/erroring result and change tactics | repeating the same failing action; giving up; hallucinating |
| **Grounded stopping / abstention** | search, find nothing, and **refuse** without burning budget | fabricating an answer; looping until the budget dies |

This taxonomy is universal. The *examples* are the domain skin:

- **Travel-booking agent.** *Tool selection:* "What's the baggage allowance?" must call
  `lookup_policy`, not `search_flights`. *Adaptive recovery:* the first fare search
  returns nothing → widen dates rather than report failure. *Grounded stopping:* "Book
  me a hotel on the moon" → decline.
- **Customer-support agent.** *Decomposition:* "Why was I double-charged and can I get
  a refund?" → pull the transaction log, *then* check the refund policy, *then*
  answer. *Efficiency:* "What are your hours?" → answer directly, don't open a ticket.

### A3. Assert the *property*, not the *path*.

Once an example forces a decision, you must decide **how strictly** to grade the
decision. There is a spectrum, and the middle is almost always right:

- **L0 — outcome only.** Grade just the final answer; read trajectory metrics at the
  *run* level but assert nothing per example. Cheap, but can't tell a good path from a
  lucky one: an agent that ignores the right tool yet scrapes the answer together
  passes silently.
- **L1 — assert the invariant.** Attach a *loose* per-example expectation that any
  competent path must satisfy — "a good trajectory uses `expand_document`," or
  "should finish in ≤ 1 round." Score it as a **separate diagnostic** axis next to the
  answer verdict.
- **L2 — golden path.** Specify the one ideal action sequence and score edit-distance
  to it. Precise but **brittle**: there are usually many valid paths, and L2 punishes
  legitimate alternatives, so it drifts toward measuring "did you do it *my* way"
  rather than "did you do it well."

**Prefer L1.** The reason is a general truth about sequential decision problems:
*there is rarely one correct trajectory, but there are invariants every good
trajectory shares.* Assert those invariants. A robot can reach a goal by many paths;
"never collides" is the invariant worth asserting, not "follow these exact waypoints."

Two honesty rules that come with L1:

1. **Trajectory assertions are heuristics, not ground truth.** The answer's correctness
   is (relatively) objective; "should have used tool X" is your *opinion* of a good
   path. A clever agent may have a better one. So a trajectory miss is a **flag to
   inspect**, never an automatic failure — and it must be scored on its own axis, not
   folded into the correctness number.
2. **Keep the assertion as loose as still discriminates.** Assert `expects_tool: X`
   (X appears somewhere), not `tool_sequence: [X, then Y, then Z]`. The moment your
   assertion forbids a reasonable path, it has become an L2 trap.

### A4. Every agentic example still needs an outcome anchor.

The seductive failure when you discover trajectory scoring is to over-rotate into it
and start grading *only* process. Don't. **A beautiful path to a wrong answer is a
failure.** Trajectory is a *second* axis that explains *how* the outcome happened — it
never replaces the outcome check. Concretely: every example carries a normal
answer/abstention expectation (the meter from the principles doc), *and* optionally a
trajectory expectation. The trajectory axis earns its keep precisely by being
orthogonal — it catches "right by luck" and "right but 5× the cost," neither of which
the outcome axis can see.

### A5. Seed *effort* negative-cases, not just *answer* negative-cases.

The principles doc (P5) says: deliberately include examples whose correct answer is
*refuse / empty*, or the system learns to always answer. Agents have a **second**
null-case, unique to them: examples whose correct *effort* is **minimal**. An agent
that calls three tools and loops twice to answer "what's the capital of France" is
**failing**, even though the answer is right — it's wasting budget, latency, and money,
and it will do the same on your production traffic. So seed easy questions and assert a
*low* effort ceiling (`max_rounds: 1`). Over-thinking is to agents what
over-confidence is to classifiers: a failure mode you must test for on purpose, because
the happy path hides it.

### A6. Span difficulty so the set *discriminates*.

An eval set exists to tell two system versions apart. A set that's all trivial
saturates at 100% (every version passes — no signal); a set that's all brutal
saturates at floor (every version fails — no signal). You want a **difficulty
gradient**, with deliberate *headroom* at the top: a few examples nothing passes yet,
so there's room to show improvement. (This is exactly why this project's `seed.yaml`
grew a v2 batch — v1's retrieval recall had saturated at 1.0 and could no longer rank
changes.) The same logic applies per-capability: include an easy tool-selection case
*and* a subtle one where the right tool is non-obvious.

### A7. Ground-truth the trajectory the way you ground-truth the answer.

A reference answer is only trustworthy if someone *verified* it against the source. A
trajectory expectation deserves the same rigor: before you write `expects_tool:
expand_document`, confirm the corpus actually makes that the right move (e.g. the fact
really does span more of the document than a single retrieved chunk shows). An
unverified trajectory assertion is worse than none — it manufactures false failures
that send you chasing phantom regressions.

---

## Part 2 — This project's `agentic.yaml` as the worked instance

Map Part 1 onto this build. `seed.yaml` (43 questions) was written for a
**RAG→generate** pipeline — it grades retrieval recall and answer correctness, and it
stays **frozen as the regression suite** (the "do no harm" floor: the agent must not
lose ground on what the simple system already did). The **new** `agentic.yaml` is the
capability layer, organized by the A2 taxonomy.

### The schema (L1, per A3)

Each question keeps the familiar outcome fields and adds an **optional** `capability`
tag and an **optional** `trajectory` block:

```yaml
- id: a01
  capability: tool_selection      # which taxonomy row this example targets (A2)
  type: factual
  question: "..."
  match: any
  expected_sources: [RERANKING.md]    # outcome anchor — retriever (A4)
  expected_answer: "..."              # outcome anchor — generator (A4)
  should_abstain: false
  trajectory:                         # OPTIONAL loose invariant (A3); omit = outcome-only
    expects_tool: expand_document     # this tool should appear in a good path
    max_rounds: 2                     # effort ceiling (A5)
    min_distinct_searches: 1
  notes: "Why this example exists / which failure mode it probes."
```

Anything without a `trajectory:` block is graded outcome-only (L0 for that row) — so
simple regression questions stay simple. The block is scored as a **separate
diagnostic** (trajectory PASS/FAIL), never mixed into the correctness rate (A3/A4).

> **Note — `expects_tool` binds to the tool's *code name*, on purpose (DD-027).**
> This couples the eval case to the implementation identifier in `agent/tools.py`
> rather than to a stable capability label — exactly the implementation-vs-interface
> coupling Principle A3 warns about. We accept it deliberately: with only four
> hand-rolled tools the name↔capability mapping is ~1:1, so the looser alternatives (a
> registry-owned capability vocabulary, or behavioral-effect assertions) would be
> indirection with no current payoff (YAGNI). The price is brittleness if tool names
> churn. **Revisit when they do** — most likely at **A2 (native tool-calling)**, or
> when a tool splits/merges or several tools share one capability — and promote
> `expects_tool` to an `expects_capability` label the registry declares. As a cheap
> guardrail in the meantime, the loader validates `expects_tool` against the live
> registry so a rename fails *loudly*, not silently.

### Mapping the taxonomy onto *this* corpus

The corpus is unusually well-suited because it contains **two self-referential RAG
systems** plus general RAG-concept docs (`CHUNKING.md`, `RERANKING.md`,
`VECTOR_DB_INTERNALS.md`), which gives natural multi-hop and distractor structure:

| Capability | Example archetype here | Why it's the right move |
|---|---|---|
| **Single-shot efficiency** | a `seed`-style one-fact question (e.g. "what's the default embedding dim?") with `max_rounds: 1` | one search answers it; more is waste (A5) |
| **Decomposition** | a genuine **3-hop** spanning `CHUNKING` + `RERANKING` + `VECTOR_DB_INTERNALS` | each doc holds one necessary third; one search can't cover all |
| **Tool selection — `expand_document`** | a fact that spans **more of a doc than the ±1 parent window holds** | the on-by-default expansion (DD-020) under-covers it, so the agent must *choose* to pull the whole doc — this is the discretion `expand_document` adds over the fixed retrieval policy |
| **Tool selection — `list_sources`** | an orientation query: "which corpus docs cover reranking?" | answered by *seeing the corpus*, not by content search |
| **Grounded stopping** | an abstention case (like `q43`'s absent HNSW knob values) reached *through the loop* | agent must search, find nothing numeric, and refuse without spinning to budget |

**Verification discipline (A7).** Every `expected_source`/`expected_answer` is
confirmed by reading the cited doc — inheriting `seed.yaml`'s rule ("do not add a
source you have not read"). Trajectory assertions stay **loose**: `expects_tool` +
`max_rounds`, never a pinned sequence, because the loop legitimately has several good
paths (e.g. it might `list_sources` first *or* go straight to `search`).

### Which meters read it

Same "one dataset, many meters" idea (principles doc P2), now with a trajectory meter
that actually has examples to bite on:

- **Outcome** — answer-correctness judge + abstention check (reused from
  `answer_correctness.py`).
- **Trajectory (diagnostic)** — per-question PASS/FAIL on the `trajectory` block,
  computed from the run's `Trajectory` object (`tool_calls`, `rounds_used`, …).
- **Operational** — controller/generator tokens & rounds, already instrumented (X1).
- **Capability slice** — scores grouped by the `capability` tag, so the report says
  *"tool-selection 4/5, decomposition 2/4"* instead of one undifferentiated number.

This is what makes the upcoming A1 A/B (baseline `[search, finish]` vs the four-tool
set) **non-blind**: the tool-selection rows are, by construction, the ones only the
four-tool arm can pass — so a real difference can finally show up in the numbers.

---

## Part 3 — Applying this to a *new* agent system (checklist)

When you build an eval set for a different agent, don't copy this project's questions —
regenerate them from the questions that produced them:

1. **List the action space.** Write down every tool. *Each non-trivial tool needs ≥ 1
   example where it is the uniquely correct move* — or you can't measure whether it
   helps. (A1, A2)
2. **Walk the capability taxonomy** (efficiency, decomposition, tool-selection,
   recovery, grounded-stopping) and ask which rows are empty for your agent. Fill the
   gaps. (A2)
3. **For each example, pick the loosest trajectory invariant that still
   discriminates** — assert the property, not the path; prefer L1. (A3)
4. **Anchor every example with an outcome check.** Trajectory is a second axis, never a
   replacement. (A4)
5. **Seed effort negative-cases** — easy tasks with a low effort ceiling — alongside
   the usual refuse/empty answer negative-cases. Over-thinking is an agent-specific
   failure you must test on purpose. (A5)
6. **Build a difficulty gradient with headroom** so the set can rank versions instead
   of saturating. (A6)
7. **Verify the trajectory expectations against the source**, not just the answers. An
   unverified path assertion manufactures phantom regressions. (A7)

Answer those seven for your system and you've transferred the skill; the tool names and
topics will follow from the domain.

---

*See also: `docs/evals/EVALUATION_PRINCIPLES.md` (what to measure and why),
`docs/evals/ANSWER_QUALITY.md` (the outcome meters in depth),
`docs/agents/AGENT_ROADMAP.md` (X1 trajectory evals; the A1 multi-tool A/B this set
unblocks).*
