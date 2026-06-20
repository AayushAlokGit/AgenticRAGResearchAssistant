# Stop Conditions for Agent Loops

> A teaching doc on one of the four parts of any agent loop (action space · controller ·
> budget · **stop conditions**). It answers: *when should a retrieve→reason→act loop stop
> looping?* — and why a naive answer to that question silently caps the quality of every
> multi-step agent.
>
> General-first, as always: Part 1 is the transferable principle, Part 2 maps it onto this
> project's `agent/loop.py`, Part 3 extends it to other kinds of agents. RAG is the worked
> example; the reasoning applies to any tool-using loop — a coding agent, a web-research
> agent, a support bot that escalates.

---

## The one-sentence thesis

**An agent loop must stop on evidence that it is *globally stuck*, never on a single *local*
stumble — because the cheapest stop condition (abort the moment one step is unproductive) is
indistinguishable from giving up exactly when the agent was one move away from the answer.**

---

## Part 1 — The general principles

### Why stop conditions exist at all

A loop that calls a model to pick its next action will, without a brake, run forever or until
it bankrupts you. So *every* agent needs stop conditions. There are only three kinds, and a
robust agent uses all three together:

1. **Hard budget** — a fixed ceiling on rounds/tokens/wall-clock. The non-negotiable backstop;
   it bounds worst-case cost no matter how confused the agent gets. Blunt by design.
2. **Voluntary stop** — the agent itself decides it is done (a `finish` action). The *best*
   stop: the agent has the most context about whether it has enough. But it can be wrong in
   both directions (stop too early, or never stop).
3. **Progress-based early stop** — a heuristic that detects "the agent is spinning" and stops
   it before the hard budget. This is the subtle one, and the one that goes wrong.

### The core distinction: local stumble vs. global stuckness

A progress heuristic watches for "no progress." The trap is **defining no-progress at the
wrong scale.** There are two very different events:

| | A **local stumble** | **Global stuckness** |
|---|---|---|
| What happened | *one* action was unproductive | *several* actions in a row got nowhere |
| What it means | the agent's last move was poorly chosen | the agent has no viable next move |
| Correct response | **try something different** | **stop** |

The cardinal error is to wire a *global* decision ("stop the whole loop") behind a *local*
signal ("this one step was unproductive"). That is a hair-trigger: it fires on the single
event that, for a multi-step agent, is *normal and recoverable*. A multi-hop agent whose
first phrasing misses the second-hop document hasn't failed — it just needs to rephrase. Abort
there and you punish the agent precisely when it was about to succeed.

> **Principle 1 — fire on the streak, not the stumble.** Count *consecutive* unproductive
> rounds and stop only past a threshold. Reset the count on any productive round, so the trip
> means "N dead-ends back-to-back," not "one dead-end ever."

### Make the signal recoverable, not fatal

Patience alone is necessary but not sufficient. If you merely grant the agent another round
without telling it *why* its last move was wasted, a near-deterministic controller (low
temperature) may just repeat the same move — and now you've spent a round and trip the guard
anyway. The fix is to turn the failure into **information**:

> **Principle 2 — feed the failure back as an observation.** When a step is unproductive, tell
> the controller so, in the state it sees next round ("that returned nothing new — change
> approach or finish"). A stop condition should be the *last* resort after the agent has been
> given a chance to self-correct, not the *first* response to a stumble.

This is the same idea as treating a tool error as an observation the agent can react to, rather
than a crash. Recoverable-by-default; fatal only when recovery demonstrably isn't happening.

### The tension to name: patience costs money

More patience = better coverage (the agent gets to reformulate) but higher cost (every extra
round is more model calls, and the controller is usually the priciest part of the loop). So
"just let it run to the budget" is not free — it converts every genuinely-stuck run into a
full-budget burn. The three brakes are a layered defense, each catching what the one above
misses:

```
  voluntary finish   →  catches the healthy case (cheapest, smartest)
  progress guard     →  catches genuine spinning before the ceiling (bounded waste)
  hard budget        →  catches everything else (worst-case cap)
```

> **Principle 3 — layer the brakes; tune the middle one.** Keep the hard budget fixed and the
> voluntary stop encouraged; the patience threshold is the dial you A/B. Too low → you abort
> recoverable runs (under-answer); too high → you pay for spinning (over-spend).

### How you'd know it's wrong (measure it)

A mis-scaled progress guard is invisible in aggregate — the agent still answers, just from
*partial* evidence, and a fluent partial answer looks like a fluent full one. The tell is in
the **trajectory data**: bucket outcome correctness by *exit reason*. If runs that exited via
the progress guard score materially worse than runs that exited via voluntary `finish`, the
guard is stopping good trajectories early. That gap is the whole diagnosis.

---

## Part 2 — This project: the oscillation guard

Our loop (`src/agentic_rag/agent/loop.py`) has all three brakes: `max_rounds` (hard budget),
a `finish` tool (voluntary), and an **oscillation guard** (progress-based) — a retrieval that
returns hits but *no new* chunks means we re-found only evidence we already hold.

**The bug (the textbook version of Principle 1).** The original guard hard-`break`ed the
instant *one* retrieval was redundant. That is the local-signal/global-decision error exactly.
The failure analysis (`docs/evals/AGENT_FAILURES.md`) caught it in the trajectory data:

- **Exit-reason correlation:** `oscillation` exits were **68%** correct vs `finish` **82%** —
  the gap Principle "how you'd know" predicts.
- **The smoking gun (question a10, a 2-hop question).** Two runs, *both* did 3 searches. In the
  failing run the 3rd search happened to re-hit known chunks → guard tripped → the loop was
  forced to answer with only `DESIGN.md` + `normalizer.md`, **never retrieving `resolver.md`**.
  The agent's own answer admitted *"the provided context does not contain enough information to
  explain… the Resolver's job."* It knew it wasn't done; the guard had already given up. The
  passing run's 3 searches each landed new docs (incl. `resolver.md`), never tripped, chose
  `finish`, and answered fully.

**The fix (decision D).** Three changes that map one-to-one onto the principles:

1. **Patience counter (Principle 1).** A module constant `OSCILLATION_PATIENCE = 2` and a
   `consecutive_redundant` counter. Trip only after 2 redundant rounds *in a row*; reset to 0
   on any retrieval that adds new evidence. `max_rounds` stays the hard backstop. (The
   cumulative `redundant_searches` trajectory metric is kept separate and unchanged — *streak*
   drives the stop, *total* is for diagnosis.)
2. **Feedback observation (Principle 2).** On a redundant round, append a `[NOTE: this search
   returned only chunks already in EVIDENCE — reformulate / switch tool / finish]` marker to
   the observation the controller sees next round — so the extra round Principle 1 bought is
   actually *usable* instead of a silent repeat.
3. **Prompt pointer.** `prompts/agent_react.md` points its standing "don't repeat a fruitless
   search" rule at that explicit `[NOTE]` marker, so the controller treats it as authoritative
   rather than having to infer redundancy by diffing the evidence block itself.

**The dial.** `OSCILLATION_PATIENCE` is the tunable middle brake (Principle 3). 2 is the
starting bet; it is an eval-gated knob, raised only if the data shows recoverable runs still
aborting, lowered if abstention questions start spinning to budget.

---

## Part 3 — Extending to other agents

The "skin" changes; the three principles don't. What counts as a "redundant round" is the only
domain-specific part — it's *any signal that the last action moved the agent no closer to done.*

- **Coding agent.** Unproductive round = an edit that left the same test failing with the same
  error (or a command that changed nothing). Local stumble: one failed attempt — normal, try a
  different fix. Global stuckness: the *same* error N times in a row → stop and escalate.
  Principle 2 = feed the failing test output back as the next observation (most coding agents
  already do this; it's why they recover).
- **Web-research agent.** Redundant round = a query returning already-seen URLs/snippets. The
  recovery is identical to RAG: reformulate, broaden, or switch source. Same hair-trigger risk
  if you abort on the first overlap.
- **Support / workflow bot.** Unproductive round = a lookup that returns no new account state.
  Here the *voluntary* stop matters most — escalate-to-human is itself a stop action — but the
  progress guard still belongs underneath it so a confused bot doesn't loop forever on a
  customer who can't be matched.

**The portable checklist for a new agent loop:**

1. Do I have all three brakes — hard budget, voluntary stop, *and* a progress guard?
2. Is my progress guard counting a **streak**, not a single stumble? Does a productive round
   reset it?
3. When a step is unproductive, do I **feed that back** to the controller before stopping?
4. Is the patience threshold a **named, eval-gated knob** — not a magic number buried in a
   branch?
5. Can I **bucket outcomes by exit reason** to detect the guard stopping good runs early?

If any answer is "no," the loop is one unlucky step away from giving up mid-investigation —
and you won't see it in the aggregate score, only in the trajectories.
