# Context Engineering

> A portable mental model. RAG is the worked example here, but the principles
> apply to **any LLM agent that runs over more than one turn** — a coding agent, a
> computer-use/browser agent, a customer-support bot, a long-horizon research agent.
> Learn Part 1 as the skill; treat Part 2 as one dialect of it; use Part 3 when you
> start a different system.

This doc exists because of two questions worth taking seriously, asked from opposite
sides: *is context engineering just token-budgeting and reordering — too trivial to
study?* and *is it just an umbrella for a sprawl of efficiency tricks — too varied to
have a spine?* The honest answer to both: it's a **structured umbrella.** Under the
open-ended sprawl of named techniques (RAG, reranking, summarization, sub-agents,
memory, lazy-loading…) sits a small fixed skeleton — a couple of failure modes that
explain *why* (P1–P2), a *closed* set of seven levers that are the *what* (P3), and
three dials that decide *which* levers a given system actually needs (P4). Every
technique you can name is just an instance of one of the seven. So it's neither
trivial nor chaotic: the **levers are few and universal, but which ones you reach for
is dictated by your system's shape, not by how "advanced" you want to be** — the skill
is dial-reading, not trick-collecting. Our system is small, so it correctly uses the
small end of the toolbox. That's not the whole toolbox.

---

## Part 1 — The general principles

### P1. The context window is the agent's whole working memory, and attention is the scarce resource.

A model has no state between calls except what you put in the window *this* call. So
the window isn't just "the prompt" — it's the system prompt **plus** the tool
definitions **plus** the conversation history **plus** retrieved documents **plus**
any memory **plus** examples. Every one of those competes for a fixed budget, and —
more subtly — competes for the model's *attention* even when it fits.

This is why the field renamed itself. "Prompt engineering" implied the unit of design
was the instruction text. "Context engineering" names the real unit: **the entire
window, assembled fresh every turn.** The job is deciding, each turn, what earns a
slot.

### P2. There are two distinct failure modes — survival and quality — and they pull in opposite directions.

- **Survival:** the context physically overflows the window (or a per-request token
  ceiling). Hard failure — the call errors or truncates. The fix is *less*: drop,
  trim, summarize.
- **Quality:** the context fits, but it's *full of the wrong stuff* — distractors,
  redundancy, stale turns — and the model's answer degrades even though nothing
  errored. The fix is *better*, which is not the same as *less*.

Beginners conflate these and optimize for "smaller context." The real target is
**the highest signal-per-token**, which sometimes means adding a chunk and sometimes
means cutting three. Two named quality effects you will hit:

- **Lost in the middle** (Liu et al., 2023): models attend most to the *start* and
  *end* of a long context and skim the middle. So *where* a fact sits changes whether
  it's used — ordering is a real lever, not cosmetics.
- **Distraction / dilution:** every irrelevant token is a chance for the model to
  cross-wire facts or hedge. More evidence can make answers *worse* by spreading
  attention thinner. (We measured this directly — see Part 2.)

### P3. The work decomposes into a handful of universal levers.

Everything anyone does under "context engineering," for any agent, is one of these:

| Lever | The question it answers | Typical techniques |
|---|---|---|
| **Select** | What goes in? | retrieval, ranking, reranking, filtering, top-k |
| **Order** | Where in the window? | most-important-last, interleaving, recency ordering |
| **Compress** | How to fit more signal per token? | summarization, compaction of old turns, deduplication |
| **Isolate** | What gets its *own* window? | sub-agents, scratchpads, separate tool contexts |
| **Persist** | What survives across turns/sessions? | external memory, notes-to-self, scratchpad files |
| **Offload** | What stays *out* until needed? | tools/files as context, lazy-loading, retrieve-on-demand |
| **Format** | How is it encoded? | structure over prose, delimiters, tight tool schemas |

That's the whole map. It's small *on purpose* — the skill is not memorizing tricks,
it's recognizing which lever a given problem calls for.

### P4. Which levers you need is set by three dials: horizon, tool-count, agent-count.

The levers above are not all equally relevant to every system. They light up as you
turn three dials:

1. **Horizon** — how many turns / how long the agent runs. Long horizon forces
   **Compress** (you can't keep 200 raw turns) and **Persist** (state must outlive
   the window).
2. **Tool-count** — how many tools/actions exist. Many tools force **Offload** and
   **Format** (50 verbose tool schemas can eat the whole budget before any work
   starts).
3. **Agent-count** — single vs multi-agent. More than one forces **Isolate** (give a
   sub-task a clean window, return only its conclusion).

A short-horizon, few-tools, single-agent system *correctly* uses only Select / Order
/ light Compress. It is not "behind" — it's matched to its shape. Crank a dial and the
corresponding lever stops being optional.

### P5. Non-RAG examples (to prove the levers transfer).

- **Coding agent (e.g. Claude Code).** Does *not* load the repo into context —
  `grep`/`read` on demand (**Offload**); summarizes a long session when it fills up,
  "auto-compact" (**Compress**); spawns sub-agents that explore in their own window
  and hand back a short answer (**Isolate**); keeps tool schemas terse and lazy-loads
  rarely-used ones (**Format** + **Offload**).
- **Computer-use / browser agent.** Keeps only the *latest* screenshot and drops old
  ones because images are token-expensive (**Compress**); compresses the
  accessibility tree (**Format**); keeps a running "what I've done so far" note
  (**Persist**).
- **Customer-support bot.** Retrieves policy snippets (**Select**); summarizes the
  earlier conversation once it gets long (**Compress**); reads back the customer's
  prior tickets/profile (**Persist**).

Same seven levers, different *skin*. The domain picks the dialect; the levers are
constant.

---

## Part 2 — This RAG project as one worked instance

Now map Part 1 onto what we've actually built. The general lever is on the left; what
this project does (and the design-decision / experiment it traces to) is on the right.

**Our system's dials (P4):** short-horizon (`max_rounds=5`), few-tools
(`[search, list_sources, finish]`), single-agent. So by P4 we *should* be living in
Select / Order / light Compress — and we are.

| Lever | In this project | Status |
|---|---|---|
| **Select** | hybrid retrieval (DD-009) → cross-encoder rerank (DD-014) → parent-child expansion (DD-020) → arrival-order budget trim (`select_within_budget`) | ✅ the bulk of our work |
| **Order** | arrival order in the scratchpad (search-1 hits, then search-2…); the ordering A/B (arrival vs interleaved vs grouped) is the open lever | 🟡 partly, one A/B open |
| **Compress** | the controller sees a compact **router-view** (source + 300-char snippet), not full text (`router_view`); the final answer is char-budgeted | 🟡 a *baby* version — see below |
| **Isolate** | none — single agent, one shared scratchpad | ❌ (dial is at 1) |
| **Persist** | an episodic soft cache of past Q→A episodes, recalled across sessions (Module 4) | ✅ built after this module (DD-045→050) |
| **Offload** | none — we stuff retrieved chunks straight into the window | ❌ |
| **Format** | source-labeled passages (`[source: X]`) so the model can cite; JSON tool-action schema for the controller | partial |

**The router-view is a textbook Compress + Offload, and it's why the loop works at
all.** The agent accumulates evidence across rounds. An early version stuffed every
gathered chunk's full text back into the controller prompt each round — which blew
past Groq's 6K-token-per-request ceiling (a **survival** failure, P2). The fix: the
controller only *routes* (decide what to search next / whether to stop), so it
doesn't need full text — a source + short snippet is enough to recognize what's been
found. Full text is kept out of the controller's window and only assembled for the
*final* answer step. That's Compress (snippet not full) + a hint of Offload (the full
text lives in the scratchpad, not the controller prompt). See `agent/loop.py` and
DD-022.

**We measured P2's quality side directly — and it bit us three times.** Module 3's
whole arc (DD-023) was trying to improve *Select/Order* and discovering that "more/
fairer/all evidence" is not free:

- **Controller self-curation** (let the model pick which chunks to keep): LOST —
  it over-pruned and starved itself. A Select lever, mis-set.
- **Round-robin-by-source** (fairer Order/Select across sources): LOST as a
  trade-off — perfect grounding but capped completeness.
- **Removing the budget** (keep *all* evidence): LOST — extra context *diluted*
  answers (P2's distraction effect) rather than completing them. This is the cleanest
  proof in the project that quality ≠ less *and* ≠ more; it's *right*.

The plain arrival-order trim beat all three. The lesson is pure Part 1: the budget
that began as a *survival* hack now earns its keep as a *quality* filter.

**Module 3 proper: both applicable levers measured, both null/negative (DD-043/044).** When
we formally opened this module we added a **context-cost eval axis** (structural chunks+chars/Q
plus billed prompt-tokens/Q — the silent gauge for cost-moving levers) and tested the two levers
our dials light up. **ORDER** (`relevance_last`, most-relevant nearest the question) came back
DEAD FLAT — at ~8 chunks the window is too short for lost-in-the-middle to bite. **Structural
DEDUP** (lossless merge of overlapping parent-expansion windows) cut tokens −27% but **regressed
completeness *and* faithfulness at n=3** (pc −0.079, faith −0.095). The sharp part: we *proved*
it dropped zero facts, so the harm was pure loss of **repetition-as-salience** — duplicated
evidence was implicitly emphasising facts, and the generator leaned on it. This is P4 made
concrete: a short-horizon / few-tools / single-agent system genuinely has little
context-engineering headroom — not a failure to find a win, a *measurement* that the headroom
isn't here. (It also sharpens P2: even *provably-lossless* compression can cost quality, because
attention responds to repetition, not just to information content.)

**What we deliberately have NOT built (and why it's correct not to, for now):**

- **Real Compress (LLM compaction).** True compaction = an LLM *summarizes* the
  scratchpad/history into fewer tokens. We do the cheap structural version (snippets +
  trim) but not summarization. At `max_rounds=5` the scratchpad still rarely gets big
  enough to need it (a batched round can gather more at once — DD-041 — but the char-budget
  trim keeps it bounded). *This is the first deferred lever we have the machinery to measure* — a
  summarize-the-scratchpad A/B is the natural next experiment.
- **Persist (memory).** Module 4 — built *after* this module (DD-045→050): an episodic soft
  cache that lets the system get smarter *across* sessions, the project's stated goal. See
  `docs/memory/MEMORY_ENGINEERING.md`.
- **Isolate (sub-agents).** Only pays off at agent-count > 1. We're single-agent by
  design (hand-rolled to learn the loop), so it's correctly absent.
- **Offload (tools/files as context).** Our corpus is small and retrieval already
  *is* a form of fetch-on-demand. Becomes relevant with many tools or a corpus too big
  to even index naively.

---

## Part 3 — Applying this to a *new* system (the anti-overfit checklist)

When you start something that isn't this project, don't reach for "router-view and a
char budget." Reach for the questions that *generated* those choices:

1. **Where's the attention budget, and is the failure survival or quality?** Is the
   window overflowing (need *less*), or fitting-but-degrading (need *better*)? They
   have opposite fixes. (P1, P2)
2. **What are my three dials?** Horizon, tool-count, agent-count. Long horizon → you
   owe Compress + Persist. Many tools → Offload + Format. Multi-agent → Isolate.
   Don't build a lever your dials don't call for. (P4)
3. **For each lever the dials light up, what's the *cheapest* version that works?**
   Structural compression (snippets, dedup) before LLM summarization; retrieval before
   a memory system. Earn the expensive version. (mirrors the eval cost-ladder)
4. **Can I measure it?** Context-engineering changes fail *silently* — a diluted
   answer looks like a good one. Tie every lever change to an eval delta (quality
   **and** cost), or you're tuning blind. (See `EVALUATION_PRINCIPLES.md`.)
5. **Am I optimizing for "less" when I should optimize for "right"?** The most common
   mistake. The target is signal-per-token, which sometimes means adding.

If you can answer those for a new system, you've transferred the skill. The specific
techniques (router-views, screenshot-dropping, auto-compact) will follow from which
dial is turned up.

---

*See also: `evals/EVALUATION_PRINCIPLES.md` (how to measure any change you make here),
`DESIGN_DECISIONS.md` DD-022 (the agent loop + router-view) and DD-023 (the Module 3
selection A/Bs that all lost), and `docs/EXPERIMENTS.md` (the change→measure→revert
log).*
