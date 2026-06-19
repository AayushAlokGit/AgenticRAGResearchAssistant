# Building Production-Grade Agents — Roadmap & Progress Log

This is the **curriculum and the running log** for evolving this project's agent loop from a
basic single-tool ReAct loop into a production-grade agent — and, more importantly, for
*learning* every concept that goes into one. It is general-first (the transferable idea),
then mapped onto this project as one worked instance, in the spirit of
`docs/evals/EVALUATION_PRINCIPLES.md`.

It doubles as a **cross-session memory**: each increment records what we built, why, the
trade-off, and the measured result, so any future session can pick up exactly where we left
off. **When you finish an increment, update the Progress Log at the bottom.**

---

## The mental model: an agent is a policy over an action space

Strip away the hype and an LLM agent is four things in a loop:

1. **An action space** — the set of tools the agent may invoke (plus "stop"). This is the
   single most defining property. A loop with one tool is a *workflow*; a loop choosing among
   many tools and composing them is an *agent*.
2. **A controller (policy)** — the model that, given the current state, picks the next action.
3. **State / memory** — what the agent has observed so far (the scratchpad), fed back each turn.
4. **A control regime** — budgets, stop conditions, error handling, guardrails: the harness
   that keeps the loop bounded, safe, and recoverable.

Everything in this roadmap is an upgrade to one of those four. We name which one each time.

### The non-negotiable: measure the trajectory, not just the answer

Per `CLAUDE.md` ("no unmeasured changes"), every increment must be eval-gated. But agent
changes are different from RAG changes: improving *planning* or *tool selection* may barely
move final-answer correctness while drastically changing **how** the agent got there —
rounds used, tool calls made, redundant work, cost per question. So agent evals need
**trajectory/process metrics** alongside the existing answer-correctness/faithfulness:

- **efficiency** — rounds used, LLM calls/question, tokens/question (already partly logged)
- **tool-call validity** — fraction of actions that parsed + dispatched without error
- **redundant work** — searches that added nothing new (the oscillation signal)
- **outcome** — answer-correctness + faithfulness (the existing rungs)

A technique that holds correctness flat but cuts cost/Q in half is a *win* we could not see
with answer-correctness alone. We build a thin trajectory-eval dimension alongside Phase A.

---

## Starting point (as of 2026-06-18)

`src/agentic_rag/agent/loop.py` is a clean hand-rolled **ReAct loop** with: a controller
(`decide_next_action`), a deduped scratchpad fed back each round, a budget (`max_rounds`),
three stop conditions (finish / budget / oscillation), compaction-lite (compact router view
for the controller, char-budget for the generator), and role tiering (separate
controller/generator models, DD-025).

**The defining limitation:** the action space is `{search, finish}` — effectively *one* real
tool. Tool *selection*, *composition*, and *error recovery* — the heart of production agents —
don't exist yet because there's nothing to select between. So the curriculum starts there.

---

## The curriculum (ordered by dependency)

### Phase A — The action space *(the foundation everything else needs)*
- **A1. Multi-tool agent + a tool registry.** Generalize the single `search` action into a
  typed tool set with a dispatcher. Tools: `search`, `expand_document` (small→big on demand),
  `list_sources`, `finish`. Hand-rolled first (understand dispatch/schemas before the API).
  *Upgrades: action space.* **← we are here**
- **A2. Native tool-calling vs hand-rolled JSON.** Swap hand-parsed JSON actions for the
  provider's tool-use API; A/B them so you feel what it abstracts (schema enforcement, the
  tool-call/tool-result message protocol, parallel calls). *This is Module 5's "typed tool
  schemas." Upgrades: controller interface.*
- **A3. Tool robustness / harness.** Tool errors become observations the agent recovers from;
  argument validation; per-tool retries/fallbacks; guardrails. *Upgrades: control regime.*

### Phase B — Smarter control
- **B1. Planning / decomposition.** An explicit plan-and-execute step (decompose a multi-hop
  question up front) A/B'd vs reactive ReAct. Trade-off: foresight vs adaptivity.
- **B2. Reflection / self-critique.** The agent judges whether its evidence actually answers
  the question and revises (Reflexion). Targets the synthesis-partial failures (DD-024).

### Phase C — State & memory *(project Module 4)*
- **C1. Structured working memory** — typed scratchpad, observations vs. actions, sub-goal state.
- **C2. Long-term memory across sessions** — what to persist, retrieve, and forget.

### Phase D — Scaling out
- **D1. Multi-agent orchestration** — planner/worker, subagents, delegation, when it helps vs hurts.

### Cross-cutting
- **X1. Trajectory evals** — process metrics (above). Built alongside A1, extended as needed.

---

## Progress Log *(newest first; update when an increment lands)*

- **2026-06-18 — New baseline lineage declared: "agentic system" (vs the old "pure RAG +
  simple ReAct").** Decision (user): stop comparing against the old champion's absolute
  numbers; the prior system's results (DD-001..DD-026) stand as the *pure-RAG + simple-ReAct*
  record, and we now baseline a fresh *agentic* system. First change under the new lineage:
  `agent.answer_char_budget: 16000 → 0` (final-answer evidence trim OFF). Rationale: the trim
  was a Groq-6K survival hack; DD-023 found it incidentally helped as a quality filter on the
  OLD pipeline, but the multi-tool pipeline (esp. `expand_document`) has different
  evidence-volume dynamics, so that result doesn't transfer. **PENDING A/B:** re-test 0 vs
  16000 on the current pipeline before re-enabling; write a fresh DD on the result.

- **2026-06-18 — A1 multi-tool registry BUILT + smoke-tested live; eval A/B still PENDING.**
  - New `agent/tools.py`: `Tool`/`ToolResult`/`ToolContext`/`ToolRegistry` + four tools
    (`search`, `expand_document`, `list_sources`, `finish`). Pydantic arg schemas (commodity
    part borrowed; loop stays hand-rolled). `finish` modeled as a terminal tool so the action
    grammar is uniform (sets up A2). Tool descriptions + JSON grammar rendered FROM the
    registry into the prompt (single source of truth).
  - `agent/loop.py` generalized: controller emits `{action: <tool>, args: {...}}`, parsed +
    validated against the registry (`parse_and_validate`); unknown tool / bad args / bad JSON
    → re-ask (≤3), then fail-safe finish. Dispatch is registry-driven (no per-tool branch).
    Oscillation guard now trips only when a RETRIEVAL re-found only known evidence (no-hit
    tools like list_sources don't false-trip it).
  - X1 trajectory eval landed: `Trajectory` on `AnswerResult` (rounds_used, exit_reason,
    tool_calls, tool_errors, redundant_searches); aggregated in `answer_correctness.report()`
    + a console line + per-question record in the run JSON. Naive path leaves it None.
  - Config: `agent.tools` knob (default `[search, finish]` = pre-A1 champion). Recorded in
    `agent_config_snapshot`. Models held at `gemini-2.5-flash` (controller+generator) per
    user — flash-lite considered (~4× cheaper) but deferred; ablation constraint dropped.
  - Smoke test: `python -m agentic_rag.agent.loop --tools search,expand_document,list_sources,finish`
    ran clean — searched, hit the oscillation guard on a duplicate retrieval, answered
    correctly, trajectory line printed. Tools/parse/validate unit-checked.
  - **NEXT (pending):** run the eval A/B — baseline `[search,finish]` vs the four-tool set —
    on answer-correctness + the new trajectory metrics; keep-or-revert; then write the DD.
