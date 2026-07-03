# LangGraph — orchestration as a stateful graph

> A portable mental model for agent-orchestration frameworks, with LangGraph as the
> concrete instance. The transferable skill is *how to think about a framework that runs
> your control flow for you* — the specific API is the dialect. This is the "and here is
> what the framework abstracts" chapter of the build: we hand-rolled the agent loop first
> (`agent/loop.py`), then re-expressed it in LangGraph (`agent/graph.py`) and proved parity,
> so every abstraction below maps to a twin we built by hand. See `docs/RETROSPECTIVE.md`.

The reason to learn a framework like this *after* hand-rolling the loop is that the
framework's value only becomes legible once you know what it's replacing. Hand-rolled, an
agent is a Python `for` loop with `if/break` over some mutable state. LangGraph's single
idea is to **flip that from imperative to declarative**: you *describe* the agent as a graph
— nodes (work) wired by edges (control flow) over a shared state — and the framework runs
the loop. That inversion is the whole thing; everything else is consequence.

---

## Part 1 — The general idea (framework-agnostic)

### P1. Imperative control flow → declarative control flow.
Hand-rolled, *you* own execution: you write the loop, the branches, the stop conditions, and
you thread state through by mutation. In a graph framework, you **declare the structure** —
"these are the nodes, these are the edges, here's the shared state" — and hand execution to
the framework. You stop writing `while`/`for`/`if`/`break` and start writing *nodes and the
rules connecting them*. The framework becomes the interpreter of your state machine.

### P2. Why give up your loop? Because execution becomes a service.
Once the framework owns execution, capabilities that are painful to hand-roll come **for
free**, because they're properties of *how the graph is run*, not of your business logic:
persistence (checkpoint/resume), streaming, tracing/observability, parallelism, retries, and
human-in-the-loop pauses. The trade you make: a little indirection and a new vocabulary, in
exchange for not re-implementing that machinery yourself. That trade is only worth it when
your system's shape actually needs those capabilities (P8).

### P3. The execution model: a graph is run in super-steps (actors + BSP).
Under the hood these frameworks borrow the **Pregel / bulk-synchronous-parallel** model. Work
happens in **super-steps**: in each tick, every currently-active node runs (potentially in
parallel), then the edges fire to decide which nodes are active next tick. In a sequential
graph, that's just "one node per tick." This matters for two reasons: it's why a **cycle**
(an edge back to an earlier node) is a first-class loop, and it's the unit a runaway-loop
backstop counts (P7).

### P4. State + reducers: the framework merges, you don't mutate.
Instead of mutating shared variables, each node **receives the whole state and returns a
partial update** — a dict of only the fields it wants to change. The framework merges that
update in. *How* a field merges is set by its **reducer**: no reducer means overwrite
(last-write-wins); a reducer means accumulate (append, sum, dedup-merge). Three consequences
that generalize:
- **The reducer is a property of the field, not the node.** Whichever node writes `evidence`,
  the same merge rule applies — so accumulation stays coherent no matter who contributes.
- **The merge is deferred.** A node that needs the *result* of the merge (e.g. "how many
  chunks are genuinely new this round?") can't ask the reducer — it must compute that itself
  against the current state, because the reducer only runs *after* the node returns.
- **A reducer can be a command-interpreter.** Keep one shared reducer but have it inspect the
  update and branch (append / replace-by-id / clear). That's how you get per-situation
  behavior without per-node reducers (the canonical example is a chat framework's
  "append a message, unless it's a delete/replace command").

### P5. Nodes are just functions; the framework doesn't own your work.
A node is `state -> {updates}` — ordinary code. Crucially, an orchestration framework
abstracts the **control flow and nothing else**: it does *not* force you to adopt its LLM
client, its vector store, or its retrieval layer. Your nodes can call your own hand-built
substrate. This is the cleanest line to hold: **let the framework orchestrate; keep owning
the pieces you understand.**

### P6. Branching and loops are edges, not `if`/`while`.
A normal edge is unconditional ("after A, always B"). A **conditional edge** is a *router*: a
pure function `state -> next-node-name`. That's your old `if/elif/break`, extracted into a
function the framework calls to decide where to go. A router that can point back to an earlier
node is what makes a **loop** — the loop is now graph *structure*, not a language construct.

### P7. Two layers of stop condition: your budget, and the framework's backstop.
Your *domain* stop conditions (a round budget, an oscillation guard) live in your routers and
stop the loop cleanly. Separately, the framework enforces a coarse **runaway backstop** (a cap
on total super-steps) that aborts with an error if a cycle never terminates. These are not the
same and shouldn't be conflated: the domain guard is the real logic; the backstop is a seatbelt
for a *bug* in that logic. Set the backstop comfortably *above* what your legitimate budget
needs, so it only ever fires on an actual defect. (This is the same "true unit vs coarse proxy /
defense-in-depth" idea as a spend-cap behind a round-cap — see `docs/harness/HARNESS_ENGINEERING.md`.)

### P8. When to reach for it.
A framework earns its place when your system's shape needs what P2 buys: **cycles**,
**persistence across turns/sessions**, **human-in-the-loop**, **multi-agent** coordination,
**streaming**, or durable/resumable execution. It does *not* earn its place for a single model
call or a strictly linear pipeline — there, it's indirection with no payoff. Match the tool to
the shape, not to its sophistication.

---

## Part 2 — LangGraph concretely (the dialect)

LangGraph is the instance of Part 1 we used. Minimal vocabulary:

| Concept (Part 1) | LangGraph API |
|---|---|
| State schema | a `TypedDict` (or Pydantic model) passed to `StateGraph(Schema)` |
| Reducer | `Annotated[T, reducer_fn]` on a field; no annotation ⇒ overwrite |
| Node | `builder.add_node("name", fn)` where `fn(state) -> dict` |
| Unconditional edge | `builder.add_edge("a", "b")`, with `START` / `END` sentinels |
| Conditional edge (router) | `builder.add_conditional_edges("a", router_fn, {label: node})` |
| Compile / run | `graph = builder.compile()`, then `graph.invoke(initial_state)` |
| Runaway backstop | `graph.invoke(..., config={"recursion_limit": N})` (default 25; `GraphRecursionError` on exceed) |
| Persistence | a **checkpointer** (`InMemorySaver`, `SqliteSaver`, …) passed to `compile()` |
| Observability | **LangSmith** auto-traces — *zero code* (see below) |

A few specifics worth internalizing:

- **`Annotated[T, fn]` is `typing`, not magic.** `Annotated[list, operator.add]` says "the type
  is `list`, and here's metadata: use `operator.add` to merge." To type-checkers it's just
  `list`; LangGraph reads the metadata at graph-build time. It **compiles to a channel** —
  `Annotated[list, operator.add]` → a `BinaryOperatorAggregate` channel, a plain field → a
  `LastValue` channel. There is no `reducers={...}` shortcut on `StateGraph`; `Annotated` (or
  the lower-level `langgraph.channels` API) is the way. If the inline `Annotated` reads noisy,
  alias it: `Evidence = Annotated[list[Hit], merge_evidence]` and use `evidence: Evidence`.

- **Nodes need dependencies that aren't state** (LLMs, a retriever, prompts). Two idioms:
  **closures** — a `build_graph(deps...)` that returns node functions closing over the deps
  (explicit, deps visible) — or LangGraph's `RunnableConfig`/`configurable` (framework-native,
  more indirection). We chose closures.

- **LangSmith tracing is env-var-driven, not code.** A compiled LangGraph *is* a
  langchain-core `Runnable`, so when `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` are in the
  environment (we load them from `.env` via `load_dotenv`), langchain-core auto-attaches a
  tracing callback at `invoke` time and ships a run tree (root `LangGraph` → a run per node) to
  LangSmith in the background. You write *no* tracing code. (The hand-rolled loop, being plain
  Python and not a Runnable, does **not** auto-trace — which is exactly what the framework
  bought us.) This is why we didn't hand-roll a `Span`/`Tracer`: tracing infra is commodity.

---

## Part 3 — This project: the hand-rolled → LangGraph mapping

The capstone re-expressed `agent/loop.py` as `agent/graph.py`, **built alongside** the champion
(not replacing it) and reusing the *entire* substrate. The mapping is the payoff of hand-rolling
first:

| Hand-rolled (`loop.py`) | LangGraph twin (`graph.py`) |
|---|---|
| `run_agent`'s `for round` loop | compiled graph with a **cycle** (`controller → tools → controller`) |
| `Scratchpad` / `steps` / `Usage` | typed `AgentState`, merged by **reducers** (evidence dedup-append *reuses* `Scratchpad.add`; steps append; usage/counts sum) |
| `decide_next_action` (controller) | a **controller node** + a conditional-edge **router** |
| tool dispatch | a **tools node** |
| `answer_from_evidence` + grounding check | a terminal **answer node** |
| `max_rounds` budget, oscillation, empty-finish guards | **router logic** + node logic |
| logging trail | **LangSmith** spans (free) |
| our provider router + cache, retriever, tools, prompts | **reused unchanged** — nodes call them directly |

**What was reused vs replaced.** LangGraph replaced *only the control flow*. The provider
router + temp-0 cache, the retriever, the tools, the prompts, and the core functions
(`decide_next_action`, `answer_from_evidence`) were reused verbatim inside nodes — the concrete
proof of P5.

**A subtlety the port surfaced (P4's deferred-merge).** The oscillation guard needs the count of
*genuinely new* chunks per round. Hand-rolled, `scratchpad.add` dedups inline and returns that
count. In the graph, the `evidence` reducer dedups only *after* the node returns — so the tools
node can't learn the new-count from it. The node had to **dedup the batch itself** against
`state["evidence"]` to drive the guard. That's the general consequence of deferred merging: a
node that acts on the merge result must compute it locally.

**The result.** A cold A/B (`agentic_eval --graph` vs the champion, both `--no-cache`, 25 Q)
gave **24/25 identical verdicts**, identical faithfulness, identical grounding — parity proven
(DD-053). The trajectory differences that remained were **retriever cross-process
non-determinism** (DD-051), not the implementation: both agents call the same decide/answer
functions, so given identical evidence they decide identically.

**Memory, now mapped (by reusing substrate).** Episodic memory is integrated the P5 way: our
validated `EpisodicStore` is reused as-is, and the graph just orchestrates two hooks — a
`memory_read` node at entry (recall top-k → set the `recalled` state field → the controller
injects it as a hint) and a `memory_write` node at exit (write the answered Q→A). They're wired
**only when a store is present**, so the no-memory graph stays byte-identical to the one we proved
parity on. The **framework-native** twin — LangGraph's `BaseStore` (cross-thread memory with
built-in semantic search) or a `checkpointer` (state persistence / resumable threads) — is the
alternative we prototyped and **reverted**, for a sharp transferable reason: **the framework twin
isn't a superset.** LangGraph ships *no* SQLite Store — only `InMemoryStore` (ephemeral,
per-process) and `PostgresStore` (a separate package needing a running server). So rewiring the
graph onto the native Store *lost* the on-disk cross-session durability our `EpisodicStore`
(JSON + cosine) gives for free — the framework assumes you'll bring a real database for
persistence. (The rewrite worked in-process: an `InMemoryStore` with an embedding index,
dependency-injected via `compile(store=…)`, recalled a paraphrase at score 0.877.) The takeaway:
reach for the framework's Store when you want its *infrastructure* — Postgres-scale durability,
per-user namespacing, resumable threads; keep your own when you need lightweight **local**
persistence and only want the graph to call it at the right boundaries. Reusing a working
substrate isn't the lesser choice — here it was the *better* one.

---

## Part 4 — Applying this to a *new* system (the checklist)

1. **Does the shape need a framework at all?** (P8) Cycles / persistence / human-in-the-loop /
   multi-agent / streaming ⇒ maybe. A single call or linear pipeline ⇒ no.
2. **What is my State, and which fields accumulate?** Design the schema first; pick a reducer per
   field (overwrite vs append/sum/merge). Remember reducers are per-*field* and merge *after* the
   node — if a node needs the merged result, it computes it itself.
3. **Where are my branches and loops?** Each becomes a conditional-edge **router** (`state ->
   next node`). Loops are edges back to earlier nodes.
4. **What are my two stop layers?** A domain budget in a router, and the framework's runaway
   backstop set *above* it (so it only catches bugs).
5. **What stays mine?** Keep owning the LLM/retrieval/tools substrate — let the framework
   orchestrate, not annex. Adopt its free machinery (tracing, persistence) rather than hand-rolling
   commodity plumbing.
6. **Can I prove equivalence?** If you're migrating a hand-built system, run the *same eval*
   through both and compare — the migration should preserve behavior, and any divergence is a lead
   (implementation bug vs upstream noise).

If you can answer these, you've transferred the skill: you can pick up *any* graph-orchestration
framework and know what it's doing for you, what it isn't, and whether you should be using it.

---

*See also: `docs/RETROSPECTIVE.md` (where the capstone sits in the arc), `agent/loop.py` (the
hand-rolled twin) and `agent/graph.py` (the port), `docs/harness/HARNESS_ENGINEERING.md` (the
domain-budget-vs-framework-backstop idea, and why tracing is commodity), and `DESIGN_DECISIONS.md`
DD-053 (the parity result).*
