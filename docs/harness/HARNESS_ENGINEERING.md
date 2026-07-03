# Harness Engineering

> A portable mental model. RAG is the worked example here, but the principles
> apply to **any system that puts a model call into production** — a coding
> agent, a customer-support bot, a batch classifier, a document pipeline, an
> LLM-backed API endpoint. Learn Part 1 as the skill; treat Part 2 as one dialect
> of it; use Part 3 when you start a different system.

This doc exists because of two worries worth taking seriously, from opposite
sides: *is the harness just "plumbing" — retries and logging and config, the
boring boilerplate around the interesting part?* and *is it just a bottomless
grab-bag — retries, caches, rate-limiters, fallbacks, dashboards, feature flags,
circuit breakers, queues — too varied to have a spine?* The honest answer to
both: it's a **structured umbrella**, and it is the layer the other three docs
quietly assume. Context engineering decides *what goes into a call*; memory
decides *what persists across calls*; evaluation decides *whether the answer is
good.* The harness is everything that makes **the call itself dependable** —
the operational shell that turns a single model invocation into a service you
can run, trust, afford, and see inside. Under the sprawl of named techniques
sits a small fixed skeleton: one defining shift (P1), a *closed* set of six
concerns (P2), one reframe that generates most of the work (P3), one safety
pattern with three rings (P4), one rule about units and metering (P5), and one
hard rule about how you measure any of it (P6). So it's neither trivial nor
chaotic — and, like the other disciplines, **how much of it you build is
dictated by your system's shape, not by how production-grade you want to look.**

---

## Part 1 — The general principles

### P1. A harness turns a model *call* into a *service* — and how much you need scales with autonomy × scale × stakes.

A raw `answer = model(prompt)` is a **capability with no guarantees.** It can
fail (rate limits, timeouts, malformed output); it can cost anything; it can take
any amount of time; if it drives tools it can *do* anything; and when it goes
wrong you often can't see why. That's fine for a notebook demo. It is not a
service. The harness is the machinery that wraps that call in the operational
guarantees production needs: it will retry, it will fall back, it won't overspend,
it won't ship a malformed result, and it will tell you what it did.

The key idea is **orthogonality**: the harness is a different axis from
capability. Better prompts, better models, better retrieval improve *what the
answer is*. The harness governs *whether you can depend on getting one at all* —
reliability, cost, safety, observability. You can have a brilliant answer path
wrapped in no harness (a demo) or a mediocre one wrapped in a solid harness (a
boring, dependable service). Production needs both axes.

How *much* harness is not a fixed amount — it scales with three properties of
your system:

- **Autonomy** — the more the model decides its own next step (agentic loops,
  tool use), the more orchestration and safety you need. A single-shot call needs
  almost none; a multi-round agent needs budgets, stop conditions, guardrails.
- **Scale** — the more calls you make, the more caching, reliability, and cost
  metering pay off. One call a day: skip it. A thousand a minute: every ring matters.
- **Stakes** — the more consequential the actions (writes, spends, user-facing
  side-effects), the more the safety ring earns its place.

A low-autonomy, low-scale, low-stakes tool needs a *thin* harness and building a
thick one is wasted motion (the harness has its own failure surface). Crank a
dial and a concern stops being optional.

*Non-RAG anchor:* the engine vs. the rest of the car. A more powerful engine
(model) makes you faster; brakes, airbags, the fuel gauge, and the dashboard
(harness) are what make the car something you'd actually drive on a road. Nobody
ships a car that's all engine.

### P2. The harness is a closed set of six concerns — concentric rings around the model call.

This is the spine. Everything anyone builds under "harness," for any system, is
one of these six — each answering a distinct "what do I need / what could go
wrong" question, each a ring you can add or omit independently:

| Concern | The question it answers | Typical techniques |
|---|---|---|
| **Interface** | Is the I/O well-formed and typed? | tool/function schemas, structured output, argument validation |
| **Orchestration** | How do single calls become a task? | the agent loop, control flow, budgets, stop conditions |
| **Reliability** | What happens when a call fails? | retries + backoff, provider fallback, timeouts, circuit breakers |
| **Efficiency** | Am I paying for the same work twice? | caching (content-addressed), memoization, request batching |
| **Safety** | Can the autonomous system do harm? | guardrails at the input / action / output rings (P4) |
| **Observability** | Can I see what it did and what it cost? | cost + latency meters, tracing, structured logging |

Two things to notice. The rings are **independent** — you can have strong
Reliability and zero Observability, or a great Interface with no Safety — which
is exactly why they're worth naming separately: it lets you audit *which* ring is
missing rather than treating "the harness" as one blob. And they're **not
capability** — none of these changes *what the answer says*; they change whether
you can depend on, afford, and inspect it. That orthogonality is why they get
measured on a different axis (P6).

### P3. The reframe that generates most of the harness: a model call is unreliable I/O, not a function call.

`model(prompt)` *looks* like a function call — you call it, you get a value back.
It *behaves* like a network request to a nondeterministic, rate-limited,
occasionally-failing remote service. That single reframing — **treat the call as
I/O, not computation** — generates most of the harness by itself, because I/O has
a well-known discipline:

- I/O **fails transiently** → retries with backoff, timeouts, a fallback endpoint.
- I/O is **expensive/slow** → cache it, batch it, don't call it twice for the same input.
- I/O is **outside your process** → meter it, log it, trace it; you can't step through it.
- I/O is **untrusted** → validate what comes back before you act on it.

Map those four lines back to P2's rings and you've re-derived Reliability,
Efficiency, Observability, and Safety from first principles. The failure of *not*
making this reframe is the classic one: the demo that works flawlessly on your
machine and falls over in production on the first `429`, the first timeout, the
first time the model returns prose where you expected JSON. It worked because you
tested it like a function; it broke because it was always I/O.

### P4. Safety is a guardrail at three rings — input, action, output — and you observe before you enforce.

The Safety concern (P2) has enough structure to deserve its own principle,
because it's where an *autonomous* system earns trust. A guardrail is a
**constraint checked at a boundary the system cannot cross without passing
through it.** There are exactly three such boundaries:

| Ring | Bounds | Question | Example checks |
|---|---|---|---|
| **Input** | what comes in | Is the request/args well-formed and permitted? | typed/validated arguments, schema conformance, allow-lists |
| **Action** | what the loop does | Can it run away or exceed its blast radius? | spend/round/time caps, rate limits, confirmation gates on side-effects |
| **Output** | what ships | Does the result conform to a checkable contract? | grounding/citation checks, format validation, safety filters, abstention |

Two general rules cut across all three. **Denominate the guardrail in the true
unit** (P5) — an action-ring cap should bound the resource you actually care
about (tokens, dollars, seconds), not a convenient proxy (round-count). And
**observe before you enforce**: a new guardrail should first *flag and measure*
how often it would fire on known-good inputs before it's allowed to *change*
behavior — an over-eager check that mangles correct outputs is a new failure mode,
not a safety win. Prefer a **cheap deterministic check** (does the citation point
at a real retrieved source?) over a second expensive model call (ask a judge if
it's faithful) whenever the contract is mechanically checkable.

### P5. Denominate every control in the resource it governs — and you can't govern what you don't meter.

Every harness control is *about* a resource: a spend-cap is about tokens/dollars,
a latency SLA is about seconds, a rate-limiter is about calls-per-second, a cache
is about repeated work. Two rules follow.

**Meter first.** Observability is not a nice-to-have you add at the end; it is the
**precondition** for every other concern. You cannot cap spend without a token
meter, enforce a latency budget without a latency meter, or even *know* your cache
is helping without a cost meter. The number comes before the control that acts on
it. This is why P2 lists Observability as a peer ring, not an afterthought — it's
the ring the others read from.

**Use the true unit, not a proxy.** A control denominated in a convenient stand-in
leaks. "Stop after N rounds" bounds *iterations*, but if one round can fan out ten
parallel calls, it does not bound *spend* — a proxy guardrail lets the thing you
actually care about escape. Bound spend in tokens, latency in seconds, blast
radius in side-effects. The proxy is fine as a *coarse* backstop; it is not the
guardrail you reach for when the resource matters.

### P6. The harness is measured on the operational axis — cost, latency, reliability — never on correctness.

This is the harness's version of the measurement rule every one of these docs
lands on. A harness change is, by design, **invisible to a correctness eval**: a
cached answer is byte-identical to an uncached one; a retry that eventually
succeeds produces the same output as a first-try success; a fallback tier answers
when the primary would have failed *entirely*; a guardrail that never trips on
good inputs changes nothing you'd see in accuracy. So if you run your accuracy
eval over a harness change and see no movement, that is **not** evidence it did
nothing — it's evidence your instrument is pointed at the wrong axis.

Measure the harness on the axis it actually moves:

- **Cost** — tokens (and dollars), calls, cache hit-rate.
- **Latency** — wall-clock per request, tail latencies.
- **Reliability** — success rate, retry counts, fallback frequency, error taxonomy.

The trap and the fix mirror context-engineering's *silent cost axis* and memory's
*sequenced eval*: **the eval must match the layer being changed.** A caching win
shows up as −90% tokens at *identical* correctness — you have to be watching the
cost line to see it at all, and watching the correctness line to confirm it cost
you nothing. Two axes, read together.

*Non-RAG examples (to prove the levers transfer):*
- **Coding agent.** Interface = a typed `edit_file(path, patch)` schema so the
  model can't emit a malformed edit; Action = a cap on files touched / commands
  run per task; Output = "the patch applies cleanly" before it's shown; Reliability
  = retry the API on a 529; measured on *tokens per task and success rate*, not on
  whether the code is elegant.
- **Support bot.** Safety-output = a filter that blocks answers citing no knowledge-base
  article; Efficiency = cache the embedding + answer for FAQ-shaped questions;
  Observability = per-conversation cost so finance can see the unit economics.
- **Batch classifier.** Reliability + Efficiency dominate (high scale, low autonomy):
  retries, a cheap fallback model, and a cache are the whole harness; there's no
  loop to orchestrate and little to guard.

Same six concerns, same three guardrail rings, same operational axis — different
*skin*. The domain picks the dialect; the skeleton is constant.

---

## Part 2 — This RAG project as one worked instance

Now map Part 1 onto what we've actually built. The general idea is on the left;
this project's dialect and the design-decision it traces to is on the right.

**Our dials (P1) — read them honestly.** *Autonomy:* medium — a real
multi-round retrieve→reason→act loop, so orchestration and guardrails genuinely
apply. *Scale:* low — eval-sized traffic, not production QPS, though the eval
*re-runs* often enough that caching pays for the developer loop. *Stakes:* low —
**every tool is read-only** (search / list_sources / finish), so there is nothing
to gate for side-effects. By P1, a system with these dials wants a **thin-to-medium
harness**: real orchestration and observability, real efficiency for the dev loop,
but a *light* safety ring and no need for the heavy machinery a high-stakes,
high-QPS agent would demand. That is exactly why **Module 5 is scoped as a light
pass** — touch each ring, fill the genuine gaps, make the pattern explicit — not a
production-hardening sprint.

**The six concerns, mapped to this system:**

| Concern | In this project | Status / DD |
|---|---|---|
| **Interface** | typed Pydantic tool args (`search`, `list_sources`, `finish`), validated in `decide_next_action` before any tool runs | ✅ built (Modules 1–2) |
| **Orchestration** | the **hand-rolled** retrieve→reason→act loop with `max_rounds` budget, the oscillation guard, and the empty-finish guard | ✅ built (Module 2; DD-039) — deliberately hand-rolled *before* LangGraph, to learn what the framework abstracts |
| **Reliability** | the multi-tier provider router (Gemini → Groq via `provider_order`), per-provider retries with backoff (`max_retries: 5`), request `timeout` | ✅ built (DD-004/013 lineage) |
| **Efficiency** | content-addressed **LLM response cache** — `sha256(provider+model+params+messages)`, sqlite, temp-0-gated, default on | ✅ Module 5 slice 1 (**DD-051**) |
| **Safety** | input ring = typed args; action ring = round-budget + oscillation guard + a token-denominated **spend-cap**; output ring = the abstention path + an inline **citation-grounding** flag (`harness/guardrails.py`) | ✅ Module 5 slice 2 (**DD-052**) |
| **Observability** | `Usage` (calls + prompt/completion tokens + `latency_s`), per-role cost buckets (controller vs generator vs judge), the Module-3 context-cost eval axis, run JSON | ✅ built incrementally; **tracing** (a span tree) is the optional next slice |

**The honest scoping story.** Notice how much of the harness was *already built
incrementally across Modules 1–4* — typed tools came with the agent loop, the
router came early (DD-004), cost metering grew alongside the evals. That's the
real lesson of Module 5: **a harness is rarely built as "the harness module"; it
accretes ring by ring as the system needs each guarantee.** Module 5 is the pass
where we *name* the rings, audit which are thin, and fill the two or three that
never got their own attention (caching, the latency meter, the guardrail gaps).

**Guardrails, the slice that landed (P4), made concrete.** The three rings, audited:
we already held one guardrail from each — typed args (input), the
round-budget + oscillation guard (action), the abstention path (output). The slice
filled the two real gaps: a **spend-cap** denominated in *controller tokens* rather
than rounds (P5's true-unit rule — batching means one round ≠ one unit of spend),
and an **inline grounding check** that deterministically confirms every
`[filename]` the answer cites is actually in the retrieved evidence (P4's
cheap-deterministic-over-a-judge rule; the eval's LLM faithfulness judge measures
the same thing *offline*, so running it inline would double generator cost for no
new information). Both land **flag-first** (observe before enforce, P4). The
**confirmation-gate** — the classic "ask before a destructive action" guardrail —
is *taught, not built*: every tool here is read-only, so there is nothing to gate;
a web-search or file-writing tool would make it real (YAGNI until then).

A sharp calibration lesson fell out of the spend-cap (DD-052). A circuit breaker is
tuned **above the worst *legitimate* case, never at the average** — a cap at the mean
controller cost (7.3k tok/Q) would trip on ~half the eval; we set it to 30k, ~2× the
observed *max* (15.3k). And you can only read that ceiling from a **cold** run: a warm
cache reports *zero* tokens on a hit (M5-1), so it's blind to the very number you're
calibrating — a small worked example of P5's "meter first, in the true unit." The cap
is deliberately **dormant** on today's eval (`max_rounds=5` already bounds per-question
tokens to ~15k); it's defense-in-depth for what a *round* cap can't see — a batched
round fanning out N searches, a raised `max_rounds`, scratchpad bloat, retry burn, or
adversarial input. A guardrail that never trips in normal operation isn't useless; it's
a circuit breaker doing its job by *not* firing.

**The measurement, made concrete (P6).** DD-051 is the cleanest demonstration of
the operational-axis rule this project has: the cache cut tokens −92.5% and
latency −97.8% while correctness stayed **byte-identical** (e2e/pc/faith
unchanged; a second warm run was a literally-free 100%-hit replay). Read on the
correctness line alone, caching "did nothing." Read on the cost + latency lines —
the meters we threaded through `Usage` and `agentic_eval` for exactly this — it's
a decisive win. The `latency_s` field exists *because* P5 says meter-before-control
and P6 says the harness moves the operational axis.

**What's deferred, and why.** **Native tool-calling** (the provider's own
function-calling API) remains unbuilt. The **LangGraph capstone** (DD-053) re-expressed
the loop's *control flow* as a graph but deliberately **reused the hand-rolled JSON
controller** (`decide_next_action`) unchanged — so it re-derived each hand-built ring and
mapped it to its LangGraph twin (the payoff of having built them by hand first) *without*
adopting native tool-calling. Swapping in the provider's function-calling API would
re-introduce the very abstraction we hand-rolled the loop to understand (CLAUDE.md's
"hand-rolled before framework magic"), so it stays a deliberate, open next step.

---

## Part 3 — Applying this to a *new* system (the anti-overfit checklist)

When you start something that isn't this project, don't reach for "a sqlite
response cache and a Pydantic tool schema." Reach for the questions that
*generated* those choices:

1. **How much harness does this system even need?** Score it on autonomy × scale
   × stakes (P1). A one-shot, low-traffic, read-only tool wants a *thin* harness;
   building a thick one is a self-inflicted failure surface. Don't add a ring the
   dials don't ask for.
2. **Which of the six rings am I missing?** Walk Interface / Orchestration /
   Reliability / Efficiency / Safety / Observability (P2) and name the weakest.
   Auditing ring-by-ring beats treating "the harness" as one undifferentiated blob.
3. **Am I treating the model call as I/O?** (P3) If you're calling it like a
   function — no timeout, no retry, no fallback, no validation of what comes
   back — you have a demo, not a service. Add the I/O discipline before traffic
   finds the gap for you.
4. **What's my true-unit for each control, and am I metering it?** (P5) Bound
   spend in tokens/dollars, latency in seconds, blast radius in side-effects —
   never a proxy. And build the meter *before* the control that reads it; you
   can't cap what you can't count.
5. **For any autonomous action: what are my three guardrail rings — and do I
   observe before I enforce?** (P4) Validate input, bound the loop's blast radius
   in its true unit, verify output against a checkable contract. Prefer a cheap
   deterministic check to a second model call. Flag-and-measure a new guardrail
   before you let it change behavior.
6. **Am I measuring the harness on the operational axis?** (P6) Cost, latency,
   reliability — *not* correctness. If your only eval is an accuracy eval, every
   harness change will read as "no effect," and you'll conclude the instrument's
   blindness is the feature's uselessness.

If you can answer those six for a new system, you've transferred the skill. The
specific techniques (which cache, which retry policy, which schema library, which
tracer) follow from which rings the dials turned up.

---

*See also: `docs/context/CONTEXT_ENGINEERING.md` (what goes **into** a call — the
harness is the machinery **around** it; orthogonal axes), `docs/memory/MEMORY_ENGINEERING.md`
(what **persists** across calls — note caching here is memoization, a different kind
of persistence than learning), `docs/evals/EVALUATION_PRINCIPLES.md` (P3 metric
families — the operational axis P6 reads from), `docs/ProjectIdea.md` (Module 5
spec), and `DESIGN_DECISIONS.md` (DD-051 caching and the guardrails slice as they
land).*
