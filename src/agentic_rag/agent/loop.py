"""Module 2: a hand-rolled, MULTI-TOOL ReAct agent loop over the retriever.

The naive pipeline (rag/answer.py) retrieves ONCE and answers — blind to its own results.
This wraps that with a model-driven control loop: the controller looks at what it has done
and gathered so far and chooses the next ACTION from a set of TOOLS — search again, pull a
whole document, list the corpus, or finish and answer. That's retrieve -> reason -> act, the
ReAct pattern, and it's what lets a multi-hop question recover from a bad first retrieval.

Built by hand (CLAUDE.md) to make the four parts of any agent loop explicit:
  1. Action space — the TOOLS the agent may take, held in a registry (see agent/tools.py).
  2. Controller   — decide_next_action: the model picks the next tool + args from the state.
  3. Budget       — max_rounds: a hard cap on rounds (the circuit breaker).
  4. Stop conds   — finish chosen, OR budget spent, OR a retrieval re-found only old evidence.
Plus the SCRATCHPAD (deduped evidence) and an OBSERVATION history (what each action returned),
both fed back to the controller each round so it reasons over the whole trajectory.

A1 GENERALIZATION: the controller no longer emits a fixed {action: search|finish, query};
it emits {action: <tool name>, args: {...}} and we DISPATCH through the registry. Adding a
tool is a registry entry, not a new branch here — which is exactly the seam the provider's
tool-use API will slot into (A2). Which tools are available is config (agent.tools), so the
action space is an eval-gated knob; the default ['search','finish'] reproduces the pre-A1
champion, so the A/B isolates "richer action space" as the one variable.

Context engineering still lives here (Module-3-lite): the controller sees a COMPACT router
view (source + snippet per chunk) not full text, and the final answer is char-budgeted to
fit the model's request ceiling. Heavier compaction stays for Module 3 proper.

Run a single question and watch the loop:
    python -m agentic_rag.agent.loop "your multi-hop question here"
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel, ValidationError

from agentic_rag.agent.tools import DEFAULT_TOOLS, Tool, ToolContext, ToolRegistry, build_registry
from agentic_rag.llm.provider import Usage
from agentic_rag.rag.answer import (AnswerResult, Trajectory, assemble_context, generate_answer,
                                    load_prompt)
from agentic_rag.rag.vector_store import Hit

logger = logging.getLogger(__name__)

# How many times to re-ask the controller for a parseable+valid action before giving up and
# FINISHing (a malformed/invalid action is an instrument failure, not a reason to crash). In
# A3 we'll instead feed the error back as an observation so the agent can self-correct.
CONTROLLER_MAX_ATTEMPTS = 3

# How many CONSECUTIVE redundant retrievals (a search that re-found only evidence we already
# hold) we tolerate before tripping the oscillation guard and stopping. One redundant search
# is a local stumble — the agent's phrasing missed, and the right move is to REFORMULATE, not
# abort (a multi-hop question often needs a second, differently-worded retrieval). We only
# conclude the agent is genuinely STUCK after this many in a row. The count resets to 0 on any
# retrieval that adds new evidence, so a productive round clears the slate.
OSCILLATION_PATIENCE = 2

# Chars of each chunk shown to the CONTROLLER in its router view — enough to recognize what
# was found, small enough that the controller prompt stays bounded across many rounds.
# KNOWN LIMITATION (DD-030): this is a PREFIX snippet, so a fact past char 300 is invisible to
# the CONTROLLER (the GENERATOR still sees full hit.text, so answers aren't lost — only routing
# is blind to chunk tails). Fix if it ever bites: a relevance-CENTERED snippet (show the
# query-matching span, not the prefix). Deferred — no measured failure traces to this yet.
CONTROLLER_SNIPPET_CHARS = 300

# Tolerate the model wrapping its JSON in prose or ```json fences: grab the first {...} block.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class AgentStep:
    """One turn of the loop, kept for the observation history and diagnosis."""
    thought: str
    action: str           # the tool name chosen
    args: str = ""        # human-readable args (e.g. 'query=...'), for history/logging
    observation: str = "" # what the tool returned (shown to the controller next round)
    new_chunks: int = 0   # evidence chunks this action ADDED to the scratchpad (0 = no progress)


@dataclass
class Scratchpad:
    """The evidence gathered across rounds: deduped, kept in arrival order.

    Encapsulates an invariant that was previously maintained by hand with two parallel
    structures (a list + a `seen` set kept in sync by a free function): a chunk is added at
    most once, identified by (source, chunk_index). One type owns the rule.
    """
    hits: List[Hit] = field(default_factory=list)
    _seen: set = field(default_factory=set)  # (source, chunk_index) already held

    def add(self, hits: List[Hit]) -> List[Hit]:
        """Add hits not already held; return the newly-added ones (for logging/stop conditions)."""
        new_hits = []
        for hit in hits:
            key = (hit.source, hit.chunk_index)
            if key not in self._seen:
                self._seen.add(key)
                self.hits.append(hit)
                new_hits.append(hit)
        return new_hits

    def __len__(self) -> int:
        return len(self.hits)


@dataclass
class Decision:
    """The controller's chosen next action, already parsed AND validated against the registry."""
    thought: str
    tool: Tool
    args: BaseModel                              # validated Pydantic args for `tool`
    usage: Usage = field(default_factory=Usage)  # controller token cost (sums retry attempts)
    failed_attempts: int = 0                     # parse/validate failures before this decision


# ───────────────────────────── the controller (part 2) ─────────────────────────────

def decide_next_action(controller_llm, react_prompt: str, registry: ToolRegistry, question: str,
                       scratchpad: List[Hit], steps: List[AgentStep],
                       rounds_left: int) -> Decision:
    """Ask the model for the next action; re-ask if it returns no parseable+valid tool call.

    An action is valid only if it (a) parses as JSON, (b) names a known tool, and (c) its args
    satisfy that tool's schema. If every attempt fails, return a FINISH decision — better to
    answer from what we have than crash or spin on a broken controller (the evals' fail-safe).
    """
    user_message = build_controller_prompt(question, scratchpad, steps, rounds_left)
    messages = [
        {"role": "system", "content": react_prompt},
        {"role": "user", "content": user_message},
    ]
    # DEBUG (file only): prompt size matters — this is the value that 413'd past the old 6K
    # ceiling before the router view trimmed it; logging it lets you watch headroom per round.
    logger.debug("agent controller: prompt=%d chars over %d evidence chunk(s), %d round(s) left",
                 len(user_message), len(scratchpad), rounds_left)

    usage = Usage()      # sum the cost of every attempt (a re-ask still cost tokens)
    failed = 0
    for attempt in range(1, CONTROLLER_MAX_ATTEMPTS + 1):
        completion = controller_llm.complete(messages)
        usage = usage + completion.usage
        raw = completion.text or ""
        logger.debug("agent controller raw (attempt %d/%d): %r",
                     attempt, CONTROLLER_MAX_ATTEMPTS, raw.strip()[:300])
        decision = parse_and_validate(raw, registry)
        if decision is not None:
            decision.usage = usage
            decision.failed_attempts = failed
            return decision
        failed += 1
        logger.warning("agent controller: unparseable/invalid action (attempt %d/%d): %r",
                       attempt, CONTROLLER_MAX_ATTEMPTS, raw.strip()[:120])
    return Decision(thought="(controller output unparseable; finishing)", tool=registry.get("finish"),
                    args=registry.get("finish").args_model(), usage=usage, failed_attempts=failed)


def build_controller_prompt(question: str, scratchpad: List[Hit],
                            steps: List[AgentStep], rounds_left: int) -> str:
    """The dynamic state shown to the controller each turn: question, history, evidence, budget.

    The available TOOLS are NOT here — they're injected once into the system prompt (a fixed
    instruction), while this user message carries only what changes round to round.
    """
    return (
        f"QUESTION:\n{question}\n\n"
        f"ACTIONS TAKEN SO FAR (and what each returned):\n{format_history(steps)}\n\n"
        f"EVIDENCE GATHERED SO FAR (source + snippet of each chunk):\n{router_view(scratchpad)}\n\n"
        f"Rounds remaining: {rounds_left}. Choose ONE action and output it as a single JSON object."
    )


def format_history(steps: List[AgentStep]) -> str:
    """Render the action+observation trail so the controller can see what it already tried
    (and not repeat a query, or re-list sources) — observations are how non-evidence tools
    like list_sources feed information back into the loop."""
    if not steps:
        return "(none yet)"
    lines = []
    for i, step in enumerate(steps, start=1):
        call = step.action if not step.args else f"{step.action}({step.args})"
        lines.append(f"{i}. {call} -> {step.observation}")
    return "\n".join(lines)


def router_view(scratchpad: List[Hit]) -> str:
    """A COMPACT view of the evidence for the controller: source + a short snippet per chunk.

    The controller routes (what to do next) — it doesn't write the answer, so it doesn't need
    full chunk text, just enough to see WHAT has been found. This keeps the controller prompt
    bounded no matter how many rounds run (the 'stuff everything' version 413'd past the ceiling).
    """
    if not scratchpad:
        return "(nothing retrieved yet)"
    lines = []
    for hit in scratchpad:
        snippet = " ".join(hit.text[:CONTROLLER_SNIPPET_CHARS].split())  # collapse whitespace
        lines.append(f"[{hit.source} #{hit.chunk_index}] {snippet}...")
    return "\n".join(lines)


def parse_and_validate(raw: str, registry: ToolRegistry) -> Optional[Decision]:
    """Parse the controller's JSON action and validate it against the registry.

    Returns a Decision (tool + validated args) on success, or None if it can't be read or
    isn't a valid call (caller re-asks). Three failure modes collapse to None: no JSON,
    unknown tool, args that don't satisfy the tool's schema.
    """
    if not raw:
        return None
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    action = str(data.get("action", "")).strip()
    thought = str(data.get("thought", "")).strip()
    raw_args = data.get("args", {})
    if not isinstance(raw_args, dict):
        raw_args = {}

    tool = registry.get(action)
    if tool is None:
        return None  # hallucinated/unknown tool name
    try:
        args = tool.validate_args(raw_args)
    except ValidationError:
        return None  # right tool, bad/missing arguments
    return Decision(thought=thought, tool=tool, args=args)


# ───────────────────────────── the loop (parts 1, 3, 4) ─────────────────────────────

def run_agent(question: str, retriever, llm, react_prompt: str, answer_prompt: str,
              top_k: int, max_rounds: int, answer_char_budget: int,
              registry: ToolRegistry, store, controller_llm=None) -> AnswerResult:
    """Run the retrieve -> reason -> act loop, then answer from the gathered evidence.

    `llm` writes the final answer (the generator); `controller_llm` makes the routing
    decisions (which tool, with what args). They may be DIFFERENT models — we tier compute by
    task difficulty (DD-025); defaults to the same model for both (the champion) when not split.
    `registry` is the action space; `store` backs the tools that read the corpus directly.
    """
    if controller_llm is None:
        controller_llm = llm
    # Inject the available tools into the system prompt ONCE (a fixed instruction). Plain
    # string replace, not .format(), so literal JSON braces elsewhere in the prompt are safe.
    react_prompt = react_prompt.replace("{tools}", registry.render_for_prompt())
    ctx = ToolContext(retriever=retriever, store=store, top_k=top_k)

    logger.info("agent: START | budget<=%d round(s), tools=%s, top_k=%d, answer<=%d chars | Q: %s",
                max_rounds, registry.names(), top_k, answer_char_budget, question)
    scratchpad = Scratchpad()    # the evidence so far, deduped, ordered by arrival
    steps: List[AgentStep] = []
    exit_reason = "budget"       # default: the loop ran out of rounds without a FINISH
    controller_usage = Usage()
    tool_calls: dict = {}        # tool name -> times invoked (trajectory)
    tool_errors = 0              # parse/validate failures across all rounds (trajectory)
    redundant_searches = 0       # retrievals that re-found only old evidence (trajectory, cumulative)
    consecutive_redundant = 0    # redundant retrievals IN A ROW; resets on progress (oscillation guard)

    for round_index in range(max_rounds):
        rounds_left = max_rounds - round_index
        logger.info("agent: --- round %d/%d --- scratchpad=%d chunk(s)",
                    round_index + 1, max_rounds, len(scratchpad))

        decision = decide_next_action(controller_llm, react_prompt, registry, question,
                                      scratchpad.hits, steps, rounds_left)
        controller_usage = controller_usage + decision.usage
        tool_errors += decision.failed_attempts
        tool = decision.tool
        tool_calls[tool.name] = tool_calls.get(tool.name, 0) + 1
        logger.info("agent: think: %s", decision.thought or "(no thought given)")

        if tool.terminal:  # finish
            steps.append(AgentStep(decision.thought, tool.name, observation="(finish)"))
            logger.info("agent: action=FINISH (model judged the evidence sufficient)")
            exit_reason = "finish"
            break

        # DISPATCH the chosen tool, then merge any evidence it produced (deduped).
        args_repr = ", ".join(f"{k}={v}" for k, v in decision.args.model_dump().items())
        logger.info("agent: action=%s args={%s}", tool.name, args_repr)
        result = tool.run(decision.args, ctx)
        new_hits = scratchpad.add(result.hits)

        # A retrieval that returned hits but added NOTHING new re-found only evidence we already
        # hold. Detect it once here; it drives both the controller feedback (next) and the
        # oscillation streak (below). A no-hit tool like list_sources legitimately adds nothing,
        # so a round is "redundant" only when the tool DID return hits and all were duplicates.
        is_redundant = bool(result.hits and not new_hits)

        # FEED THE SIGNAL BACK: append a note to the observation the controller will see next
        # round (via format_history). Without this, the stored observation just says "found N
        # chunks", so at temperature 0 the controller has no reason to change course and may
        # re-issue the same query. Telling it "these were duplicates — reformulate/switch/finish"
        # is what turns the extra round Change 1 bought into an actual recovery.
        observation = result.observation
        if is_redundant:
            observation += ("\n[NOTE: NO NEW EVIDENCE FOUND. Reformulate with DIFFERENT terms, try another tool, or"
                            " finish if you already have enough to answer.]")
        steps.append(AgentStep(decision.thought, tool.name, args_repr, observation, len(new_hits)))
        logger.info("agent: -> %s | +%d new chunk(s) | scratchpad=%d",
                    observation.splitlines()[0], len(new_hits), len(scratchpad))

        # Stop condition (oscillation guard): ONE redundant round is a local stumble (bad
        # phrasing), not proof the agent is stuck — so we count CONSECUTIVE redundant rounds and
        # only stop once they reach OSCILLATION_PATIENCE. Any retrieval that DID add evidence
        # resets the count, so the trip means "several unproductive rounds in a row", not "one
        # duplicate ever".
        if is_redundant:
            redundant_searches += 1
            consecutive_redundant += 1
            if consecutive_redundant >= OSCILLATION_PATIENCE:
                logger.info("agent: %d redundant retrieval(s) in a row — stopping (oscillation guard)",
                            consecutive_redundant)
                exit_reason = "oscillation"
                break
            logger.info("agent: retrieval re-found only known evidence (%d/%d before stop) — continuing",
                        consecutive_redundant, OSCILLATION_PATIENCE)
        elif new_hits:
            consecutive_redundant = 0  # progress clears the slate

    n_actions = len(steps)
    logger.info("agent: loop done — %d round(s) used, calls=%s, %d chunk(s) gathered, exit=%s",
                n_actions, tool_calls, len(scratchpad), exit_reason)

    result = answer_from_evidence(question, scratchpad, retriever, llm, answer_prompt,
                                  top_k, answer_char_budget)

    # Attribute the controller (routing) cost to its own bucket; the generator bucket was set
    # by the answer call above. Kept separate so a tiered run shows each role's cost (DD-025).
    result.controller_usage = controller_usage
    result.trajectory = Trajectory(rounds_used=n_actions, exit_reason=exit_reason,
                                   tool_calls=tool_calls, tool_errors=tool_errors,
                                   redundant_searches=redundant_searches)
    total_calls = result.controller_usage.calls + result.generator_usage.calls
    total_tokens = result.controller_usage.total_tokens + result.generator_usage.total_tokens
    preview = result.answer.strip().replace("\n", " ")[:100]
    logger.info("agent: answer ready — %d char(s) from %d chunk(s), %d LLM call(s)/%d tokens: %s",
                len(result.answer), len(result.retrieved), total_calls, total_tokens, preview)
    return result


def select_within_budget(scratchpad: List[Hit], char_budget: int) -> List[Hit]:
    """Keep chunks (in arrival order) until the running character budget is exhausted.

    Token budgeting, Module-3-lite: the final-answer prompt must fit the model's per-request
    ceiling. Arrival order interleaves the tool results, so trimming the overflow tends to
    keep at least the earlier evidence of each hop. KNOWN RISK: a long chain can still lose
    late-sub-topic evidence — we MEASURE multi-hop and add a coverage-aware selector only if
    it regresses (see DD-019/DD-023). char_budget <= 0 disables trimming.
    """
    if char_budget <= 0:
        return list(scratchpad)
    selected = []
    used = 0
    for hit in scratchpad:
        # Always keep at least one chunk, even if it alone exceeds the budget.
        if selected and used + len(hit.text) > char_budget:
            break
        selected.append(hit)
        used += len(hit.text)
    return selected


def answer_from_evidence(question: str, scratchpad: Scratchpad, retriever, llm, answer_prompt: str,
                         top_k: int, answer_char_budget: int) -> AnswerResult:
    """Write the final cited answer from the gathered evidence (the loop's second job).

    The evidence is TRIMMED to fit the model's per-request ceiling (token budgeting) and
    answered with the same cited-answer step as the naive pipeline. If the loop finished
    without retrieving anything, fall back to one direct retrieval so we never answer on
    empty context.
    """
    if not scratchpad.hits:
        logger.info("agent: empty scratchpad — falling back to a single naive retrieval")
        return generate_answer(question, retriever, llm, answer_prompt, top_k)
    selected = select_within_budget(scratchpad.hits, answer_char_budget)
    if len(selected) < len(scratchpad):
        logger.info("agent: compaction — trimmed %d -> %d chunk(s) to fit the %d-char budget",
                    len(scratchpad), len(selected), answer_char_budget)
    return generate_from_scratchpad(question, selected, llm, answer_prompt)


def generate_from_scratchpad(question: str, scratchpad: List[Hit], llm, answer_prompt: str) -> AnswerResult:
    """Generate the cited answer from the agent's gathered evidence (no further retrieval)."""
    context = assemble_context(scratchpad)
    user_message = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
    # DEBUG (file only): the final-answer prompt is the OTHER place that must fit the model's
    # per-request ceiling — select_within_budget keeps it bounded, this lets you watch it.
    logger.debug("agent answer: prompt=%d chars over %d chunk(s)", len(user_message), len(scratchpad))
    completion = llm.complete([
        {"role": "system", "content": answer_prompt},
        {"role": "user", "content": user_message},
    ])
    return AnswerResult(question=question, answer=completion.text or "",
                        retrieved=scratchpad, generator_usage=completion.usage)


# ───────────────────── the naive/agentic switch used by the evals ─────────────────────

@dataclass
class Answerer:
    """Answers a question either naively (one retrieval) or via the agent loop.

    Built ONCE (the deps are expensive) and reused per question. `agentic` is the A/B switch:
    same retriever, same final-answer prompt, same model — what changes is whether retrieval
    is a fixed single shot or a model-driven, multi-tool loop. `registry`/`store` back the
    agent's action space (unused on the naive path).
    """
    retriever: object
    llm: object              # the GENERATOR (writes the final answer)
    answer_prompt: str
    top_k: int
    agentic: bool
    react_prompt: str = ""
    max_rounds: int = 3
    answer_char_budget: int = 10000
    controller_llm: object = None     # the agent's routing brain; defaults to llm (same model)
    registry: ToolRegistry = None     # the agent's action space (agent.tools)
    store: object = None              # backs corpus-reading tools (expand_document, list_sources)

    def answer(self, question: str) -> AnswerResult:
        if self.agentic:
            return run_agent(question, self.retriever, self.llm, self.react_prompt,
                             self.answer_prompt, self.top_k, self.max_rounds,
                             self.answer_char_budget, self.registry, self.store,
                             controller_llm=self.controller_llm)
        return generate_answer(question, self.retriever, self.llm, self.answer_prompt, self.top_k)


def build_agent_deps(config: dict, tool_names: List[str]):
    """Build the agent's two corpus-facing dependencies: the action space and a store handle.

    Shared by `build_answerer` and the CLI `main` so the wiring lives in one place (change the
    store construction once). The store is built here, not dug out of the retriever, so the
    tools that read the corpus directly (expand_document, list_sources) have an explicit handle.
    """
    from agentic_rag.config import resolve_path
    from agentic_rag.rag.vector_store import ChromaVectorStore

    registry = build_registry(tool_names)
    store = ChromaVectorStore(resolve_path(config["vector_store"]["path"]),
                              config["vector_store"]["collection"])
    return registry, store


def build_answerer(config: dict, retriever, llm, controller_llm=None) -> Answerer:
    """Construct the Answerer from config; `agent.enabled` decides naive vs agentic.

    `llm` is the generator. `controller_llm` (optional) is the routing model — pass a
    separate one to tier the agent (DD-025); if None, the controller reuses the generator.
    """
    answer_prompt = load_prompt(config, "answer_with_citations")
    top_k = config["retrieval"]["top_k"]
    agent_cfg = config.get("agent", {})
    agentic = bool(agent_cfg.get("enabled", False))
    react_prompt = load_prompt(config, "agent_react") if agentic else ""
    max_rounds = agent_cfg.get("max_rounds", 3)
    answer_char_budget = agent_cfg.get("answer_char_budget", 10000)

    registry = None
    store = None
    if agentic:
        # The action space (agent.tools); default reproduces the pre-A1 champion (search+finish).
        registry, store = build_agent_deps(config, agent_cfg.get("tools", DEFAULT_TOOLS))

    return Answerer(retriever, llm, answer_prompt, top_k, agentic, react_prompt, max_rounds,
                    answer_char_budget, controller_llm=controller_llm, registry=registry, store=store)


# ───────────────────────────── manual single-question run ─────────────────────────────

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Run the agentic loop on one question (forces agent on).")
    parser.add_argument("question", nargs="+", help="The question to answer.")
    parser.add_argument("--max-rounds", type=int, default=None, help="Override agent.max_rounds.")
    parser.add_argument("--tools", default=None,
                        help="Comma-separated tool set to use (e.g. search,expand_document,list_sources,finish).")
    args = parser.parse_args()

    from agentic_rag.config import load_config
    from agentic_rag.llm.provider import build_llm
    from agentic_rag.logging_setup import configure_run_logging
    from agentic_rag.rag.retriever import build_retriever

    configure_run_logging("agent/loop")
    config = load_config()
    retriever = build_retriever(config)
    llm = build_llm(config, role="generator")
    controller_llm = build_llm(config, role="controller")
    react_prompt = load_prompt(config, "agent_react")
    answer_prompt = load_prompt(config, "answer_with_citations")
    top_k = config["retrieval"]["top_k"]
    agent_cfg = config.get("agent", {})
    max_rounds = args.max_rounds if args.max_rounds is not None else agent_cfg.get("max_rounds", 3)
    answer_char_budget = agent_cfg.get("answer_char_budget", 10000)
    tool_names = [t.strip() for t in args.tools.split(",")] if args.tools else agent_cfg.get("tools", DEFAULT_TOOLS)
    registry, store = build_agent_deps(config, tool_names)

    question = " ".join(args.question)
    result = run_agent(question, retriever, llm, react_prompt, answer_prompt, top_k, max_rounds,
                       answer_char_budget, registry, store, controller_llm=controller_llm)

    print(f"\nQ: {result.question}\n")
    print(f"A: {result.answer}\n")
    if result.trajectory:
        t = result.trajectory
        print(f"Trajectory: {t.rounds_used} round(s), exit={t.exit_reason}, calls={t.tool_calls}, "
              f"tool_errors={t.tool_errors}, redundant={t.redundant_searches}")
    print(f"Gathered {len(result.retrieved)} chunk(s):")
    for i, hit in enumerate(result.retrieved, start=1):
        print(f"  {i}. [{hit.source}] (score={hit.score:.3f})")


if __name__ == "__main__":
    main()
