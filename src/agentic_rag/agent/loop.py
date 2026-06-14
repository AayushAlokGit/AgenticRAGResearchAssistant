"""Module 2 v1: a hand-rolled ReAct agent loop over the existing retriever.

The naive pipeline (rag/answer.py) retrieves ONCE and answers — query-blind to its own
results. This wraps that with a model-driven control loop: the model looks at what it has
retrieved so far and decides whether to SEARCH again (with a fresh, reformulated query) or
FINISH and answer. That's retrieve -> reason -> retrieve, the standard ReAct pattern, and
it's what lets a multi-hop question recover from a bad first retrieval.

It is built by hand (CLAUDE.md) to make the four parts of any agent loop explicit:
  1. Tools        — the actions the agent may take. v1: SEARCH(query) and FINISH.
  2. Controller   — decide_next_action: the model picks the next action from the state.
  3. Budget       — max_rounds: a hard cap on search rounds (the circuit breaker).
  4. Stop conds   — FINISH chosen, OR budget spent, OR a search adds nothing new (oscillation).
Plus the SCRATCHPAD (the deduped evidence gathered so far), fed back to the controller each
round so it reasons over history, not just the latest hit.

A thin slice of Module 3 (context engineering) is already here, because the agent itself
creates the pressure that needs it (the unbounded scratchpad blew past the small models'
~6K tokens-per-request ceiling). Two lightweight controls, no summarization yet:
  - the CONTROLLER gets a compact ROUTER VIEW (source + snippet per chunk), not full text —
    it decides what to search next, it doesn't need to re-read everything each round;
  - the FINAL ANSWER is TOKEN-BUDGETED (select_within_budget) so the prompt fits the model.
Heavier context engineering (compression, coverage-aware selection, ordering) stays for
Module 3 proper.

The agent reuses the SAME retriever and the SAME cited-answer prompt as the naive path, so
when the eval A/Bs naive-vs-agentic the ONLY variable that changes is the retrieval control.

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

from agentic_rag.llm.provider import Usage
from agentic_rag.rag.answer import AnswerResult, assemble_context, generate_answer, load_prompt
from agentic_rag.rag.vector_store import Hit

logger = logging.getLogger(__name__)

# How many times to re-ask the controller for a parseable JSON action before giving up and
# FINISHing (a malformed action is an instrument failure, not a reason to crash the loop).
CONTROLLER_MAX_ATTEMPTS = 3

# Chars of each chunk shown to the CONTROLLER in its router view — enough to recognize what
# was found, small enough that the controller prompt stays bounded across many rounds.
CONTROLLER_SNIPPET_CHARS = 300

# Tolerate the model wrapping its JSON in prose or ```json fences: grab the first {...} block.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class AgentStep:
    """One turn of the loop, kept for diagnosis (how many rounds, what queries, did it help)."""
    thought: str
    action: str          # "search" | "finish"
    query: str = ""
    new_chunks: int = 0  # chunks this search ADDED to the scratchpad (0 = no progress)


@dataclass
class Decision:
    """The controller's chosen next action."""
    thought: str
    action: str          # "search" | "finish"
    query: str = ""
    usage: Usage = field(default_factory=Usage)  # controller token cost to reach this decision
                                                  # (sums the retry attempts, if any)


# ───────────────────────────── the controller (part 2) ─────────────────────────────

def decide_next_action(llm, react_prompt: str, question: str, scratchpad: List[Hit],
                       steps: List[AgentStep], rounds_left: int) -> Decision:
    """Ask the model for the next action, re-asking if it returns no parseable JSON.

    If every attempt fails to parse, return a FINISH decision — better to answer from what
    we have than to crash or loop on a broken controller (same fail-safe stance as the evals).
    """
    user_message = build_controller_prompt(question, scratchpad, steps, rounds_left)
    messages = [
        {"role": "system", "content": react_prompt},
        {"role": "user", "content": user_message},
    ]
    # DEBUG (file only): prompt size matters here — this is the value that 413'd past the 6K
    # ceiling before the router_view trimmed it; logging it lets you watch headroom per round.
    logger.debug("agent controller: prompt=%d chars over %d evidence chunk(s), %d round(s) left",
                 len(user_message), len(scratchpad), rounds_left)

    # Sum the cost of every controller attempt (a retry on unparseable output still cost
    # tokens) so the returned Decision carries the true price of reaching it.
    usage = Usage()
    for attempt in range(1, CONTROLLER_MAX_ATTEMPTS + 1):
        completion = llm.complete(messages)
        usage = usage + completion.usage
        raw = completion.text or ""
        logger.debug("agent controller raw (attempt %d/%d): %r",
                     attempt, CONTROLLER_MAX_ATTEMPTS, raw.strip()[:300])
        decision = parse_action(raw)
        if decision is not None:
            decision.usage = usage
            return decision
        logger.warning("agent controller: unparseable action (attempt %d/%d): %r",
                       attempt, CONTROLLER_MAX_ATTEMPTS, raw.strip()[:120])
    return Decision(thought="(controller output unparseable; finishing)", action="finish", usage=usage)


def build_controller_prompt(question: str, scratchpad: List[Hit], steps: List[AgentStep],
                            rounds_left: int) -> str:
    """The dynamic state shown to the controller each turn: question, history, evidence, budget."""
    evidence = router_view(scratchpad)

    search_lines = []
    for i, step in enumerate(steps, start=1):
        if step.action == "search":
            search_lines.append(f'{i}. searched "{step.query}" -> {step.new_chunks} new chunk(s)')
    history = "\n".join(search_lines) if search_lines else "(none)"

    return (
        f"QUESTION:\n{question}\n\n"
        f"SEARCHES ALREADY DONE:\n{history}\n\n"
        f"EVIDENCE GATHERED SO FAR (source + snippet of each chunk):\n{evidence}\n\n"
        f"Search rounds remaining: {rounds_left}. Output the next action as one JSON object."
    )


def router_view(scratchpad: List[Hit]) -> str:
    """A COMPACT view of the evidence for the controller: source + a short snippet per chunk.

    The controller routes (what to search next / whether to finish) — it doesn't write the
    answer, so it doesn't need full chunk text, just enough to see WHAT has been found. This
    keeps the controller prompt bounded no matter how many rounds run, which is what the
    'stuff everything' version failed to do (it 413'd past the 6K-TPM per-request ceiling).
    """
    if not scratchpad:
        return "(nothing retrieved yet)"
    lines = []
    for hit in scratchpad:
        snippet = " ".join(hit.text[:CONTROLLER_SNIPPET_CHARS].split())  # collapse whitespace
        lines.append(f"[{hit.source} #{hit.chunk_index}] {snippet}...")
    return "\n".join(lines)


def parse_action(raw: str) -> Optional[Decision]:
    """Parse the controller's JSON action. Return None if it can't be read (caller retries)."""
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

    action = str(data.get("action", "")).strip().lower()
    thought = str(data.get("thought", "")).strip()
    query = str(data.get("query", "")).strip()

    if action == "search" and query:
        return Decision(thought, "search", query)
    if action == "finish":
        return Decision(thought, "finish")
    return None  # unknown action, or a search with no query


# ───────────────────────────── the loop (parts 1, 3, 4) ─────────────────────────────

def run_agent(question: str, retriever, llm, react_prompt: str, answer_prompt: str,
              top_k: int, max_rounds: int, answer_char_budget: int) -> AnswerResult:
    """Run the retrieve -> reason -> retrieve loop, then answer from the gathered evidence."""
    logger.info("agent: START | budget<=%d round(s), top_k=%d, answer<=%d chars | Q: %s",
                max_rounds, top_k, answer_char_budget, question)
    scratchpad: List[Hit] = []   # the evidence so far, deduped, best-effort ordered by arrival
    seen = set()                 # (source, chunk_index) already in the scratchpad
    steps: List[AgentStep] = []
    exit_reason = "budget"       # default: the for-loop ran out of rounds without a FINISH
    controller_usage = Usage()   # token cost of all the controller (reasoning) calls

    for round_index in range(max_rounds):
        rounds_left = max_rounds - round_index
        logger.info("agent: --- round %d/%d --- scratchpad=%d chunk(s)",
                    round_index + 1, max_rounds, len(scratchpad))

        decision = decide_next_action(llm, react_prompt, question, scratchpad, steps, rounds_left)
        controller_usage = controller_usage + decision.usage
        logger.info("agent: think: %s", decision.thought or "(no thought given)")

        if decision.action == "finish":
            steps.append(AgentStep(decision.thought, "finish"))
            logger.info("agent: action=FINISH (model judged the evidence sufficient)")
            exit_reason = "finish"
            break

        # SEARCH: retrieve with the model's reformulated query, merge only NEW chunks.
        logger.info("agent: action=SEARCH query=%r", decision.query)
        hits = retriever.query(decision.query, top_k)
        new_hits = add_new_hits(hits, scratchpad, seen)
        steps.append(AgentStep(decision.thought, "search", decision.query, len(new_hits)))
        new_sources = ", ".join(f"{h.source}#{h.chunk_index}" for h in new_hits) or "none"
        logger.info("agent: -> %d new chunk(s) [%s] | scratchpad=%d",
                    len(new_hits), new_sources, len(scratchpad))

        # Stop condition (oscillation guard): a search that added nothing new means we're
        # spinning — answer with what we have rather than burn the rest of the budget.
        if not new_hits:
            logger.info("agent: no new evidence this round — stopping early (oscillation guard)")
            exit_reason = "oscillation"
            break

    n_searches = sum(1 for s in steps if s.action == "search")
    logger.info("agent: loop done — %d round(s) used, %d search(es), %d chunk(s) gathered, exit=%s",
                len(steps), n_searches, len(scratchpad), exit_reason)

    # Final answer over the gathered evidence, TRIMMED to fit the model's per-request ceiling
    # (token budgeting). Uses the same cited-answer step as the naive pipeline, so only the
    # retrieval control differs in the A/B.
    if scratchpad:
        selected = select_within_budget(scratchpad, answer_char_budget)
        if len(selected) < len(scratchpad):
            logger.info("agent: compaction — trimmed %d -> %d chunk(s) to fit the %d-char budget",
                        len(scratchpad), len(selected), answer_char_budget)
        result = generate_from_scratchpad(question, selected, llm, answer_prompt)
    else:
        # Finished without retrieving anything — fall back to one direct retrieval so we never
        # answer on empty context.
        logger.info("agent: empty scratchpad — falling back to a single naive retrieval")
        result = generate_answer(question, retriever, llm, answer_prompt, top_k)

    # Total generator cost of this question = all the controller reasoning calls + the final
    # answer call. This is what makes "cost of agency" visible per question (naive=1 call).
    result.usage = controller_usage + result.usage
    preview = result.answer.strip().replace("\n", " ")[:100]
    logger.info("agent: answer ready — %d char(s) from %d chunk(s), %d LLM call(s)/%d tokens: %s",
                len(result.answer), len(result.retrieved), result.usage.calls,
                result.usage.total_tokens, preview)
    return result


def select_within_budget(scratchpad: List[Hit], char_budget: int) -> List[Hit]:
    """Keep chunks (in arrival order) until the running character budget is exhausted.

    Token budgeting, Module-3-lite: the final-answer prompt must fit the model's per-request
    ceiling. Arrival order interleaves the sub-query results (search 1's hits, then search
    2's, ...), so trimming the overflow tends to keep at least the earlier hits of each hop.
    KNOWN RISK: a long multi-hop chain can still lose late-sub-topic evidence — we MEASURE
    multi-hop and add a coverage-aware selector only if it regresses (see DD-019: reranking
    the pool against the whole question would systematically drop second-hop chunks, which is
    why we do NOT do that here). char_budget <= 0 disables trimming.
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


def add_new_hits(hits: List[Hit], scratchpad: List[Hit], seen: set) -> List[Hit]:
    """Append hits not already in the scratchpad; return the newly-added ones (for logging)."""
    new_hits = []
    for hit in hits:
        key = (hit.source, hit.chunk_index)
        if key not in seen:
            seen.add(key)
            scratchpad.append(hit)
            new_hits.append(hit)
    return new_hits


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
                        retrieved=scratchpad, usage=completion.usage)


# ───────────────────── the naive/agentic switch used by the evals ─────────────────────

@dataclass
class Answerer:
    """Answers a question either naively (one retrieval) or via the agent loop.

    Built ONCE (the deps are expensive) and reused per question. `agentic` is the A/B switch:
    same retriever, same final-answer prompt, same model — the ONLY thing that changes is
    whether retrieval is a fixed single shot or a model-driven loop. That keeps the eval a
    clean one-variable comparison (naive baseline vs the agentic loop).
    """
    retriever: object
    llm: object
    answer_prompt: str
    top_k: int
    agentic: bool
    react_prompt: str = ""
    max_rounds: int = 3
    answer_char_budget: int = 10000

    def answer(self, question: str) -> AnswerResult:
        if self.agentic:
            return run_agent(question, self.retriever, self.llm, self.react_prompt,
                             self.answer_prompt, self.top_k, self.max_rounds, self.answer_char_budget)
        return generate_answer(question, self.retriever, self.llm, self.answer_prompt, self.top_k)


def build_answerer(config: dict, retriever, llm) -> Answerer:
    """Construct the Answerer from config; `agent.enabled` decides naive vs agentic."""
    answer_prompt = load_prompt(config, "answer_with_citations")
    top_k = config["retrieval"]["top_k"]
    agent_cfg = config.get("agent", {})
    agentic = bool(agent_cfg.get("enabled", False))
    react_prompt = load_prompt(config, "agent_react") if agentic else ""
    max_rounds = agent_cfg.get("max_rounds", 3)
    answer_char_budget = agent_cfg.get("answer_char_budget", 10000)
    return Answerer(retriever, llm, answer_prompt, top_k, agentic, react_prompt, max_rounds, answer_char_budget)


# ───────────────────────────── manual single-question run ─────────────────────────────

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Run the agentic loop on one question (forces agent on).")
    parser.add_argument("question", nargs="+", help="The question to answer.")
    parser.add_argument("--max-rounds", type=int, default=None, help="Override agent.max_rounds.")
    args = parser.parse_args()

    from agentic_rag.config import load_config
    from agentic_rag.llm.provider import build_llm
    from agentic_rag.logging_setup import configure_run_logging
    from agentic_rag.rag.retriever import build_retriever

    configure_run_logging("agent/loop")
    config = load_config()
    retriever = build_retriever(config)
    llm = build_llm(config, role="generator")
    react_prompt = load_prompt(config, "agent_react")
    answer_prompt = load_prompt(config, "answer_with_citations")
    top_k = config["retrieval"]["top_k"]
    agent_cfg = config.get("agent", {})
    max_rounds = args.max_rounds if args.max_rounds is not None else agent_cfg.get("max_rounds", 3)
    answer_char_budget = agent_cfg.get("answer_char_budget", 10000)

    question = " ".join(args.question)
    result = run_agent(question, retriever, llm, react_prompt, answer_prompt, top_k, max_rounds, answer_char_budget)

    print(f"\nQ: {result.question}\n")
    print(f"A: {result.answer}\n")
    print(f"Gathered {len(result.retrieved)} chunk(s):")
    for i, hit in enumerate(result.retrieved, start=1):
        print(f"  {i}. [{hit.source}] (score={hit.score:.3f})")


if __name__ == "__main__":
    main()
