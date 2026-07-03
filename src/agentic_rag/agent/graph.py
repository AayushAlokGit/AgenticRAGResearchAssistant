"""The agent loop, re-expressed as a LangGraph graph (the capstone).

Built ALONGSIDE `agent/loop.py`, not replacing it: the hand-rolled loop stays the champion
until this proves parity on the eval. The point is to see exactly what LangGraph abstracts —
so we keep the ENTIRE hand-rolled substrate (the provider router + temp-0 cache, the
retriever, the tools, the prompts, the EpisodicStore) and let LangGraph replace ONLY the
control flow. LangGraph orchestrates; it does not force LangChain's LLM/vector-store layer.

The mapping (hand-built -> LangGraph twin):
  run_agent's `for round` loop      -> a compiled graph with a CYCLE (controller -> tools -> controller)
  Scratchpad / steps / Usage        -> this typed STATE, merged by REDUCERS
  decide_next_action (controller)   -> a controller NODE + a conditional edge (router)   [step 4/5]
  tool dispatch                     -> a tools NODE                                       [step 4]
  answer_from_evidence              -> a terminal answer NODE                             [step 4]
  max_rounds / oscillation guard    -> router logic + recursion_limit                     [step 5]

THIS FILE, so far: step 3 — the State and its reducers only. Nodes/edges land next.
"""
from __future__ import annotations

import operator
from typing import Annotated, List, TypedDict

from langgraph.graph import END, START, StateGraph

from agentic_rag.agent.loop import (OSCILLATION_PATIENCE, AgentStep, Decision, Scratchpad,
                                    answer_from_evidence, decide_next_action, provenance_label)
from agentic_rag.agent.tools import ToolContext
from agentic_rag.harness.guardrails import check_citation_grounding
from agentic_rag.llm.provider import Usage
from agentic_rag.rag.vector_store import Hit


# ── Reducers ──────────────────────────────────────────────────────────────────────────
# A reducer decides how a state field MERGES when a node returns an update for it. No
# reducer = OVERWRITE (last write wins); a reducer lets updates ACCUMULATE. These three are
# where the graph reuses the hand-rolled invariants instead of re-deriving them.

def merge_evidence(existing: List[Hit], new: List[Hit]) -> List[Hit]:
    """Dedup-append new evidence onto the old — the Scratchpad invariant, reused verbatim.

    A chunk is held at most once, keyed by (source, chunk_index), earliest occurrence wins.
    Rather than re-implement that here, seed a Scratchpad with what we have and add the new
    hits: `Scratchpad.add` owns the rule (it also preserves each hit's provenance label).
    """
    pad = Scratchpad()
    pad.add(existing)   # already-labeled hits; add() won't re-stamp provenance
    pad.add(new)        # dedup-append; duplicates of held chunks are dropped
    return pad.hits


def add_usage(existing: Usage, new: Usage) -> Usage:
    """Sum token/latency cost across nodes — Usage already defines `+` (calls, tokens, latency)."""
    return existing + new


def merge_counts(existing: dict, new: dict) -> dict:
    """Accumulate a name->count map (tool_calls) across rounds."""
    merged = dict(existing)
    for name, count in new.items():
        merged[name] = merged.get(name, 0) + count
    return merged


# ── State ─────────────────────────────────────────────────────────────────────────────
# run_agent's local variables, promoted to a typed dict that flows through the graph. Each
# node reads it and returns a PARTIAL update; the reducer above (or overwrite) merges it in.

class AgentState(TypedDict):
    question: str                                        # set once (overwrite)
    evidence: Annotated[List[Hit], merge_evidence]       # scratchpad: dedup-append
    steps: Annotated[List[AgentStep], operator.add]      # action/observation trail: append
    controller_usage: Annotated[Usage, add_usage]        # routing cost: summed across rounds
    tool_calls: Annotated[dict, merge_counts]            # name->count, for the trajectory
    rounds: int                                          # rounds taken so far (overwrite; controller ++)
    exit_reason: str                                     # finish | budget | oscillation | ... (overwrite)
    answer: str                                          # final cited answer (overwrite; answer node)
    pending: list                                        # retrieval Decisions the controller chose, for the tools node (overwrite)
    seeded_on_empty: bool                                # empty-finish guard has fired (fires at most once)
    consecutive_redundant: int                           # redundant rounds IN A ROW (overwrite; resets on progress)
    redundant_searches: Annotated[int, operator.add]     # cumulative redundant rounds (trajectory)
    # Set once by the answer node — what the AnswerResult adapter needs beyond `answer`:
    retrieved: list                                      # the trimmed+ordered window the generator saw
    generator_usage: Usage                               # the answer call's token/latency cost
    grounded: bool                                       # output-ring guardrail: no fabricated citations
    ungrounded_citations: list
    # Deferred: recalled -> episodic memory hint (memory step)


# ── Nodes ─────────────────────────────────────────────────────────────────────────────
# A node is `state -> {partial update}`. Each one wraps an EXISTING hand-rolled function;
# the graph only re-orchestrates. Nodes need deps (LLMs, retriever, tools, prompts) that
# aren't part of the flowing state, so we build them as CLOSURES over those deps (chosen
# over LangGraph's RunnableConfig — plainer, and the deps stay visible).

def make_nodes(retriever, controller_llm, generator_llm, registry, store,
               react_prompt: str, answer_prompt: str, top_k: int, max_rounds: int,
               answer_char_budget: int, ordering: str = "arrival"):
    """Return (controller_node, tools_node, answer_node), each closing over the deps above."""
    # Inject the tool list into the system prompt ONCE (a fixed instruction), same as run_agent.
    react_prompt = react_prompt.replace("{tools}", registry.render_for_prompt())
    ctx = ToolContext(retriever=retriever, store=store, top_k=top_k)

    def controller_node(state: AgentState) -> dict:
        """DECIDE — ask the controller for the next action(s). Twin of run_agent's decide + batch split.

        `finish` is honored only when SOLE: if the batch also has retrieval actions, we run those and
        drop the finish (you don't gather and stop in one breath). No retrieval left => we're done.
        """
        rounds_left = max_rounds - state["rounds"]
        turn = decide_next_action(controller_llm, react_prompt, registry, state["question"],
                                  state["evidence"], state["steps"], rounds_left)
        retrieval = [d for d in turn.decisions if not d.tool.terminal]
        update = {
            "controller_usage": turn.usage,
            "rounds": state["rounds"] + 1,
            "pending": retrieval,
        }
        if not retrieval:  # finish-only turn
            update["tool_calls"] = {"finish": 1}
            # EMPTY-FINISH GUARD (DD-039): never answer from an empty scratchpad. If the controller
            # picks finish before ANY evidence exists, seed one search and let it decide again with
            # evidence in hand — instead of degrading to a naive single retrieval. Fires at most once
            # (seeded_on_empty), so it can't spin. In the graph this = inject a search into `pending`
            # so the existing router sends us to `tools` and back to `controller`.
            if not state["evidence"] and not state.get("seeded_on_empty"):
                search = registry.get("search")
                seed = Decision(thought="[empty-finish guard] seed a search on the question",
                                tool=search, args=search.args_model(query=state["question"]))
                update["pending"] = [seed]
                update["seeded_on_empty"] = True
                # The step records what actually happened here — the guard firing — NOT "finish"
                # (the controller ATTEMPTED finish, but it was overridden; the seeded search itself
                # gets its own step from the tools node).
                update["steps"] = [AgentStep(turn.decisions[0].thought, "empty-finish-guard",
                                   observation="[GUARD: controller chose finish with no evidence — "
                                               "overrode it and seeded a search on the question.]")]
            else:
                finish = turn.decisions[0]
                update["exit_reason"] = "finish"
                update["steps"] = [AgentStep(finish.thought, "finish", observation="(finish)")]
        return update

    def tools_node(state: AgentState) -> dict:
        """ACT — run the retrieval actions the controller chose, plus the round-level oscillation guard.

        `round_new` (chunks genuinely new this round) must be computed HERE: the evidence reducer
        dedups only AFTER the node returns, so the node can't learn the new-count from it — it dedups
        the batch itself against state["evidence"] (and within the batch) to drive the guard.
        """
        existing_keys = {(h.source, h.chunk_index) for h in state["evidence"]}
        seen_this_round: set = set()
        new_hits: list[Hit] = []
        steps: list[AgentStep] = []
        counts: dict = {}
        round_new = 0
        round_returned_hits = False
        for d in state["pending"]:
            result = d.tool.run(d.args, ctx)
            round_returned_hits = round_returned_hits or bool(result.hits)
            label = provenance_label(d.tool, d.args)
            action_new = 0
            for hit in result.hits:
                hit.retrieved_by = label       # stamp provenance; the evidence reducer dedups after
                key = (hit.source, hit.chunk_index)
                if key not in existing_keys and key not in seen_this_round:
                    seen_this_round.add(key)
                    action_new += 1
            round_new += action_new
            new_hits.extend(result.hits)
            args_repr = ", ".join(f"{k}={v}" for k, v in d.args.model_dump().items())
            steps.append(AgentStep(d.thought, d.tool.name, args_repr, result.observation, action_new))
            counts[d.tool.name] = counts.get(d.tool.name, 0) + 1

        update: dict = {"evidence": new_hits, "steps": steps, "tool_calls": counts}

        # OSCILLATION GUARD: a round that RETURNED hits but added NOTHING new is redundant. Feed the
        # signal back onto the last step (so the controller sees it next round) and track the streak;
        # PATIENCE consecutive redundant rounds => stop (exit_reason=oscillation). Progress resets it.
        is_redundant = round_returned_hits and round_new == 0
        if is_redundant:
            steps[-1].observation += ("\n[NOTE: NO NEW EVIDENCE this round. Reformulate with DIFFERENT "
                                      "terms, try another tool, or finish if you already have enough.]")
            consecutive = state["consecutive_redundant"] + 1
            update["consecutive_redundant"] = consecutive
            update["redundant_searches"] = 1   # delta; the reducer sums it into the running total
            if consecutive >= OSCILLATION_PATIENCE:
                update["exit_reason"] = "oscillation"
        elif round_new:
            update["consecutive_redundant"] = 0   # progress clears the streak
        return update

    def answer_node(state: AgentState) -> dict:
        """ANSWER — write the cited answer + run the output-ring grounding guard. Twin of the tail of
        run_agent: answer_from_evidence, then check_citation_grounding (flag-only). Captures the
        generator cost + trimmed window the adapter needs to build an AnswerResult for the eval."""
        scratchpad = Scratchpad(hits=list(state["evidence"]))
        result = answer_from_evidence(state["question"], scratchpad, retriever, generator_llm,
                                      answer_prompt, top_k, answer_char_budget, ordering)
        grounding = check_citation_grounding(result.answer, result.retrieved)
        return {
            "answer": result.answer,
            "retrieved": result.retrieved,
            "generator_usage": result.generator_usage,
            "grounded": grounding.is_grounded,
            "ungrounded_citations": sorted(grounding.ungrounded),
        }

    return controller_node, tools_node, answer_node


# ── Edges + compile ───────────────────────────────────────────────────────────────────
# The control flow, declared as graph structure instead of a Python for-loop. The CYCLE is
# tools -> controller; the max_rounds budget lives in the router after tools (the twin of
# `for round_index in range(max_rounds)`). exit_reason defaults to "budget" and is overwritten
# to "finish" by the controller node, so the budget path needs no mutation here.

def initial_state(question: str) -> AgentState:
    """The starting state for one question — every field seeded (reducers need a base value)."""
    return {
        "question": question,
        "evidence": [],
        "steps": [],
        "controller_usage": Usage(),
        "tool_calls": {},
        "rounds": 0,
        "exit_reason": "budget",   # the default; a finish/oscillation/spend_cap stop overwrites it
        "answer": "",
        "pending": [],
        "seeded_on_empty": False,
        "consecutive_redundant": 0,
        "redundant_searches": 0,
        "retrieved": [],
        "generator_usage": Usage(),
        "grounded": True,
        "ungrounded_citations": [],
    }


def build_graph(retriever, controller_llm, generator_llm, registry, store,
                react_prompt: str, answer_prompt: str, top_k: int, max_rounds: int,
                answer_char_budget: int, ordering: str = "arrival"):
    """Compile the agent graph. `max_rounds` is captured by the routers below (the budget)."""
    controller_node, tools_node, answer_node = make_nodes(
        retriever, controller_llm, generator_llm, registry, store,
        react_prompt, answer_prompt, top_k, max_rounds, answer_char_budget, ordering)

    def route_after_controller(state: AgentState) -> str:
        # No retrieval chosen => the controller finished; otherwise go run the tools.
        return "answer" if not state["pending"] else "tools"

    def route_after_tools(state: AgentState) -> str:
        # Oscillation first (matches the hand-rolled order): a spinning loop stops before the budget.
        if state["consecutive_redundant"] >= OSCILLATION_PATIENCE:
            return "answer"
        # Budget check (the loop's exit condition): stop after max_rounds controller turns.
        if state["rounds"] >= max_rounds:
            return "answer"
        return "controller"

    builder = StateGraph(AgentState)
    builder.add_node("controller", controller_node)
    builder.add_node("tools", tools_node)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "controller")
    builder.add_conditional_edges("controller", route_after_controller,
                                  {"tools": "tools", "answer": "answer"})
    builder.add_conditional_edges("tools", route_after_tools,
                                  {"controller": "controller", "answer": "answer"})
    builder.add_edge("answer", END)
    return builder.compile()


# ── Adapter: make the graph answer like run_agent, so the eval can A/B it ────────────────
# The eval scores an object with `.answer(question) -> AnswerResult`. GraphAnswerer invokes the
# compiled graph and repackages its final state into that exact shape, so agentic_eval scores the
# graph identically to the hand-rolled Answerer (same judge, faithfulness, partial-credit).

class GraphAnswerer:
    """Drop-in for loop.Answerer, backed by the LangGraph agent."""

    def __init__(self, graph, max_rounds: int):
        self.graph = graph
        self.recursion_limit = max_rounds * 2 + 5  # framework backstop above our own budget

    def answer(self, question: str):
        from agentic_rag.rag.answer import AnswerResult, Trajectory
        final = self.graph.invoke(initial_state(question),
                                  config={"recursion_limit": self.recursion_limit})
        trajectory = Trajectory(
            rounds_used=final["rounds"],
            exit_reason=final["exit_reason"],
            tool_calls=final["tool_calls"],
            tool_errors=0,                       # not tracked in the graph yet (deferred)
            redundant_searches=final["redundant_searches"],
            grounded=final["grounded"],
            ungrounded_citations=final["ungrounded_citations"],
        )
        return AnswerResult(
            question=question,
            answer=final["answer"],
            retrieved=final["retrieved"],
            controller_usage=final["controller_usage"],
            generator_usage=final["generator_usage"],
            trajectory=trajectory,
        )


def build_graph_answerer(config: dict, retriever, generator_llm, controller_llm):
    """Assemble deps from config (mirrors loop.build_answerer) and compile a GraphAnswerer."""
    from agentic_rag.agent.loop import build_agent_deps
    from agentic_rag.agent.tools import DEFAULT_TOOLS
    from agentic_rag.rag.answer import load_prompt

    react_prompt = load_prompt(config, "agent_react")
    answer_prompt = load_prompt(config, "answer_with_citations")
    top_k = config["retrieval"]["top_k"]
    agent_cfg = config.get("agent", {})
    max_rounds = agent_cfg.get("max_rounds", 3)
    answer_char_budget = agent_cfg.get("answer_char_budget", 10000)
    registry, store = build_agent_deps(config, agent_cfg.get("tools", DEFAULT_TOOLS))
    ordering = config.get("context", {}).get("ordering", "arrival")
    graph = build_graph(retriever, controller_llm, generator_llm, registry, store,
                        react_prompt, answer_prompt, top_k, max_rounds, answer_char_budget, ordering)
    return GraphAnswerer(graph, max_rounds)


# ── Manual single-question run (the LangGraph twin of loop.py's main) ────────────────────

def main() -> None:
    import argparse
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Run the LangGraph agent on one question.")
    parser.add_argument("question", nargs="+", help="The question to answer.")
    parser.add_argument("--max-rounds", type=int, default=None, help="Override agent.max_rounds.")
    parser.add_argument("--no-cache", action="store_true", help="Force the LLM cache OFF.")
    args = parser.parse_args()

    from dotenv import load_dotenv

    from agentic_rag.agent.loop import build_agent_deps
    from agentic_rag.agent.tools import DEFAULT_TOOLS
    from agentic_rag.config import load_config
    from agentic_rag.llm.provider import build_llm
    from agentic_rag.logging_setup import configure_run_logging
    from agentic_rag.rag.answer import load_prompt
    from agentic_rag.rag.retriever import build_retriever

    load_dotenv()  # loads GROQ/GOOGLE keys AND the LANGSMITH_* vars so the invoke below auto-traces
    configure_run_logging("agent/graph")
    config = load_config()
    cache_enabled = False if args.no_cache else None
    retriever = build_retriever(config)
    controller_llm = build_llm(config, role="controller", cache_enabled=cache_enabled)
    generator_llm = build_llm(config, role="generator", cache_enabled=cache_enabled)
    react_prompt = load_prompt(config, "agent_react")
    answer_prompt = load_prompt(config, "answer_with_citations")
    top_k = config["retrieval"]["top_k"]
    agent_cfg = config.get("agent", {})
    max_rounds = args.max_rounds if args.max_rounds is not None else agent_cfg.get("max_rounds", 3)
    answer_char_budget = agent_cfg.get("answer_char_budget", 10000)
    registry, store = build_agent_deps(config, agent_cfg.get("tools", DEFAULT_TOOLS))
    ordering = config.get("context", {}).get("ordering", "arrival")

    graph = build_graph(retriever, controller_llm, generator_llm, registry, store,
                        react_prompt, answer_prompt, top_k, max_rounds, answer_char_budget, ordering)
    question = " ".join(args.question)
    # recursion_limit = super-steps LangGraph will run before erroring; our cycle spends 2 per round
    # (controller + tools), so max_rounds*2 + a little slack is a safe backstop behind our own budget.
    final = graph.invoke(initial_state(question),
                         config={"recursion_limit": max_rounds * 2 + 5})

    print("\n" + "=" * 70)
    print("ANSWER:", final["answer"])
    print("-" * 70)
    print(f"rounds={final['rounds']} exit={final['exit_reason']} "
          f"evidence={len(final['evidence'])} chunk(s) | tool_calls={final['tool_calls']} | "
          f"controller_tokens={final['controller_usage'].total_tokens}")
    print("trace -> smith.langchain.com  (project AgenticRAG)")


if __name__ == "__main__":
    main()
