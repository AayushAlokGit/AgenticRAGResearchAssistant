"""Naive RAG answer pipeline: retrieve -> assemble context -> generate cited answer.

This is the simplest end-to-end question-answering path (CLAUDE.md build sequence step 4):

    question -> hybrid retrieve top_k -> stuff chunks into the prompt -> LLM answers

It is NOT yet agentic — there's no retrieve->reason->retrieve loop, no budget, no tool
selection. It's the naive baseline that the agentic layer (module 2) will later wrap and
improve, each step earning its place against the eval set.

Grounding is enforced by the prompt (`prompts/answer_with_citations.md`): answer only from
the provided passages, cite source filenames, and reply exactly "Not enough information."
when the context can't support an answer — which is what makes the abstention questions
(q10-q12) measurable later.

Run:
    python -m agentic_rag.rag.answer "your question here"
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from agentic_rag.config import load_config, resolve_path
from agentic_rag.llm.provider import Usage, build_llm
from agentic_rag.rag.retriever import build_retriever
from agentic_rag.rag.vector_store import Hit


@dataclass
class Trajectory:
    """Process metrics for ONE agent run (X1 trajectory eval): HOW the agent reached the
    answer, not just whether it was right. An agentic change (planning, a new tool) can hold
    answer-correctness flat while halving cost/Q or eliminating redundant work — invisible to
    the answer-correctness rung alone, visible here. The naive path leaves this None.
    """
    rounds_used: int = 0
    exit_reason: str = ""             # finish | budget | oscillation | spend_cap
    tool_calls: dict = field(default_factory=dict)  # tool name -> times invoked
    tool_errors: int = 0             # actions that failed to parse/validate before dispatch
    redundant_searches: int = 0      # searches that added no new evidence (spinning)
    # OUTPUT-ring guardrail (Module 5, flag-only): did the answer cite only sources it was actually
    # given? `grounded` false + the offending filenames means a fabricated citation. Recorded, not
    # enforced — a diagnostic the eval reads; the answer is returned unchanged either way.
    grounded: bool = True
    ungrounded_citations: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "rounds_used": self.rounds_used,
            "exit_reason": self.exit_reason,
            "tool_calls": dict(self.tool_calls),
            "tool_errors": self.tool_errors,
            "redundant_searches": self.redundant_searches,
            "grounded": self.grounded,
            "ungrounded_citations": list(self.ungrounded_citations),
        }


@dataclass
class AnswerResult:
    question: str
    answer: str
    retrieved: List[Hit]              # the chunks fed to the model, best-first
    # Token cost split by ROLE so we can tier models (DD-025): the controller (routing) and
    # the generator (final answer) may be different models with different prices, so summing
    # them would hide which role the cost is in. Naive path has no controller -> stays zero.
    controller_usage: Usage = field(default_factory=Usage)
    generator_usage: Usage = field(default_factory=Usage)
    trajectory: Optional["Trajectory"] = None  # agent process metrics (X1); None for naive path


def load_prompt(config: dict, name: str) -> str:
    """Load a versioned prompt by name (e.g. 'answer_with_citations') from the prompts dir.

    Only substantive prompt content (the instructions) is versioned as a file. The trivial
    CONTEXT/QUESTION wrapper around the dynamic values is built in code (see
    answer_question) — it's structural scaffolding, not prompt content worth A/B testing.
    """
    prompt_path = resolve_path(config["prompts"]["dir"]) / f"{name}.md"
    return prompt_path.read_text(encoding="utf-8")


def assemble_context(hits: List[Hit]) -> str:
    """Format retrieved chunks into labeled passages the model can cite by filename."""
    passages = []
    for hit in hits:
        passages.append(f"[source: {hit.source}]\n{hit.text}")
    return "\n\n".join(passages)


def generate_answer(question: str, retriever, llm, system_prompt: str, top_k: int) -> AnswerResult:
    """Core generation given PREBUILT dependencies.

    Split out so a caller answering many questions (e.g. the answer-quality eval) can build
    the retriever + LLM once and reuse them, instead of reloading the embedder per question.
    """
    hits = retriever.query(question, top_k)

    # Assemble the two-message prompt: versioned instructions as the system message, and the
    # dynamic context + question wrapped in code as the user message.
    context = assemble_context(hits)
    user_message = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    completion = llm.complete([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ])
    return AnswerResult(question=question, answer=completion.text or "",
                        retrieved=hits, generator_usage=completion.usage)


def answer_question(question: str, config: Optional[dict] = None) -> AnswerResult:
    """Convenience one-shot: build everything from config and answer a single question."""
    if config is None:
        config = load_config()
    retriever = build_retriever(config)
    llm = build_llm(config)
    system_prompt = load_prompt(config, "answer_with_citations")
    top_k = config["retrieval"]["top_k"]
    return generate_answer(question, retriever, llm, system_prompt, top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the naive RAG pipeline a question.")
    parser.add_argument("question", nargs="+", help="The question to answer.")
    parser.add_argument("--show-context", action="store_true",
                        help="Also print the retrieved chunks that were fed to the model.")
    args = parser.parse_args()

    # LLM answers can contain non-ASCII (em-dashes, etc.); the Windows console defaults to
    # cp1252 and would crash on those. Force UTF-8 output where the stream supports it.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    question = " ".join(args.question)
    result = answer_question(question)

    print(f"\nQ: {result.question}\n")
    print(f"A: {result.answer}\n")

    print(f"Retrieved {len(result.retrieved)} chunk(s):")
    for i, hit in enumerate(result.retrieved, start=1):
        print(f"  {i}. [{hit.source}] (score={hit.score:.3f})")
        if args.show_context:
            snippet = hit.text[:200].replace("\n", " ")
            print(f"     {snippet}...")


if __name__ == "__main__":
    main()
