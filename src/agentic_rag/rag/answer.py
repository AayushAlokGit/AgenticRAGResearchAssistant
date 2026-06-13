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
from dataclasses import dataclass
from typing import List, Optional

from agentic_rag.config import load_config, resolve_path
from agentic_rag.llm.provider import build_llm
from agentic_rag.rag.retriever import build_retriever
from agentic_rag.rag.vector_store import Hit


@dataclass
class AnswerResult:
    question: str
    answer: str
    retrieved: List[Hit]   # the chunks fed to the model, best-first


def load_answer_prompt(config: dict) -> str:
    """Load the versioned system prompt that instructs grounding + citations + abstention."""
    prompt_path = resolve_path(config["prompts"]["answer"])
    return prompt_path.read_text(encoding="utf-8")


def assemble_context(hits: List[Hit]) -> str:
    """Format retrieved chunks into labeled passages the model can cite by filename."""
    passages = []
    for hit in hits:
        passages.append(f"[source: {hit.source}]\n{hit.text}")
    return "\n\n".join(passages)


def answer_question(question: str, config: Optional[dict] = None) -> AnswerResult:
    if config is None:
        config = load_config()

    # 1. Retrieve (hybrid, by default) the top_k chunks for the question.
    retriever = build_retriever(config)
    top_k = config["retrieval"]["top_k"]
    hits = retriever.query(question, top_k)

    # 2. Assemble the labeled context and the two-message prompt.
    system_prompt = load_answer_prompt(config)
    context = assemble_context(hits)
    user_message = f"CONTEXT:\n{context}\n\nQUESTION: {question}"

    # 3. Generate the answer.
    llm = build_llm(config)
    answer = llm.complete([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ])

    return AnswerResult(question=question, answer=answer, retrieved=hits)


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
