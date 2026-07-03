"""Re-grade STORED eval answers under the CURRENT judge model, without re-running the agent.

Why this exists: the judge model was switched (groq llama-4-scout -> gemini-2.5-flash-lite)
after the llama judge degenerated to all-PARTIAL on the coverage-gate run. A Gemini-judged run
is not comparable to a llama-judged baseline, and the agent ANSWERS in the stored runs are fine —
only the grading was broken. So we replay the SAME stored (question, answer, reference, retrieved)
through the new judge. This isolates exactly one variable (the judge) and costs judge tokens only,
no controller/generator loop.

Usage:
    python -m scripts.regrade eval_runs/agentic/v2/<file>.json [<file2>.json ...]
"""
from __future__ import annotations

import json
import sys

from agentic_rag.config import load_config, resolve_path
from agentic_rag.evals.answer_correctness import judge_correctness
from agentic_rag.evals.faithfulness import judge_faithfulness
from agentic_rag.llm.provider import build_llm
from agentic_rag.rag.answer import load_prompt
from agentic_rag.rag.vector_store import build_store

SCORE = {"CORRECT": 1.0, "PARTIALLY_CORRECT": 0.5, "INCORRECT": 0.0}


def build_text_lookup(config: dict) -> dict:
    """{(source, chunk_index): text} for every stored chunk, so we can rebuild the EXACT context
    the generator saw. The run JSONs store only (source, chunk_index) for each retrieved chunk,
    not the text, so faith re-grading needs the corpus text back — and the vector store has it."""
    store = build_store(config)
    return {(c["source"], c["chunk_index"]): c["text"] for c in store.all_chunks()}


def faith_context(retrieved: list, text_lookup: dict) -> str:
    """Rebuild the generator's context (rag.answer.assemble_context shape) from the stored
    (source, chunk_index) pairs, looking the text back up from the store."""
    passages = []
    for r in retrieved:
        text = text_lookup.get((r["source"], r["chunk_index"]), "")
        passages.append(f"[source: {r['source']}]\n{text}")
    return "\n\n".join(passages)


def regrade_one(run_path: str, judge_llm, corr_prompt: str, faith_prompt: str,
                text_lookup: dict) -> dict:
    data = json.load(open(run_path, encoding="utf-8"))
    pqs = data["per_question"]

    corr_num = 0.0
    faith_supported = 0
    faith_graded = 0

    for pq in pqs:
        # --- correctness (mirror the harness: abstention handled without the judge) ---
        if pq["should_abstain"]:
            corr_num += 1.0 if pq["abstained"] else 0.0
        elif pq["abstained"]:
            corr_num += 0.0  # false abstention on an answerable question
        else:
            verdict, _reason, _u = judge_correctness(
                judge_llm, corr_prompt, pq["question"], pq["expected_answer"], pq["answer"])
            corr_num += SCORE.get(verdict, 0.0)

        # --- faithfulness (only for questions that actually produced an answer) ---
        if not pq["abstained"] and pq.get("retrieved"):
            context = faith_context(pq["retrieved"], text_lookup)
            fverdict, _fr, _fu = judge_faithfulness(
                judge_llm, faith_prompt, pq["question"], context, pq["answer"])
            if fverdict in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"):
                faith_graded += 1
                if fverdict == "SUPPORTED":
                    faith_supported += 1

    return {
        "correctness": corr_num / len(pqs),
        "faithfulness": faith_supported / faith_graded if faith_graded else 0.0,
        "faith_graded": faith_graded,
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    config = load_config()
    judge_llm = build_llm(config, role="judge")
    corr_prompt = load_prompt(config, "judge_correctness")
    faith_prompt = load_prompt(config, "judge_faithfulness")
    judge_model = config["llm"]["roles"]["judge"]["model"]
    text_lookup = build_text_lookup(config)
    print(f"Re-grading under judge = {judge_model}  ({len(text_lookup)} chunks in lookup)\n")
    print(f"{'file':28s} {'corr':6s} {'faith':6s} faith_n")
    for path in sys.argv[1:]:
        r = regrade_one(path, judge_llm, corr_prompt, faith_prompt, text_lookup)
        label = path.split("/")[-1].split("\\")[-1]
        print(f"{label:28s} {r['correctness']:.3f}  {r['faithfulness']:.3f}  {r['faith_graded']}")


if __name__ == "__main__":
    main()
