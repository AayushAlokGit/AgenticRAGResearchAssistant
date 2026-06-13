# Answer-Quality Evaluation: Correctness vs Faithfulness

> A portable distinction. It applies to **any system whose output should be grounded in
> provided evidence** — RAG, grounded summarization, a tool-using agent that cites
> sources. This project's answer-quality eval is the worked example in Part 5.
> See also `EVALUATION_PRINCIPLES.md` (the general meta-principles this builds on).

The trap this doc exists to prevent: assuming that **"the answer matched the expected
answer"** is enough to trust a grounded generation system. It isn't. Matching the
reference and being grounded in the evidence are **two different questions**, and the gap
between them hides the most dangerous failure mode.

## 1. Two different questions

- **Correctness** (reference-based): does the answer match a *known-correct reference
  answer*? Requires ground truth. Catches wrong answers.
- **Faithfulness** (reference-free, evidence-grounded): is every claim in the answer
  *supported by the evidence the system was actually given* (the retrieved context)?
  Requires no reference. Catches ungrounded answers — i.e. hallucination.

## 2. They are orthogonal — the 2×2

|  | **Faithful** (grounded in context) | **Unfaithful** (not in context) |
|---|---|---|
| **Correct** (matches reference) | ✅ ideal | ⚠️ right answer, but from the model's memory, not retrieval |
| **Incorrect** (contradicts reference) | context was wrong/insufficient; model reported it honestly | ❌ hallucinated a wrong answer |

A correctness judge only sees the rows (correct vs incorrect). A faithfulness judge only
sees the columns (grounded vs not). Neither alone sees the full square.

## 3. What correctness alone catches — and misses

**Catches:** hallucinations that *contradict* the reference. (In this project, q22: the
model claimed the local store has hybrid BM25+vector search; the reference says it
doesn't → graded INCORRECT.)

**Misses — the dangerous cell (Correct + Unfaithful), "ungrounded-but-correct":** the
model answers correctly **from its training knowledge instead of the retrieved context**.
A large model already "knows" many facts from pretraining, so it can answer right even
when retrieval handed it a useless chunk. Correctness says CORRECT; the RAG system
*didn't actually work*. This is insidious because it's invisible on easy questions — and
the moment the model's memory is wrong or **the corpus deliberately disagrees with the
world**, it will confidently hallucinate, with no warning.

**Also misses:**
- **Fabricated extra detail** — the answer states the reference facts (so: CORRECT) *plus*
  an invented claim the reference never mentions, so there's nothing to contradict.
- **Citation hallucination** — cites a source that doesn't support the claim. Correctness
  never looks at citations vs context.

## 4. Why faithfulness generalizes (and correctness can't)

Correctness **requires a reference answer for every question**. Faithfulness needs only
the answer + the context it was generated from. So faithfulness:

- works on **questions you have no ground truth for** — i.e. real user traffic in
  production, where there's no `expected_answer` to compare against;
- can run as a **live guardrail / monitor**, not just an offline eval.

This is why mature RAG eval frameworks (e.g. Ragas) score **faithfulness** and **answer
correctness** as *separate* metrics — they catch different failures and have different
requirements.

## 5. Abstention — the third axis

A grounded system must sometimes **refuse** ("not enough information") rather than answer.
That's a separate, often **deterministic** check (did it output the refusal?), scored two ways:

- **Failure to abstain:** it answered an out-of-corpus question → fabrication.
- **False abstention:** it refused a question it *should* have answered (e.g. the
  fact-bearing chunk wasn't retrieved) — a failure the *retrieval* metric can't see, since
  document-level recall can "pass" while the answer-bearing passage was missed.

## 6. In this project

`evals/answer_correctness.py` (named for what it measures) scores **correctness (LLM-judge
vs `expected_answer`) + abstention (deterministic)**. Faithfulness is the **next layer** —
a separate eval whose judge compares the answer to the *retrieved chunks* instead of to the
reference.

Honest caveats already noted in-code:
- The judge is a **different model** from the generator (DD-013: `llama-3.3-70b` generates,
  `gpt-oss-120b` judges) — a different family avoids **self-evaluation bias** and gives the
  judge its own provider rate-limit bucket. (Pointing both roles at one model brings the
  bias back; the eval's report line flags which case is active.)
- Generation + judging are **non-deterministic**, so this metric has run-to-run variance,
  unlike the deterministic retrieval recall. Look for clear movement, not small deltas.
- q22 (caught by correctness) and the q01/q25/q26 false abstentions (retrieval gaps
  cascading into the answer layer) are the concrete instances of the cells above.

## 7. Applying this to a new system (the transferable rule)

1. **Does the output need to be grounded in provided evidence?** If yes, you owe *both*
   correctness and faithfulness — they're orthogonal; one can't substitute for the other.
2. **Use faithfulness for anything without ground truth** (production traffic, new
   queries). Use correctness where you have verified reference answers (offline eval).
3. **Score refuse/abstain as its own axis** — track both failure-to-abstain and
   false-abstention.
4. **Watch the Correct+Unfaithful cell specifically** — a system that's "right" by
   ignoring its evidence is a time bomb, and only faithfulness reveals it.
