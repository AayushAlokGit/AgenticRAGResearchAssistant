"""Eval harness (code).

Turns "did that change help?" into a number. The eval *datasets* and human docs live
in the top-level ``evals/`` directory; the *runnable code* lives here in the package so
it's importable and installs with everything else.

Scoring is added in layers (see ``evals/README.md`` and ``docs/evals/EVALUATION_PRINCIPLES.md``):
  1. retrieval recall  — ``retrieval.py`` (now): deterministic, no LLM.
  2. answer quality     — later: faithfulness / relevance via LLM-judge.
  3. system cost        — later: steps / tokens / latency.
"""
