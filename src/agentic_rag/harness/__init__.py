"""Harness — the operational shell around the model call (Module 5).

Not a single thing but a conceptual layer with several rings (see
``docs/harness/HARNESS_ENGINEERING.md``): Interface, Orchestration, Reliability,
Efficiency, Safety, Observability. Several rings already live in their natural
home — the typed tools + loop in ``agent/``, the multi-tier provider router and
the response cache in ``llm/``. This package is the home for the **Safety ring**
(``guardrails.py``) and, later, the **Observability ring** (tracing).
"""
