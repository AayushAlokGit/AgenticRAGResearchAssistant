"""Agentic RAG Research Assistant.

A from-scratch, learning-focused agentic RAG system. Subpackages map to the five
modules in docs/ProjectIdea.md, built in order:

    llm      - provider interface + fallback router (part of the harness)
    rag      - module 1: RAG substrate (ingestion, chunking, retrieval)
    agent    - module 2: agentic loop
    context  - module 3: context engineering
    memory   - module 4: working + long-term memory
    harness  - module 5: orchestration loop, tools, guardrails, tracing

Status: scaffolding only - no implementation yet.
"""

__version__ = "0.0.1"
