"""LLM access layer (part of the harness).

Houses the thin provider interface and, later, the provider-fallback router
(primary -> secondary -> local Ollama). Build router-shaped but single-tier first;
expand once the naive loop + evals can prove the router does not degrade quality.

Status: not yet implemented - see ProjectIdea.md (harness) and CLAUDE.md sequence step 2.
"""
