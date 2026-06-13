"""LLM access layer (part of the harness).

Houses the thin provider interface and, later, the provider-fallback router
(primary -> secondary -> local Ollama). Build router-shaped but single-tier first;
expand once the naive loop + evals can prove the router does not degrade quality.

Status: thin provider + single-tier router implemented — `provider.py` (GroqProvider,
LLMRouter, build_llm). Groq-only for now (DD-004); router-shaped so fallback tiers slot
in later. Reads GROQ_API_KEY from a gitignored .env.
"""
