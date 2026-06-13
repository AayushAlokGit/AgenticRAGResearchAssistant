"""LLM access layer — thin provider interface + single-tier router.

Built "router-shaped but single-tier" (DD-004): ``provider_order`` in config is a list, and
the router tries providers in order until one answers. Today only Groq has a key, so the
list has one tier — but adding a fallback (Anthropic, local Ollama) later is a config +
one provider class, not a rewrite.

A provider exposes one method: ``complete(messages) -> str``, where ``messages`` is the
OpenAI-style list of ``{"role": ..., "content": ...}`` dicts (roles: system / user /
assistant). Keeping the surface this small is deliberate — the agent loop shouldn't care
which provider answered.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List

from agentic_rag.config import resolve_path

logger = logging.getLogger(__name__)

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


class LLMError(RuntimeError):
    """Raised when no provider can answer (e.g. all tiers failed, or none configured)."""


class GroqProvider:
    name = "groq"

    def __init__(self, model: str, api_key: str, temperature: float, max_tokens: int,
                 max_retries: int = 5, timeout: float = 30.0):
        # Imported lazily so importing this module stays cheap and doesn't require the SDK
        # until an LLM is actually built.
        from groq import Groq

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # max_retries/timeout drive the SDK's built-in exponential backoff + jitter on
        # retryable errors (429 rate limit, 5xx, timeouts, connection drops).
        self.client = Groq(api_key=api_key, max_retries=max_retries, timeout=timeout)

    def complete(self, messages: List[Message]) -> str:
        logger.debug("groq completion: model=%s, %d message(s)", self.model, len(messages))
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        elapsed = time.perf_counter() - start
        usage = getattr(response, "usage", None)
        if usage is not None:
            logger.info("groq completion ok: %.2fs, tokens prompt=%s completion=%s",
                        elapsed, usage.prompt_tokens, usage.completion_tokens)
        else:
            logger.info("groq completion ok: %.2fs", elapsed)
        return response.choices[0].message.content


class LLMRouter:
    """Tries providers in priority order until one succeeds; raises if all fail.

    Single-tier today (just Groq). The retry-across-providers logic is here now so that
    when a second tier exists, fallback is automatic — the value of building it
    router-shaped up front.
    """

    def __init__(self, providers: List):
        self.providers = providers

    def complete(self, messages: List[Message]) -> str:
        errors = []
        for provider in self.providers:
            try:
                return provider.complete(messages)
            except Exception as exc:  # noqa: BLE001 — we genuinely want to try the next tier
                logger.warning("provider %s failed (will try next tier if any): %s", provider.name, exc)
                errors.append(f"{provider.name}: {exc}")
        raise LLMError("All LLM providers failed: " + " | ".join(errors))


def build_llm(config: dict) -> LLMRouter:
    """Construct the router from config, reading API keys from the environment (.env)."""
    # Load .env so GROQ_API_KEY is available even when not exported in the shell.
    from dotenv import load_dotenv

    load_dotenv(resolve_path(".env"))

    llm_cfg = config["llm"]
    temperature = llm_cfg["temperature"]
    max_tokens = llm_cfg["max_tokens"]
    max_retries = llm_cfg.get("max_retries", 5)
    timeout = llm_cfg.get("timeout", 30.0)

    providers = []
    for provider_name in llm_cfg["provider_order"]:
        if provider_name == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                continue  # no key → skip this tier (router-shaped: other tiers can still run)
            model = llm_cfg["models"]["groq"]
            providers.append(GroqProvider(model, api_key, temperature, max_tokens, max_retries, timeout))
        else:
            # Other providers (anthropic, ollama) aren't wired yet — no keys exist. They're
            # listed in config for the future; skipping keeps the router single-tier for now.
            continue

    if not providers:
        raise LLMError(
            "No LLM providers available. Set GROQ_API_KEY in .env (see .env.example)."
        )
    return LLMRouter(providers)
