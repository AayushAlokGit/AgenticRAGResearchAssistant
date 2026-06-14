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
import random
import time
from dataclasses import dataclass
from typing import Dict, List

from agentic_rag.config import resolve_path

logger = logging.getLogger(__name__)

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


@dataclass
class Usage:
    """Token cost of one or more LLM calls. Provider-neutral counts (tokens, not dollars):
    a count is comparable across Groq/Gemini, whereas a $ figure needs a price table that
    drifts. ``calls`` lets us see the cost of agency directly — naive=1 call/question, the
    agent loop=N. Usages add, so a per-call Usage rolls up into a per-question / per-run total.
    """
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            calls=self.calls + other.calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class Completion:
    """What ``complete()`` returns: the generated text PLUS what it cost.

    Cost is returned (not stashed as hidden provider state) so it's an explicit, visible
    property of every call — the caller threads/sums ``usage`` to get per-question and
    per-run totals (Option A; see DD-024).
    """
    text: str
    usage: Usage


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

    def complete(self, messages: List[Message]) -> Completion:
        logger.debug("groq completion: model=%s, %d message(s)", self.model, len(messages))
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        elapsed = time.perf_counter() - start
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage = Usage(calls=1,
                          prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
                          completion_tokens=getattr(raw_usage, "completion_tokens", 0) or 0)
            logger.info("groq completion ok: %.2fs, tokens prompt=%d completion=%d",
                        elapsed, usage.prompt_tokens, usage.completion_tokens)
        else:
            usage = Usage(calls=1)  # SDK gave no usage; still count the call
            logger.info("groq completion ok: %.2fs (no usage reported)", elapsed)
        return Completion(text=response.choices[0].message.content or "", usage=usage)


def _messages_to_gemini(messages: List[Message]):
    """Translate OpenAI-style messages into Gemini's (system_instruction, contents) shape.

    Gemini differs from the OpenAI format two ways: the system prompt is a SEPARATE field
    (not a message), and the assistant role is called "model". So we pull all system
    messages into one instruction string and map the rest into Gemini "Content" dicts.
    """
    system_parts = []
    contents = []
    for message in messages:
        role = message.get("role")
        text = message.get("content") or ""
        if role == "system":
            system_parts.append(text)
        else:
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": text}]})
    return "\n\n".join(system_parts), contents


def _gemini_is_retryable(exc) -> bool:
    """Is a Gemini error transient (worth retrying) rather than a real client error?

    Rate limits (429) and 5xx are transient. A raw connection drop (e.g. WinError 10054,
    "connection forcibly closed") has NO HTTP status — treat it as transient too. A 4xx other
    than 429 (bad request, auth) is a real error — don't retry it.
    """
    code = getattr(exc, "code", None)
    if code is None:
        code = getattr(exc, "status_code", None)
    if code is None:
        return True  # no HTTP status → network/connection blip → retry
    try:
        code = int(code)
    except (TypeError, ValueError):
        return True
    return code == 429 or 500 <= code < 600


class GeminiProvider:
    """Google Gemini via the unified google-genai SDK. Same complete(messages)->str surface
    as GroqProvider, so the router and the rest of the system don't care which one answered.
    """
    name = "gemini"

    def __init__(self, model: str, api_key: str, temperature: float, max_tokens: int,
                 max_retries: int = 5, timeout: float = 30.0):
        # Lazy import so this module stays cheap to import without the SDK installed.
        from google import genai
        from google.genai import types

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._types = types
        # HttpOptions.timeout is in MILLISECONDS (our config timeout is seconds).
        self.client = genai.Client(api_key=api_key,
                                   http_options=types.HttpOptions(timeout=int(timeout * 1000)))

    def complete(self, messages: List[Message]) -> Completion:
        system_text, contents = _messages_to_gemini(messages)
        # Gemini 2.5 models "think" by DEFAULT, and thinking tokens are drawn from
        # max_output_tokens — so they can starve/truncate the visible output. flash/flash-lite
        # let us disable thinking entirely (budget=0); gemini-2.5-pro cannot (nonzero minimum),
        # so we CAP its thinking small enough to leave room for the answer/verdict within
        # max_output_tokens. (Gemini keeps thinking out of response.text either way, so it never
        # pollutes a judge's verdict — unlike Groq reasoning models.)
        thinking_budget = 0 if "flash" in self.model else 512
        gen_config = self._types.GenerateContentConfig(
            system_instruction=system_text or None,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            thinking_config=self._types.ThinkingConfig(thinking_budget=thinking_budget),
        )
        logger.debug("gemini completion: model=%s, %d message(s)", self.model, len(messages))
        start = time.perf_counter()
        # The google-genai SDK doesn't transparently retry transient drops the way Groq's SDK
        # does, and a single blip would otherwise crash a whole eval run — so retry here with
        # exponential backoff before giving up (then the router falls through to the next tier).
        response = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=contents, config=gen_config,
                )
                break
            except Exception as exc:  # noqa: BLE001 — classify, then retry or re-raise
                if attempt == self.max_retries or not _gemini_is_retryable(exc):
                    raise
                # Exponential backoff with jitter (half fixed + half random): backs off
                # meaningfully while spreading retries so they don't synchronize/collide.
                backoff = min(2 ** attempt, 30)
                wait = backoff / 2 + random.uniform(0, backoff / 2)
                logger.warning("gemini transient error (attempt %d/%d), retrying in %.1fs: %s",
                               attempt, self.max_retries, wait, str(exc)[:140])
                time.sleep(wait)
        elapsed = time.perf_counter() - start
        raw_usage = getattr(response, "usage_metadata", None)
        if raw_usage is not None:
            # Gemini names them differently: prompt_token_count / candidates_token_count. The
            # candidates count EXCLUDES thinking tokens (those are thoughts_token_count) — we
            # disable/cap thinking, so completion_tokens here is the visible-answer cost.
            usage = Usage(calls=1,
                          prompt_tokens=getattr(raw_usage, "prompt_token_count", 0) or 0,
                          completion_tokens=getattr(raw_usage, "candidates_token_count", 0) or 0)
            logger.info("gemini completion ok: %.2fs, tokens prompt=%d completion=%d",
                        elapsed, usage.prompt_tokens, usage.completion_tokens)
        else:
            usage = Usage(calls=1)  # no usage reported; still count the call
            logger.info("gemini completion ok: %.2fs (no usage reported)", elapsed)
        return Completion(text=response.text or "", usage=usage)


class LLMRouter:
    """Tries providers in priority order until one succeeds; raises if all fail.

    Single-tier today (just Groq). The retry-across-providers logic is here now so that
    when a second tier exists, fallback is automatic — the value of building it
    router-shaped up front.
    """

    def __init__(self, providers: List):
        self.providers = providers

    def complete(self, messages: List[Message]) -> Completion:
        errors = []
        for provider in self.providers:
            try:
                # The Completion (text + usage) of whichever tier answered. Failed tiers
                # cost no usage here — they typically return no tokens before raising.
                return provider.complete(messages)
            except Exception as exc:  # noqa: BLE001 — we genuinely want to try the next tier
                logger.warning("provider %s failed (will try next tier if any): %s", provider.name, exc)
                errors.append(f"{provider.name}: {exc}")
        raise LLMError("All LLM providers failed: " + " | ".join(errors))


def role_model(config: dict, role: str | None = None) -> str:
    """Model NAME for a role (used to record which model produced a run).

    A role entry may be a bare model string (shorthand) or a ``{provider, model}`` dict;
    either way this returns the model name. A role with no entry uses the default provider's
    default model. See ``role_provider_model`` for the (provider, model) pair.
    """
    return role_provider_model(config, role)[1]


def role_provider_model(config: dict, role: str | None = None) -> tuple[str, str]:
    """Resolve a role to its (provider, model).

    Roles (e.g. "generator", "judge") let one config run different parts of the system on
    different models/providers — separate token buckets, and a judge from a different model
    family (no self-eval bias). Two config spellings are accepted:
      - explicit:  generator: {provider: gemini, model: gemini-2.5-flash}
      - shorthand: generator: llama-3.1-8b-instant   (model on the DEFAULT provider)
    A role with no entry uses the default provider's default model.
    """
    llm_cfg = config["llm"]
    default_provider = llm_cfg["provider_order"][0]
    roles = llm_cfg.get("roles", {})
    entry = roles.get(role) if role else None
    if isinstance(entry, dict):
        return entry["provider"], entry["model"]
    if entry:
        return default_provider, entry
    return default_provider, llm_cfg["models"][default_provider]


def _build_provider(name: str, model: str, temperature: float, max_tokens: int,
                    max_retries: int, timeout: float):
    """Construct one provider, or return None if its API key is absent / it isn't wired yet.

    Returning None (instead of raising) lets a missing key simply DROP that tier, so the
    router falls through to the next configured provider.
    """
    if name == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        return GroqProvider(model, api_key, temperature, max_tokens, max_retries, timeout)
    if name == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None
        return GeminiProvider(model, api_key, temperature, max_tokens, max_retries, timeout)
    # anthropic / ollama: listed in config for the future, not wired yet.
    return None


def build_llm(config: dict, role: str | None = None) -> LLMRouter:
    """Construct the router from config, reading API keys from the environment (.env).

    The role's chosen (provider, model) is the PRIMARY tier; the remaining providers in
    ``provider_order`` (each with their own default model) are fallback tiers. Providers
    whose API key isn't set are skipped, so a missing key just removes that tier.
    """
    # Load .env so GROQ_API_KEY / GOOGLE_API_KEY are available even when not exported.
    from dotenv import load_dotenv

    load_dotenv(resolve_path(".env"))

    llm_cfg = config["llm"]
    temperature = llm_cfg["temperature"]
    max_tokens = llm_cfg["max_tokens"]
    max_retries = llm_cfg.get("max_retries", 5)
    timeout = llm_cfg.get("timeout", 30.0)

    primary_provider, primary_model = role_provider_model(config, role)
    plan = [(primary_provider, primary_model)]
    for provider_name in llm_cfg["provider_order"]:
        if provider_name != primary_provider:
            plan.append((provider_name, llm_cfg["models"].get(provider_name)))

    providers = []
    for provider_name, model in plan:
        if model is None:
            continue  # no default model configured for this fallback tier
        provider = _build_provider(provider_name, model, temperature, max_tokens, max_retries, timeout)
        if provider is not None:
            providers.append(provider)

    if not providers:
        raise LLMError(
            "No LLM providers available. Set the API key for your configured provider in .env "
            "(GOOGLE_API_KEY for gemini, GROQ_API_KEY for groq; see .env.example)."
        )
    return LLMRouter(providers)
