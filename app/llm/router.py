"""Selects the active LLM provider from Settings, with automatic fallback.

Implements the LLMProvider interface so the Controller's
``llm_factory(settings)`` can return this router unchanged.

Backend choice (``settings.llm_backend``):
  "groq"     -> Groq (fast, free tier)
  "deepseek" -> DeepSeek (cheap, stable)

Fallback semantics (important detail):
  stream_chat is a generator. We can only safely fall back to the other
  provider if the primary fails BEFORE producing any visible text -
  otherwise the user would see a half-answer from provider A followed by
  a full answer from provider B. So we pull the FIRST chunk inside a
  try/except: if that first pull raises (auth error, 429 out-of-tokens,
  connection error), we transparently switch to the fallback provider.
  Once the first chunk has been yielded we're committed to that provider
  and any later error propagates normally.

This is what makes "Groq ran out of free tokens" a non-event: the very
next press silently answers via DeepSeek instead.
"""
from __future__ import annotations

import traceback
from typing import Iterator

from ..config import Settings
from .base import LLMProvider


class LLMRouter(LLMProvider):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.last_used: str = ""

    def _build(self, name: str) -> LLMProvider:
        if name == "deepseek":
            from .deepseek_provider import DeepSeekProvider

            return DeepSeekProvider(
                api_key=self.settings.deepseek_api_key,
                model=self.settings.deepseek_model,
                base_url=self.settings.deepseek_base_url,
            )
        from .groq_provider import GroqProvider

        return GroqProvider(
            api_key=self.settings.groq_api_key,
            model=self.settings.groq_model,
            base_url=self.settings.groq_base_url,
        )

    def _order(self) -> list[str]:
        if self.settings.llm_backend == "deepseek":
            return ["deepseek", "groq"]
        return ["groq", "deepseek"]

    def stream_chat(
        self,
        system_prompt: str,
        user_message: str,
        prior_messages: list[dict] | None = None,
    ) -> Iterator[str]:
        order = self._order()
        first_err: Exception | None = None

        for i, name in enumerate(order):
            # Build provider (may raise if key is missing for this one).
            try:
                provider = self._build(name)
            except Exception as e:
                if first_err is None:
                    first_err = e
                print(f"[llm-router] backend {name!r} unavailable: {e}", flush=True)
                continue

            gen = provider.stream_chat(
                system_prompt, user_message, prior_messages=prior_messages
            )
            # Try to pull the FIRST chunk under guard so we can still
            # fall back cleanly. After this point we're committed.
            try:
                first_chunk = next(gen)
            except StopIteration:
                # Provider produced zero chunks without error (empty
                # answer). Treat as a soft failure and try the fallback,
                # but only if there IS a fallback left.
                if i < len(order) - 1:
                    print(
                        f"[llm-router] {name!r} returned empty; trying fallback...",
                        flush=True,
                    )
                    continue
                self.last_used = name
                return
            except Exception as e:
                if first_err is None:
                    first_err = e
                traceback.print_exc()
                print(
                    f"[llm-router] {name!r} failed before first token: {e}; "
                    + ("trying fallback..." if i < len(order) - 1 else "no fallback left."),
                    flush=True,
                )
                continue

            # Success: this provider is producing output. Commit to it.
            self.last_used = name
            if i > 0:
                print(f"[llm-router] fell back to {name!r} backend.", flush=True)
            yield first_chunk
            yield from gen
            return

        # Every backend failed before producing a token.
        if first_err is not None:
            raise first_err
