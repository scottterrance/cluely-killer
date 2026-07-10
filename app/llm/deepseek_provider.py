"""DeepSeek cloud provider.

Uses DeepSeek's OpenAI-compatible chat completions endpoint at
https://api.deepseek.com/v1. Two models are available:

  deepseek-chat       - general purpose (V3 family). Fast, OpenAI-class
                        quality, the right default for interview use.
  deepseek-reasoner   - reasoning model (R1 family). Higher quality on
                        complex 'why?' / 'design ...' questions but
                        emits chain-of-thought first which we discard,
                        so latency-to-first-visible-token is higher.

Get a key at https://platform.deepseek.com/api_keys. Pricing as of
writing is ~$0.14/M input tokens / $0.28/M output for deepseek-chat,
so a typical interview Q+A is well under a tenth of a cent.

Implementation note: SSE format is identical to OpenAI / OpenRouter so
this is essentially the OpenRouter provider stripped of the fallback
chain (DeepSeek's models are stable, no need to chain).
"""
from __future__ import annotations

import json
from typing import Iterator

import httpx

from .base import LLMProvider

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


class DeepSeekProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int = 400,
    ):
        if not api_key:
            raise ValueError(
                "DeepSeek API key is missing. Get one at "
                "https://platform.deepseek.com/api_keys and paste it in "
                "Settings -> AI Provider."
            )
        self.api_key = api_key
        self.model = (model or DEFAULT_MODEL).strip()
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        # Output-token ceiling. LLM latency is ~proportional to tokens
        # generated, so a lower cap directly lowers answer time. The
        # prompt's brevity rule drives typical length; this is the hard
        # backstop against a runaway answer.
        self.max_tokens = int(max_tokens) if max_tokens else 400

    def stream_chat(
        self,
        system_prompt: str,
        user_message: str,
        prior_messages: list[dict] | None = None,
    ) -> Iterator[str]:
        msgs: list[dict] = [{"role": "system", "content": system_prompt}]
        if prior_messages:
            msgs.extend(prior_messages)
        msgs.append({"role": "user", "content": user_message})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": msgs,
            "stream": True,
            "temperature": 0.6,
            "max_tokens": self.max_tokens,
            "top_p": 0.95,
        }

        try:
            response_ctx = httpx.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
        except httpx.RequestError as e:
            raise RuntimeError(
                f"DeepSeek connection failed: {type(e).__name__}: {e}"
            )

        with response_ctx as response:
            if response.status_code != 200:
                try:
                    body_text = response.read().decode("utf-8", errors="replace")
                except Exception:
                    body_text = ""
                raise RuntimeError(
                    f"DeepSeek HTTP {response.status_code}: {body_text[:300]}"
                )
            for raw in response.iter_lines():
                if not raw or raw.startswith(":"):
                    continue
                if not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                try:
                    delta = obj["choices"][0]["delta"]
                except (KeyError, IndexError, TypeError):
                    continue
                # deepseek-reasoner emits 'reasoning_content' (chain of
                # thought) BEFORE 'content'. For an interview overlay
                # we want only the final answer; the reasoning is
                # internal scratchpad we deliberately discard.
                text = delta.get("content") or ""
                if text:
                    yield text
