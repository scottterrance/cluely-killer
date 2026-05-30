"""Groq cloud LLM provider.

Groq runs open models (Llama, etc.) on their LPU hardware at very high
token-throughput with a low time-to-first-token, which is exactly what
an interview overlay wants. The endpoint is OpenAI-compatible, so the
SSE streaming format is identical to DeepSeek's - this provider is the
DeepSeek provider pointed at Groq with a different default model.

Get a free key at https://console.groq.com/keys. The free tier is
generous but rate-limited; when you exhaust it, the app can fall back
to DeepSeek automatically (see Controller.fallback_llm_factory) or you
can switch the LLM backend manually in Settings -> AI Provider.

Recommended chat models (as of writing):
  llama-3.3-70b-versatile   - strong general default, great quality/speed
  llama-3.1-8b-instant      - fastest, lighter quality
  openai/gpt-oss-120b       - heavier, higher quality, a bit slower
Pick whatever your account has access to in the Settings model box.
"""
from __future__ import annotations

import json
from typing import Iterator

import httpx

from .base import LLMProvider

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ):
        if not api_key:
            raise ValueError(
                "Groq API key is missing. Get a free one at "
                "https://console.groq.com/keys and paste it in "
                "Settings -> AI Provider."
            )
        self.api_key = api_key
        self.model = (model or DEFAULT_MODEL).strip()
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")

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
            "max_tokens": 400,
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
                f"Groq connection failed: {type(e).__name__}: {e}"
            )

        with response_ctx as response:
            if response.status_code != 200:
                try:
                    body_text = response.read().decode("utf-8", errors="replace")
                except Exception:
                    body_text = ""
                raise RuntimeError(
                    f"Groq HTTP {response.status_code}: {body_text[:300]}"
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
                text = delta.get("content") or ""
                if text:
                    yield text
