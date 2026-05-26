"""OpenRouter cloud provider.

OpenRouter is an aggregator that exposes ~50 model providers behind a
single OpenAI-compatible endpoint. It has a generous free tier
(`...:free` model IDs) and crucially does NOT use the same Cloudflare
WAF rules as Groq, so it works from most cloud / VPS / VPN IP ranges
that Groq blocks.

Get a key (free): https://openrouter.ai/keys
Default model: meta-llama/llama-3.3-70b-instruct:free
"""
from __future__ import annotations

import json
from typing import Iterator

import httpx

from .base import LLMProvider

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# Recommended HTTP headers per OpenRouter docs - they're optional but
# unlock per-app analytics on your dashboard and are good citizenship.
_REFERER = "https://github.com/scottterrance/cluely-killer"
_TITLE = "cluely-killer"


class OpenRouterProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "meta-llama/llama-3.3-70b-instruct:free",
        base_url: str = DEFAULT_BASE_URL,
    ):
        if not api_key:
            raise ValueError(
                "OpenRouter API key is missing. Get a free key at "
                "https://openrouter.ai/keys and paste it in Settings -> AI Provider."
            )
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

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
            "HTTP-Referer": _REFERER,
            "X-Title": _TITLE,
        }
        body = {
            "model": self.model,
            "messages": msgs,
            "stream": True,
            "temperature": 0.6,
            "max_tokens": 400,
            "top_p": 0.95,
        }

        # Server-Sent Events stream. Each line is either:
        #   data: {<chat-completion delta JSON>}
        #   data: [DONE]
        # or a blank keep-alive comment line beginning with ':'.
        with httpx.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=httpx.Timeout(60.0, connect=10.0),
        ) as response:
            if response.status_code != 200:
                # Read body for the user-facing message; the controller
                # turns this into a friendly hint via _friendly_error().
                payload = response.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"OpenRouter HTTP {response.status_code}: {payload[:300]}"
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
                    delta = obj["choices"][0]["delta"].get("content") or ""
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
                if delta:
                    yield delta
