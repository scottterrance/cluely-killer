"""OpenRouter cloud provider with automatic fallback chain.

OpenRouter's free-tier models (`...:free`) get rate-limited upstream
all the time, especially the popular ones (llama-3.3-70b is the worst
because every free user on Earth points at it). Mid-interview that
would be fatal, so the provider accepts a comma-separated list of
models and tries each in order. On HTTP 429 from one model we silently
fall through to the next; only when the whole chain is exhausted do
we surface the rate-limit error to the candidate.

Model field examples:
  meta-llama/llama-3.3-70b-instruct:free
  meta-llama/llama-3.3-70b-instruct:free, google/gemma-2-9b-it:free
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


class _RateLimited(Exception):
    """Internal sentinel: this model is 429, try the next one."""


def _split_models(spec: str | list[str]) -> list[str]:
    if isinstance(spec, str):
        return [m.strip() for m in spec.split(",") if m.strip()]
    return [m.strip() for m in spec if m and m.strip()]


class OpenRouterProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str | list[str] = "meta-llama/llama-3.3-70b-instruct:free",
        base_url: str = DEFAULT_BASE_URL,
    ):
        if not api_key:
            raise ValueError(
                "OpenRouter API key is missing. Get a free key at "
                "https://openrouter.ai/keys and paste it in Settings -> AI Provider."
            )
        self.api_key = api_key
        self.models = _split_models(model)
        if not self.models:
            raise ValueError(
                "OpenRouter: no model specified. "
                "Put at least one model id in Settings -> AI Provider -> OpenRouter -> Model(s)."
            )
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

        for i, model in enumerate(self.models):
            try:
                yield from self._stream_one(model, msgs)
                return  # success - first model that responds wins
            except _RateLimited:
                remaining = self.models[i + 1:]
                if remaining:
                    print(
                        f"[openrouter] {model!r} rate-limited; "
                        f"falling back to {remaining[0]!r}",
                        flush=True,
                    )
                    continue
                # Whole chain exhausted - surface as the standard 429 so the
                # controller's _friendly_error maps it to the 'Rate limit'
                # status hint.
                raise RuntimeError(
                    f"OpenRouter HTTP 429: all {len(self.models)} model"
                    f"{'s' if len(self.models) != 1 else ''} in your fallback chain "
                    f"are rate-limited. Add more comma-separated models in Settings, "
                    f"wait a minute, or switch provider."
                )

    # ------------------------------------------------------------------
    def _stream_one(self, model: str, msgs: list[dict]) -> Iterator[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": _REFERER,
            "X-Title": _TITLE,
        }
        body = {
            "model": model,
            "messages": msgs,
            "stream": True,
            "temperature": 0.6,
            "max_tokens": 400,
            "top_p": 0.95,
        }
        with httpx.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=httpx.Timeout(60.0, connect=10.0),
        ) as response:
            if response.status_code == 429:
                try:
                    response.read()
                except Exception:
                    pass
                raise _RateLimited(model)
            if response.status_code != 200:
                payload = response.read().decode("utf-8", errors="replace")
                # OpenRouter sometimes wraps a 429 from upstream in a 200
                # with an error payload, or surfaces it via 'Provider
                # returned error' text. Treat those as rate-limited too.
                low = payload.lower()
                if '"code":429' in payload or "rate-limited" in low:
                    raise _RateLimited(model)
                raise RuntimeError(
                    f"OpenRouter HTTP {response.status_code} ({model}): {payload[:300]}"
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
                # Body-level error inside the SSE stream.
                if isinstance(obj, dict) and "error" in obj:
                    err = obj["error"]
                    if isinstance(err, dict):
                        if err.get("code") == 429:
                            raise _RateLimited(model)
                        raise RuntimeError(
                            f"OpenRouter ({model}): {err.get('message', err)}"
                        )
                    raise RuntimeError(f"OpenRouter ({model}): {err}")
                try:
                    delta = obj["choices"][0]["delta"].get("content") or ""
                except (KeyError, IndexError, TypeError):
                    continue
                if delta:
                    yield delta
