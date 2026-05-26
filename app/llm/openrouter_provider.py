"""OpenRouter cloud provider with automatic fallback chain.

OpenRouter's free-tier models (`...:free`) fail in two common ways:
  1. Rate-limited upstream (HTTP 429, especially the popular ones like
     llama-3.3-70b which every free user on Earth points at).
  2. Quietly removed (HTTP 404 'No endpoints found' - the free model
     list rotates, gemma-2-9b-it disappeared this way).

For an interview tool either one is fatal if it stops the chain. So the
provider takes a comma-separated list of model ids and treats ANY
non-success outcome on a single model as a "skip this one" signal,
falling through to the next. Only when the whole chain is exhausted
do we surface the last error to the candidate.

Model field examples:
  meta-llama/llama-3.3-70b-instruct:free
  qwen/qwen3-next-80b-a3b-instruct:free, deepseek/deepseek-v4-flash:free
"""
from __future__ import annotations

import json
from typing import Iterator

import httpx

from .base import LLMProvider

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# OpenRouter accepts optional `HTTP-Referer` and `X-Title` headers for
# per-app analytics on their dashboard. We DELIBERATELY leave them empty
# now: any string here would be a network-level fingerprint that
# identifies the app to anyone inspecting outbound HTTPS metadata
# (corporate firewall, monitoring proxy). The user can opt back in by
# editing this file or by setting them via OPENROUTER_HTTP_REFERER /
# OPENROUTER_X_TITLE env vars before launch if they want analytics.
import os as _os
_REFERER = _os.getenv("OPENROUTER_HTTP_REFERER", "")
_TITLE = _os.getenv("OPENROUTER_X_TITLE", "")


class _SkipModel(Exception):
    """Internal sentinel: this model failed pre-content; try the next one.

    Carries a short human-readable reason so the exhausted-chain error
    message can tell the user which kind of failure they're hitting
    (rate-limit, model removed, server error, etc.).
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _split_models(spec: str | list[str]) -> list[str]:
    if isinstance(spec, str):
        return [m.strip() for m in spec.split(",") if m.strip()]
    return [m.strip() for m in spec if m and m.strip()]


def _classify_status(status: int, body: str) -> str:
    """One-line description of why a non-200 response is unusable."""
    low = body.lower()
    if status == 429 or '"code":429' in body or "rate-limited" in low:
        return f"rate-limited (HTTP {status})"
    if status == 404 or "no endpoints found" in low:
        return f"model not found / removed (HTTP {status})"
    if status == 401:
        return "API key invalid (HTTP 401)"
    if status == 403:
        return "forbidden (HTTP 403)"
    if 500 <= status < 600:
        return f"upstream server error (HTTP {status})"
    return f"HTTP {status}: {body[:120]}"


class OpenRouterProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str | list[str] = "qwen/qwen3-next-80b-a3b-instruct:free",
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

        last_reason = "no models tried"
        for i, model in enumerate(self.models):
            try:
                yield from self._stream_one(model, msgs)
                return  # success - first model that responds wins
            except _SkipModel as e:
                last_reason = e.reason
                remaining = self.models[i + 1:]
                if remaining:
                    print(
                        f"[openrouter] {model!r} -> {e.reason}; "
                        f"falling back to {remaining[0]!r}",
                        flush=True,
                    )
                    continue
                # Whole chain exhausted. Surface a 429-flavoured RuntimeError
                # so the controller's _friendly_error maps it to the standard
                # rate-limit hint when that's what dominated.
                raise RuntimeError(
                    f"OpenRouter HTTP 429: all {len(self.models)} model"
                    f"{'s' if len(self.models) != 1 else ''} in your fallback chain failed. "
                    f"Last reason: {last_reason}. "
                    f"Edit the comma-separated model list in Settings -> AI Provider, "
                    f"wait a minute, or switch provider (Ollama works offline)."
                )

    # ------------------------------------------------------------------
    def _stream_one(self, model: str, msgs: list[dict]) -> Iterator[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Only attach app-fingerprint headers if the user explicitly
        # opted in via env vars (default empty = no fingerprint).
        if _REFERER:
            headers["HTTP-Referer"] = _REFERER
        if _TITLE:
            headers["X-Title"] = _TITLE
        body = {
            "model": model,
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
            raise _SkipModel(f"connect failed: {type(e).__name__}: {e}")

        yielded_any = False
        with response_ctx as response:
            # ---- Pre-content checks: any non-200 -> skip this model ----
            if response.status_code != 200:
                try:
                    body_text = response.read().decode("utf-8", errors="replace")
                except Exception:
                    body_text = ""
                raise _SkipModel(_classify_status(response.status_code, body_text))

            # ---- 200 OK; iterate the SSE body ----
            try:
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
                    # OpenRouter sometimes wraps an upstream failure as a
                    # 200 OK with the error inlined in the SSE body.
                    if isinstance(obj, dict) and "error" in obj:
                        err = obj["error"]
                        if isinstance(err, dict):
                            err_code = err.get("code")
                            err_msg = err.get("message", str(err))
                        else:
                            err_code = None
                            err_msg = str(err)
                        if not yielded_any:
                            # Pre-content: safe to fall back.
                            raise _SkipModel(
                                f"in-stream error code={err_code}: {err_msg[:140]}"
                            )
                        # Mid-stream failure: we already showed the user
                        # partial output, can't fall back without dupes.
                        raise RuntimeError(
                            f"OpenRouter ({model}) mid-stream error: {err_msg}"
                        )
                    try:
                        delta = obj["choices"][0]["delta"].get("content") or ""
                    except (KeyError, IndexError, TypeError):
                        continue
                    if delta:
                        yielded_any = True
                        yield delta
            except httpx.RequestError as e:
                if yielded_any:
                    raise RuntimeError(
                        f"OpenRouter ({model}) connection lost mid-stream: {e}"
                    )
                raise _SkipModel(f"connection error: {type(e).__name__}: {e}")
