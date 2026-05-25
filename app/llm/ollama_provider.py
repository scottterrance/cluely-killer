"""Ollama local-model provider.

Install Ollama (https://ollama.com), then `ollama pull llama3.1:8b`
and start the daemon (it auto-starts on Windows after install).
"""
from __future__ import annotations

from typing import Iterator

import ollama

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "llama3.1:8b", host: str = "http://localhost:11434"):
        self.model = model
        self.client = ollama.Client(host=host)

    def stream_chat(self, system_prompt: str, user_message: str) -> Iterator[str]:
        stream = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            stream=True,
            options={
                "temperature": 0.6,
                "num_predict": 400,
                "top_p": 0.95,
            },
        )
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "") if isinstance(chunk, dict) else ""
            if content:
                yield content
