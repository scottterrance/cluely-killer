"""Groq cloud provider.

Free tier, ~500+ tok/s on Llama-3.x — currently the fastest free
streaming option for an interview overlay.
"""
from __future__ import annotations

from typing import Iterator

from groq import Groq

from .base import LLMProvider


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        if not api_key:
            raise ValueError("Groq API key is missing. Set it in Settings or .env (GROQ_API_KEY).")
        self.client = Groq(api_key=api_key)
        self.model = model

    def stream_chat(self, system_prompt: str, user_message: str) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            stream=True,
            temperature=0.6,
            max_tokens=400,
            top_p=0.95,
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content
            except (IndexError, AttributeError):
                delta = None
            if delta:
                yield delta
