"""LLM provider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator


class LLMProvider(ABC):
    @abstractmethod
    def stream_chat(self, system_prompt: str, user_message: str) -> Iterator[str]:
        """Yield text chunks as they arrive from the model."""
