"""Rolling conversation memory for follow-up questions.

Keeps the last N (question, answer) turns and serializes them as
OpenAI-style role messages so the LLM can ground follow-ups like
"can you give me an example?" or "tell me more about that".

Memory is intentionally process-local and NOT persisted to disk:
when the app restarts or the user presses Ctrl+R, every turn is gone.
Privacy + clean slate per interview.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass


@dataclass
class Turn:
    question: str
    answer: str


class ConversationHistory:
    def __init__(self, max_turns: int = 5):
        self._max = max_turns
        self._turns: deque[Turn] = deque(maxlen=max_turns)
        self._lock = threading.Lock()

    def add(self, question: str, answer: str) -> None:
        question = (question or "").strip()
        answer = (answer or "").strip()
        if not question or not answer:
            return
        with self._lock:
            self._turns.append(Turn(question, answer))

    def as_messages(self) -> list[dict]:
        """Prior turns as alternating user / assistant messages."""
        with self._lock:
            out: list[dict] = []
            for t in self._turns:
                out.append({"role": "user", "content": t.question})
                out.append({"role": "assistant", "content": t.answer})
            return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()
