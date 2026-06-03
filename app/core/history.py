"""Rolling conversation memory for follow-up questions.

Keeps the last N (question, answer) turns and serializes them as
OpenAI-style role messages so the LLM can ground follow-ups like
"can you give me an example?" or "tell me more about that".

Enhanced in this version:
- Default max_turns raised from 5 to 8 to support deep technical drill-down
  sequences (e.g., a system design question followed by 5-6 probing follow-ups).
- Answers are stored with a truncated version for the history payload to avoid
  exceeding context limits: the first 600 chars are kept (enough for the LLM
  to recall the substance without bloating the prompt).
- Memory is intentionally process-local and NOT persisted to disk:
  when the app restarts or the user presses Ctrl+R, every turn is gone.
  Privacy + clean slate per interview.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

# Maximum characters to retain per answer in the history payload.
# Keeps the context window lean while preserving enough substance for
# the model to understand what was already said.
_MAX_ANSWER_CHARS_IN_HISTORY = 600


@dataclass
class Turn:
    question: str
    answer: str


class ConversationHistory:
    def __init__(self, max_turns: int = 8):
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
        """Prior turns as alternating user / assistant messages.

        Answers are truncated to _MAX_ANSWER_CHARS_IN_HISTORY characters
        to keep the context payload lean for deep interview sessions with
        many follow-up questions.
        """
        with self._lock:
            out: list[dict] = []
            for t in self._turns:
                out.append({"role": "user", "content": t.question})
                # Truncate long answers to avoid bloating the prompt
                answer_payload = t.answer
                if len(answer_payload) > _MAX_ANSWER_CHARS_IN_HISTORY:
                    answer_payload = (
                        answer_payload[:_MAX_ANSWER_CHARS_IN_HISTORY]
                        + "... [truncated for context efficiency]"
                    )
                out.append({"role": "assistant", "content": answer_payload})
            return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()
