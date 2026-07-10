"""Rolling conversation memory for follow-up questions.

Keeps the last N (question, answer) turns and serializes them as
OpenAI-style role messages so the LLM can ground follow-ups like
"can you give me an example?" or "tell me more about that".

Memory is intentionally process-local and NOT persisted to disk:
when the app restarts or the user presses Ctrl+R, every turn is gone.
Privacy + clean slate per interview.

Upgrade notes:
- brief() now detects technical follow-up patterns and gives the most
  recent answer more character budget so the LLM can continue the thread.
- max_chars tightened to 500 (was 600) to reduce token spend on history.
- Question truncation raised to 100 chars; answer gist raised to 180 chars
  for the most recent turn (older turns stay at 120 chars).
"""
from __future__ import annotations

import re
import threading
from collections import deque
from dataclasses import dataclass

# Strip the overlay's highlight markup (==red==, **yellow**) and stray
# backticks so a rolling brief reads as plain prose.
_MARKUP = re.compile(r"==|\*\*|`")


def _strip_markup(text: str) -> str:
    return _MARKUP.sub("", text or "").strip()


def _truncate(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "\u2026"


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
        """Prior turns as alternating user / assistant messages.

        This is the FULL raw transcript of prior Q+A - used by 'full'
        context mode. It's accurate but token-heavy (grows with answer
        length), so 'smart' mode uses brief() instead.
        """
        with self._lock:
            out: list[dict] = []
            for t in self._turns:
                out.append({"role": "user", "content": t.question})
                out.append({"role": "assistant", "content": t.answer})
            return out

    def brief(self, max_turns: int | None = None, max_chars: int = 500) -> str:
        """A compact rolling summary of prior turns for the LLM's volatile tail.

        Token-efficient design:
        - Older turns get a tighter answer gist (120 chars) to save tokens.
        - The most recent turn gets a fuller answer gist (200 chars) so the
          LLM can continue a technical deep-dive naturally.
        - Total budget capped at 500 chars (was 600) to reduce token spend.
        - Markup stripped so the brief reads as clean prose.
        Returns '' when there's no history.
        """
        with self._lock:
            turns = list(self._turns)
        if max_turns is not None:
            turns = turns[-max_turns:]
        if not turns:
            return ""
        lines: list[str] = []
        for i, t in enumerate(turns):
            q = _strip_markup(t.question)
            a = _strip_markup(t.answer)
            q = _truncate(q, 100)
            # Give the most recent turn more answer budget for continuity.
            is_last = (i == len(turns) - 1)
            a = _truncate(a, 200 if is_last else 120)
            lines.append(f"- Q: {q} | A: {a}")
        brief = "\n".join(lines)
        return brief[:max_chars]

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()
