"""Rolling conversation memory for follow-up questions.

Keeps the last N (question, answer) turns and serializes them as
OpenAI-style role messages so the LLM can ground follow-ups like
"can you give me an example?" or "tell me more about that".

Memory is intentionally process-local and NOT persisted to disk:
when the app restarts or the user presses Ctrl+R, every turn is gone.
Privacy + clean slate per interview.
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

    def brief(self, max_turns: int | None = None, max_chars: int = 600) -> str:
        """A compact, mechanical rolling summary of prior turns.

        Roadmap #9: instead of replaying every prior answer verbatim
        (which balloons the prompt and slows time-to-first-token), we
        send a short digest the model can use for continuity on follow-up
        questions ("elaborate on that"). No LLM call - this is pure string
        work, so it adds zero latency. Markdown emphasis from the stored
        answers (==red==, **yellow**) is stripped so the brief stays
        clean. Returns '' when there's no history.
        """
        with self._lock:
            turns = list(self._turns)
        if max_turns is not None:
            turns = turns[-max_turns:]
        if not turns:
            return ""
        lines: list[str] = []
        for t in turns:
            q = _strip_markup(t.question)
            a = _strip_markup(t.answer)
            # Keep the question short and the answer to its gist.
            q = _truncate(q, 90)
            a = _truncate(a, 160)
            lines.append(f"- They asked: {q} | I answered: {a}")
        brief = "\n".join(lines)
        return brief[:max_chars]

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()
