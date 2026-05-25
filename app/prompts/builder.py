"""System prompt construction + the example scheduler.

Splitting these from the LLM client keeps the prompt logic testable and
swappable without touching provider code.
"""
from __future__ import annotations

import random

BASE_INSTRUCTIONS = """You are a real-time interview assistant. The user is the candidate; you generate the candidate's spoken answer to whatever the interviewer just asked.

Hard rules — these are non-negotiable:
- 3 to 5 sentences. Never longer.
- Speak in first person as the candidate ("I", "my").
- Be confident, natural, conversational. Sound like a smart human, not a textbook or chatbot.
- Wrap the 2 to 4 most important keywords or phrases in **markdown bold** so they pop visually for the candidate.
- Output ONLY the answer text. No preamble, no "Great question", no headers, no explanation, no quotation marks.
- If the input is unclear or not a question, output the single word: SKIP
"""

EXAMPLE_INSTRUCTION = (
    "\n- Include exactly ONE short concrete example, anecdote, or metric (one sentence) "
    "to make the answer memorable. Anchor it to my background where possible."
)


def build_system_prompt(
    resume: str,
    job_desc: str,
    about: str,
    custom: str,
    include_example: bool,
) -> str:
    parts: list[str] = [BASE_INSTRUCTIONS]
    if include_example:
        parts.append(EXAMPLE_INSTRUCTION)
    if custom and custom.strip():
        parts.append("\nAdditional instructions from the candidate:\n" + custom.strip())
    if about and about.strip():
        parts.append("\n--- About me ---\n" + about.strip())
    if resume and resume.strip():
        parts.append("\n--- My resume ---\n" + resume.strip())
    if job_desc and job_desc.strip():
        parts.append("\n--- Target job ---\n" + job_desc.strip())
    return "\n".join(parts)


class ExampleScheduler:
    """Decides when to inject the 'include example' instruction.

    Triggers on every 3rd or 4th answer (randomly chosen between cycles)
    so the cadence feels natural rather than mechanical.
    """

    def __init__(self) -> None:
        self._counter = 0
        self._next_at = random.choice([3, 4])

    def should_include(self) -> bool:
        self._counter += 1
        if self._counter >= self._next_at:
            self._counter = 0
            self._next_at = random.choice([3, 4])
            return True
        return False
