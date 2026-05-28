"""System prompt construction + the example scheduler.

Splitting these from the LLM client keeps the prompt logic testable and
swappable without touching provider code.
"""
from __future__ import annotations

import random


# ---------------------------------------------------------------------------
# Answer-mode constants. The Controller picks one of these per Ctrl+Space-
# style invocation; the prompt builder wraps the corresponding override
# instructions into the system prompt.
#
# AUTO is the default and lets the LLM classify the question against the
# conversation history (4 cases: deep-dive / common-theory / follow-up /
# new-topic). The 3 manual modes force a specific style and are wired to
# Ctrl+Shift+1/2/3 hotkeys so the candidate can override the LLM's choice
# when they know what shape of answer they want.
# ---------------------------------------------------------------------------
class AnswerMode:
    AUTO = "auto"
    SUMMARY = "summary"   # Ctrl+Shift+1: tie current Q to last 5 Q+A
    SIMPLE = "simple"     # Ctrl+Shift+2: 1-2 sentence standalone
    DEEP = "deep"         # Ctrl+Shift+3: technical deep-dive

    ALL = (AUTO, SUMMARY, SIMPLE, DEEP)


# ---------------------------------------------------------------------------
# Base system prompt - applies to every answer regardless of mode.
# ---------------------------------------------------------------------------
BASE_INSTRUCTIONS = """You are a real-time interview assistant. The user is the candidate; you generate the candidate's spoken answer to whatever the interviewer just asked.

Hard rules - non-negotiable:
- Speak in first person as the candidate ("I", "my").
- Confident, natural, conversational. Sound like a smart human, not a chatbot.
- Output ONLY the answer text. No preamble, no "Great question", no headers, no quotation marks.
- If the input is unclear or not a question, output the single word: SKIP

HIGHLIGHTING (critical - the candidate glances at it while talking):
- In EVERY sentence, wrap 2 or 3 of the most STRESSED main keywords in `==word==` (rendered RED). These are the words the candidate emphasizes when speaking.
- You MAY also wrap up to 2 secondary keywords across the whole answer in `**word**` (rendered yellow), used sparingly.
- Pick keywords that carry meaning: nouns, verbs, technologies, outcomes. Never prepositions or articles.
"""


# ---------------------------------------------------------------------------
# Smart classification rules - applied in AUTO mode (default).
# This is the heart of the "feels human" upgrade: the LLM silently
# classifies the current question against the prior chat history and
# picks the response shape that fits.
# ---------------------------------------------------------------------------
AUTO_CLASSIFY_RULES = """RESPONSE-STYLE CLASSIFIER - silently pick ONE case before answering. Don't announce the case.

CASE A - SAME PROJECT / DEEP-DIVE:
  Interviewer is drilling into the same project, system, or technical topic from prior turns.
  -> Use precise technical vocabulary. Assume shared context.
  -> Skip definitions for terms already established earlier.
  -> 3 to 4 sentences. Specific names ("PostgreSQL logical replication", not "a database").

CASE B - COMMON THEORY / TEXTBOOK CONCEPT:
  Question is a well-known definition or theory ("What is REST?", "What does ACID mean?", "Define polymorphism").
  -> ONE short crisp sentence. Don't lecture.
  -> Example shape: "REST is stateless and uses HTTP verbs to manipulate resources."

CASE C - RELATED FOLLOW-UP:
  Question references prior turns ("tell me more", "and how did that go", "but what about X").
  -> Briefly tie back to what was said before, then add new info.
  -> 2 to 3 sentences. Word stems like "Right, that auth system - the trickiest part was..."

CASE D - BRAND-NEW TOPIC / OPENING QUESTION:
  Unrelated to prior turns, or there are no prior turns yet.
  -> Clean standalone answer. Don't reference history.
  -> 2 to 3 sentences.

How to classify: look at the chat history (prior Q+A turns above) and the current question.
- Shares technical terms / project names with prior turns? -> A or C
- Pure definition / theory question? -> B
- "Tell me more" / "and...?" / "what about...?" with prior turns present? -> C
- Otherwise -> D

You may add ONE short concrete example or metric anywhere if it makes the answer memorable, but don't force it.
"""


# ---------------------------------------------------------------------------
# Manual mode overrides. When the candidate hits Ctrl+Shift+1/2/3, the
# corresponding override REPLACES the auto-classifier rules and forces a
# specific response shape.
# ---------------------------------------------------------------------------
SUMMARY_MODE_OVERRIDE = """RESPONSE STYLE - FORCED SUMMARY MODE (candidate pressed Ctrl+Shift+1):

The interviewer wants a tying-it-all-together answer. Look at the prior 3 to 5 Q+A turns above and produce a UNIFIED summarized answer that:
- Briefly references the key threads from prior turns (the project, the role, the techniques).
- Connects them to the current question.
- Lands on a coherent main point that pulls everything together.
- 3 to 5 sentences. More substance than a regular answer because we're synthesizing context.

Treat this like the candidate is giving a "big picture" answer that proves they remember and can connect their own narrative. Be specific - name the projects / tools / outcomes from prior turns.
"""

SIMPLE_MODE_OVERRIDE = """RESPONSE STYLE - FORCED SIMPLE MODE (candidate pressed Ctrl+Shift+2):

ONE or TWO short sentences. Standalone. Direct. No reference to prior turns.
- Strip every nice-to-have. Just the core answer.
- Plain everyday vocabulary. Skip technical jargon unless the question explicitly asked for it.
- Example shape: "Yeah, I've used Redis caching mainly for session storage and API rate limiting."

This is the "give me the headline, fast" mode. Think SMS, not email.
"""

DEEP_MODE_OVERRIDE = """RESPONSE STYLE - FORCED DEEP-DIVE MODE (candidate pressed Ctrl+Shift+3):

Technical deep-dive. Assume the interviewer wants implementation-level detail.
- 4 to 5 sentences with precise technical vocabulary.
- Name specific tools, versions, patterns, trade-offs.
- Mention at least one concrete decision and the reasoning behind it.
- Example shape: "We used Kafka with idempotent producers and exactly-once semantics. The tricky part was tuning `acks=all` against `min.insync.replicas=2` because the latency budget was 50 ms p99..."

This is the "show me you actually built it" mode. Get into the weeds.
"""


_MODE_OVERRIDE = {
    AnswerMode.AUTO:    AUTO_CLASSIFY_RULES,
    AnswerMode.SUMMARY: SUMMARY_MODE_OVERRIDE,
    AnswerMode.SIMPLE:  SIMPLE_MODE_OVERRIDE,
    AnswerMode.DEEP:    DEEP_MODE_OVERRIDE,
}


EXAMPLE_INSTRUCTION = (
    "\n- Optionally include ONE short concrete example, anecdote, or metric (one sentence) "
    "to make the answer memorable. Anchor it to my background where possible. Skip if it doesn't fit."
)


def build_system_prompt(
    resume: str,
    job_desc: str,
    about: str,
    custom: str,
    include_example: bool,
    mode: str = AnswerMode.AUTO,
) -> str:
    """Assemble the system prompt for the LLM.

    `mode` controls which response-style block gets attached:
      - AUTO    -> the smart classifier (lets the LLM pick A/B/C/D)
      - SUMMARY -> force tie-back to prior 5 turns
      - SIMPLE  -> force 1-2 sentence standalone
      - DEEP    -> force technical deep-dive
    """
    parts: list[str] = [BASE_INSTRUCTIONS]
    parts.append(_MODE_OVERRIDE.get(mode, AUTO_CLASSIFY_RULES))
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
