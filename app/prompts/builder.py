"""System prompt construction + the example scheduler.

Splitting these from the LLM client keeps the prompt logic testable and
swappable without touching provider code.
"""
from __future__ import annotations

import random

# Answer-length directives. LLM generation is sequential (one token at a
# time), so answer LATENCY is ~proportional to output length. Shorter =
# faster. This is THE speed lever on a fast-STT setup where the LLM is
# the bottleneck. 'concise' roughly halves DeepSeek time vs 'detailed'.
_LENGTH_RULES = {
    "concise": "- Answer in 1 to 2 SHORT sentences. Brevity is the top priority - it makes the answer appear FAST. Cut every non-essential word.",
    "normal": "- 2 to 3 sentences. Concise but complete.",
    "detailed": "- 3 to 5 sentences. Never longer.",
}
# max_tokens ceiling per brevity (defense against a runaway answer; the
# prompt above drives the typical length). Used by the LLM provider.
LENGTH_MAX_TOKENS = {"concise": 110, "normal": 220, "detailed": 400}


def _base_instructions(brevity: str) -> str:
    length_rule = _LENGTH_RULES.get(brevity, _LENGTH_RULES["concise"])
    return f"""You are a real-time interview assistant. The user is the candidate; you generate the candidate's spoken answer to whatever the interviewer just asked.

Hard rules - these are non-negotiable:
{length_rule}
- Speak in first person as the candidate ("I", "my").
- Be confident, natural, conversational. Sound like a smart human, not a textbook or chatbot.

HIGHLIGHTING (this is critical - the candidate glances at it while talking):
- In EVERY sentence, wrap 2 or 3 of the most STRESSED, main keywords in `==word==` (rendered RED). These are the words the candidate should emphasize when speaking.
- You may ALSO wrap up to 2 secondary keywords across the whole answer in `**word**` (rendered yellow), used sparingly.
- Choose the keywords that carry the most meaning - nouns, verbs, numbers, technologies, outcomes - never prepositions or articles.

CONVERSATION CONTINUITY:
- If prior turns are present (the chat history), assume the new question is a follow-up. Reference earlier specifics naturally instead of repeating my whole story.

OUTPUT:
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
    brevity: str = "concise",
) -> str:
    # PREFIX-CACHE ORDERING (matters for latency + cost):
    # DeepSeek (and most providers) cache the longest IDENTICAL leading
    # span of the prompt across requests and skip recomputing it - a
    # cache hit cuts time-to-first-token and is billed ~10x cheaper.
    # So everything that stays CONSTANT within a session goes first:
    #   base rules -> custom -> about-me -> resume -> JD
    # and the only VOLATILE bit (the every-3rd-turn example toggle) goes
    # LAST. The base rules vary only with `brevity` (a session-level
    # setting), so they stay cache-stable within a session.
    parts: list[str] = [_base_instructions(brevity)]
    if custom and custom.strip():
        parts.append("\nAdditional instructions from the candidate:\n" + custom.strip())
    if about and about.strip():
        parts.append("\n--- About me ---\n" + about.strip())
    if resume and resume.strip():
        parts.append("\n--- My resume ---\n" + resume.strip())
    if job_desc and job_desc.strip():
        parts.append("\n--- Target job ---\n" + job_desc.strip())
    # Volatile tail - keep this the ONLY thing that varies turn-to-turn.
    if include_example:
        parts.append(EXAMPLE_INSTRUCTION)
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
