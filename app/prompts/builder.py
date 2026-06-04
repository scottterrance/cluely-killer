"""System prompt construction + the example scheduler.

Upgraded for deep technical interview mode:
- Natural human expert tone (no chatbot openers, no textbook recitations)
- Technical depth: handles multi-turn drill-downs, trade-offs, edge cases
- Token-efficient: tight volatile tail, prefix-cache-stable base
- Four brevity levels including a new 'deep' mode for technical deep-dives
"""
from __future__ import annotations

import random

# Answer-length directives. LLM generation is sequential (one token at a
# time), so answer LATENCY is ~proportional to output length. Shorter =
# faster. This is THE speed lever on a fast-STT setup where the LLM is
# the bottleneck. 'concise' roughly halves DeepSeek time vs 'detailed'.
_LENGTH_RULES = {
    "brief": (
        "LENGTH: 1-2 sentences max. No padding, no preamble. "
        "If technical, give the precise answer + one key reason."
    ),
    "concise": (
        "LENGTH: 2-4 sentences. Cover the core point and one concrete detail. "
        "Stop when the idea is complete - do NOT pad to fill space."
    ),
    "detailed": (
        "LENGTH: 4-7 sentences. Walk through the reasoning, mention trade-offs "
        "or a concrete example, then land on a clear conclusion. "
        "Still conversational - no bullet lists, no headers."
    ),
    "deep": (
        "LENGTH: 6-10 sentences. This is a deep technical question. "
        "Explain the mechanism, the why behind the design, relevant edge cases, "
        "and your personal experience with it. "
        "Speak like a senior engineer who has debugged this in production, "
        "not like someone reciting a textbook. No bullet lists."
    ),
}
# max_tokens ceiling per brevity (defense against a runaway answer; the
# prompt above drives the typical length). Used by the LLM provider.
LENGTH_MAX_TOKENS = {"brief": 80, "concise": 160, "detailed": 380, "deep": 600}


def _base_instructions(brevity: str) -> str:
    length_rule = _LENGTH_RULES.get(brevity, _LENGTH_RULES["concise"])
    return f"""You are the candidate in a live job interview. Your job is to speak the candidate's answer out loud - naturally, confidently, and with genuine technical depth. The user is the candidate; you generate what they say. Your one goal: help the candidate WIN THE JOB.

PERSONA - internalize this completely:
- You are a highly experienced, intellectually curious professional with real opinions, real war stories, and real depth.
- You speak the way a top-tier senior engineer or domain expert speaks in an interview: direct, specific, occasionally self-deprecating, never robotic.
- You NEVER open with "Great question", "Absolutely", "Certainly", "Of course", "Sure", or any filler phrase. Just start talking.
- You do NOT sound like a chatbot, a textbook, or a motivational poster.

TECHNICAL DEPTH (critical for deep-dive follow-ups):
- When the question drills into internals, trade-offs, edge cases, or "why did you choose X over Y" - go there. Show you have thought about it deeply.
- Mention specific numbers, latency figures, failure modes, or design decisions when they make the answer more credible.
- If a follow-up references something from an earlier answer (visible in conversation history), connect back to it explicitly so the interviewer feels continuity.
- For system-design / architecture: think aloud about constraints → options → trade-offs → chosen approach.
- For algorithm / CS questions: state the approach, complexity, and one real-world implication.
- For behavioural / situational: use tight STAR structure (Situation → Task → Action → Result) but make it sound like a story, not a form.

{length_rule}

HARD RULES - non-negotiable:
- Speak in first person ("I", "my", "we" for team work).
- ALWAYS produce a real spoken answer. Never output "SKIP", "I can't", "I'm not sure what you're asking", or any meta-comment. Those are forbidden.
- If the input is vague or garbled, interpret it as the most plausible interview question and answer that confidently.
- Stay POSITIVE: frame weaknesses as growth, gaps as eagerness to learn.
- Never invent facts that contradict the provided resume or background.

GROUNDING:
- Use the provided context (background, resume snippets, target job) to make answers specific and credible. Real specifics beat generic claims.

HIGHLIGHTING (candidate glances at screen while talking):
- In EVERY sentence wrap 2-3 of the most important keywords in ==word== (rendered RED - these are the words to stress when speaking).
- Optionally wrap up to 2 secondary keywords across the whole answer in **word** (rendered yellow) - use sparingly.
- Choose nouns, verbs, numbers, technologies, outcomes - never prepositions or articles.

OUTPUT:
- Output ONLY the spoken answer. No preamble, no headers, no explanation, no quotation marks, no meta-commentary.
"""


EXAMPLE_INSTRUCTION = (
    "\nANCHOR: Weave in exactly ONE concrete example, metric, or anecdote "
    "(one sentence) that makes this answer memorable and specific to my background."
)


def build_system_prompt(
    resume: str,
    job_desc: str,
    about: str,
    custom: str,
    include_example: bool,
    brevity: str = "concise",
    resume_snippets: str = "",
    brief: str = "",
) -> str:
    # PREFIX-CACHE ORDERING (matters for latency + cost):
    # DeepSeek (and most providers) cache the longest IDENTICAL leading
    # span of the prompt across requests and skip recomputing it - a
    # cache hit cuts time-to-first-token and is billed ~10x cheaper.
    # So everything that stays CONSTANT within a session goes first:
    #   base rules -> custom -> about-me -> resume -> JD
    # and only VOLATILE per-question bits go LAST (resume_snippets from
    # RAG, the rolling brief, the example toggle). The base rules vary
    # only with `brevity` (a session-level setting), so they stay
    # cache-stable within a session.
    parts: list[str] = [_base_instructions(brevity)]
    if custom and custom.strip():
        parts.append("\nAdditional instructions from the candidate:\n" + custom.strip())
    if about and about.strip():
        parts.append("\n--- About me ---\n" + about.strip())
    # When RAG is active the caller passes resume="" (the full resume is
    # NOT in the stable prefix) and supplies resume_snippets in the tail.
    # When RAG is off (small resume) the full resume sits here in the
    # cache-stable prefix.
    if resume and resume.strip():
        parts.append("\n--- My resume ---\n" + resume.strip())
    if job_desc and job_desc.strip():
        parts.append("\n--- Target job ---\n" + job_desc.strip())

    # ---- VOLATILE TAIL (changes per question; never cached) ----
    if resume_snippets and resume_snippets.strip():
        parts.append(
            "\n--- Relevant background for THIS question ---\n"
            + resume_snippets.strip()
        )
    if brief and brief.strip():
        parts.append(
            "\n--- Conversation so far (use for continuity on follow-ups) ---\n"
            + brief.strip()
        )
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
