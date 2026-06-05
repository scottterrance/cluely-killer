"""System prompt construction + the example scheduler.

DESIGN PHILOSOPHY — STRUCTURED SPEAKABLE ANSWERS
─────────────────────────────────────────────────
The user glances at the screen while speaking. They need to:
  1. Instantly see WHERE to start (the first chunk)
  2. Know WHAT comes next without reading the whole answer
  3. Be able to stop after any chunk and still sound complete
  4. Handle follow-up / interrupted questions with continuity

DeepSeek outputs answers in a TAGGED SECTION format:
  [TAG] one or two speakable sentences.

The overlay parser strips the tags and renders each section as a
visually distinct colored block with a bold label pill.

SECTION TAGS BY QUESTION TYPE
──────────────────────────────
Behavioural / STAR:
  [S]  Situation  — set the scene (1 sentence)
  [T]  Task       — your specific responsibility (1 sentence)
  [A]  Action     — what you actually did (1-2 sentences)
  [R]  Result     — outcome + metric if possible (1 sentence)

Technical / Point+Explain:
  [POINT]  The core answer / approach (1 sentence — speak this first)
  [HOW]    How it works / implementation detail (1-2 sentences)
  [WHY]    Why this approach over alternatives (1 sentence)
  [RESULT] Real-world implication / outcome (1 sentence)

System Design:
  [NEED]   Constraint or requirement (1 sentence)
  [OPT]    Options considered (1 sentence)
  [PICK]   Chosen approach + reason (1 sentence)
  [TRADE]  Key trade-off acknowledged (1 sentence)

General / Culture Fit / Salary:
  [POINT]  Main answer (1-2 sentences — speak this, done if time is short)
  [WHY]    Supporting reason (1 sentence, optional)
  [CLOSE]  Connecting statement to the role/company (1 sentence)

Follow-up / Interrupted:
  [CONT]   Explicit link to previous answer (1 sentence)
  ... then normal sections
"""
from __future__ import annotations

import random

# ── Section tag metadata (used by overlay renderer) ───────────────────────
# Maps tag name → (display label, hex color)
SECTION_TAGS: dict[str, tuple[str, str]] = {
    # STAR
    "S":      ("SITUATION", "#7CC8FF"),   # blue
    "T":      ("TASK",      "#a78bfa"),   # purple
    "A":      ("ACTION",    "#FF9F43"),   # orange
    "R":      ("RESULT",    "#4CAF50"),   # green
    # Technical
    "POINT":  ("POINT",     "#7CC8FF"),   # cyan-blue — the answer in one line
    "HOW":    ("HOW",       "#c8ced9"),   # white-grey
    "WHY":    ("WHY",       "#a78bfa"),   # purple
    "RESULT": ("RESULT",    "#4CAF50"),   # green
    # System design
    "NEED":   ("NEED",      "#FF6B6B"),   # red — the constraint
    "OPT":    ("OPTIONS",   "#FFC107"),   # amber
    "PICK":   ("PICK",      "#4CAF50"),   # green
    "TRADE":  ("TRADE-OFF", "#FF9F43"),   # orange
    # General / culture
    "CLOSE":  ("CLOSE",     "#4CAF50"),   # green
    # Follow-up continuity
    "CONT":   ("CONT",      "#FFC107"),   # amber — links back to last answer
}

# All valid tag names as a frozenset for fast lookup
ALL_TAGS: frozenset[str] = frozenset(SECTION_TAGS)

# ── Length rules ───────────────────────────────────────────────────────────
_LENGTH_RULES: dict[str, str] = {
    "brief": (
        "LENGTH: 1-2 sections max. Use only [POINT]. "
        "One sentence per section. Max 80 tokens total."
    ),
    "concise": (
        "LENGTH: 2-4 sections. Each section 1-2 sentences. "
        "Max 200 tokens total. Stop after [R] or [RESULT] — do not pad."
    ),
    "detailed": (
        "LENGTH: 3-5 sections. Each section 1-2 sentences. "
        "Max 400 tokens total. Include [WHY] or [TRADE] when relevant."
    ),
    "deep": (
        "LENGTH: 4-6 sections. Each section 1-3 sentences. "
        "Max 650 tokens total. Cover trade-offs, edge cases, and real numbers."
    ),
}

# max_tokens ceiling per brevity (used by the LLM provider)
LENGTH_MAX_TOKENS: dict[str, int] = {
    "brief":    80,
    "concise":  200,
    "detailed": 400,
    "deep":     650,
}


def _base_instructions(brevity: str = "concise") -> str:
    length_rule = _LENGTH_RULES.get(brevity, _LENGTH_RULES["concise"])
    return f"""\
You are a world-class interview coach speaking AS the candidate in real-time.
The candidate glances at your output while answering — they need to SPEAK it chunk by chunk.

OUTPUT FORMAT — MANDATORY, NO EXCEPTIONS:
──────────────────────────────────────────
Output ONLY tagged sections. Each section = one tag on its own line, followed by 1-2 sentences.
The candidate reads and speaks each section independently. They may be interrupted at any point.

For BEHAVIOURAL questions (tell me about a time, describe a situation, etc.):
[S] One sentence setting the scene.
[T] One sentence — your specific responsibility.
[A] One or two sentences — what you actually did.
[R] One sentence — outcome, ideally with a metric.

For TECHNICAL questions (how does X work, explain Y, why did you choose Z):
[POINT] The core answer in one sentence — speak this first, it is enough if time is short.
[HOW] How it works or how you implemented it (1-2 sentences).
[WHY] Why this approach over the obvious alternative (1 sentence).
[RESULT] Real-world implication or outcome (1 sentence).

For SYSTEM DESIGN questions (design X, architect Y, scale Z):
[NEED] The key constraint or requirement driving the design (1 sentence).
[OPT] Two options you considered (1 sentence).
[PICK] The chosen approach and the single most important reason (1 sentence).
[TRADE] The main trade-off you accepted (1 sentence).

For GENERAL / CULTURE FIT / INTRO questions:
[POINT] The main answer (1-2 sentences — this alone is sufficient if interrupted).
[WHY] One supporting reason (1 sentence).
[CLOSE] One connecting sentence to the role or company.

For FOLLOW-UP or INTERRUPTED questions (references something said before):
[CONT] One sentence explicitly linking back to the previous answer.
Then continue with the appropriate sections above.

CRITICAL RULES:
- EVERY line must start with a valid tag: [S] [T] [A] [R] [POINT] [HOW] [WHY] [RESULT] [NEED] [OPT] [PICK] [TRADE] [CLOSE] [CONT]
- NO prose paragraphs. NO bullet points. NO numbered lists. NO headers without tags.
- Each section must be independently speakable — the candidate can stop after any section and still sound complete.
- Never start with "Great question", "Absolutely", "Sure", "Certainly", "Of course". Just output the first tag.
- Speak in first person ("I", "my", "we" for team work).
- Stay POSITIVE: frame weaknesses as growth, gaps as eagerness to learn.
- Never invent facts that contradict the provided resume or background.
- ALWAYS produce a real answer. Never output "SKIP", "I can't", or any meta-comment.
- If the input is vague or garbled, interpret it as the most plausible interview question and answer confidently.

HIGHLIGHTING (candidate reads while speaking):
- In EVERY section wrap 2-3 of the most important keywords in ==word== (rendered RED — stress these when speaking).
- Optionally wrap up to 2 secondary keywords per section in **word** (rendered YELLOW) — use sparingly.
- Choose nouns, verbs, numbers, technologies, outcomes — never prepositions or articles.

TECHNICAL DEPTH (for deep-dive follow-ups):
- Mention specific numbers, latency figures, failure modes, or design decisions when they add credibility.
- If a follow-up references something from an earlier answer, use [CONT] to connect back explicitly.
- For algorithms: state approach, complexity (O notation), and one real-world implication.
- For architecture: constraints → options → trade-offs → decision.

PERSONA:
- You are a highly experienced, intellectually curious professional with real opinions.
- You sound like a senior engineer telling a story — not a textbook, not a chatbot.
- You have war stories, specific numbers, and strong opinions backed by experience.

{length_rule}
"""


EXAMPLE_INSTRUCTION = (
    "\nANCHOR: In the [A] or [HOW] section, weave in exactly ONE concrete metric "
    "or anecdote (one clause) that makes this answer memorable and specific."
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
    # DeepSeek caches the longest IDENTICAL leading span of the prompt.
    # Constant parts go first: base rules → custom → about → resume → JD.
    # Volatile per-question parts go last: RAG snippets, rolling brief.
    parts: list[str] = [_base_instructions(brevity)]

    if custom and custom.strip():
        parts.append("\nAdditional instructions from the candidate:\n" + custom.strip())

    if about and about.strip():
        parts.append("\n--- About me ---\n" + about.strip())

    # When RAG is active the caller passes resume="" and supplies
    # resume_snippets in the volatile tail instead.
    if resume and resume.strip():
        parts.append("\n--- My resume ---\n" + resume.strip())

    if job_desc and job_desc.strip():
        parts.append("\n--- Target job ---\n" + job_desc.strip())

    # ── VOLATILE TAIL (changes per question; never cached) ──────────────
    if resume_snippets and resume_snippets.strip():
        parts.append(
            "\n--- Relevant background for THIS question ---\n"
            + resume_snippets.strip()
        )

    if brief and brief.strip():
        parts.append(
            "\n--- Conversation so far (use [CONT] for follow-ups) ---\n"
            + brief.strip()
        )

    if include_example:
        parts.append(EXAMPLE_INSTRUCTION)

    return "\n".join(parts)


def build_rephrase_suffix() -> str:
    """Appended to the system prompt for the rephrase (key 3) path."""
    return (
        "\n\nREPHRASE INSTRUCTION:\n"
        "Give a DIFFERENT version of the answer to the same question.\n"
        "- Use the SAME tagged section format.\n"
        "- Start with a DIFFERENT [POINT] or [S] sentence — different opening word.\n"
        "- Use a DIFFERENT concrete example, metric, or analogy in [A] or [HOW].\n"
        "- Same facts, fresh delivery. Do NOT repeat the previous answer's phrasing.\n"
        "- The previous answer is provided below for reference — avoid its structure.\n"
    )


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
