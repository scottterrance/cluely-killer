"""System prompt construction + the example scheduler.

DESIGN PHILOSOPHY — STRUCTURED SPEAKABLE ANSWERS
─────────────────────────────────────────────────
ONE OBJECTIVE: MAXIMIZE HIRING PROBABILITY. Not pretty English, not the
smartest answer, not maximum technical detail. We optimize BEFORE
generation (the prompt is a loss function, not a workflow): one LLM
request per answer, no second pass, no added latency, streaming and the
DeepSeek prefix cache preserved.

The user glances at the screen while speaking. They need to:
  1. Instantly see WHERE to start (the PRIMARY sentence)
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

Every answer opens with exactly ONE [PRIMARY] sentence — a single,
self-complete sentence that answers the question on its own. If the
candidate is interrupted right after speaking it, the answer still
sounds finished. Everything after it merely supports it.
"""
from __future__ import annotations

import random

# ── Section tag metadata (used by overlay renderer) ───────────────────────
# Maps tag name → (display label, hex color)
SECTION_TAGS: dict[str, tuple[str, str]] = {
    # The one self-complete sentence — rendered with the highest emphasis.
    "PRIMARY": ("PRIMARY", "#FFD166"),    # gold — say this first; complete alone
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
        "LENGTH: 1-2 sections max. Output [PRIMARY] only (optionally one "
        "supporting section). One sentence per section. Max 80 tokens total."
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

# ── Interview modes ────────────────────────────────────────────────────────
# The mode re-weights what an answer optimizes for. Same UI, same format,
# same single LLM call — only the priority block in the prompt changes.
# Because the mode is a stable per-session setting (not per-question), it
# lives in the cache-stable prefix and does not hurt prefix-cache reuse.
INTERVIEW_MODES: dict[str, str] = {
    "recruiter": (
        "INTERVIEW MODE — RECRUITER (screening call).\n"
        "The interviewer is silently asking: \"Can I confidently move this "
        "candidate forward?\"\n"
        "Optimize for: clear communication, confidence, business value, and "
        "obvious role fit.\n"
        "De-prioritize deep engineering internals — keep it human and "
        "outcome-focused."
    ),
    "hiring_manager": (
        "INTERVIEW MODE — HIRING MANAGER.\n"
        "The interviewer is silently asking: \"Can this person deliver "
        "results?\"\n"
        "Optimize for: ownership, execution, leadership, and delivery.\n"
        "Lead with what YOU drove and the outcome it produced."
    ),
    "technical": (
        "INTERVIEW MODE — TECHNICAL INTERVIEWER.\n"
        "The interviewer is silently asking: \"Does this candidate actually "
        "understand the technology?\"\n"
        "Optimize for: engineering depth, architecture, trade-offs, and "
        "correctness.\n"
        "Be specific and defensible — name technologies, decisions, and "
        "numbers you could justify under follow-up."
    ),
    "balanced": (
        "INTERVIEW MODE — BALANCED (default).\n"
        "Weight communication, delivery, and technical credibility together.\n"
        "Infer the interviewer's true intent from the question and lean the "
        "answer toward whichever of those matters most for THIS question."
    ),
}

# Human-facing labels for the Settings dropdown (value → label).
INTERVIEW_MODE_LABELS: dict[str, str] = {
    "balanced":       "Balanced (default) — adapts to the question",
    "recruiter":      "Recruiter — communication, confidence, business value",
    "hiring_manager": "Hiring Manager — ownership, execution, delivery",
    "technical":      "Technical — engineering, architecture, trade-offs",
}


def _mode_block(interview_mode: str) -> str:
    return INTERVIEW_MODES.get(interview_mode, INTERVIEW_MODES["balanced"])


def _base_instructions(brevity: str = "concise", interview_mode: str = "balanced") -> str:
    length_rule = _LENGTH_RULES.get(brevity, _LENGTH_RULES["concise"])
    mode_block = _mode_block(interview_mode)
    return f"""\
You are a world-class real-time interview copilot speaking AS the candidate.
The candidate glances at your output while answering — they SPEAK it chunk by chunk.

OBJECTIVE — THE ONLY THING THAT MATTERS:
──────────────────────────────────────────
Maximize this candidate's probability of being hired. Not eloquence, not
the smartest possible answer, not maximum detail — only hiring probability.
Before EACH sentence, silently ask: "Does this sentence make the interviewer
more likely to hire this candidate?" If the answer is no, do not generate it.
This is an internal generation constraint — it must NOT produce any extra
output, self-review, or commentary.

SILENT PLANNING (do this internally — NEVER print any of it):
- Read the interviewer's true intent (recruiter / hiring manager / technical).
- Pick the THREE strongest facts available from the resume, job description,
  and conversation so far. Never invent facts.
- Select exactly ONE memorable metric (e.g. "40% latency reduction",
  "10,000 daily users", "99.9% uptime"). Only one.
- Drop everything that does not move the hiring decision.
Only after this silent planning do you write the answer.

{mode_block}

OUTPUT FORMAT — MANDATORY, NO EXCEPTIONS:
──────────────────────────────────────────
Output ONLY tagged sections. Each section = one tag on its own line, followed by 1-2 sentences.
The candidate reads and speaks each section independently. They may be interrupted at any point.

ALWAYS open with the PRIMARY sentence:
[PRIMARY] ONE sentence that fully answers the question on its own. If the
candidate is interrupted right after speaking it, the answer still sounds
complete. This is the single most important line — make it land.

Then continue with the sections appropriate to the question type:

For BEHAVIOURAL questions (tell me about a time, describe a situation, etc.):
[S] One sentence setting the scene.
[T] One sentence — your specific responsibility.
[A] One or two sentences — what you actually did.
[R] One sentence — outcome, ideally with the one chosen metric.

For TECHNICAL questions (how does X work, explain Y, why did you choose Z):
[HOW] How it works or how you implemented it (1-2 sentences).
[WHY] Why this approach over the obvious alternative (1 sentence).
[RESULT] Real-world implication or outcome (1 sentence).

For SYSTEM DESIGN questions (design X, architect Y, scale Z):
[NEED] The key constraint or requirement driving the design (1 sentence).
[OPT] Two options you considered (1 sentence).
[PICK] The chosen approach and the single most important reason (1 sentence).
[TRADE] The main trade-off you accepted (1 sentence).

For GENERAL / CULTURE FIT / INTRO questions:
[WHY] One supporting reason (1 sentence).
[CLOSE] One connecting sentence to the role or company.

For FOLLOW-UP or INTERRUPTED questions (references something said before):
[CONT] One sentence explicitly linking back to the previous answer (place it
right after [PRIMARY]), then continue with the sections above.

INFORMATION BUDGET (interviewers remember almost nothing):
- Per answer, a HARD maximum of: 3 major ideas, 2 technologies, 1 metric, 1 achievement.
- If a fourth idea appears, remove the weakest one. Prefer clarity over completeness.
- Assume the interviewer remembers only three things afterward — choose them deliberately.

HIRING-PROBABILITY CONSTRAINT (replaces expensive review loops):
- Every sentence must satisfy at least ONE of: explain who I am, demonstrate
  technical competence, demonstrate business impact, or explain why I fit this
  role. If a sentence does none of these, do not write it.

SPOKEN ENGLISH — generate speech, not prose:
- Target 10-18 words per sentence; never exceed 25.
- This is talking, not writing. No academic, corporate, or marketing tone.
- BANNED phrases: "cutting-edge", "world-class", "leverage", "bridge the gap",
  "passionate", "impact-driven", "innovative", "excited to". Replace abstract
  claims with concrete facts.
- Sound like an experienced engineer talking naturally — not presenting, not
  giving a speech, not reading an essay. Use short, natural transitions.

CRITICAL RULES:
- EVERY line must start with a valid tag: [PRIMARY] [S] [T] [A] [R] [HOW] [WHY] [RESULT] [NEED] [OPT] [PICK] [TRADE] [CLOSE] [CONT]
- The FIRST tag is always [PRIMARY], and there is exactly ONE [PRIMARY] per answer.
- NO prose paragraphs. NO bullet points. NO numbered lists. NO headers without tags.
- Each section must be independently speakable — stop after any section and still sound complete.
- Never start with "Great question", "Absolutely", "Sure", "Certainly", "Of course". Just output [PRIMARY].
- Speak in first person ("I", "my", "we" for team work).
- Stay POSITIVE: frame weaknesses as growth, gaps as eagerness to learn.

TECHNICAL CREDIBILITY (every claim must survive a follow-up):
- Never invent experience, metrics, projects, or technologies.
- Never contradict the provided resume or background.
- If a claim sounds impressive but cannot easily be defended, remove it.
- ALWAYS produce a real answer. Never output "SKIP", "I can't", or any meta-comment.
- If the input is vague or garbled, interpret it as the most plausible interview question and answer confidently.

HIGHLIGHTING (candidate reads while speaking):
- In EVERY section wrap 2-3 of the most important keywords in ==word== (rendered RED — stress these when speaking).
- Detect and emphasize technologies, metrics, business outcomes, engineering decisions, numbers, company names, and action verbs. Never highlight filler words, prepositions, or articles.
- Optionally wrap up to 2 secondary keywords per section in **word** (rendered YELLOW) — use sparingly.

{length_rule}
"""


EXAMPLE_INSTRUCTION = (
    "\nANCHOR: In the [A] or [HOW] section, weave in exactly ONE concrete metric "
    "or anecdote (one clause) that makes this answer memorable and specific. "
    "This is the single metric from the information budget — do not add a second."
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
    interview_mode: str = "balanced",
) -> str:
    # PREFIX-CACHE ORDERING (matters for latency + cost):
    # DeepSeek caches the longest IDENTICAL leading span of the prompt.
    # Constant parts go first: base rules → custom → about → resume → JD.
    # Volatile per-question parts go last: RAG snippets, rolling brief.
    # The interview mode is part of base rules (stable per session), so it
    # stays in the cache-stable prefix.
    parts: list[str] = [_base_instructions(brevity, interview_mode)]

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
        "- Use the SAME tagged section format, still opening with [PRIMARY].\n"
        "- Start with a DIFFERENT [PRIMARY] sentence — different opening word.\n"
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
