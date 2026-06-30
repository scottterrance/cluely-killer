"""System prompt construction based on the Core Optimization Principle.

DESIGN PHILOSOPHY: MAXIMIZE HIRING PROBABILITY
─────────────────────────────────────────────
ONE OBJECTIVE: MAXIMIZE HIRING PROBABILITY.
Not the best answer. Not the most technical. Not the most complete.
Every generated sentence must increase the probability that the interviewer
moves the candidate to the next stage.
"""
from __future__ import annotations

# ── Dependencies for other modules ──────────────────────────────────────────
# overlay.py imports these to render the UI and header badges.
# We keep them for compatibility but the prompt no longer uses [TAGS].
SECTION_TAGS: dict[str, tuple[str, str]] = {
    "PRIMARY": ("PRIMARY", "#FFD166"),  # gold
}
ALL_TAGS: frozenset[str] = frozenset(SECTION_TAGS)

# main.py imports these for LLM configuration and the controller.
LENGTH_MAX_TOKENS: dict[str, int] = {
    "brief": 80,
    "concise": 200,
    "detailed": 400,
    "deep": 650,
}

class ExampleScheduler:
    """Kept for main.py compatibility; the prompt no longer uses examples."""
    def should_include(self) -> bool: return False

# settings_dialog.py imports these for the UI dropdowns.
INTERVIEW_MODE_LABELS: dict[str, str] = {
    "balanced": "Balanced (default)",
    "recruiter": "Recruiter (screening)",
    "hiring_manager": "Hiring Manager (delivery)",
    "technical": "Technical (engineering depth)",
}

INTERVIEW_MODE_SHORT: dict[str, str] = {
    "balanced": "BALANCED",
    "recruiter": "RECRUITER",
    "hiring_manager": "HIRING MGR",
    "technical": "TECHNICAL",
}

def build_rephrase_suffix() -> str:
    """Return the instruction appended to the prompt for Key '3' (rephrase)."""
    return (
        "\n\nREPHRASE INSTRUCTION:\n"
        "Give a DIFFERENT version of your previous answer. Change the opening, "
        "use a different supporting example, or change the emphasis. "
        "Apply the HIRING-PROBABILITY TEST: if a sentence doesn't directly "
        "increase the chance of a 'Hire' decision, delete it. "
        "Same facts, fresh delivery. Do NOT repeat yourself verbatim."
    )

def build_system_prompt(
    resume: str = "",
    job_desc: str = "",
    about: str = "",
    custom: str = "",
    include_example: bool = False,
    brevity: str = "concise",
    resume_snippets: str = "",
    brief: str = "",
    interview_mode: str = "balanced",
) -> str:
    """The core 'hiring-probability' system prompt based on Core Optimization Principle."""

    # Select the silent evaluation criteria based on the active mode.
    mode_eval_criteria = {
        "recruiter": "communication -> confidence -> role fit. Can I confidently move this candidate forward?",
        "hiring_manager": "ownership -> business impact -> execution. Can this person deliver results?",
        "technical": "engineering judgment -> scalability -> trade-offs. Does this candidate actually understand the tech?",
        "balanced": "auto-adapt to the question's hidden evaluation criteria (e.g. Behavioral -> judgment; Coding -> correctness).",
    }.get(interview_mode, "maximize hiring probability.")

    # Information budget constraints
    budget = {
        "brief": "1 key idea, 1 metric (if any).",
        "concise": "2 key ideas, 1 metric, 1 technology.",
        "detailed": "3 key ideas, 1 metric, 2 technologies, 1 achievement.",
        "deep": "3 key ideas, 1 metric, 2 technologies, 1 achievement. Add depth to trade-offs.",
    }.get(brevity, "3 key ideas, 1 metric, 2 technologies.")

    prompt = f"""CORE OPTIMIZATION PRINCIPLE
Do NOT optimize for the best answer.
Do NOT optimize for the most technical answer.
Do NOT optimize for the most complete answer.

Instead, optimize for one objective:
Maximize the probability that the interviewer moves the candidate to the next interview stage.
Every generated sentence MUST increase that probability.

SILENT INTERVIEW INTENT PLANNING
Before generating the answer, perform this planning silently. Never output these steps.
Step 1 - Identify Interview Intent: Determine what the interviewer is actually evaluating.
Active Mode Evaluation Criteria: {mode_eval_criteria}
Step 2 - Choose Evidence: Select only the strongest, most persuasive evidence from the candidate's profile. Never invent information.
Step 3 - Build the Answer:
1. Direct answer.
2. Supporting evidence.
3. Business impact or lesson learned.
4. Natural closing when appropriate.

HIRING CONFIDENCE CONSTRAINTS
Before generating every sentence, silently ask: "What does this sentence prove?"
Every sentence must prove at least one of:
- I can communicate clearly.
- I can solve problems.
- I deliver business value.
- I work well with other people.
- I fit this role.
If a sentence proves none of these, do NOT generate it.

INFORMATION BUDGET
Interviewers remember very little. Prefer clarity over completeness.
Current Budget ({brevity}): {budget}
Maximum across all modes: 3 key ideas, 1 memorable metric, 2 technologies, 1 major achievement.

NATURAL CONVERSATION
- Generate spoken English, not written English.
- Sound like an experienced engineer having a conversation.
- Avoid speeches, marketing language, and unnecessary buzzwords.
- Use short, natural sentences (10-18 words; never more than 25).

INTERNAL OBJECTIVE
The interviewer should finish the answer thinking:
"I understand who this person is."
"I believe their experience."
"I can imagine working with them."

FORMATTING RULES
- NO TAGS like [STAR], [WHY], [S], [T], [A], [R], [POINT], [HOW], etc.
- Use ONLY the [PRIMARY] tag for the first, self-complete sentence.
- Use ONLY standard Markdown for emphasis.
- Use ==double equals== for key terms the candidate should stress when speaking.
- Use **bold** for secondary emphasis.
- Output the answer as a few short, spoken paragraphs.

CANDIDATE PROFILE
About me: {about or "Not provided"}
Resume: {resume or "Not provided"}
{f"Relevant Resume Snippets: {resume_snippets}" if resume_snippets else ""}
Job Description: {job_desc or "Not provided"}
{f"Conversation Context: {brief}" if brief else ""}
{f"Custom Instructions: {custom}" if custom else ""}
"""
    return prompt.strip()
