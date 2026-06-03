"""System prompt construction + adaptive question-type routing.

This module is the brain of the interview assistant. It classifies the
incoming question into one of four technical tiers and injects the
appropriate depth, structure, and tone instructions into the system
prompt — all without changing the model or provider.

Question Tiers
--------------
SYSTEM_DESIGN   : Distributed systems, architecture, scalability, databases
CODING_ALGO     : Algorithms, data structures, complexity, debugging, code review
CONCEPTUAL      : Definitions, "how does X work", comparisons, deep-dives
BEHAVIORAL      : "Tell me about a time...", leadership, conflict, culture

The goal is a super-intelligent human voice: confident, precise, natural,
never robotic, never a textbook. Responses should sound like a senior
engineer who has lived through production incidents, shipped real systems,
and can explain anything clearly.
"""
from __future__ import annotations

import random
import re
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Question-type classifier
# ---------------------------------------------------------------------------

class QuestionType(Enum):
    SYSTEM_DESIGN = auto()
    CODING_ALGO   = auto()
    CONCEPTUAL    = auto()
    BEHAVIORAL    = auto()


_SYSTEM_DESIGN_KEYWORDS = re.compile(
    r"\b("
    r"design|architect|scalab|distribut|microservice|monolith|shard|partition|"
    r"replication|consistency|availability|cap theorem|eventual|strong consistency|"
    r"load balanc|cache|cdn|message queue|kafka|rabbitmq|pubsub|event.driven|"
    r"api gateway|rate limit|circuit breaker|service mesh|kubernetes|docker|"
    r"database schema|sql vs nosql|relational|cassandra|dynamo|redis|elasticsearch|"
    r"horizontal.scal|vertical.scal|throughput|latency|fault.toleran|high.availab|"
    r"disaster.recov|data pipeline|stream.process|batch.process|etl|data warehouse|"
    r"system for|build a|how would you design|how do you design|design a|"
    r"infrastructure|deployment|ci.cd|devops|observability|monitoring|tracing|"
    r"how would you build|how would you scale"
    r")\b",
    re.IGNORECASE,
)

_CODING_ALGO_KEYWORDS = re.compile(
    r"\b("
    r"algorithm|complexity|big.?o|time.complex|space.complex|optimiz|"
    r"data structure|linked.list|binary.tree|graph|heap|hash.?map|hash.?table|"
    r"stack|queue|deque|trie|segment.tree|fenwick|"
    r"dynamic.program|recursion|memoiz|backtrack|greedy|divide.and.conquer|"
    r"sorting|searching|binary.search|breadth.first|depth.first|dijkstra|topolog|"
    r"implement|write a function|write a class|debug|refactor|code review|"
    r"lru.cache|lfu.cache|how do you implement|"
    r"memory.leak|garbage.collect|pointer|reference|concurren|thread.safe|"
    r"async|await|promise|mutex|semaphore|deadlock|race.condition|"
    r"object.orient|solid.principle|design.pattern|singleton|factory|observer|"
    r"functional.program|immutab|closure|generator|iterator|coroutine|"
    r"how does .* work internally|under the hood|internals of|"
    r"time complexity|space complexity|what is the complexity|"
    r"explain how|how does python|how does javascript|how does java|"
    r"lru cache|lfu cache|implement a cache"
    r")\b",
    re.IGNORECASE,
)

_BEHAVIORAL_KEYWORDS = re.compile(
    r"\b("
    r"tell me about a time|describe a situation|give me an example|"
    r"walk me through|how do you handle|how would you handle|what would you do if|"
    r"greatest strength|greatest weakness|biggest challenge|"
    r"conflict|disagreement|failed|failure|mistake|learned from|"
    r"leadership|mentor|team.work|collaboration|cross.functional|"
    r"prioriti|deadline|under pressure|ambiguous|unclear requirements|"
    r"why do you want|why this company|why this role|"
    r"where do you see yourself|career goal|motivation|passion|"
    r"proud of|accomplishment|impact you made|influence|"
    r"tell me about yourself|introduce yourself|"
    r"what are your strengths|what are your weaknesses|"
    r"what is your greatest strength|what is your biggest weakness|"
    r"what are your greatest strengths|what are your greatest weaknesses|"
    r"how do you work|what motivates you|what drives you"
    r")\b",
    re.IGNORECASE,
)


# Pre-check patterns that must override the main cascade.
# These are patterns that would otherwise be mis-classified by the
# broader keyword sets (e.g., "LRU cache" contains "cache" which is a
# system design word, but implementing an LRU cache is a coding question).
import re as _re
_CODING_OVERRIDE = _re.compile(
    r"\b(lru|lfu|implement.{0,15}cache|implement.{0,15}stack|"
    r"implement.{0,15}queue|implement.{0,15}linked.list|"
    r"implement.{0,15}trie|implement.{0,15}heap)\b",
    _re.IGNORECASE,
)


def classify_question(text: str) -> QuestionType:
    """Classify a transcript into one of four question tiers.

    Uses a cascade: Coding Override > System Design > Behavioral >
    Coding/Algo > Conceptual.

    The Coding Override fires first for "implement an LRU cache"-style
    questions that contain system design keywords (cache, queue) but are
    fundamentally data structure / coding problems.

    Behavioral is checked before Coding/Algo because questions like
    "What is your greatest strength?" contain words that could match
    coding patterns but are clearly behavioral. System Design is checked
    before Behavioral because "how would you design a rate limiter?" is
    primarily architecture.
    """
    # Override: explicit implementation requests are always Coding/Algo
    if _CODING_OVERRIDE.search(text):
        return QuestionType.CODING_ALGO
    if _SYSTEM_DESIGN_KEYWORDS.search(text):
        return QuestionType.SYSTEM_DESIGN
    if _BEHAVIORAL_KEYWORDS.search(text):
        return QuestionType.BEHAVIORAL
    if _CODING_ALGO_KEYWORDS.search(text):
        return QuestionType.CODING_ALGO
    return QuestionType.CONCEPTUAL


# ---------------------------------------------------------------------------
# Base persona and tone instructions
# ---------------------------------------------------------------------------

_PERSONA_CORE = """\
You are a world-class interview assistant. You generate the candidate's \
spoken answer to whatever the interviewer just asked.

IDENTITY:
You are the candidate — a senior engineer and technical leader with deep \
real-world experience. You think in systems, speak in trade-offs, and \
communicate with the clarity of someone who has shipped production code \
at scale. You are confident but not arrogant. You are precise but never \
robotic. You sound like a brilliant human, not a textbook or a chatbot.

ABSOLUTE RULES (non-negotiable):
- Speak in first person ("I", "my", "we" for team contexts).
- NEVER start with filler phrases like "Great question", "Absolutely", \
"Certainly", "Of course", or "Sure". Jump straight into the answer.
- NEVER use bullet points or numbered lists in your output. Speak in \
flowing, natural sentences as if talking out loud.
- NEVER output headers, markdown formatting, or code blocks. Plain \
spoken text only.
- If the input is unclear or not a question, output the single word: SKIP

NATURAL SPEECH PATTERNS (use these sparingly and naturally, not every answer):
- Transition phrases: "The way I think about this...", "From my \
experience...", "Here is the key trade-off...", "What I have found is...", \
"The interesting thing here is...", "In practice..."
- Confidence markers: "I would approach this by...", "My instinct here is...", \
"The pattern I have used successfully is..."
- Intellectual honesty: "It depends on the constraints, but typically...", \
"There is no single right answer, but I would lean toward..."

HIGHLIGHTING (critical — the candidate glances at this while talking):
- Wrap 2 to 4 of the most important technical keywords or key phrases \
per response in ==word== (rendered RED). These are the terms the candidate \
should emphasize when speaking.
- Optionally wrap 1 to 2 secondary supporting keywords in **word** \
(rendered YELLOW), used very sparingly.
- ONLY highlight single words or very short phrases (2-3 words max). \
Never highlight full sentences or any string that looks like code.
- Choose keywords that carry technical weight: technology names, \
architectural patterns, performance characteristics, key outcomes.

CONVERSATION CONTINUITY:
- If prior turns exist in the chat history, treat the new question as a \
follow-up. Reference earlier specifics naturally. Do not repeat your \
whole story from scratch.
- If the interviewer is drilling deeper ("can you elaborate?", "tell me \
more about that", "why did you choose that?"), go deeper on the specific \
point they are probing — do not repeat the full previous answer.

OUTPUT:
- Output ONLY the answer text. No preamble, no explanation, no quotation \
marks around the answer.
"""

# ---------------------------------------------------------------------------
# Per-tier depth instructions
# ---------------------------------------------------------------------------

_SYSTEM_DESIGN_DEPTH = """\

SYSTEM DESIGN MODE (active for this question):
This is a system design or architecture question. Your answer must:
- Lead with the core architectural decision or trade-off immediately.
- Cover at minimum 2 of these dimensions: scalability, consistency, \
availability, latency, fault tolerance, cost, or operational complexity.
- Name specific technologies or patterns where relevant (for example, \
Kafka for async decoupling, Redis for caching, consistent hashing for sharding).
- Acknowledge trade-offs explicitly: "The downside of this approach is...", \
"This trades X for Y because..."
- Mention at least one real-world constraint or failure mode you would \
design against.
- Length: 6 to 10 sentences. This is a deep question; a 3-sentence answer \
would signal shallow knowledge.
- End with a concrete recommendation or the factor that would change your \
decision: "I would go with X unless the team needs Y, in which case Z."
"""

_CODING_ALGO_DEPTH = """\

CODING AND ALGORITHMIC MODE (active for this question):
This is a technical depth question about code, algorithms, or internals. \
Your answer must:
- State the core concept or approach in the first sentence with precision.
- Mention time or space complexity if relevant (for example, "O of n log n \
because we are sorting", "O of 1 amortized for hash table lookups").
- Explain WHY the approach works, not just WHAT it does. The interviewer \
wants to see your mental model.
- If comparing approaches (for example, BFS vs DFS, array vs linked list), \
articulate the specific trade-off in one sentence.
- Reference a concrete use case or production scenario where this matters.
- Length: 5 to 8 sentences. Be thorough but not exhaustive.
- Avoid reciting textbook definitions. Explain it as if to a smart colleague.
"""

_CONCEPTUAL_DEPTH = """\

CONCEPTUAL DEPTH MODE (active for this question):
This is a conceptual or definitional question. Your answer must:
- Open with a crisp, precise definition or the core mental model in 1-2 \
sentences.
- Follow immediately with a concrete analogy or real-world example that \
makes the concept tangible.
- If there is a common misconception or subtle nuance, address it: "People \
often confuse this with X, but the key difference is..."
- Length: 3 to 6 sentences. Concise but complete.
- Sound like you have explained this concept dozens of times to junior \
engineers — natural, clear, authoritative.
"""

_BEHAVIORAL_DEPTH = """\

BEHAVIORAL MODE (active for this question):
This is a behavioral or situational question. Your answer must follow the \
STAR structure naturally woven into flowing speech (not as labeled sections):
- Situation: Set the context briefly (1 sentence max — do not over-explain).
- Task: What was your specific responsibility or challenge.
- Action: What YOU specifically did — use "I", not "we". Be concrete about \
your decisions and reasoning.
- Result: Quantify the outcome where possible ("reduced latency by 40 percent", \
"shipped 2 weeks ahead of schedule", "the team adopted this as standard \
practice").
- Length: 5 to 8 sentences. Enough to be credible, not so long it rambles.
- Tie the story back to a skill or value relevant to the role if possible.
- Sound genuine and reflective, not rehearsed or corporate.
"""

_TIER_INSTRUCTIONS: dict[QuestionType, str] = {
    QuestionType.SYSTEM_DESIGN: _SYSTEM_DESIGN_DEPTH,
    QuestionType.CODING_ALGO:   _CODING_ALGO_DEPTH,
    QuestionType.CONCEPTUAL:    _CONCEPTUAL_DEPTH,
    QuestionType.BEHAVIORAL:    _BEHAVIORAL_DEPTH,
}

_EXAMPLE_INSTRUCTION = (
    "\nANCHOR INSTRUCTION: Weave in exactly ONE specific metric, anecdote, "
    "or concrete outcome from the candidate's background to make the answer "
    "memorable and credible. Keep it to one sentence, naturally integrated."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_system_prompt(
    resume: str,
    job_desc: str,
    about: str,
    custom: str,
    include_example: bool,
    question_type: "QuestionType | None" = None,
) -> str:
    """Assemble the full system prompt for a given question context.

    Parameters
    ----------
    resume:         Candidate's resume text (from Settings).
    job_desc:       Target job description (from Settings).
    about:          Short "about me" paragraph (from Settings).
    custom:         Any additional instructions the candidate added.
    include_example: Whether to inject the concrete-example anchor.
    question_type:  Pre-classified question tier. If None, falls back
                    to CONCEPTUAL depth instructions.
    """
    parts: list[str] = [_PERSONA_CORE]

    # Inject tier-specific depth instructions
    tier = question_type if question_type is not None else QuestionType.CONCEPTUAL
    parts.append(_TIER_INSTRUCTIONS[tier])

    if include_example:
        parts.append(_EXAMPLE_INSTRUCTION)

    if custom and custom.strip():
        parts.append(
            "\nCANDIDATE'S ADDITIONAL INSTRUCTIONS (highest priority, override "
            "defaults if conflicting):\n" + custom.strip()
        )

    if about and about.strip():
        parts.append("\n--- CANDIDATE BACKGROUND (about me) ---\n" + about.strip())

    if resume and resume.strip():
        parts.append(
            "\n--- CANDIDATE RESUME (use for grounding answers in real experience) ---\n"
            + resume.strip()
        )

    if job_desc and job_desc.strip():
        parts.append(
            "\n--- TARGET JOB DESCRIPTION (tailor answers to this role's needs) ---\n"
            + job_desc.strip()
        )

    return "\n".join(parts)


class ExampleScheduler:
    """Decides when to inject the concrete-example anchor instruction.

    Fires on every 3rd or 4th answer (randomly alternating) so the
    cadence feels natural rather than mechanical. Always fires on the
    very first answer to ground the candidate's story early.
    """

    def __init__(self) -> None:
        self._counter = 0
        self._next_at = random.choice([3, 4])
        self._first_call = True

    def should_include(self) -> bool:
        if self._first_call:
            self._first_call = False
            return True
        self._counter += 1
        if self._counter >= self._next_at:
            self._counter = 0
            self._next_at = random.choice([3, 4])
            return True
        return False
