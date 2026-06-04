"""analysis.py — Real-time interview analysis helpers.

Two lightweight, zero-token, zero-latency tools that run locally:

1. FillerWordDetector
   Scans the interviewer's transcript for filler words / phrases
   ("um", "uh", "like", "you know", "sort of", etc.) and returns a
   count + list of unique fillers found. Useful for coaching yourself
   to listen more carefully and notice when the interviewer is
   thinking aloud vs asking a real question.

2. QuestionClassifier
   Classifies each question into one of five types using keyword rules
   (no LLM call, instant):
     - BEHAVIOURAL  ("tell me about a time", "describe a situation")
     - TECHNICAL    ("implement", "algorithm", "complexity", "debug")
     - SYSTEM_DESIGN ("design a system", "scale", "architecture")
     - CULTURE_FIT  ("why this company", "values", "team", "fit")
     - SALARY       ("compensation", "salary", "expect", "pay")
     - GENERAL      (catch-all)

   The classifier also returns a recommended brevity hint so the
   controller can auto-suggest a depth level without the user having
   to switch manually.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# 1. Filler Word Detector
# ---------------------------------------------------------------------------

# Each entry is (display_label, regex_pattern).
# Patterns are word-boundary anchored and case-insensitive.
_FILLERS: list[tuple[str, str]] = [
    ("um",          r"\bum+\b"),
    ("uh",          r"\buh+\b"),
    ("hmm",         r"\bhmm+\b"),
    ("like",        r"\blike\b"),
    ("you know",    r"\byou\s+know\b"),
    ("sort of",     r"\bsort\s+of\b"),
    ("kind of",     r"\bkind\s+of\b"),
    ("basically",   r"\bbasically\b"),
    ("literally",   r"\bliterally\b"),
    ("actually",    r"\bactually\b"),
    ("I mean",      r"\bI\s+mean\b"),
    ("right",       r"\bright\b"),
    ("okay so",     r"\bokay\s+so\b"),
    ("so",          r"\bso\b"),
    ("just",        r"\bjust\b"),
]

_FILLER_RES: list[tuple[str, re.Pattern]] = [
    (label, re.compile(pat, re.IGNORECASE))
    for label, pat in _FILLERS
]


@dataclass
class FillerReport:
    total_count: int                  # total filler occurrences
    unique_fillers: list[str]         # e.g. ["um", "like", "you know"]
    per_filler: dict[str, int]        # e.g. {"um": 3, "like": 5}
    confidence_score: int             # 0-100: 100 = no fillers (perfect clarity)
    label: str                        # human-readable e.g. "Clear ✓" / "3 fillers"


def detect_fillers(text: str) -> FillerReport:
    """Scan ``text`` for filler words and return a FillerReport."""
    per: dict[str, int] = {}
    for label, pat in _FILLER_RES:
        matches = pat.findall(text)
        if matches:
            per[label] = len(matches)
    total = sum(per.values())
    unique = list(per.keys())

    # Confidence score: starts at 100, docked by 5 per unique filler type
    # and 2 per extra occurrence beyond the first of each type.
    dock = 0
    for label, cnt in per.items():
        dock += 5 + (cnt - 1) * 2
    score = max(0, 100 - dock)

    if total == 0:
        label_str = "Clear"
    elif total == 1:
        label_str = f"1 filler ({unique[0]})"
    else:
        label_str = f"{total} fillers"

    return FillerReport(
        total_count=total,
        unique_fillers=unique,
        per_filler=per,
        confidence_score=score,
        label=label_str,
    )


# ---------------------------------------------------------------------------
# 2. Question Type Classifier
# ---------------------------------------------------------------------------

QuestionType = Literal[
    "BEHAVIOURAL", "TECHNICAL", "SYSTEM_DESIGN", "CULTURE_FIT", "SALARY", "GENERAL"
]

# (type, recommended_brevity, patterns)
# Patterns are matched against the lowercased question text.
_Q_RULES: list[tuple[QuestionType, str, list[str]]] = [
    ("SYSTEM_DESIGN", "deep", [
        r"design (a |an |the )?system",
        r"design (a |an |the )?(service|platform|api|database|pipeline|architecture)",
        r"how (would|do) you (scale|architect|build|design)",
        r"scalab",
        r"distributed",
        r"microservice",
        r"load balanc",
        r"high.availab",
        r"fault.toleran",
        r"caching strateg",
        r"system design",
    ]),
    ("TECHNICAL", "detailed", [
        r"\bimplement\b",
        r"\balgorithm\b",
        r"\bcomplexity\b",
        r"\bdata structure\b",
        r"\brecursion\b",
        r"\bdynamic programming\b",
        r"\bbig.?o\b",
        r"\bdebug\b",
        r"\boptimiz",
        r"\brefactor",
        r"\bcode (review|quality)\b",
        r"\bthread(ing|safe)\b",
        r"\bconcurren",
        r"\bdeadlock\b",
        r"\bmemory leak\b",
        r"\bsql\b",
        r"\bindex(ing)?\b",
        r"\bapi (design|versioning)\b",
        r"\brest(ful)?\b",
        r"\bci.?cd\b",
        r"\bdocker\b",
        r"\bkubernetes\b",
        r"\btest(ing)? (strateg|coverage|driven)\b",
        r"\bunit test\b",
        r"\bsecurity\b",
        r"\bencrypt",
        r"\bperforman",
        r"\blatency\b",
        r"\bthroughput\b",
        r"\bhow (does|do|would|did) you",
        r"\bexplain (how|why|what)\b",
        r"\bwhat is (the )?(difference|tradeoff|advantage|disadvantage)\b",
        r"\bcompare\b",
        r"\bwhy (did you|would you|do you) (choose|use|prefer|pick)\b",
    ]),
    ("BEHAVIOURAL", "concise", [
        r"tell me about a time",
        r"describe a (situation|time|moment|experience)",
        r"give me an example",
        r"have you ever",
        r"what (did|would) you do when",
        r"how (did|do) you handle",
        r"walk me through",
        r"biggest (challenge|mistake|failure|success|achievement|accomplishment)",
        r"most (difficult|challenging|proud|rewarding)",
        r"conflict (with|between)",
        r"disagreed? with",
        r"leadership",
        r"mentor",
        r"influence",
        r"prioriti(ze|zation|se)",
        r"under pressure",
        r"tight deadline",
        r"failed",
        r"learned from",
    ]),
    ("SALARY", "brief", [
        r"\bsalar(y|ies)\b",
        r"\bcompensation\b",
        r"\bpay\b",
        r"\bexpect(ation|ed|ing)?\b",
        r"\bpackage\b",
        r"\bbonus\b",
        r"\bequity\b",
        r"\bstock\b",
        r"\bbenefits?\b",
        r"\brate\b",
        r"\bhourly\b",
        r"\bannual\b",
    ]),
    ("CULTURE_FIT", "concise", [
        r"why (this company|us|our company|do you want to work)",
        r"why (do you want to|are you interested in) (join|work|leave)",
        r"what (do you know about|attracted you to)",
        r"our (values|mission|culture|team)",
        r"where do you see yourself",
        r"career (goal|plan|aspiration|path)",
        r"work.?life balance",
        r"remote",
        r"team (size|dynamic|culture|environment)",
        r"management style",
        r"what (are you looking for|do you want) in",
        r"strengths? and weakness",
        r"tell me about yourself",
        r"introduce yourself",
        r"walk me through your (background|resume|experience)",
    ]),
]

_Q_COMPILED: list[tuple[QuestionType, str, list[re.Pattern]]] = [
    (qtype, brevity, [re.compile(p, re.IGNORECASE) for p in pats])
    for qtype, brevity, pats in _Q_RULES
]


@dataclass
class ClassificationResult:
    question_type: QuestionType
    recommended_brevity: str          # "brief" / "concise" / "detailed" / "deep"
    confidence: float                 # 0.0-1.0 (rule match strength)
    display_label: str                # e.g. "⚙ Technical" / "🏗 System Design"
    depth_hint: str                   # short coaching note shown in overlay


_TYPE_LABELS: dict[QuestionType, str] = {
    "BEHAVIOURAL":   "★ Behavioural",
    "TECHNICAL":     "⚙ Technical",
    "SYSTEM_DESIGN": "🏗 System Design",
    "CULTURE_FIT":   "◎ Culture Fit",
    "SALARY":        "$ Salary",
    "GENERAL":       "· General",
}

_TYPE_HINTS: dict[QuestionType, str] = {
    "BEHAVIOURAL":   "Use STAR: Situation → Task → Action → Result",
    "TECHNICAL":     "State approach, complexity, real-world implication",
    "SYSTEM_DESIGN": "Constraints → Options → Trade-offs → Decision",
    "CULTURE_FIT":   "Be specific, authentic, connect to their mission",
    "SALARY":        "Give a range, anchor high, mention total comp",
    "GENERAL":       "",
}


def classify_question(text: str) -> ClassificationResult:
    """Classify ``text`` into a QuestionType using keyword rules."""
    scores: dict[QuestionType, int] = {}
    for qtype, _brevity, patterns in _Q_COMPILED:
        hit = sum(1 for p in patterns if p.search(text))
        if hit:
            scores[qtype] = hit

    if not scores:
        return ClassificationResult(
            question_type="GENERAL",
            recommended_brevity="concise",
            confidence=0.0,
            display_label=_TYPE_LABELS["GENERAL"],
            depth_hint=_TYPE_HINTS["GENERAL"],
        )

    best_type = max(scores, key=lambda t: scores[t])
    best_hits = scores[best_type]
    # Find the brevity for the best type
    best_brevity = next(
        brevity for qtype, brevity, _ in _Q_COMPILED if qtype == best_type
    )
    # Confidence: 1 hit = 0.6, 2 = 0.8, 3+ = 1.0
    confidence = min(1.0, 0.5 + best_hits * 0.2)

    return ClassificationResult(
        question_type=best_type,
        recommended_brevity=best_brevity,
        confidence=confidence,
        display_label=_TYPE_LABELS[best_type],
        depth_hint=_TYPE_HINTS[best_type],
    )
