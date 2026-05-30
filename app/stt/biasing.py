"""Build a Whisper biasing vocabulary from the user's own context.

Most transcription errors in a live interview are on *proper nouns and
technical terms* - your name, the company, frameworks, acronyms, product
names. Whisper lets us bias decoding toward an expected vocabulary via
its ``initial_prompt`` (and, in faster-whisper >= 1.0, ``hotwords``).

Feeding a glossary of the candidate's own terms measurably lowers word
error rate on exactly those tokens (reported ~5% average WER improvement,
and larger relative gains on proper-noun-dense audio). We already have
the resume / JD / about-me text sitting in Settings, so we extract a
keyword list from it for free - no new dependencies, no network.

The heuristic is deliberately simple and conservative (we'd rather miss
a term than poison the prompt with junk):

  * CamelCase / mixed-case tokens   -> PyTorch, GraphQL, TensorFlow, K8s
  * ALL-CAPS acronyms (2-6 letters) -> AWS, REST, SQL, CI, CD, GPU
  * alphanumeric tokens             -> S3, EC2, GPT, OAuth2, H100
  * Capitalized multi-word spans    -> Amazon Web Services, Goldman Sachs
  * Other Capitalized words         -> Kubernetes, Django, Kafka

Common English words that merely happen to be capitalized (sentence
starts, "I", "The", month names, etc.) are filtered out so they don't
crowd out the genuinely useful terms within Whisper's small prompt
budget (~224 tokens; we cap well under that).
"""
from __future__ import annotations

import re

# Capitalized-but-useless words: sentence starters, pronouns, filler that
# routinely appears capitalized in resumes/JDs and would waste prompt budget.
_STOPWORDS = {
    "I", "A", "An", "The", "My", "We", "Our", "You", "Your", "It", "He",
    "She", "They", "This", "That", "These", "Those", "And", "But", "Or",
    "For", "With", "From", "Into", "Over", "As", "At", "By", "In", "On",
    "Of", "To", "Is", "Are", "Was", "Were", "Be", "Been", "Being", "Have",
    "Has", "Had", "Do", "Does", "Did", "Will", "Would", "Can", "Could",
    "Should", "May", "Might", "Must", "Not", "No", "Yes", "If", "Then",
    "Else", "When", "While", "Where", "Who", "What", "Why", "How",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Responsible", "Experience", "Experienced", "Skills", "Summary",
    "Education", "Worked", "Working", "Led", "Built", "Developed",
    "Managed", "Designed", "Created", "Implemented", "Company", "Team",
    "Role", "Job", "Position", "Years", "Year", "Present", "Current",
}

# CamelCase / mixed internal caps, e.g. PyTorch, GraphQL, JavaScript.
_MIXED_CASE = re.compile(r"\b[A-Za-z]*[a-z][A-Z][A-Za-z]*\b")
# ALL-CAPS acronyms 2-6 chars, e.g. AWS, REST, SQL, GPU, CI.
_ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")
# Alphanumeric tokens, e.g. S3, EC2, GPT, OAuth2, H100, Llama3.
_ALNUM = re.compile(r"\b[A-Za-z]+\d[A-Za-z0-9]*\b|\b[A-Z][a-z]*\d+\b")
# Capitalized multi-word spans, e.g. "Amazon Web Services".
_CAP_SPAN = re.compile(r"\b(?:[A-Z][a-zA-Z]+(?:\s+|$)){2,4}")
# Single Capitalized word, e.g. Kubernetes, Django, Kafka.
_CAP_WORD = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")


def _is_clean_single(token: str) -> bool:
    """Reject mangled tokens that pollute the glossary and make Whisper
    hallucinate (the 'DevOpsunmatched' / 'VisualizationEducation' problem
    from PDF text where words get glued together with no space).

    Rules for a SINGLE word token:
      * length 2..14 (real tech terms are short; >14 is almost always
        two words glued together).
      * a CamelCase token may have AT MOST one lowercase->uppercase
        transition. 'PyTorch'/'GraphQL'/'TensorFlow' have exactly one;
        'DevOpsunmatched' technically has one too, so we ALSO cap the
        trailing lowercase run: if the part after the last capital is a
        long word (>=7 letters) the token is almost certainly two words
        glued, so drop it. That keeps 'TensorFlow'(flow=4),
        'SharePoint'(point=5) and drops 'DevOpsunmatched'(unmatched=9).
    """
    if not (2 <= len(token) <= 14):
        return False
    # Count internal lowercase->uppercase transitions.
    transitions = sum(
        1 for a, b in zip(token, token[1:]) if a.islower() and b.isupper()
    )
    if transitions >= 2:
        return False
    if transitions == 1:
        # Trailing lowercase run after the final uppercase letter.
        last_upper = max(i for i, c in enumerate(token) if c.isupper())
        tail = token[last_upper + 1:]
        if len(tail) >= 7:  # 'unmatched', 'education' -> glued word
            return False
    return True


def extract_keywords(text: str, limit: int = 32) -> list[str]:
    """Pull a deduped, order-preserving keyword list out of free text."""
    if not text or not text.strip():
        return []

    found: list[str] = []
    seen: set[str] = set()

    def _add(token: str, *, is_span: bool = False) -> None:
        token = token.strip()
        if not token:
            return
        # Drop single capitalized stopwords, but keep them inside spans.
        if token in _STOPWORDS:
            return
        if is_span:
            # Multi-word span: each word must itself be sane, and the
            # whole phrase not absurdly long.
            words = token.split()
            if not (2 <= len(words) <= 4) or len(token) > 40:
                return
            if any(len(w) > 18 for w in words):
                return
        else:
            if not _is_clean_single(token):
                return
        key = token.lower()
        if key in seen:
            return
        seen.add(key)
        found.append(token)

    # High-signal patterns first so they win the limited budget.
    for m in _MIXED_CASE.findall(text):
        _add(m)
    for m in _ALNUM.findall(text):
        _add(m)
    for m in _ACRONYM.findall(text):
        _add(m)
    for m in _CAP_SPAN.findall(text):
        # A span like "Amazon Web Services" - keep the whole phrase, but
        # skip if every word is a stopword.
        phrase = " ".join(w for w in m.split())
        words = phrase.split()
        if words and not all(w in _STOPWORDS for w in words):
            _add(phrase, is_span=True)
    for m in _CAP_WORD.findall(text):
        _add(m)

    return found[:limit]


def build_vocab_from_context(
    about: str = "",
    resume: str = "",
    job_desc: str = "",
    custom: str = "",
    limit: int = 32,
) -> list[str]:
    """Combine all four context fields into one keyword list.

    JD first (interviewer phrasing tends to mirror the job posting),
    then resume, then about-me, then any custom prompt. Order matters
    because we truncate at ``limit`` - the earliest sources win the
    budget.
    """
    combined = "\n".join(
        part for part in (job_desc, resume, about, custom) if part
    )
    return extract_keywords(combined, limit=limit)


def build_initial_prompt(keywords: list[str]) -> str | None:
    """Render keywords as a Whisper ``initial_prompt`` glossary string.

    Returns None if there's nothing to bias with, so callers can pass
    it straight through to faster-whisper (which treats None as "no
    prompt").

    Kept SHORT on purpose: a long initial_prompt is a well-known trigger
    for Whisper's repetition/hallucination loops (the 'NET STOP CA USA'
    garbage). We cap to the first ~18 terms and ~240 chars.
    """
    if not keywords:
        return None
    glossary = ", ".join(keywords[:18])
    prompt = f"Glossary: {glossary}."
    return prompt[:240]
