"""Dependency-free retrieval of the most relevant resume chunks.

Roadmap #9: instead of stuffing the candidate's ENTIRE resume into every
prompt, we embed it once-ish (here: tokenize + score on the fly, no
external model) and inject only the 2-3 chunks most relevant to the
current question. Two wins:

  * Accuracy: the model answers with the candidate's RELEVANT specifics
    (the right project / metric / tech) instead of drowning in the whole
    CV and giving generic replies.
  * Speed (minor): a shorter prompt has slightly lower time-to-first-
    token. The big speed levers are elsewhere; this is mostly quality.

Why no embeddings model? It would add a heavy dependency + load time and
run on the same CPU we are trying to keep free for Whisper. For
resume-sized text (a few KB) a classic TF-IDF-ish lexical score over
paragraph chunks is fast, deterministic, offline, and plenty good at
"which paragraph mentions what this question is about".

IMPORTANT - prefix caching: RAG output is VOLATILE (changes per
question), so the caller must place it in the prompt's volatile TAIL,
never in the cached prefix. When the resume is small (< rag_min_chars)
the caller should skip RAG entirely and keep the full resume in the
stable prefix - that keeps DeepSeek's prefix cache warm and costs
nothing since a small resume is cheap to send whole.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9+.#/-]*")

# Extremely common words carry no retrieval signal; ignore them so the
# score reflects meaningful term overlap (tech, nouns, verbs).
_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "as", "is", "are", "was", "were", "be", "been", "being", "i", "my",
    "me", "we", "our", "you", "your", "it", "this", "that", "these", "those",
    "he", "she", "they", "them", "do", "does", "did", "have", "has", "had",
    "will", "would", "can", "could", "should", "may", "might", "must", "not",
    "from", "by", "about", "into", "over", "than", "then", "so", "if", "what",
    "how", "why", "when", "where", "who", "which", "tell", "me", "us", "give",
    "your", "yourself", "describe", "explain",
}


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


@dataclass
class Chunk:
    text: str
    tokens: list[str]


def _chunk_resume(resume: str) -> list[Chunk]:
    """Split a resume into paragraph-ish chunks.

    Splits on blank lines first; if the resume is one big blob (common
    from PDF extraction), falls back to grouping consecutive lines into
    ~line-windows so we still get multiple scorable units.
    """
    raw = (resume or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if len(parts) < 2:
        # One blob -> group every ~3 non-empty lines into a chunk.
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        parts = []
        for i in range(0, len(lines), 3):
            parts.append(" ".join(lines[i : i + 3]))
    chunks = []
    for p in parts:
        toks = [t for t in _tokens(p) if t not in _STOP]
        if toks:
            chunks.append(Chunk(text=p, tokens=toks))
    return chunks


def retrieve_resume_snippets(
    resume: str,
    query: str,
    top_k: int = 3,
    max_chars: int = 700,
) -> str:
    """Return the resume chunks most relevant to ``query``, joined.

    Scoring: for each chunk, sum over query terms of
        (term frequency in chunk) * idf(term)
    where idf down-weights terms that appear in many chunks (so a word
    that's everywhere in the resume doesn't dominate). Deterministic and
    offline. Returns '' if there's nothing useful.
    """
    chunks = _chunk_resume(resume)
    if not chunks:
        return ""
    q_terms = [t for t in _tokens(query) if t not in _STOP]
    if not q_terms:
        # No usable query terms (e.g. "tell me about yourself") -> just
        # return the FIRST chunks, which on a resume is the summary/top.
        picked = chunks[:top_k]
        return _join_capped([c.text for c in picked], max_chars)

    n_docs = len(chunks)
    # Document frequency per term.
    df: dict[str, int] = {}
    for c in chunks:
        for t in set(c.tokens):
            df[t] = df.get(t, 0) + 1

    def idf(term: str) -> float:
        # Smoothed idf; terms not in the resume get a neutral-ish weight.
        d = df.get(term, 0)
        return math.log((n_docs + 1) / (d + 1)) + 1.0

    scored: list[tuple[float, int, Chunk]] = []
    q_set = set(q_terms)
    for idx, c in enumerate(chunks):
        tf: dict[str, int] = {}
        for t in c.tokens:
            if t in q_set:
                tf[t] = tf.get(t, 0) + 1
        if not tf:
            continue
        score = sum(freq * idf(term) for term, freq in tf.items())
        # Normalize a little by chunk length so a giant chunk doesn't win
        # just by being long.
        score /= math.sqrt(len(c.tokens) + 1)
        scored.append((score, idx, c))

    if not scored:
        # Question shares no terms with the resume - fall back to the top
        # of the resume (summary) so we still ground the answer.
        return _join_capped([c.text for c in chunks[:top_k]], max_chars)

    # Highest score first; tie-break by original order (earlier = more
    # likely the summary/headline).
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked = [c.text for _s, _i, c in scored[:top_k]]
    return _join_capped(picked, max_chars)


def _join_capped(parts: list[str], max_chars: int) -> str:
    out: list[str] = []
    total = 0
    for p in parts:
        if total + len(p) > max_chars and out:
            break
        out.append(p)
        total += len(p)
    joined = "\n".join(out).strip()
    return joined[:max_chars]
