"""Semantic cache for repeated interview questions (roadmap #8).

~30% of interview questions recur across interviews and within one
("Tell me about yourself", "Why this company", "Greatest weakness",
"Where do you see yourself in 5 years"). Re-calling the LLM for these
wastes the one thing we're fighting for - time. This cache stores the
answers we've already generated and, when a NEW question is
semantically close to a stored one, serves the saved answer instantly
(zero LLM latency, and consistent phrasing the candidate has effectively
rehearsed).

Matching - why TF-IDF cosine, not an embeddings model:
  An embeddings model would be more semantically precise but adds a
  heavy dependency + load time and runs on the same CPU/GPU we want free
  for Whisper. For short interview questions, a classic TF-IDF cosine
  over the cached questions is fast, deterministic, offline, and good
  enough: "tell me about yourself" vs "so, tell me a bit about yourself"
  score very high; unrelated questions score near zero. A threshold
  gates hits so only genuinely-similar questions reuse an answer.

Context fingerprint - correctness guard:
  A cached answer is only valid for the SAME candidate context. If the
  resume / JD / about-me / custom prompt / brevity changes (e.g. the
  user switches personas), the old answers no longer apply. We stamp
  every entry with a fingerprint of that context and ignore entries
  whose fingerprint doesn't match the current one. set_fingerprint()
  is called at startup and whenever settings are saved.

Persistence:
  Stored as JSON at ~/.cluely_killer/semantic_cache.json so good answers
  survive restarts. Bounded to ``max_entries`` (LRU-ish by recency).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from pathlib import Path

CACHE_DIR = Path.home() / ".cluely_killer"
CACHE_FILE = CACHE_DIR / "semantic_cache.json"

_WORD = re.compile(r"[a-z0-9']+")

# Question words carry little discriminative signal across interview
# questions (almost all start with one), so we down-weight common filler
# but KEEP topic words. We do not stopword aggressively - short questions
# have few tokens to begin with.
_FILLER = {
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "and", "or",
    "is", "are", "do", "you", "your", "me", "i", "so", "just", "a", "bit",
    "can", "could", "would", "please", "us", "about", "tell", "give",
    "that", "this", "it", "with", "as", "be", "have", "has",
}


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall((text or "").lower())]


def _vec(text: str) -> dict[str, float]:
    """Term-frequency vector (filler-down-weighted, length-normalized)."""
    toks = _tokens(text)
    if not toks:
        return {}
    tf: dict[str, float] = {}
    for t in toks:
        w = 0.3 if t in _FILLER else 1.0
        tf[t] = tf.get(t, 0.0) + w
    # L2 normalize so cosine is just the dot product.
    norm = math.sqrt(sum(v * v for v in tf.values())) or 1.0
    return {k: v / norm for k, v in tf.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    # Iterate the smaller dict for speed.
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def context_fingerprint(*parts: str) -> str:
    """Stable short hash of the candidate-context strings."""
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").strip().encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


class SemanticCache:
    def __init__(self, threshold: float = 0.82, max_entries: int = 200):
        self.threshold = threshold
        self.max_entries = max_entries
        self._fingerprint = ""
        self._lock = threading.Lock()
        # Each entry: {q, vec(dict), answer, fp, ts}
        self._entries: list[dict] = []
        self._load()

    # -- context fingerprint -------------------------------------------
    def set_fingerprint(self, fp: str) -> None:
        with self._lock:
            self._fingerprint = fp or ""

    # -- lookup / store ------------------------------------------------
    def lookup(self, question: str) -> tuple[str, float] | None:
        """Return (answer, score) for the best match >= threshold whose
        fingerprint matches the current context, else None.
        """
        qv = _vec(question)
        if not qv:
            return None
        best_ans = None
        best_score = 0.0
        with self._lock:
            fp = self._fingerprint
            for e in self._entries:
                if fp and e.get("fp") and e["fp"] != fp:
                    continue
                score = _cosine(qv, e["vec"])
                if score > best_score:
                    best_score = score
                    best_ans = e["answer"]
            # Touch recency of the matched entry so it survives eviction.
            if best_ans is not None and best_score >= self.threshold:
                for e in self._entries:
                    if e["answer"] is best_ans:
                        e["ts"] = time.time()
                        break
        if best_ans is not None and best_score >= self.threshold:
            return best_ans, best_score
        return None

    def add(self, question: str, answer: str) -> None:
        """Store a successful Q->A under the current fingerprint.

        If a near-duplicate question already exists for this context, we
        UPDATE its answer (keep the cache from filling with paraphrases).
        """
        q = (question or "").strip()
        a = (answer or "").strip()
        if not q or not a:
            return
        qv = _vec(q)
        if not qv:
            return
        with self._lock:
            fp = self._fingerprint
            # Update an existing very-close entry instead of duplicating.
            for e in self._entries:
                if e.get("fp") == fp and _cosine(qv, e["vec"]) >= 0.95:
                    e["answer"] = a
                    e["q"] = q
                    e["ts"] = time.time()
                    self._save_locked()
                    return
            self._entries.append(
                {"q": q, "vec": qv, "answer": a, "fp": fp, "ts": time.time()}
            )
            # Evict oldest beyond the cap.
            if len(self._entries) > self.max_entries:
                self._entries.sort(key=lambda e: e.get("ts", 0.0))
                self._entries = self._entries[-self.max_entries:]
            self._save_locked()

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            self._save_locked()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # -- persistence ---------------------------------------------------
    def _load(self) -> None:
        try:
            if CACHE_FILE.exists():
                data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                entries = data.get("entries", []) if isinstance(data, dict) else []
                clean: list[dict] = []
                for e in entries:
                    q = e.get("q"); a = e.get("answer")
                    if not q or not a:
                        continue
                    # Recompute the vector (don't trust persisted floats /
                    # let the weighting evolve with code changes).
                    clean.append(
                        {"q": q, "vec": _vec(q), "answer": a,
                         "fp": e.get("fp", ""), "ts": e.get("ts", 0.0)}
                    )
                self._entries = clean[-self.max_entries:]
                print(f"[semcache] loaded {len(self._entries)} cached answer(s)", flush=True)
        except Exception as e:
            print(f"[semcache] load failed ({e}); starting empty", flush=True)
            self._entries = []

    def _save_locked(self) -> None:
        # Caller holds self._lock. Persist without the (recomputable) vec.
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            payload = {
                "entries": [
                    {"q": e["q"], "answer": e["answer"], "fp": e.get("fp", ""),
                     "ts": e.get("ts", 0.0)}
                    for e in self._entries
                ]
            }
            CACHE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[semcache] save failed ({e})", flush=True)
