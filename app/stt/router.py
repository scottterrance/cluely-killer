"""Selects the active STT engine from Settings, with automatic fallback.

Implements the same STTEngine interface the Controller already uses
(``samplerate`` / ``transcribe`` / ``set_bias``), so it's a drop-in for
the old direct WhisperEngine.

Backend choice (``settings.stt_backend``):
  "cloud" -> Groq Whisper large-v3-turbo (fast, needs net + Groq quota)
  "local" -> bundled faster-whisper (offline, uses CPU)

Engines are built LAZILY and cached:
  * The local engine raises if the model folder is missing - we must
    not construct it eagerly when the user is cloud-only and never
    downloaded the model.
  * The cloud engine raises if there's no Groq key.

Fallback: if the primary backend raises during transcribe (e.g. Groq
returns 429 "rate limit / out of tokens", or the network is down), the
router transparently tries the OTHER backend once. This is what keeps
the app working the moment your Groq free-tier quota runs out - no
settings change needed mid-interview.
"""
from __future__ import annotations

import traceback

import numpy as np

from ..config import Settings
from .base import STTEngine


class STTRouter(STTEngine):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.samplerate = 16000
        self._local: STTEngine | None = None
        self._cloud: STTEngine | None = None
        self._vocab: list[str] = []
        # Tracks which backend actually served the last transcribe(), for
        # status/logging. Not relied on for control flow.
        self.last_used: str = ""

    # -- lazy builders -------------------------------------------------
    def _get_local(self) -> STTEngine:
        if self._local is None:
            from .whisper_engine import WhisperEngine

            eng = WhisperEngine(
                model_size=self.settings.whisper_model,
                device=self.settings.whisper_device,
                compute_type=self.settings.whisper_compute,
                allow_auto_download=self.settings.whisper_allow_auto_download,
            )
            eng.set_bias(self._vocab)
            self._local = eng
        return self._local

    def _get_cloud(self) -> STTEngine:
        if self._cloud is None:
            from .groq_stt import GroqSTTEngine

            eng = GroqSTTEngine(
                api_key=self.settings.groq_api_key,
                model=self.settings.groq_stt_model,
                base_url=self.settings.groq_base_url,
            )
            eng.set_bias(self._vocab)
            self._cloud = eng
        return self._cloud

    # -- ordering ------------------------------------------------------
    def _order(self) -> list[str]:
        """Primary first, then the other as fallback."""
        if self.settings.stt_backend == "local":
            return ["local", "cloud"]
        return ["cloud", "local"]

    def _engine(self, name: str) -> STTEngine:
        return self._get_cloud() if name == "cloud" else self._get_local()

    # -- STTEngine API -------------------------------------------------
    def set_bias(self, keywords: list[str]) -> None:
        self._vocab = list(keywords or [])
        # Push to any engines already built; lazily-built ones pick it
        # up at construction time.
        if self._local is not None:
            self._local.set_bias(self._vocab)
        if self._cloud is not None:
            self._cloud.set_bias(self._vocab)

    def invalidate(self) -> None:
        """Drop cached engines so the next use rebuilds from current
        Settings. Call after the user edits keys / models / backend in
        the Settings dialog. The local model is NOT dropped (reloading
        the ~1.6 GB model is expensive and its config rarely changes);
        only the cheap-to-rebuild cloud client is reset so a new Groq
        key/model takes effect immediately.
        """
        self._cloud = None

    def transcribe(self, audio: np.ndarray) -> str:
        order = self._order()
        first_err: Exception | None = None
        for i, name in enumerate(order):
            try:
                engine = self._engine(name)
            except Exception as e:
                # Couldn't even build this backend (missing model / key).
                # Remember the first failure, try the next.
                if first_err is None:
                    first_err = e
                print(f"[stt-router] backend {name!r} unavailable: {e}", flush=True)
                continue
            try:
                text = engine.transcribe(audio)
                self.last_used = name
                if i > 0:
                    print(f"[stt-router] fell back to {name!r} backend.", flush=True)
                return text
            except Exception as e:
                if first_err is None:
                    first_err = e
                traceback.print_exc()
                print(
                    f"[stt-router] {name!r} transcribe failed: {e}; "
                    f"trying fallback..." if i == 0 else f"[stt-router] {name!r} also failed.",
                    flush=True,
                )
                continue
        # Both backends failed.
        if first_err is not None:
            raise first_err
        return ""
