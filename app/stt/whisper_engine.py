"""faster-whisper wrapper.

Built-in Silero VAD handles low-volume / whispered speech; beam_size=1
keeps latency low. English-only is enforced for speed.

This build loads the model from a flat local directory bundled next
to the .exe (./models/whisper-<size>/) instead of going through
HuggingFace's cache+revision machinery. Two reasons:

1. PyInstaller has been observed to mangle the HF cache structure
   (refs/main, blobs/, snapshots/<sha>/) when bundling, breaking
   offline lookups in the resulting .exe even when the source files
   are correct.
2. faster-whisper accepts a directory path as its first argument and
   loads model.bin / config.json / tokenizer.json directly. No HF
   network calls at all - not even the metadata revision check.

setup-model.ps1 prepares ./models/whisper-<size>/ as a flat directory.

ACCURACY BIASING
----------------
``set_bias()`` feeds a glossary of the candidate's own terms (extracted
from resume / JD / about-me) into Whisper's ``initial_prompt`` and, when
the installed faster-whisper supports it, ``hotwords``. This lowers word
error rate on the proper nouns and tech jargon that dominate interview
mistakes. See app/stt/biasing.py.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel


def _bundled_model_path(model_size: str) -> Path:
    """Resolve <app_dir>/models/whisper-<size>/.

    app_dir is the directory of cluely-killer.exe when frozen, otherwise
    the project root in dev. run.py exports this via CLUELY_APP_DIR.
    """
    app_dir = os.environ.get("CLUELY_APP_DIR")
    if app_dir:
        base = Path(app_dir)
    elif getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        # dev: this file is at app/stt/whisper_engine.py -> repo root is parents[2]
        base = Path(__file__).resolve().parents[2]
    return base / "models" / f"whisper-{model_size}"


class WhisperEngine:
    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cpu",
        compute_type: str = "int8",
        allow_auto_download: bool = False,
    ):
        self.samplerate = 16000
        self._initial_prompt: str | None = None
        self._hotwords: str | None = None

        # Prefer the flat bundled directory if it has a model.bin in it.
        bundled = _bundled_model_path(model_size)
        model_bin = bundled / "model.bin"
        if model_bin.exists():
            print(
                f"[whisper] loading from bundled local path: {bundled} "
                f"(model.bin {model_bin.stat().st_size // 1024 // 1024} MB)",
                flush=True,
            )
            target = str(bundled)
        elif allow_auto_download:
            # DEV-ONLY one-time convenience: let faster-whisper pull the
            # model from HuggingFace. NEVER enabled in the shipped .exe
            # (whisper_allow_auto_download defaults to False), so an
            # end user can never trigger a multi-GB download mid-interview.
            print(
                f"[whisper] WARNING: {bundled} has no model.bin and "
                f"allow_auto_download=True; faster-whisper will download "
                f"{model_size!r} from HuggingFace (dev only).",
                flush=True,
            )
            target = model_size
        else:
            # Shipped path: the model MUST be present locally. Fail loud
            # and actionable instead of silently attempting a download
            # (which would also be blocked by HF_HUB_OFFLINE=1 in run.py).
            raise RuntimeError(
                f"Whisper model not found at:\n    {bundled}\n\n"
                f"The app is configured NOT to download models at runtime. "
                f"Place the '{model_size}' model files (model.bin, config.json, "
                f"tokenizer.json, vocabulary.txt) in that folder, then relaunch.\n"
                f"See README / setup-model.ps1 for how to download it once."
            )

        self.model = WhisperModel(target, device=device, compute_type=compute_type)

        # faster-whisper >= 1.0 added a `hotwords` kwarg to transcribe();
        # detect it once so we degrade gracefully on older installs.
        try:
            params = inspect.signature(self.model.transcribe).parameters
            self._supports_hotwords = "hotwords" in params
        except (ValueError, TypeError):
            self._supports_hotwords = False

    # ------------------------------------------------------------------
    def set_bias(self, keywords: list[str]) -> None:
        """Install a biasing vocabulary (call at startup + on settings save).

        Safe to call repeatedly; the next transcribe() picks up the new
        terms. Passing an empty list clears biasing.
        """
        from .biasing import build_initial_prompt

        self._initial_prompt = build_initial_prompt(keywords)
        # hotwords wants a plain space/comma-joined phrase string.
        self._hotwords = ", ".join(keywords) if keywords else None
        n = len(keywords)
        print(
            f"[whisper] bias vocabulary set: {n} term(s)"
            + (f" e.g. {keywords[:6]}" if n else ""),
            flush=True,
        )

    # ------------------------------------------------------------------
    def transcribe(self, audio: np.ndarray) -> str:
        if audio is None or audio.size < self.samplerate * 0.5:
            return ""
        # Whisper expects float32 in [-1, 1].
        audio = audio.astype(np.float32, copy=False)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak == 0.0:
            return ""
        # Light normalization helps with very quiet voices without distorting loud ones.
        if peak < 0.2:
            audio = audio / peak * 0.6

        kwargs = dict(
            language="en",
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
            condition_on_previous_text=False,
            initial_prompt=self._initial_prompt,
        )
        if self._supports_hotwords and self._hotwords:
            kwargs["hotwords"] = self._hotwords

        segments, _info = self.model.transcribe(audio, **kwargs)
        return " ".join(seg.text.strip() for seg in segments).strip()
