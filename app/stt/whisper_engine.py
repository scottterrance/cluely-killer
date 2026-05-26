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
"""
from __future__ import annotations

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
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.samplerate = 16000

        # Prefer the flat bundled directory if it has a model.bin in it.
        # Falling back to the model_size string lets dev mode still work
        # before setup-model.ps1 has been run (faster-whisper will try
        # an HF download in that case).
        bundled = _bundled_model_path(model_size)
        model_bin = bundled / "model.bin"
        if model_bin.exists():
            print(
                f"[whisper] loading from bundled local path: {bundled} "
                f"(model.bin {model_bin.stat().st_size // 1024 // 1024} MB)",
                flush=True,
            )
            target = str(bundled)
        else:
            print(
                f"[whisper] WARNING: {bundled} has no model.bin; "
                f"falling back to HF download for {model_size!r}.",
                flush=True,
            )
            target = model_size

        self.model = WhisperModel(target, device=device, compute_type=compute_type)

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

        segments, _info = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
