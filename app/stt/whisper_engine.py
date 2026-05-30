"""faster-whisper wrapper (local, offline).

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

SPEED
-----
On CPU, faster-whisper latency is roughly LINEAR in audio length. The
two biggest local-STT speedups are therefore:
  * ``cpu_threads`` - use all physical cores for the CTranslate2 backend.
  * Transcribing only the relevant audio. ``transcribe(audio,
    isolate_last=True)`` isolates just the interviewer's last utterance
    (see app/stt/audio_proc.py), cutting on-press transcription time 2-4x.

ACCURACY BIASING
----------------
``set_bias()`` feeds a glossary of the candidate's own terms (resume /
JD / about-me) into Whisper's ``initial_prompt`` and, when supported,
``hotwords`` - lowering word error on proper nouns / jargon.
"""
from __future__ import annotations

import inspect
import os
import sys
import time
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from .audio_proc import isolate_last_utterance


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


def _default_cpu_threads() -> int:
    """Use all physical-ish cores, capped to keep one free for the UI /
    audio capture threads. 0 lets CTranslate2 decide; we pick an explicit
    value so a busy machine still dedicates real parallelism to STT.
    """
    n = os.cpu_count() or 4
    # Leave 1 core for Qt + the WASAPI capture thread; floor at 2.
    return max(2, n - 1)


def _cuda_available() -> bool:
    """Best-effort check for a usable CUDA GPU for CTranslate2.

    CTranslate2 ships its own CUDA runtime detection. We ask it directly
    so we don't depend on torch/nvidia-smi being importable. Any failure
    (no CUDA build, no driver, no device) returns False so we fall back
    to CPU cleanly.
    """
    try:
        import ctranslate2  # noqa

        count = ctranslate2.get_cuda_device_count()
        return int(count) > 0
    except Exception as e:
        print(f"[whisper] CUDA check: not available ({e})", flush=True)
        return False


def _resolve_device_and_compute(device: str, compute_type: str) -> tuple[str, str]:
    """Map the user's device/compute settings to concrete CTranslate2 args.

    device:
      "auto" -> "cuda" if a CUDA GPU is detected, else "cpu"
      "cuda"/"gpu" -> force CUDA (will error at model-build if unavailable;
                      caller handles the fallback)
      "cpu" -> force CPU

    compute_type:
      "auto" -> "float16" on GPU (fast + accurate on tensor cores),
                "int8" on CPU (fast + small)
      anything else -> used verbatim (e.g. "int8_float16", "float32")
    """
    dev = (device or "auto").strip().lower()
    if dev in ("gpu", "cuda"):
        resolved_dev = "cuda"
    elif dev == "cpu":
        resolved_dev = "cpu"
    else:  # auto
        resolved_dev = "cuda" if _cuda_available() else "cpu"

    comp = (compute_type or "auto").strip().lower()
    if comp == "auto":
        resolved_comp = "float16" if resolved_dev == "cuda" else "int8"
    else:
        resolved_comp = comp
    return resolved_dev, resolved_comp


class WhisperEngine:
    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "auto",
        allow_auto_download: bool = False,
        cpu_threads: int = 0,
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

        threads = cpu_threads if cpu_threads and cpu_threads > 0 else _default_cpu_threads()
        resolved_dev, resolved_comp = _resolve_device_and_compute(device, compute_type)

        # Build the model. On GPU (cuda) cpu_threads is irrelevant. If a
        # GPU build fails for ANY reason (missing CUDA/cuDNN DLLs, OOM,
        # driver mismatch), fall back to CPU/int8 so the app still works
        # instead of crashing - the user just doesn't get the speedup.
        def _build(dev: str, comp: str) -> "WhisperModel":
            print(
                f"[whisper] building model on device={dev} compute={comp}"
                + (f" cpu_threads={threads}" if dev == "cpu" else ""),
                flush=True,
            )
            kwargs = dict(device=dev, compute_type=comp)
            if dev == "cpu":
                kwargs["cpu_threads"] = threads
            return WhisperModel(target, **kwargs)

        try:
            self.model = _build(resolved_dev, resolved_comp)
            self.device = resolved_dev
            self.compute_type = resolved_comp
        except Exception as e:
            if resolved_dev == "cuda":
                print(
                    f"[whisper] GPU init failed ({type(e).__name__}: {e}); "
                    f"falling back to CPU/int8. For GPU you need the NVIDIA "
                    f"CUDA + cuDNN runtime DLLs on PATH (see the GPU setup note).",
                    flush=True,
                )
                self.model = _build("cpu", "int8")
                self.device = "cpu"
                self.compute_type = "int8"
            else:
                raise

        print(
            f"[whisper] READY on {self.device.upper()} "
            f"(compute={self.compute_type}). "
            + ("GPU active - expect sub-second transcription." if self.device == "cuda"
               else "CPU mode - transcription speed scales with audio length."),
            flush=True,
        )

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
        # hotwords wants a plain comma-joined phrase string.
        self._hotwords = ", ".join(keywords) if keywords else None
        n = len(keywords)
        print(
            f"[whisper] bias vocabulary set: {n} term(s)"
            + (f" e.g. {keywords[:6]}" if n else ""),
            flush=True,
        )

    # ------------------------------------------------------------------
    def transcribe(self, audio: np.ndarray, isolate_last: bool = False) -> str:
        if audio is None or audio.size < self.samplerate * 0.5:
            return ""
        audio = audio.astype(np.float32, copy=False)
        # On the press path, isolate just the interviewer's last question
        # so Whisper transcribes ~5-8s instead of the whole window. This
        # is the biggest local-STT latency win (latency ~ audio length).
        if isolate_last:
            audio = isolate_last_utterance(audio, self.samplerate, max_seconds=15.0)

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

        # Timing receipt: this is the single most important number for
        # diagnosing local-STT latency. faster-whisper's transcribe()
        # returns a generator; the actual compute happens as we iterate
        # the segments, so we time the FULL drain, not just the call.
        audio_secs = audio.size / self.samplerate
        t0 = time.monotonic()
        segments, _info = self.model.transcribe(audio, **kwargs)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        elapsed = time.monotonic() - t0
        rtf = (elapsed / audio_secs) if audio_secs > 0 else 0.0
        print(
            f"[whisper] transcribed {audio_secs:.1f}s audio in {elapsed:.2f}s "
            f"on {self.device.upper()} (RTF {rtf:.2f}x; <1.0 = faster than real-time)",
            flush=True,
        )
        return text
