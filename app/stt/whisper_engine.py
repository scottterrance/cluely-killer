"""faster-whisper wrapper.

Built-in Silero VAD handles low-volume / whispered speech; beam_size=1
keeps latency low. English-only is enforced for speed.
"""
from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel


class WhisperEngine:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.samplerate = 16000
        # First call downloads the model (cached under ~/.cache/huggingface).
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

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
