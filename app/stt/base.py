"""Common Speech-to-Text interface.

Both the local faster-whisper engine and the Groq cloud engine implement
this so the rest of the app (Controller) can treat them interchangeably.
The Controller only ever touches three things on an STT engine:

  * ``samplerate``        - int, the sample rate it expects audio at (16k)
  * ``transcribe(audio)`` - np.float32 mono array -> transcript string
  * ``set_bias(words)``   - install a biasing vocabulary (may be a no-op)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class STTEngine(ABC):
    samplerate: int = 16000

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a mono float32 [-1, 1] numpy array to text."""

    def set_bias(self, keywords: list[str]) -> None:
        """Install a biasing vocabulary. Default: no-op."""
        return None
