"""Thread-safe rolling audio buffer.

Always-on capture appends here; the answer worker pulls the last N seconds
when the user presses the hotkey.
"""
from __future__ import annotations

import threading

import numpy as np


class RollingAudioBuffer:
    def __init__(self, samplerate: int = 16000, max_seconds: float = 60.0):
        self.samplerate = samplerate
        self.max_samples = int(samplerate * max_seconds)
        self._buf = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()

    def append(self, data: np.ndarray) -> None:
        if data is None or data.size == 0:
            return
        flat = data.flatten().astype(np.float32, copy=False)
        with self._lock:
            self._buf = np.concatenate([self._buf, flat])
            if self._buf.size > self.max_samples:
                self._buf = self._buf[-self.max_samples:]

    def get_last_seconds(self, seconds: float) -> np.ndarray:
        n = int(seconds * self.samplerate)
        with self._lock:
            if self._buf.size == 0:
                return np.zeros(0, dtype=np.float32)
            return self._buf[-n:].copy() if self._buf.size >= n else self._buf.copy()

    def clear(self) -> None:
        with self._lock:
            self._buf = np.zeros(0, dtype=np.float32)

    def duration_seconds(self) -> float:
        with self._lock:
            return self._buf.size / self.samplerate
