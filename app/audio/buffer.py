"""Thread-safe rolling audio buffer.

Always-on capture appends here; the answer worker pulls the last N seconds
when the user presses the hotkey. The live transcriber consumer also reads
from here, but cursor-based via `total_samples()` + `get_samples_since()`
so it can keep an absolute position even though the underlying ring drops
old data once max_seconds is exceeded.
"""
from __future__ import annotations

import threading

import numpy as np


class RollingAudioBuffer:
    def __init__(self, samplerate: int = 16000, max_seconds: float = 60.0):
        self.samplerate = samplerate
        self.max_samples = int(samplerate * max_seconds)
        self._buf = np.zeros(0, dtype=np.float32)
        # Monotonically increasing count of every sample EVER appended.
        # Lets cursor consumers (live transcriber) survive ring truncation.
        self._total_samples: int = 0
        self._lock = threading.Lock()

    def append(self, data: np.ndarray) -> None:
        if data is None or data.size == 0:
            return
        flat = data.flatten().astype(np.float32, copy=False)
        with self._lock:
            self._total_samples += int(flat.size)
            self._buf = np.concatenate([self._buf, flat])
            if self._buf.size > self.max_samples:
                self._buf = self._buf[-self.max_samples:]

    def get_last_seconds(self, seconds: float) -> np.ndarray:
        n = int(seconds * self.samplerate)
        with self._lock:
            if self._buf.size == 0:
                return np.zeros(0, dtype=np.float32)
            return self._buf[-n:].copy() if self._buf.size >= n else self._buf.copy()

    # --- Cursor-based access for the live transcriber ---------------
    def total_samples(self) -> int:
        with self._lock:
            return self._total_samples

    def get_samples_since(
        self,
        absolute_pos: int,
        max_count: int | None = None,
    ) -> np.ndarray:
        """Return audio from absolute_pos onward, up to current end."""
        with self._lock:
            if absolute_pos >= self._total_samples:
                return np.zeros(0, dtype=np.float32)
            oldest_abs = self._total_samples - self._buf.size
            start = max(0, absolute_pos - oldest_abs)
            end = self._buf.size
            if max_count is not None:
                end = min(end, start + max_count)
            return self._buf[start:end].copy()

    def trailing_silence_seconds(
        self, energy_thresh: float = 0.005, frame_seconds: float = 0.05
    ) -> float:
        """Return seconds of silence at the END of the current buffer.

        Cheap energy-based detector: walks 50 ms frames backward from the
        end of the buffer, counting how many are below `energy_thresh`,
        stopping at the first frame above threshold. Used by the live
        transcriber to know when the interviewer just stopped speaking
        (which is when we want to fire a speculative answer).
        """
        sr = self.samplerate
        frame = max(1, int(frame_seconds * sr))
        with self._lock:
            buf = self._buf
            if buf.size < frame * 2:
                return 0.0
            silent = 0
            i = buf.size
            while i >= frame:
                chunk = buf[i - frame:i]
                rms = float(np.sqrt(np.mean(chunk * chunk)))
                if rms > energy_thresh:
                    break
                silent += 1
                i -= frame
            return silent * frame_seconds

    def clear(self) -> None:
        with self._lock:
            self._buf = np.zeros(0, dtype=np.float32)
            # Don't reset _total_samples - downstream cursors use it as a
            # monotonic anchor; the controller snaps its own position to
            # the new buffer end after clear().

    def duration_seconds(self) -> float:
        with self._lock:
            return self._buf.size / self.samplerate
