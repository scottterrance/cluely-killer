"""Thread-safe rolling audio buffer.

Always-on capture appends here; the answer worker pulls audio out of
the buffer when the user presses a hotkey.

Two read modes are supported:

  * ``get_last_seconds(s)`` - the classic "last N seconds" slice.
    Used as a fallback for the very first press in a session before any
    marker has been set.
  * ``get_since_position(pos, max_seconds)`` - everything captured
    since the position marker `pos`, capped to `max_seconds` of audio.
    This is what the two-key answer modes use: each press records the
    buffer's position, and the next press grabs every sample the
    interviewer spoke between those two presses.

The position marker is a monotonic count of samples ever appended. It
is independent of where the rolling window currently sits, so even
after the rolling buffer wraps (older audio gets dropped to stay
under ``max_seconds``) the marker math still works - we just clip
to whatever is still resident.
"""
from __future__ import annotations

import threading

import numpy as np


class RollingAudioBuffer:
    def __init__(self, samplerate: int = 16000, max_seconds: float = 60.0):
        self.samplerate = samplerate
        self.max_samples = int(samplerate * max_seconds)
        self._buf = np.zeros(0, dtype=np.float32)
        # Monotonic count of every sample ever appended. The current
        # rolling buffer always represents [_total_appended - _buf.size,
        # _total_appended) on this timeline.
        self._total_appended = 0
        self._lock = threading.Lock()

    def append(self, data: np.ndarray) -> None:
        if data is None or data.size == 0:
            return
        flat = data.flatten().astype(np.float32, copy=False)
        with self._lock:
            self._buf = np.concatenate([self._buf, flat])
            self._total_appended += flat.size
            if self._buf.size > self.max_samples:
                self._buf = self._buf[-self.max_samples:]

    def get_last_seconds(self, seconds: float) -> np.ndarray:
        n = int(seconds * self.samplerate)
        with self._lock:
            if self._buf.size == 0:
                return np.zeros(0, dtype=np.float32)
            return self._buf[-n:].copy() if self._buf.size >= n else self._buf.copy()

    def current_position(self) -> int:
        """Sample-index marker for "everything captured up to now".

        Stash this immediately after a successful answer; the next press
        passes it back to ``get_since_position`` to retrieve every sample
        that arrived in the meantime.
        """
        with self._lock:
            return self._total_appended

    def get_range(self, start: int, end: int) -> np.ndarray:
        """Return audio for the absolute sample interval [start, end).

        Both bounds are on the same monotonic timeline as
        ``current_position()`` / ``_total_appended``. The interval is
        clipped to whatever is still resident in the rolling window:
        the buffer currently holds samples
        ``[_total_appended - _buf.size, _total_appended)``, so any part
        of [start, end) older than that has already been evicted and
        simply isn't returned.

        Used by the continuous transcriber, which walks forward through
        the timeline one chunk at a time and asks for each chunk by its
        absolute [start, end) bounds.
        """
        with self._lock:
            if end <= start or self._buf.size == 0:
                return np.zeros(0, dtype=np.float32)
            buf_start = self._total_appended - self._buf.size
            # Clip the requested interval to what's resident.
            lo = max(start, buf_start)
            hi = min(end, self._total_appended)
            if hi <= lo:
                return np.zeros(0, dtype=np.float32)
            i = lo - buf_start
            j = hi - buf_start
            return self._buf[i:j].copy()

    def get_since_position(
        self, position: int, max_seconds: float
    ) -> np.ndarray:
        """Return audio appended after ``position``, capped to ``max_seconds``.

        Clipping rules, in order:
          1. Clamp request length to (current_total - position). Negative
             or zero means nothing new since the marker.
          2. Cap to ``max_seconds`` worth of samples (the "120 s default"
             ceiling so a runaway interviewer monologue can't blow up
             the prompt).
          3. Cap to ``self._buf.size`` because that's literally all the
             audio still resident in memory - if the rolling window has
             already evicted older samples, we can only return what's
             left.

        Always returns the most-recent slice (i.e. the ``cap`` newest
        samples), which is the desired semantic when the cap is hit.
        """
        cap_samples = int(max(0.0, max_seconds) * self.samplerate)
        with self._lock:
            available = self._total_appended - max(0, position)
            n = min(max(available, 0), cap_samples, self._buf.size)
            if n <= 0:
                return np.zeros(0, dtype=np.float32)
            return self._buf[-n:].copy()

    def clear(self) -> None:
        with self._lock:
            self._buf = np.zeros(0, dtype=np.float32)
            # Note: we do NOT reset _total_appended. Keeping it
            # monotonic guarantees that a stale position marker held
            # by the controller will simply yield an empty slice on
            # the next read (since available <= 0), instead of
            # surprisingly returning audio that pre-dates the clear.

    def duration_seconds(self) -> float:
        with self._lock:
            return self._buf.size / self.samplerate
