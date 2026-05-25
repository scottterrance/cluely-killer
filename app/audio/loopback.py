"""WASAPI loopback capture.

Captures whatever Windows is currently *playing* through the default
output device (i.e. the interviewer's voice through Zoom/Meet/Teams)
and pushes mono 16 kHz float32 frames into a RollingAudioBuffer.

Uses the `soundcard` library, which wraps WASAPI loopback on Windows
without requiring a virtual cable.
"""
from __future__ import annotations

import threading
import traceback

import numpy as np
import soundcard as sc

from .buffer import RollingAudioBuffer


class LoopbackCapture:
    def __init__(
        self,
        buffer: RollingAudioBuffer,
        samplerate: int = 16000,
        block_seconds: float = 0.1,
    ):
        self.buffer = buffer
        self.samplerate = samplerate
        self.blocksize = int(samplerate * block_seconds)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: str | None = None

    @property
    def last_error(self) -> str | None:
        return self._error

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="LoopbackCapture")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            speaker = sc.default_speaker()
            mic = sc.get_microphone(str(speaker.name), include_loopback=True)
            with mic.recorder(samplerate=self.samplerate, channels=1, blocksize=self.blocksize) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=self.blocksize)
                    if data is None or data.size == 0:
                        continue
                    # Defensive: clamp to mono float32
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    self.buffer.append(data.astype(np.float32, copy=False))
        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
