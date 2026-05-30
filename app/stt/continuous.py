"""Always-on background transcriber (continuous STT).

Moves Whisper OFF the hotkey critical path. A background thread
continuously transcribes the audio buffer as the interviewer talks,
maintaining a live list of timestamped transcript segments. When the
user presses '1'/'2', the Controller just *reads text that already
exists* - STT latency on the press path drops to ~0, and the only thing
left on the clock is the LLM call.

Design
------
* One daemon thread loops ~4x/second.
* Cursor ``_done_pos`` = absolute sample index already transcribed; the
  unprocessed tail is ``[_done_pos, buffer.current_position())``.
* Flush a chunk to Whisper when EITHER the tail reaches
  ``max_chunk_seconds`` (bounded pieces for a non-stop talker) OR the
  tail is >= ``min_chunk_seconds`` and the speaker just paused (natural
  sentence boundary).
* SILENCE-SKIP: before transcribing a chunk, check its RMS. If it's
  essentially silence (no one spoke), skip Whisper entirely and just
  advance the cursor. This is critical on a slow CPU - it stops the
  loop from burning time transcribing dead air, keeping it caught up
  with real time and freeing the CPU for the actual questions.

Local model only: this loop runs many times per question and must be
free. There is no cloud STT in this build.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass

import numpy as np

from ..audio.buffer import RollingAudioBuffer
from .audio_proc import rms
from .base import STTEngine


@dataclass
class Segment:
    start: int   # absolute sample position (inclusive)
    end: int     # absolute sample position (exclusive)
    text: str
    seq: int     # monotonic id, 1-based


class ContinuousTranscriber:
    def __init__(
        self,
        buffer: RollingAudioBuffer,
        engine: STTEngine,
        samplerate: int = 16000,
        min_chunk_seconds: float = 2.5,
        max_chunk_seconds: float = 12.0,
        silence_hold_seconds: float = 0.5,
        silence_rms: float = 0.006,
        poll_interval: float = 0.25,
        max_segments: int = 400,
    ):
        self.buffer = buffer
        self.engine = engine
        self.samplerate = samplerate
        self.min_chunk = int(min_chunk_seconds * samplerate)
        self.max_chunk = int(max_chunk_seconds * samplerate)
        self.silence_hold = int(silence_hold_seconds * samplerate)
        self.silence_rms = silence_rms
        self.poll_interval = poll_interval
        self.max_segments = max_segments

        self._segments: list[Segment] = []
        self._seq = 0
        # Cursor: absolute sample index already transcribed. Start it at
        # the buffer's current position so we never try to transcribe
        # the silence that existed before launch.
        self._done_pos = buffer.current_position()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._force_flush = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        # Clear the stop flag in case this is a restart after stop().
        self._stop.clear()
        self._force_flush.clear()
        self._thread = threading.Thread(
            target=self._run, name="ContinuousTranscriber", daemon=True
        )
        self._thread.start()
        print("[continuous] background transcriber started.", flush=True)

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
        self._thread = None

    # -- public read API (called from Controller) ---------------------
    def current_seq(self) -> int:
        """The seq id of the latest finalized segment.

        Monotonic: ``reset()`` does NOT zero this (mirroring the audio
        buffer's monotonic ``_total_appended``). That guarantees a stale
        marker held by the Controller yields an empty snapshot after a
        reset instead of replaying pre-reset segments.
        """
        with self._lock:
            return self._seq

    def reset(self) -> None:
        """Forget all segments and skip the cursor to 'now'. Called on
        Ctrl+R so a new topic starts from a clean slate. The seq counter
        stays monotonic (see ``current_seq``).
        """
        with self._lock:
            self._segments.clear()
            self._done_pos = self.buffer.current_position()

    def snapshot_since(self, since_seq: int, max_seconds: float) -> tuple[str, int, float]:
        """Return (joined_text, latest_seq, covered_seconds) for every
        segment with seq > ``since_seq``.

        Text is capped to roughly the most recent ``max_seconds`` of
        audio by dropping the oldest segments first when the span is too
        long. ``latest_seq`` is what the Controller stores as its new
        marker after a successful answer.
        """
        cap_samples = int(max(0.0, max_seconds) * self.samplerate)
        with self._lock:
            picked = [s for s in self._segments if s.seq > since_seq]
            if not picked:
                latest = self._segments[-1].seq if self._segments else since_seq
                return "", latest, 0.0
            if cap_samples > 0:
                total = picked[-1].end - picked[0].start
                while len(picked) > 1 and total > cap_samples:
                    picked.pop(0)
                    total = picked[-1].end - picked[0].start
            text = " ".join(s.text for s in picked if s.text).strip()
            covered = (picked[-1].end - picked[0].start) / self.samplerate
            return text, picked[-1].seq, covered

    def has_pending_audio(self) -> bool:
        with self._lock:
            return self.buffer.current_position() - self._done_pos >= self.min_chunk

    def flush_now(self, timeout: float = 1.5) -> None:
        """Block (up to timeout) until the current tail has been
        transcribed. Called right after a hotkey press so the answer
        includes the last few words the interviewer just said, even if
        they didn't pause long enough (or speak long enough) to trigger
        an automatic flush.
        """
        target = self.buffer.current_position()
        self._force_flush.set()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                done = self._done_pos
            if done >= target:
                return
            time.sleep(0.03)

    def backlog_seconds(self) -> float:
        """Seconds of un-transcribed audio queued right now. A large
        value => the local model can't keep up with real time.
        """
        with self._lock:
            return (self.buffer.current_position() - self._done_pos) / self.samplerate

    def transcribe_recent(self, max_seconds: float) -> str:
        """Transcribe the most-recent ``max_seconds`` of audio DIRECTLY,
        bypassing the segment backlog, and fast-forward the cursor to now.

        Escape hatch for when the background loop has fallen behind:
        reading the accumulated segments would answer a stale question,
        so we grab the latest audio window, isolate the last utterance,
        transcribe it once on the calling thread, and reset the cursor.
        """
        now = self.buffer.current_position()
        want = int(max(1.0, max_seconds) * self.samplerate)
        start = max(0, now - want)
        audio = self.buffer.get_range(start, now)
        text = ""
        if audio.size >= self.samplerate * 0.4:
            try:
                # isolate_last: send only the interviewer's last question.
                text = self.engine.transcribe(audio, isolate_last=True)
            except TypeError:
                # Engine without the isolate_last kwarg (defensive).
                text = self.engine.transcribe(audio)
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
        with self._lock:
            self._done_pos = now
            if text:
                self._seq += 1
                self._segments.append(
                    Segment(start=start, end=now, text=text, seq=self._seq)
                )
                if len(self._segments) > self.max_segments:
                    self._segments = self._segments[-self.max_segments:]
        return text

    def set_bias(self, keywords: list[str]) -> None:
        """Forward biasing to the underlying engine."""
        try:
            self.engine.set_bias(keywords)
        except Exception:
            pass

    # -- worker --------------------------------------------------------
    def _looks_like_pause(self, tail_end: int) -> bool:
        """Is the most recent silence_hold window quiet (speaker paused)?"""
        win = self.buffer.get_range(tail_end - self.silence_hold, tail_end)
        if win.size < self.silence_hold * 0.5:
            return False
        return rms(win) < self.silence_rms

    def _run(self) -> None:
        ff = self._force_flush
        while not self._stop.is_set():
            try:
                self._tick(forced=ff.is_set())
                ff.clear()
            except Exception as e:  # never let the loop die
                self.last_error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            time.sleep(self.poll_interval)

    def _tick(self, forced: bool) -> None:
        now = self.buffer.current_position()
        with self._lock:
            done = self._done_pos
        pending = now - done
        if pending <= 0:
            return

        # Decide whether to flush a chunk.
        flush_end: int | None = None
        if pending >= self.max_chunk:
            flush_end = done + self.max_chunk
        elif forced:
            flush_end = now
        elif pending >= self.min_chunk and self._looks_like_pause(now):
            flush_end = now

        if flush_end is None:
            return

        audio = self.buffer.get_range(done, flush_end)
        min_audio = self.samplerate * (0.2 if forced else 0.4)
        if audio.size < min_audio:
            with self._lock:
                self._done_pos = flush_end
            return

        # SILENCE-SKIP: if this chunk is essentially silence, don't waste
        # CPU running Whisper on it (and don't risk a hallucinated
        # segment from dead air). Just advance the cursor. This keeps the
        # loop caught up on slow machines - the chunks that actually
        # contain speech are the only ones that pay the Whisper cost.
        # A forced (press-time) flush still runs even if quiet, so a
        # barely-audible question isn't dropped.
        if not forced and rms(audio) < self.silence_rms:
            with self._lock:
                self._done_pos = flush_end
            return

        text = ""
        try:
            text = self.engine.transcribe(audio)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            with self._lock:
                self._done_pos = flush_end
            return

        with self._lock:
            self._done_pos = flush_end
            if text:
                self._seq += 1
                self._segments.append(
                    Segment(start=done, end=flush_end, text=text, seq=self._seq)
                )
                if len(self._segments) > self.max_segments:
                    self._segments = self._segments[-self.max_segments:]
                print(
                    f"[continuous] seg #{self._seq} "
                    f"({(flush_end-done)/self.samplerate:.1f}s): {text[:80]!r}",
                    flush=True,
                )
