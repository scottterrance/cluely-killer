"""Always-on background transcriber (Phase 2: continuous STT).

The whole point of Phase 2 is to move Whisper OFF the hotkey critical
path. Instead of transcribing on press (which costs seconds), a
background thread continuously transcribes the audio buffer as the
interviewer talks, maintaining a live list of timestamped transcript
segments in memory. When the user presses '1'/'2', the Controller just
*reads text that already exists* - STT latency on the press path drops
to ~0, and the only thing left on the clock is the LLM call.

Design
------
* One daemon thread loops ~4x/second.
* It tracks a cursor ``_done_pos`` = the absolute sample index up to
  which audio has already been transcribed. The unprocessed tail is
  ``[_done_pos, buffer.current_position())``.
* It only flushes a chunk to Whisper when EITHER:
    - the tail reaches ``max_chunk_seconds`` (so a non-stop talker still
      gets transcribed in bounded pieces), OR
    - the tail is at least ``min_chunk_seconds`` long AND the most
      recent ``silence_hold`` seconds look like silence (i.e. the
      speaker paused - a natural sentence/question boundary).
  Transcribing on pauses keeps segments clean and gives Whisper full
  phrases rather than mid-word cuts.
* Each transcribed chunk becomes a ``Segment(start, end, text)`` where
  start/end are absolute sample positions. Segments accumulate (capped)
  so the Controller can pull "everything spoken since marker P".

Why local-only
--------------
This loop runs many times per question, so it MUST be free. It always
uses the local faster-whisper engine - never the metered Groq cloud STT
(that would burn your free-tier quota in minutes). If the local model
isn't present, continuous mode simply can't run and the app falls back
to the classic on-press transcription path (see Controller).

Thread-safety: the segment list and cursor are guarded by a lock. The
public methods (``snapshot_since``, ``current_seq``, ``reset``) are safe
to call from the Qt/Controller threads while the worker runs.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass

import numpy as np

from ..audio.buffer import RollingAudioBuffer
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
        reset instead of replaying pre-reset segments. The Controller
        re-reads this into ``_last_seq`` right after calling ``reset()``.
        """
        with self._lock:
            return self._seq

    def reset(self) -> None:
        """Forget all segments and skip the cursor to 'now'. Called on
        Ctrl+R so a new topic starts from a clean slate. The seq counter
        stays monotonic (see ``current_seq``); only the stored segments
        and the transcription cursor are reset.
        """
        with self._lock:
            self._segments.clear()
            self._done_pos = self.buffer.current_position()

    def snapshot_since(self, since_seq: int, max_seconds: float) -> tuple[str, int, float]:
        """Return (joined_text, latest_seq, covered_seconds) for every
        segment with seq > ``since_seq``.

        The text is capped to roughly the most recent ``max_seconds`` of
        audio (mirrors the on-press ``max_capture_seconds`` ceiling) by
        dropping the oldest segments first when the span is too long.
        ``latest_seq`` is what the Controller stores as its new marker
        after a successful answer.
        """
        cap_samples = int(max(0.0, max_seconds) * self.samplerate)
        with self._lock:
            picked = [s for s in self._segments if s.seq > since_seq]
            if not picked:
                latest = self._segments[-1].seq if self._segments else since_seq
                return "", latest, 0.0
            # Trim oldest-first so the captured span stays within the cap.
            if cap_samples > 0:
                total = picked[-1].end - picked[0].start
                while len(picked) > 1 and total > cap_samples:
                    picked.pop(0)
                    total = picked[-1].end - picked[0].start
            text = " ".join(s.text for s in picked if s.text).strip()
            covered = (picked[-1].end - picked[0].start) / self.samplerate
            return text, picked[-1].seq, covered

    def has_pending_audio(self) -> bool:
        """True if there's un-transcribed audio in the tail right now.
        Used by the Controller to decide whether to wait briefly for the
        worker to flush before answering.
        """
        with self._lock:
            return self.buffer.current_position() - self._done_pos >= self.min_chunk

    def flush_now(self, timeout: float = 1.5) -> None:
        """Block (up to timeout) until the current tail has been
        transcribed. Called right after a hotkey press so the answer
        includes the last few words the interviewer just said, even if
        they didn't pause long enough (or speak long enough) to trigger
        an automatic flush.

        Because the press sets the force flag, the worker's next tick
        flushes the whole tail regardless of length. We wait until the
        cursor has actually caught up to where the buffer was at press
        time (a later-arriving tail is the NEXT question, not this one).
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

    # -- worker --------------------------------------------------------
    def _rms(self, audio: np.ndarray) -> float:
        if audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))

    def _looks_like_pause(self, tail_end: int) -> bool:
        """Is the most recent silence_hold window quiet (speaker paused)?"""
        win = self.buffer.get_range(tail_end - self.silence_hold, tail_end)
        if win.size < self.silence_hold * 0.5:
            return False
        return self._rms(win) < self.silence_rms

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
        #   - max_chunk reached: hard-cap flush (bounded pieces for a
        #     non-stop talker), leaving the remainder for next tick.
        #   - forced (hotkey press): flush whatever is there NOW, even a
        #     short tail. The user explicitly asked for an answer, so we
        #     must not swallow the last few words just because they came
        #     in under min_chunk.
        #   - otherwise: flush only once we have >= min_chunk AND the
        #     speaker just paused (natural sentence boundary).
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
        # Floor on FORCED flushes is lower (0.2s) than automatic ones,
        # but we still need *something* to transcribe.
        min_audio = self.samplerate * (0.2 if forced else 0.4)
        if audio.size < min_audio:
            # Mostly evicted or too short - advance the cursor anyway so
            # we don't get stuck retrying the same gap forever.
            with self._lock:
                self._done_pos = flush_end
            return

        text = ""
        try:
            text = self.engine.transcribe(audio)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            # Advance cursor past this chunk to avoid a tight retry loop
            # on a persistently failing region.
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

    def backlog_seconds(self) -> float:
        """How many seconds of un-transcribed audio are queued right now.
        Large backlog => the local model can't keep up with real time and
        the segments are stale (answering an old question).
        """
        with self._lock:
            return (self.buffer.current_position() - self._done_pos) / self.samplerate

    def transcribe_recent(self, max_seconds: float, engine: "STTEngine | None" = None) -> str:
        """Transcribe the most-recent ``max_seconds`` of audio DIRECTLY,
        bypassing the segment backlog, and fast-forward the cursor to now.

        This is the escape hatch for slow machines: when the background
        loop has fallen behind (backlog > a few seconds), reading the
        accumulated segments would answer a stale question. Instead we
        grab the latest audio window, transcribe it once on the calling
        thread, and reset the cursor so the worker resumes cleanly from
        the present. The result is "what was just said".

        ``engine`` lets the caller pass a FASTER engine (e.g. Groq cloud
        STT) for this one synchronous call while the background loop keeps
        using the local model. If None, the background engine is used.
        """
        eng = engine or self.engine
        now = self.buffer.current_position()
        # Pull up to max_seconds of the most recent audio.
        want = int(max(1.0, max_seconds) * self.samplerate)
        start = max(0, now - want)
        audio = self.buffer.get_range(start, now)
        text = ""
        if audio.size >= self.samplerate * 0.4:
            try:
                text = eng.transcribe(audio)
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
        # Fast-forward: everything up to now is considered consumed, and
        # we record a synthetic segment so seq advances and future
        # snapshot_since(seq) calls behave consistently.
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
