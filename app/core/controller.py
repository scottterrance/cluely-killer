"""Orchestrates: hotkey -> audio slice -> Whisper -> LLM stream -> UI signals.

Three workers cooperate via the rolling audio buffer:

  1. AnswerWorker     - one-shot, started by Ctrl+Space. Transcribes the
                        last ~25 s, calls the LLM, streams chunks to UI.
  2. LiveTranscriber  - always-on. Every ~4 s pulls new audio from the
                        cursor position, transcribes it, emits each
                        segment so the overlay shows a continuously
                        updating live transcript.
  3. SpeculativeWorker - fires when LiveTranscriber sees the buffer go
                        silent after speech. It speculatively runs the
                        full transcribe + LLM pipeline IN BACKGROUND
                        and caches the streamed chunks. If the candidate
                        presses Ctrl+Space within ~5 s, AnswerWorker
                        replays the cached chunks - perceived latency
                        ~= 0. Otherwise the speculative result is
                        discarded.

All three share one WhisperEngine; the engine has its own internal lock
so concurrent calls serialize safely.
"""
from __future__ import annotations

import threading
import time
import traceback
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from ..audio.buffer import RollingAudioBuffer
from ..config import Settings
from ..llm.base import LLMProvider
from ..prompts.builder import ExampleScheduler
from ..stt.whisper_engine import WhisperEngine
from .history import ConversationHistory


def _friendly_error(exc: BaseException) -> str:
    cls = type(exc).__name__
    msg = str(exc)
    low = msg.lower()
    if "403" in msg or cls == "PermissionDeniedError" or "access denied" in low:
        return (
            "Provider blocked your IP (HTTP 403). Open Settings -> AI Provider "
            "and switch to 'ollama' (local model)."
        )
    if "401" in msg or cls == "AuthenticationError" or "invalid api key" in low:
        return "API key is invalid or revoked. Check Settings -> AI Provider."
    if "429" in msg or cls == "RateLimitError" or "rate limit" in low:
        return "Rate limit hit. Wait a minute, or switch provider in Settings."
    if cls in ("APIConnectionError", "ConnectionError") or "connection" in low:
        return "Network error reaching the LLM. Check your connection."
    if "model" in low and ("not found" in low or "404" in msg):
        return "Model not found. For Ollama, run e.g.  ollama pull llama3.1:8b"
    return f"{cls}: {msg}"[:200]


# Live transcriber tuning.
_LIVE_CHUNK_SECONDS = 4.0
_LIVE_POLL_SECONDS = 0.4

# Speculative pre-fetch tuning.
# Trigger when at least this much silence is detected at the buffer end.
_SPEC_TRIGGER_SILENCE_S = 1.2
# Audio span we hand to the speculative pipeline as "the question".
_SPEC_QUESTION_WINDOW_S = 18.0
# How long after silence we still consider the speculative result fresh.
_SPEC_FRESH_S = 6.0
# Don't fire another speculative call within this gap.
_SPEC_RECOOL_S = 3.0


class _Speculative:
    """Cache for one in-flight or completed speculative answer."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.transcript: str | None = None
        self.chunks: list[str] = []
        self.complete: bool = False
        self.error: str | None = None
        self.born_at: float = 0.0  # time.monotonic
        self.consumed: bool = False  # set True once AnswerWorker replayed it
        self.cancelled: bool = False
        self.thread: threading.Thread | None = None


class Controller(QObject):
    # UI signals
    transcript_ready = pyqtSignal(str)
    answer_started = pyqtSignal()
    answer_chunk = pyqtSignal(str)
    answer_finished = pyqtSignal()
    error = pyqtSignal(str)
    status = pyqtSignal(str)
    history_changed = pyqtSignal(int)

    transcript_appended = pyqtSignal(str)      # live segment arrived
    transcript_cleared = pyqtSignal()          # Ctrl+R wiped the view
    answer_trigger_marker = pyqtSignal()       # Ctrl+Space - draw separator

    def __init__(
        self,
        settings: Settings,
        audio_buffer: RollingAudioBuffer,
        whisper: WhisperEngine,
        llm_factory: Callable[[Settings], LLMProvider],
        scheduler: ExampleScheduler,
        prompt_builder: Callable[[Settings, bool], str],
        history: ConversationHistory,
    ):
        super().__init__()
        self.settings = settings
        self.audio_buffer = audio_buffer
        self.whisper = whisper
        self.llm_factory = llm_factory
        self.scheduler = scheduler
        self.prompt_builder = prompt_builder
        self.history = history
        self._busy = threading.Lock()

        # Live transcription bookkeeping.
        self._live_thread: threading.Thread | None = None
        self._live_stop = threading.Event()
        self._live_pos: int = 0

        # Speculative pre-fetch bookkeeping.
        self._spec_lock = threading.Lock()
        self._spec: _Speculative | None = None
        self._last_spec_at: float = 0.0
        self._was_silent_last_check: bool = False

        self.start_live_transcription()

    # ------------------------------------------------------------------
    # Live transcription
    # ------------------------------------------------------------------
    def start_live_transcription(self) -> None:
        if self._live_thread and self._live_thread.is_alive():
            return
        self._live_stop.clear()
        self._live_pos = self.audio_buffer.total_samples()
        self._live_thread = threading.Thread(
            target=self._live_run, daemon=True, name="LiveTranscriber"
        )
        self._live_thread.start()

    def stop_live_transcription(self) -> None:
        self._live_stop.set()
        if self._live_thread:
            self._live_thread.join(timeout=2.0)
            self._live_thread = None

    def _live_run(self) -> None:
        sr = self.audio_buffer.samplerate
        chunk_samples = int(sr * _LIVE_CHUNK_SECONDS)
        min_useful = int(sr * 0.5)
        while not self._live_stop.is_set():
            if self._live_stop.wait(_LIVE_POLL_SECONDS):
                break
            try:
                # 1. Live transcript: pull a fixed chunk if enough new audio.
                current = self.audio_buffer.total_samples()
                pending = current - self._live_pos
                if pending >= chunk_samples:
                    audio = self.audio_buffer.get_samples_since(self._live_pos, chunk_samples)
                    self._live_pos += chunk_samples
                    if audio.size >= min_useful:
                        text = self.whisper.transcribe(audio).strip()
                        if text:
                            self.transcript_appended.emit(text)

                # 2. Speculative trigger: silence-after-speech edge.
                self._maybe_fire_speculative()
            except Exception:
                traceback.print_exc()
                self._live_stop.wait(2.0)

    # ------------------------------------------------------------------
    # Speculative pre-fetch
    # ------------------------------------------------------------------
    def _maybe_fire_speculative(self) -> None:
        silence_s = self.audio_buffer.trailing_silence_seconds()
        is_silent_now = silence_s >= _SPEC_TRIGGER_SILENCE_S
        # Rising-edge detector: only fire on the transition from
        # speaking -> silent, not every poll while still silent.
        edge = is_silent_now and not self._was_silent_last_check
        self._was_silent_last_check = is_silent_now
        if not edge:
            return
        now = time.monotonic()
        if now - self._last_spec_at < _SPEC_RECOOL_S:
            return
        self._last_spec_at = now
        # Cancel any older speculative; they're stale by definition.
        with self._spec_lock:
            old = self._spec
            self._spec = _Speculative()
            self._spec.born_at = now
            spec = self._spec
        if old is not None:
            old.cancelled = True
        # Fire the worker. The worker will populate spec.chunks as the
        # LLM streams; trigger_answer() picks them up if user is fast.
        spec.thread = threading.Thread(
            target=self._speculative_worker, args=(spec,), daemon=True,
            name="SpeculativeWorker",
        )
        spec.thread.start()
        print(f"[spec] fired at silence={silence_s:.2f}s", flush=True)

    def _speculative_worker(self, spec: _Speculative) -> None:
        try:
            audio = self.audio_buffer.get_last_seconds(_SPEC_QUESTION_WINDOW_S)
            if audio.size < self.whisper.samplerate:
                spec.error = "not enough audio"
                spec.complete = True
                return
            transcript = self.whisper.transcribe(audio)
            if spec.cancelled:
                return
            if not transcript:
                spec.error = "no speech"
                spec.complete = True
                return
            spec.transcript = transcript
            include_example = False  # don't burn the scheduler on speculation
            system_prompt = self.prompt_builder(self.settings, include_example)
            llm = self.llm_factory(self.settings)
            prior = self.history.as_messages()
            for chunk in llm.stream_chat(system_prompt, transcript, prior_messages=prior):
                if spec.cancelled:
                    return
                with spec.lock:
                    spec.chunks.append(chunk)
            spec.complete = True
            print(
                f"[spec] complete: {len(spec.chunks)} chunks, "
                f"transcript={transcript[:80]!r}",
                flush=True,
            )
        except Exception as e:
            traceback.print_exc()
            spec.error = _friendly_error(e)
            spec.complete = True

    def _consume_speculative(self) -> _Speculative | None:
        """Atomically claim the current speculative if it's fresh + usable."""
        now = time.monotonic()
        with self._spec_lock:
            spec = self._spec
            if spec is None:
                return None
            if spec.consumed or spec.cancelled:
                return None
            if spec.error and not spec.chunks:
                return None
            if now - spec.born_at > _SPEC_FRESH_S:
                return None
            spec.consumed = True
            return spec

    # ------------------------------------------------------------------
    def trigger_answer(self) -> None:
        if not self._busy.acquire(blocking=False):
            self.status.emit("Busy - wait for current answer to finish")
            return
        threading.Thread(target=self._do_answer, daemon=True, name="AnswerWorker").start()

    def clear(self) -> None:
        self.audio_buffer.clear()
        self.history.clear()
        self.history_changed.emit(0)
        self._live_pos = self.audio_buffer.total_samples()
        with self._spec_lock:
            if self._spec is not None:
                self._spec.cancelled = True
            self._spec = None
        self._was_silent_last_check = False
        self.transcript_cleared.emit()
        self.status.emit("Audio buffer + memory cleared")

    # ------------------------------------------------------------------
    def _do_answer(self) -> None:
        try:
            self.answer_trigger_marker.emit()
            spec = self._consume_speculative()
            if spec is not None and spec.transcript:
                # FAST PATH: speculative pre-fetch hit. Replay buffered
                # chunks and any new ones still arriving.
                self.status.emit("Answering... (pre-fetched)")
                self.transcript_ready.emit(spec.transcript)
                print(
                    f"[answer] SPEC HIT: transcript={spec.transcript[:80]!r}",
                    flush=True,
                )
                self.answer_started.emit()
                replayed = 0
                while True:
                    with spec.lock:
                        new_chunks = spec.chunks[replayed:]
                        replayed = len(spec.chunks)
                        complete = spec.complete
                    for c in new_chunks:
                        self.answer_chunk.emit(c)
                    if complete:
                        break
                    time.sleep(0.05)
                full = "".join(spec.chunks).strip()
                err_msg = spec.error
                self.answer_finished.emit()
                if not err_msg and full and full.upper() != "SKIP":
                    self.history.add(spec.transcript, full)
                    self.history_changed.emit(len(self.history))
                    self.status.emit("Ready (pre-fetched)")
                else:
                    self._render_fallback(err_msg, full, replayed > 0)
                return

            # SLOW PATH: no usable speculative; do it the original way.
            self.status.emit("Transcribing...")
            audio = self.audio_buffer.get_last_seconds(self.settings.answer_window_seconds)
            if audio.size < self.whisper.samplerate:
                self.error.emit("Not enough audio yet - let the interviewer talk first.")
                return

            transcript = self.whisper.transcribe(audio)
            if not transcript:
                self.error.emit("No speech detected in the last window.")
                return
            self.transcript_ready.emit(transcript)
            print(f"[answer] transcript: {transcript[:200]!r}", flush=True)

            self.status.emit("Thinking...")
            include_example = self.scheduler.should_include()
            system_prompt = self.prompt_builder(self.settings, include_example)
            llm = self.llm_factory(self.settings)
            prior = self.history.as_messages()
            print(
                f"[answer] provider={self.settings.provider} "
                f"history_turns={len(prior)//2}",
                flush=True,
            )

            self.answer_started.emit()
            chunks: list[str] = []
            err_msg: str | None = None
            try:
                for chunk in llm.stream_chat(system_prompt, transcript, prior_messages=prior):
                    chunks.append(chunk)
                    self.answer_chunk.emit(chunk)
            except Exception as e:
                traceback.print_exc()
                err_msg = _friendly_error(e)

            full = "".join(chunks).strip()
            print(
                f"[answer] streamed {len(chunks)} chunks, "
                f"{len(full)} chars, err={err_msg!r}",
                flush=True,
            )

            self._render_fallback(err_msg, full, bool(chunks))
            self.answer_finished.emit()

            if not err_msg and full and full.upper() != "SKIP":
                self.history.add(transcript, full)
                self.history_changed.emit(len(self.history))
                self.status.emit("Ready")
            else:
                self.status.emit("See answer panel - non-success outcome.")
        except Exception as e:
            traceback.print_exc()
            self.error.emit(_friendly_error(e))
        finally:
            self._busy.release()

    def _render_fallback(self, err_msg: str | None, full: str, had_chunks: bool) -> None:
        fallback: str | None = None
        if err_msg:
            fallback = f"\u26a0  {err_msg}"
        elif not full:
            fallback = (
                "\u26a0  The LLM returned an empty response. Check Settings -> "
                "AI Provider, or switch to a different provider."
            )
        elif full.upper() == "SKIP":
            fallback = (
                "\u26a0  The model output 'SKIP' - transcribed text wasn't a clear "
                "question. Press Ctrl+R then Ctrl+Space again."
            )
        if fallback:
            sep = "\n\n" if had_chunks else ""
            self.answer_chunk.emit(sep + fallback)
