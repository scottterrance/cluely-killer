"""Orchestrates: hotkey -> audio slice -> Whisper -> LLM stream -> UI signals.

The Controller lives on the main thread but offloads the heavy work
(STT + LLM) to a worker thread so the UI never freezes. UI updates
are delivered via Qt signals, which are queued safely across threads.
"""
from __future__ import annotations

import threading
import traceback
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from ..audio.buffer import RollingAudioBuffer
from ..config import Settings
from ..llm.base import LLMProvider
from ..prompts.builder import AnswerMode, ExampleScheduler
from ..stt.whisper_engine import WhisperEngine
from .history import ConversationHistory


def _friendly_error(exc: BaseException) -> str:
    """Translate common provider errors into a one-line actionable hint."""
    cls = type(exc).__name__
    msg = str(exc)
    low = msg.lower()
    # DNS / hostname resolution failures FIRST - the underlying httpx
    # ConnectError doesn't contain the word "connection" so it would
    # otherwise fall through to the generic formatter.
    if (
        "getaddrinfo" in low
        or "name resolution" in low
        or "11001" in msg
    ):
        return (
            "DNS lookup failed - cannot reach api.deepseek.com. "
            "Check your WiFi / corporate firewall."
        )
    if "403" in msg or cls == "PermissionDeniedError" or "access denied" in low:
        return "DeepSeek blocked the request (HTTP 403). Check your API key in Settings."
    if "401" in msg or cls == "AuthenticationError" or "invalid api key" in low:
        return "API key is invalid or revoked. Check Settings -> AI Provider."
    if "429" in msg or cls == "RateLimitError" or "rate limit" in low:
        return "DeepSeek rate limit hit. Wait a minute and try again."
    if (
        cls in ("APIConnectionError", "ConnectionError", "ConnectError",
                "ConnectTimeout", "ReadTimeout", "TimeoutException")
        or "connection" in low
        or "timed out" in low
    ):
        return "Network error reaching DeepSeek. Check your internet."
    return f"{cls}: {msg}"[:200]


class Controller(QObject):
    # UI signals
    transcript_ready = pyqtSignal(str)
    answer_started = pyqtSignal()
    answer_chunk = pyqtSignal(str)
    answer_finished = pyqtSignal()
    error = pyqtSignal(str)
    status = pyqtSignal(str)
    history_changed = pyqtSignal(int)  # current turn count

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

    # ------------------------------------------------------------------
    def trigger_answer(self, mode: str = AnswerMode.AUTO) -> None:
        """Hotkey entry point. Non-blocking; safe to call from any thread.

        `mode` selects the response-style block in the system prompt:
          - AUTO    -> smart classifier (default Ctrl+Space)
          - SUMMARY -> tie-back to last 5 Q+A   (Ctrl+Shift+1)
          - SIMPLE  -> 1-2 sentence standalone  (Ctrl+Shift+2)
          - DEEP    -> technical deep-dive      (Ctrl+Shift+3)
        """
        if mode not in AnswerMode.ALL:
            mode = AnswerMode.AUTO
        if not self._busy.acquire(blocking=False):
            self.status.emit("Busy - wait for current answer to finish")
            return
        threading.Thread(
            target=self._do_answer,
            args=(mode,),
            daemon=True,
            name="AnswerWorker",
        ).start()

    def clear(self) -> None:
        self.audio_buffer.clear()
        self.history.clear()
        self.history_changed.emit(0)
        self.status.emit("Audio buffer + memory cleared")

    # ------------------------------------------------------------------
    def _do_answer(self, mode: str = AnswerMode.AUTO) -> None:
        try:
            self.status.emit(f"Transcribing... ({mode})" if mode != AnswerMode.AUTO else "Transcribing...")
            audio = self.audio_buffer.get_last_seconds(self.settings.answer_window_seconds)
            if audio.size < self.whisper.samplerate:  # less than 1 second
                self.error.emit("Not enough audio yet - let the interviewer talk first.")
                return

            transcript = self.whisper.transcribe(audio)
            if not transcript:
                self.error.emit("No speech detected in the last window.")
                return
            self.transcript_ready.emit(transcript)
            print(f"[answer] mode={mode!r} transcript: {transcript[:200]!r}", flush=True)

            self.status.emit("Thinking..." if mode == AnswerMode.AUTO else f"Thinking... ({mode})")
            include_example = self.scheduler.should_include()
            system_prompt = self.prompt_builder(self.settings, include_example, mode)
            llm = self.llm_factory(self.settings)
            prior = self.history.as_messages()
            print(
                f"[answer] provider=deepseek mode={mode} "
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

            # Decide what to surface. Critical: never leave the answer panel
            # silently empty - the candidate has no idea what happened
            # otherwise.
            fallback: str | None = None
            if err_msg:
                fallback = f"\u26a0  {err_msg}"
            elif not full:
                fallback = (
                    "\u26a0  The LLM returned an empty response. "
                    "Check your API key / quota / model name in Settings -> AI Provider, "
                    "or switch to a different provider."
                )
            elif full.upper() == "SKIP":
                fallback = (
                    "\u26a0  The model output 'SKIP' - it judged the transcribed text as not a "
                    "clear question. Press Ctrl+R to clear, then Ctrl+Space again right after "
                    "the interviewer finishes speaking."
                )

            if fallback:
                # If chunks already arrived, append the warning. If the panel
                # is empty, the warning IS the visible content.
                separator = "\n\n" if chunks else ""
                self.answer_chunk.emit(separator + fallback)

            self.answer_finished.emit()

            # Only real answers go in memory.
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
