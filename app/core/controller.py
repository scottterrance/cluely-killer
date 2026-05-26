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
from ..prompts.builder import ExampleScheduler
from ..stt.whisper_engine import WhisperEngine
from .history import ConversationHistory


def _friendly_error(exc: BaseException) -> str:
    """Translate common provider errors into a one-line actionable hint."""
    cls = type(exc).__name__
    msg = str(exc)
    low = msg.lower()
    # Groq / OpenAI-compatible HTTP errors put the status code in the message,
    # e.g. "Error code: 403 - {...}".
    if "403" in msg or cls == "PermissionDeniedError" or "access denied" in low:
        return (
            "Provider blocked your IP (HTTP 403). Many cloud / VPS IP ranges "
            "are blocked by Groq's WAF. Open Settings -> AI Provider and "
            "switch to 'ollama' (local model)."
        )
    if "401" in msg or cls == "AuthenticationError" or "invalid api key" in low:
        return "API key is invalid or revoked. Check Settings -> AI Provider."
    if "429" in msg or cls == "RateLimitError" or "rate limit" in low:
        return "Rate limit hit. Wait a minute, or switch provider in Settings."
    if cls in ("APIConnectionError", "ConnectionError") or "connection" in low:
        return "Network error reaching the LLM. Check your connection."
    # Ollama-specific: model not pulled yet
    if "model" in low and ("not found" in low or "404" in msg):
        return "Ollama model not found. Run e.g.  ollama pull llama3.1:8b"
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
    def trigger_answer(self) -> None:
        """Hotkey entry point. Non-blocking; safe to call from any thread."""
        if not self._busy.acquire(blocking=False):
            self.status.emit("Busy - wait for current answer to finish")
            return
        threading.Thread(target=self._do_answer, daemon=True, name="AnswerWorker").start()

    def clear(self) -> None:
        self.audio_buffer.clear()
        self.history.clear()
        self.history_changed.emit(0)
        self.status.emit("Audio buffer + memory cleared")

    # ------------------------------------------------------------------
    def _do_answer(self) -> None:
        try:
            self.status.emit("Transcribing...")
            audio = self.audio_buffer.get_last_seconds(self.settings.answer_window_seconds)
            if audio.size < self.whisper.samplerate:  # less than 1 second
                self.error.emit("Not enough audio yet - let the interviewer talk first.")
                return

            transcript = self.whisper.transcribe(audio)
            if not transcript:
                self.error.emit("No speech detected in the last window.")
                return
            self.transcript_ready.emit(transcript)

            self.status.emit("Thinking...")
            include_example = self.scheduler.should_include()
            system_prompt = self.prompt_builder(self.settings, include_example)
            llm = self.llm_factory(self.settings)
            prior = self.history.as_messages()

            self.answer_started.emit()
            chunks: list[str] = []
            try:
                for chunk in llm.stream_chat(system_prompt, transcript, prior_messages=prior):
                    chunks.append(chunk)
                    self.answer_chunk.emit(chunk)
            except Exception as e:
                # Show actionable status text. Keep partial answer visible so
                # the candidate sees what they got before the failure.
                traceback.print_exc()
                self.error.emit(_friendly_error(e))
                return
            finally:
                self.answer_finished.emit()

            full = "".join(chunks).strip()
            # Don't poison the memory with non-answers.
            if full and full.upper() != "SKIP":
                self.history.add(transcript, full)
                self.history_changed.emit(len(self.history))

            self.status.emit("Ready")
        except Exception as e:
            traceback.print_exc()
            self.error.emit(_friendly_error(e))
        finally:
            self._busy.release()
