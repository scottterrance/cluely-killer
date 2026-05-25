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


class Controller(QObject):
    # UI signals
    transcript_ready = pyqtSignal(str)
    answer_started = pyqtSignal()
    answer_chunk = pyqtSignal(str)
    answer_finished = pyqtSignal()
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(
        self,
        settings: Settings,
        audio_buffer: RollingAudioBuffer,
        whisper: WhisperEngine,
        llm_factory: Callable[[Settings], LLMProvider],
        scheduler: ExampleScheduler,
        prompt_builder: Callable[[Settings, bool], str],
    ):
        super().__init__()
        self.settings = settings
        self.audio_buffer = audio_buffer
        self.whisper = whisper
        self.llm_factory = llm_factory
        self.scheduler = scheduler
        self.prompt_builder = prompt_builder
        self._busy = threading.Lock()

    # ------------------------------------------------------------------
    def trigger_answer(self) -> None:
        """Hotkey entry point. Non-blocking; safe to call from any thread."""
        if not self._busy.acquire(blocking=False):
            self.status.emit("Busy — wait for current answer to finish")
            return
        threading.Thread(target=self._do_answer, daemon=True, name="AnswerWorker").start()

    def clear(self) -> None:
        self.audio_buffer.clear()
        self.status.emit("Audio buffer cleared")

    # ------------------------------------------------------------------
    def _do_answer(self) -> None:
        try:
            self.status.emit("Transcribing...")
            audio = self.audio_buffer.get_last_seconds(self.settings.answer_window_seconds)
            if audio.size < self.whisper.samplerate:  # less than 1 second
                self.error.emit("Not enough audio yet — let the interviewer talk first.")
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

            self.answer_started.emit()
            for chunk in llm.stream_chat(system_prompt, transcript):
                self.answer_chunk.emit(chunk)
            self.answer_finished.emit()
            self.status.emit("Ready")
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"{type(e).__name__}: {e}")
        finally:
            self._busy.release()
