"""Orchestrates: hotkey -> audio slice -> Whisper -> LLM stream -> UI signals.

The Controller lives on the main thread but offloads the heavy work
(STT + LLM) to a worker thread so the UI never freezes. UI updates
are delivered via Qt signals, which are queued safely across threads.

Two answer modes
----------------
``trigger_answer(mode)`` accepts:

  * ``"short"``   - send Whisper-transcribed text to the LLM with NO
    prior conversation context. Quick, self-contained answers.
  * ``"context"`` - send the same transcript along with the last 5
    Q+A pairs from ``ConversationHistory``. Use for follow-up
    questions ("elaborate on that", "what about edge cases?").

In both modes the audio that gets transcribed is "everything captured
since the last successful answer (in either mode), capped at
``settings.max_capture_seconds``". This is the position-marker
behavior implemented in :class:`RollingAudioBuffer`.
"""
from __future__ import annotations

import threading
import traceback
from typing import Callable, Literal

from PyQt6.QtCore import QObject, pyqtSignal

from ..audio.buffer import RollingAudioBuffer
from ..config import Settings
from ..llm.base import LLMProvider
from ..prompts.builder import ExampleScheduler
from ..stt.whisper_engine import WhisperEngine
from .history import ConversationHistory


AnswerMode = Literal["short", "context"]


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
            "DNS lookup failed - cannot reach the AI provider. "
            "Check your WiFi / corporate firewall."
        )
    if "403" in msg or cls == "PermissionDeniedError" or "access denied" in low:
        return "Provider blocked the request (HTTP 403). Check your API key in Settings -> AI Provider."
    if "401" in msg or cls == "AuthenticationError" or "invalid api key" in low:
        return "API key is invalid or revoked. Check Settings -> AI Provider."
    if "429" in msg or cls == "RateLimitError" or "rate limit" in low or "quota" in low:
        return (
            "Rate limit / out of tokens. If you're on Groq's free tier you may have "
            "run out - switch the LLM (and STT) backend to DeepSeek/local in "
            "Settings -> AI Provider, or wait and retry."
        )
    if (
        cls in ("APIConnectionError", "ConnectionError", "ConnectError",
                "ConnectTimeout", "ReadTimeout", "TimeoutException")
        or "connection" in low
        or "timed out" in low
    ):
        return "Network error reaching the AI provider. Check your internet."
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
    # Ground-truth backend report for the answer that just ran:
    #   (stt_label, llm_label, fell_back)
    # This reflects what ACTUALLY served the answer - including silent
    # fallback (Groq out of tokens -> DeepSeek) and the fact that
    # continuous mode always transcribes locally - NOT merely what's
    # selected in Settings. ``fell_back`` is True when the engine that
    # ran differs from the one the user selected.
    backend_used = pyqtSignal(str, str, bool)

    def __init__(
        self,
        settings: Settings,
        audio_buffer: RollingAudioBuffer,
        whisper: WhisperEngine,
        llm_factory: Callable[[Settings], LLMProvider],
        scheduler: ExampleScheduler,
        prompt_builder: Callable[[Settings, bool], str],
        history: ConversationHistory,
        transcriber=None,
    ):
        super().__init__()
        self.settings = settings
        self.audio_buffer = audio_buffer
        self.whisper = whisper
        self.llm_factory = llm_factory
        self.scheduler = scheduler
        self.prompt_builder = prompt_builder
        self.history = history
        # Optional ContinuousTranscriber (Phase 2). When present AND
        # settings.continuous_stt is on, answers read the pre-built
        # background transcript instead of transcribing on the press.
        self.transcriber = transcriber
        self._busy = threading.Lock()
        # Sample-position marker: "last sample we already consumed in a
        # successful answer". Initialized to -1 so the very first press
        # falls back to ``answer_window_seconds`` instead of trying to
        # transcribe the entire current buffer (which on app start is
        # mostly silence anyway). Used by the classic on-press path.
        self._last_marker: int = -1
        # Seq marker for the continuous path: "seq id of the last
        # transcript segment already consumed". -1 means "first press,
        # take everything currently buffered".
        self._last_seq: int = -1

    def _continuous_active(self) -> bool:
        return bool(self.transcriber) and bool(self.settings.continuous_stt)

    def apply_continuous_setting(self) -> None:
        """Start or stop the background transcriber to match the current
        ``settings.continuous_stt`` flag. Called after the Settings
        dialog is saved so toggling the checkbox takes effect live.

        Only acts if a transcriber object exists (it's created at launch
        from the local model). If the local model was unavailable at
        launch, transcriber is None and we can't turn it on mid-session -
        the app stays on the on-press path until restart.
        """
        t = self.transcriber
        if t is None:
            return
        want = bool(self.settings.continuous_stt)
        running = getattr(t, "_thread", None) is not None
        if want and not running:
            # Re-arm the seq marker to "now" so we don't replay stale
            # segments captured before it was paused, then start.
            t.reset()
            self._last_seq = t.current_seq()
            t.start()
            self.status.emit("Continuous transcription ON")
        elif not want and running:
            t.stop()
            self.status.emit("Continuous transcription OFF (transcribe on press)")

    # ------------------------------------------------------------------
    def trigger_answer(self, mode: AnswerMode = "short") -> None:
        """Hotkey entry point. Non-blocking; safe to call from any thread.

        ``mode``:
          - ``"short"``   - no prior Q+A context attached to the LLM call.
          - ``"context"`` - last 5 Q+A pairs attached as chat memory.
        """
        if mode not in ("short", "context"):
            mode = "short"
        if not self._busy.acquire(blocking=False):
            self.status.emit("Busy - wait for current answer to finish")
            return
        threading.Thread(
            target=self._do_answer,
            args=(mode,),
            daemon=True,
            name=f"AnswerWorker-{mode}",
        ).start()

    def clear(self) -> None:
        self.audio_buffer.clear()
        self.history.clear()
        # Reset to "next press starts a fresh capture window" by
        # snapping the marker to the buffer's current position. This
        # ensures the first press AFTER a clear sees only audio that
        # arrived after the clear, never any pre-clear leftovers from
        # before _total_appended advanced past the wipe point.
        self._last_marker = self.audio_buffer.current_position()
        # Continuous path: drop all background segments and re-arm the
        # seq marker so the next press only sees post-clear speech.
        if self.transcriber is not None:
            self.transcriber.reset()
            self._last_seq = self.transcriber.current_seq()
        self.history_changed.emit(0)
        self.status.emit("Audio buffer + memory cleared")

    # ------------------------------------------------------------------
    def _grab_audio(self):
        """Return (audio_np, source_label) per the marker rules.

        Classic on-press path only (continuous mode bypasses this and
        reads pre-built text instead).
        """
        if self._last_marker < 0:
            # First press of the session: no marker yet. Fall back to
            # the "classic" last-N-seconds slice so users don't have to
            # press once just to "arm" the marker.
            audio = self.audio_buffer.get_last_seconds(
                self.settings.answer_window_seconds
            )
            return audio, f"last {self.settings.answer_window_seconds:.0f}s (first press)"
        audio = self.audio_buffer.get_since_position(
            self._last_marker, self.settings.max_capture_seconds
        )
        return audio, "since-last-press"

    def _transcript_continuous(self):
        """Continuous path: read the background transcript.

        Returns (transcript, source_label, commit) where ``commit`` is a
        zero-arg callable that advances the seq marker - called only on a
        successful answer, mirroring the on-press path's marker advance.

        Backlog guard: if the local model has fallen behind real time
        (common on slower CPUs with large-v3-turbo), the accumulated
        segments are STALE - reading them would answer a question from
        many seconds ago. When the backlog exceeds a small threshold we
        bypass the queue and transcribe the most-recent window directly
        on this thread, so the answer reflects what was JUST said.
        """
        t = self.transcriber
        backlog = t.backlog_seconds()
        # Threshold: a couple of poll cycles + one chunk. Above this the
        # queue is meaningfully behind the present.
        if backlog > 6.0:
            cap = min(self.settings.max_capture_seconds, 30.0)
            # Prefer a FAST engine for this synchronous press-time call.
            # If a Groq key is configured, Groq cloud STT (216x realtime)
            # turns this into a sub-second call instead of blocking on the
            # slow local model that already fell behind. Falls back to the
            # local background engine if cloud can't be built.
            fast = None
            try:
                if self.settings.groq_api_key and hasattr(self.whisper, "_get_cloud"):
                    fast = self.whisper._get_cloud()
            except Exception:
                fast = None
            text = t.transcribe_recent(cap, engine=fast)
            latest_seq = t.current_seq()

            def commit():
                self._last_seq = latest_seq

            via = "cloud" if fast is not None else "local"
            label = f"continuous RECENT via {via} (backlog was {backlog:.0f}s)"
            return text, label, commit

        # Normal path: nudge the worker to flush the last partial chunk
        # (words said in the ~2s before the press that haven't hit a
        # natural pause yet), then read everything since our last marker.
        try:
            t.flush_now(timeout=1.5)
        except Exception:
            pass
        since = -1 if self._last_seq < 0 else self._last_seq
        text, latest_seq, covered = t.snapshot_since(
            0 if since < 0 else since, self.settings.max_capture_seconds
        )

        def commit():
            self._last_seq = latest_seq

        label = f"continuous since seq {since} ({covered:.1f}s)"
        return text, label, commit

    def _do_answer(self, mode: AnswerMode) -> None:
        try:
            self.status.emit("Transcribing...")
            commit_marker = None
            stt_label = "?"

            if self._continuous_active():
                # Phase 2 fast path: transcript already exists.
                transcript, source_label, commit_marker = self._transcript_continuous()
                if not transcript:
                    self.error.emit(
                        "Not enough new speech captured yet - let the "
                        "interviewer talk a moment, then press again."
                    )
                    return
                captured_seconds = 0.0  # STT happened in the background
                # Continuous transcription ALWAYS runs on the local model,
                # regardless of the cloud/local setting - so report it
                # honestly as local (background).
                stt_label = "local (continuous)"
            else:
                # Classic on-press path.
                audio, source_label = self._grab_audio()
                if audio.size < self.whisper.samplerate:  # less than 1 second
                    self.error.emit(
                        "Not enough new audio yet - let the interviewer talk first."
                    )
                    return
                captured_seconds = audio.size / self.whisper.samplerate
                transcript = self.whisper.transcribe(audio)
                if not transcript:
                    self.error.emit("No speech detected in the captured window.")
                    return
                # Ground truth: which STT engine actually transcribed it
                # (STTRouter sets last_used; may differ from the setting
                # if the primary backend failed and it fell back).
                used = getattr(self.whisper, "last_used", "") or self.settings.stt_backend
                stt_label = "cloud (Groq)" if used == "cloud" else "local"

            self.transcript_ready.emit(transcript)
            print(
                f"[answer] mode={mode} source={source_label} "
                f"transcript={transcript[:200]!r}",
                flush=True,
            )

            self.status.emit(f"Thinking ({mode})...")
            include_example = self.scheduler.should_include()
            system_prompt = self.prompt_builder(self.settings, include_example)
            llm = self.llm_factory(self.settings)
            # The mode picks whether prior turns are shipped to the LLM.
            # ``short``  -> isolated answer, no follow-up gravity.
            # ``context`` -> last 5 Q+A pairs as chat history.
            prior = self.history.as_messages() if mode == "context" else []
            print(
                f"[answer] llm_backend={self.settings.llm_backend} "
                f"stt_backend={self.settings.stt_backend} mode={mode} "
                f"history_turns_sent={len(prior)//2}",
                flush=True,
            )

            self.answer_started.emit()
            chunks: list[str] = []
            err_msg: str | None = None
            try:
                for chunk in llm.stream_chat(
                    system_prompt, transcript, prior_messages=prior
                ):
                    chunks.append(chunk)
                    self.answer_chunk.emit(chunk)
            except Exception as e:
                traceback.print_exc()
                err_msg = _friendly_error(e)

            full = "".join(chunks).strip()
            # Ground truth: which LLM actually produced the answer. The
            # LLMRouter sets last_used to the provider that streamed the
            # first token (may be the fallback if the primary errored).
            llm_used = getattr(llm, "last_used", "") or self.settings.llm_backend
            llm_label = "Groq" if llm_used == "groq" else "DeepSeek"
            print(
                f"[answer] streamed {len(chunks)} chunks, "
                f"{len(full)} chars, err={err_msg!r}",
                flush=True,
            )

            # Decide what to surface. Critical: never leave the answer
            # panel silently empty - the candidate has no idea what
            # happened otherwise.
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
                    "clear question. Press '1' or '2' again right after the interviewer "
                    "finishes asking, or press Ctrl+R to reset."
                )

            if fallback:
                separator = "\n\n" if chunks else ""
                self.answer_chunk.emit(separator + fallback)

            self.answer_finished.emit()

            # Emit the ground-truth backend report so the overlay badge
            # reflects what ACTUALLY ran (incl. silent fallback), not just
            # the Settings selection. Compute whether either engine
            # differs from what the user selected.
            stt_fell_back = (
                not self._continuous_active()
                and (getattr(self.whisper, "last_used", "") or self.settings.stt_backend)
                != self.settings.stt_backend
            )
            llm_fell_back = (
                (getattr(llm, "last_used", "") or self.settings.llm_backend)
                != self.settings.llm_backend
            )
            self.backend_used.emit(stt_label, llm_label, stt_fell_back or llm_fell_back)
            print(
                f"[answer] BACKENDS USED -> STT: {stt_label} | LLM: {llm_label}"
                + ("  (FELL BACK from selection)" if (stt_fell_back or llm_fell_back) else ""),
                flush=True,
            )

            success = not err_msg and full and full.upper() != "SKIP"
            if success:
                # Advance the marker BEFORE writing to history. From the
                # interviewer's clock perspective, "this question is
                # done" the moment our transcribe call finished; any
                # audio still arriving after this is the START of the
                # next question.
                if commit_marker is not None:
                    # Continuous path: advance the seq marker.
                    commit_marker()
                else:
                    # Classic path: advance the audio-position marker.
                    self._last_marker = self.audio_buffer.current_position()
                # Both modes feed history. The difference between
                # modes is whether we *read* history on the way in,
                # not whether we write to it on the way out - the
                # background memory must keep growing so future
                # ``context`` presses can rely on it.
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
