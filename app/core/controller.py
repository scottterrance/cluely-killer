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
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Literal

from PyQt6.QtCore import QObject, pyqtSignal

from ..audio.buffer import RollingAudioBuffer
from ..config import Settings
from ..llm.base import LLMProvider
from ..prompts.builder import ExampleScheduler
from ..stt.whisper_engine import WhisperEngine
from .history import ConversationHistory


AnswerMode = Literal["short", "context"]


@dataclass
class _Speculation:
    """A background-generated 'short' answer started on a pause.

    Lifecycle: created when the interviewer pauses -> a worker thread
    streams DeepSeek chunks into ``chunks`` -> the next matching '1'
    press claims it and replays ``chunks`` (then live-streams any
    remainder) instead of starting a fresh LLM call.
    """
    base_seq: int           # _last_seq at trigger time (what's "already answered")
    target_seq: int         # the pause segment seq this answers
    transcript: str         # the question text used for generation
    cancel: threading.Event
    lock: threading.Lock
    chunks: list = field(default_factory=list)
    done: bool = False
    error: str | None = None
    full: str = ""          # set when done (joined chunks, stripped)
    ttft: float | None = None
    started_at: float = 0.0


def _friendly_error(exc: BaseException) -> str:
    """Translate common DeepSeek errors into a one-line actionable hint."""
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
        return "DeepSeek blocked the request (HTTP 403). Check your API key in Settings -> AI Provider."
    if "401" in msg or cls == "AuthenticationError" or "invalid api key" in low:
        return "DeepSeek API key is invalid or revoked. Check Settings -> AI Provider."
    if "429" in msg or cls == "RateLimitError" or "rate limit" in low or "quota" in low:
        return "DeepSeek rate limit / quota hit. Wait a moment and try again."
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
    # Ground-truth report for the answer that just ran:
    #   (stt_label, llm_label, fell_back)
    # stt_label is "local (continuous)" or "local (on-press)"; llm_label
    # is "DeepSeek". fell_back is always False in this build (no backend
    # switching) but kept so the overlay badge slot is unchanged.
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
        # Wire the pause callback so the transcriber can trigger
        # speculative pre-generation when the interviewer stops talking.
        if self.transcriber is not None:
            self.transcriber.on_segment = self.on_pause_segment
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
        # ---- Speculative pre-generation state ----
        # The in-flight / completed speculation (a pre-generated short
        # answer started on an interviewer pause). Guarded by _spec_lock.
        self._spec: _Speculation | None = None
        self._spec_lock = threading.Lock()

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
        # Cancel any in-flight speculation - the topic is being reset.
        self._cancel_spec("clear")
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

        Backlog guard: if the local model has fallen behind real time,
        the accumulated segments are STALE - reading them would answer a
        question from many seconds ago. When the backlog exceeds a small
        threshold we bypass the queue and transcribe a TIGHT most-recent
        window directly (last-utterance isolated), so the answer reflects
        the question JUST asked.
        """
        t = self.transcriber
        backlog = t.backlog_seconds()
        if backlog > 6.0:
            cap = min(self.settings.max_capture_seconds, 18.0)
            text = t.transcribe_recent(cap)
            text = self._clean_transcript(text)
            latest_seq = t.current_seq()

            def commit():
                self._last_seq = latest_seq

            label = f"continuous RECENT (backlog was {backlog:.0f}s)"
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
        text = self._clean_transcript(text)

        def commit():
            self._last_seq = latest_seq

        label = f"continuous since seq {since} ({covered:.1f}s)"
        return text, label, commit

    def _clean_transcript(self, text: str) -> str:
        """Strip any leaked Whisper biasing-prompt echo from a transcript.

        Whisper sometimes parrots the glossary prompt as if spoken
        ('Glossary, SIA, NJ, UI...'). We remove that leading junk so the
        LLM sees only the real question.
        """
        if not text:
            return text
        try:
            from ..stt.biasing import build_vocab_from_context, strip_prompt_echo

            vocab = build_vocab_from_context(
                about=self.settings.about_me,
                resume=self.settings.resume_text,
                job_desc=self.settings.job_description,
                custom=self.settings.custom_system_prompt,
            )
            return strip_prompt_echo(text, vocab)
        except Exception:
            return text

    # ---- Speculative pre-generation -----------------------------------
    def _speculation_active(self) -> bool:
        return (
            bool(self.settings.speculative_enabled)
            and self._continuous_active()
        )

    def _cancel_spec(self, reason: str = "") -> None:
        """Cancel and drop the current speculation, if any."""
        with self._spec_lock:
            spec = self._spec
            self._spec = None
        if spec is not None:
            spec.cancel.set()
            if reason:
                print(f"[spec] cancelled ({reason}) target_seq={spec.target_seq}", flush=True)

    def on_pause_segment(self, seq: int) -> None:
        """Called by the transcriber when a PAUSE-finalized segment lands
        (a likely end-of-question). Kicks off a background short-mode
        answer for everything since the last consumed segment.

        Guards: feature on, continuous active, not currently answering,
        and this seq is actually newer than what we'd answer from. A new
        pause cancels any older in-flight speculation and replaces it.
        """
        if not self._speculation_active():
            return
        # If the user is mid-answer (a press is being served), don't
        # speculate - the press path owns the flow.
        if self._busy.locked():
            return
        base_seq = self._last_seq
        # Read the same transcript a '1' press would, right now.
        t = self.transcriber
        try:
            text, latest_seq, _covered = t.snapshot_since(
                0 if base_seq < 0 else base_seq, self.settings.max_capture_seconds
            )
        except Exception:
            return
        text = self._clean_transcript(text)
        if not text or len(text.strip()) < 3:
            return

        # Supersede any older speculation.
        with self._spec_lock:
            old = self._spec
            if old is not None and not old.done and old.target_seq == latest_seq:
                # Already speculating on this exact boundary; let it run.
                return
            if old is not None:
                old.cancel.set()
            spec = _Speculation(
                base_seq=base_seq,
                target_seq=latest_seq,
                transcript=text,
                cancel=threading.Event(),
                lock=threading.Lock(),
                started_at=time.monotonic(),
            )
            self._spec = spec
        threading.Thread(
            target=self._speculate, args=(spec,), daemon=True,
            name="SpeculateWorker",
        ).start()

    def _speculate(self, spec: "_Speculation") -> None:
        """Background worker: stream a short-mode answer into ``spec``."""
        try:
            include_example = False  # keep speculation cheap/deterministic
            system_prompt = self.prompt_builder(self.settings, include_example)
            llm = self.llm_factory(self.settings)
            t0 = time.monotonic()
            for chunk in llm.stream_chat(system_prompt, spec.transcript, prior_messages=[]):
                if spec.cancel.is_set():
                    print(f"[spec] worker aborted target_seq={spec.target_seq}", flush=True)
                    return
                with spec.lock:
                    if spec.ttft is None:
                        spec.ttft = time.monotonic() - t0
                    spec.chunks.append(chunk)
            with spec.lock:
                spec.full = "".join(spec.chunks).strip()
                spec.done = True
            if not spec.cancel.is_set():
                print(
                    f"[spec] ready target_seq={spec.target_seq} "
                    f"ttft={spec.ttft if spec.ttft is None else round(spec.ttft,2)}s "
                    f"chars={len(spec.full)}",
                    flush=True,
                )
        except Exception as e:
            with spec.lock:
                spec.error = _friendly_error(e)
                spec.done = True
            traceback.print_exc()

    def _claim_spec(self, mode: AnswerMode):
        """Return a usable speculation for THIS press, or None.

        Match criteria (all must hold):
          * feature active and mode == 'short' (context mode always
            generates fresh - prior turns differ from the spec).
          * a spec exists, not cancelled, no error.
          * spec.base_seq == self._last_seq  (we haven't answered since
            it was started - so it answers the right starting point).
          * spec.target_seq == transcriber.current_seq() (no NEWER pause
            segment has landed - i.e. the interviewer didn't keep talking
            after the pause we speculated on). This guards against
            answering a stale/partial question.
        On a match the spec is detached (removed from self._spec) and
        returned; the caller drains it.
        """
        if mode != "short" or not self._speculation_active():
            return None
        with self._spec_lock:
            spec = self._spec
            if spec is None or spec.error:
                return None
            if spec.cancel.is_set():
                return None
            if spec.base_seq != self._last_seq:
                return None
            try:
                cur = self.transcriber.current_seq()
            except Exception:
                return None
            if spec.target_seq != cur:
                # A newer segment arrived after the speculated pause;
                # the question likely continued. Don't serve stale.
                return None
            # Claim it.
            self._spec = None
            return spec

    def _drain_spec(self, spec: "_Speculation"):
        """Yield the spec's chunks as they arrive (replay buffered ones,
        then live-stream the remainder until done).
        """
        idx = 0
        while True:
            with spec.lock:
                avail = len(spec.chunks)
                done = spec.done
                err = spec.error
            while idx < avail:
                yield spec.chunks[idx]
                idx += 1
            if err:
                raise RuntimeError(err)
            if done and idx >= avail:
                return
            time.sleep(0.02)

    def _do_answer(self, mode: AnswerMode) -> None:
        press_t0 = time.monotonic()
        try:
            self.status.emit("Transcribing...")
            commit_marker = None
            stt_label = "?"
            stt_t0 = time.monotonic()
            claimed_spec = None

            # FASTEST PATH: a pre-generated answer from an interviewer
            # pause is already (partly) streamed. Claim it and skip STT +
            # the fresh LLM call entirely.
            claimed_spec = self._claim_spec(mode)
            if claimed_spec is not None:
                transcript = claimed_spec.transcript
                source_label = f"SPECULATED target_seq={claimed_spec.target_seq}"
                stt_label = "local (continuous, pre-gen)"
                captured_seconds = 0.0
                _target_seq = claimed_spec.target_seq

                def commit_marker():  # noqa: F811 - intentional rebinding
                    self._last_seq = _target_seq

                print(f"[spec] CLAIMED pre-generated answer target_seq={_target_seq}", flush=True)
            elif self._continuous_active():
                # Phase 2 fast path: transcript already exists.
                transcript, source_label, commit_marker = self._transcript_continuous()
                if not transcript:
                    self.error.emit(
                        "Not enough new speech captured yet - let the "
                        "interviewer talk a moment, then press again."
                    )
                    return
                captured_seconds = 0.0  # STT happened in the background
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
                # isolate_last: transcribe only the interviewer's last
                # question (~5-8s) instead of the whole window - the big
                # local-STT latency win.
                transcript = self.whisper.transcribe(audio, isolate_last=True)
                transcript = self._clean_transcript(transcript)
                if not transcript:
                    self.error.emit("No speech detected in the captured window.")
                    return
                stt_label = "local (on-press)"

            self.transcript_ready.emit(transcript)
            stt_elapsed = time.monotonic() - stt_t0
            print(
                f"[answer] mode={mode} source={source_label} "
                f"STT-stage={stt_elapsed:.2f}s transcript={transcript[:200]!r}",
                flush=True,
            )

            self.status.emit(f"Thinking ({mode})...")
            include_example = self.scheduler.should_include()
            system_prompt = self.prompt_builder(self.settings, include_example)
            # The mode picks whether prior turns are shipped to the LLM.
            # ``short``  -> isolated answer, no follow-up gravity.
            # ``context`` -> last 5 Q+A pairs as chat history.
            prior = self.history.as_messages() if mode == "context" else []
            print(
                f"[answer] mode={mode} history_turns_sent={len(prior)//2} "
                f"speculated={claimed_spec is not None}",
                flush=True,
            )

            # If we did NOT claim a spec, any in-flight speculation is now
            # stale (this fresh answer supersedes it). Cancel it so its
            # worker thread stops and can't be claimed by a later press.
            if claimed_spec is None:
                self._cancel_spec("superseded by fresh press")

            self.answer_started.emit()
            chunks: list[str] = []
            err_msg: str | None = None
            # Timing receipt for the LLM stage. ttft = time to FIRST token
            # (network + DeepSeek queue + prefill); total = full answer.
            llm_t0 = time.monotonic()
            ttft: float | None = None
            try:
                if claimed_spec is not None:
                    # Replay the pre-generated chunks (instant), then
                    # live-stream any remainder until the worker finishes.
                    stream = self._drain_spec(claimed_spec)
                else:
                    llm = self.llm_factory(self.settings)
                    stream = llm.stream_chat(
                        system_prompt, transcript, prior_messages=prior
                    )
                for chunk in stream:
                    if ttft is None:
                        ttft = time.monotonic() - llm_t0
                    chunks.append(chunk)
                    self.answer_chunk.emit(chunk)
            except Exception as e:
                traceback.print_exc()
                err_msg = _friendly_error(e)
            llm_total = time.monotonic() - llm_t0

            full = "".join(chunks).strip()
            llm_label = "DeepSeek"
            print(
                f"[answer] streamed {len(chunks)} chunks, "
                f"{len(full)} chars, err={err_msg!r}",
                flush=True,
            )
            print(
                f"[deepseek] TTFT {ttft if ttft is None else round(ttft, 2)}s | "
                f"full answer {llm_total:.2f}s",
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
                    "\u26a0  DeepSeek returned an empty response. "
                    "Check your API key / quota / model name in Settings -> AI Provider."
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

            # Report which engines ran (always local STT + DeepSeek in
            # this build). fell_back is always False - kept for the
            # overlay badge slot's signature.
            self.backend_used.emit(stt_label, llm_label, False)
            total_elapsed = time.monotonic() - press_t0
            print(
                f"[answer] BACKENDS USED -> STT: {stt_label} | LLM: {llm_label}",
                flush=True,
            )
            print(
                f"[TIMING] press->done {total_elapsed:.2f}s "
                f"= STT {stt_elapsed:.2f}s + LLM {llm_total:.2f}s "
                f"(+overhead {max(0.0, total_elapsed - stt_elapsed - llm_total):.2f}s)  "
                f">>> the bigger number is your bottleneck <<<",
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
