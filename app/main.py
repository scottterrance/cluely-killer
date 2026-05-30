"""Application bootstrap.

Wires together: settings -> audio capture -> Whisper -> DeepSeek ->
controller -> overlay -> hotkeys -> stealth.
"""
from __future__ import annotations

import os
import sys


def _say(msg: str) -> None:
    print(f"[startup] {msg}", flush=True)


def main() -> None:
    no_stealth = "--no-stealth" in sys.argv
    reset_window = "--reset-window" in sys.argv
    simple_mode = "--simple" in sys.argv

    _say(f"argv = {sys.argv}")
    _say(f"python = {sys.version.split()[0]}  exe = {sys.executable}")

    # On Windows, force COM into STA on the main thread BEFORE creating
    # QApplication. Some audio libs initialize COM as MTA on first import,
    # which causes Qt's OleInitialize to fail with 0x80010106.
    if sys.platform == "win32":
        try:
            import ctypes
            COINIT_APARTMENTTHREADED = 0x2
            ctypes.windll.ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
            _say("COM initialized as STA on main thread.")
        except Exception as e:
            _say(f"COM init skipped: {e}")

    _say("loading .env and settings...")
    from dotenv import load_dotenv

    from .config import load_settings, save_settings

    load_dotenv()
    settings = load_settings()
    if no_stealth:
        settings.exclude_from_capture = False
        _say("--no-stealth: WDA_EXCLUDEFROMCAPTURE will NOT be applied this session.")
    if reset_window:
        settings.window_x = -1
        settings.window_y = -1
        _say("--reset-window: forcing center of primary screen.")
    if simple_mode:
        _say("--simple: using a normal titled window.")

    # API keys from .env if present (only loaded if the user hasn't
    # already entered one in Settings).
    ds_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if ds_key and not settings.deepseek_api_key:
        settings.deepseek_api_key = ds_key
        save_settings(settings)
        _say("loaded DEEPSEEK_API_KEY from .env")
    ds_base = os.getenv("DEEPSEEK_BASE_URL", "").strip()
    if ds_base:
        settings.deepseek_base_url = ds_base
        _say(f"DEEPSEEK_BASE_URL override = {ds_base!r}")
    ds_model_env = os.getenv("DEEPSEEK_MODEL", "").strip()
    if ds_model_env:
        settings.deepseek_model = ds_model_env
        _say(f"DEEPSEEK_MODEL override = {ds_model_env!r}")

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and not settings.groq_api_key:
        settings.groq_api_key = groq_key
        save_settings(settings)
        _say("loaded GROQ_API_KEY from .env")
    groq_model_env = os.getenv("GROQ_MODEL", "").strip()
    if groq_model_env:
        settings.groq_model = groq_model_env
        _say(f"GROQ_MODEL override = {groq_model_env!r}")

    _say("creating QApplication...")
    from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPixmap
    from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

    class HotkeyDispatcher(QObject):
        answer_short_requested = pyqtSignal()
        answer_context_requested = pyqtSignal()
        toggle_requested = pyqtSignal()
        clear_requested = pyqtSignal()
        settings_requested = pyqtSignal()
        quit_requested = pyqtSignal()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("cluely-killer")

    import signal

    def _on_sigint(*_args):
        print("\n[shutdown] Ctrl+C received, quitting...", flush=True)
        QApplication.instance().quit()

    signal.signal(signal.SIGINT, _on_sigint)
    _signal_kick = QTimer()
    _signal_kick.start(200)
    _signal_kick.timeout.connect(lambda: None)

    _say("QApplication created.")

    # ---- Audio pipeline (always on) ----
    _say("starting WASAPI loopback capture thread...")
    from .audio.buffer import RollingAudioBuffer
    from .audio.loopback import LoopbackCapture

    buffer = RollingAudioBuffer(
        samplerate=16000,
        # The rolling buffer must be at least max_capture_seconds long,
        # otherwise audio that arrives between two answer presses could
        # be evicted before the next press reads it. config.py already
        # migrates persisted values, but enforce it here too as a
        # belt-and-braces guard for in-memory edits.
        max_seconds=max(
            settings.buffer_seconds,
            settings.max_capture_seconds + 5,
        ),
    )
    capture = LoopbackCapture(buffer, samplerate=16000)
    capture.start()
    _say("audio capture started.")

    # ---- STT (router: cloud Groq turbo OR local faster-whisper) ----
    _say(
        f"initializing STT (backend={settings.stt_backend!r}; "
        f"local model '{settings.whisper_model}', cloud '{settings.groq_stt_model}')..."
    )
    from .stt.router import STTRouter

    whisper = STTRouter(settings)
    # Eagerly build the primary backend so model-load / key errors show
    # up at startup (and the local model warms up) rather than on the
    # first hotkey press. Failure here is non-fatal: the router will try
    # the other backend on first use. We warm the first backend that can
    # actually be built so a cloud-primary user with no local model (or
    # vice-versa) still gets a warm engine.
    for _b in whisper._order():
        try:
            whisper._engine(_b)
            _say(f"STT backend '{_b}' ready (primary warm).")
            break
        except Exception as e:
            _say(f"STT backend '{_b}' not ready ({e}); trying next...")

    # ---- Continuous background transcription (Phase 2) ----
    # A daemon thread transcribes the audio buffer as the interviewer
    # talks, using the LOCAL whisper model ONLY (never the metered cloud
    # STT). On a hotkey press the answer reads this pre-built transcript,
    # so Whisper is off the press critical path. If the local model
    # isn't available we disable continuous mode and fall back to the
    # classic transcribe-on-press path.
    transcriber = None
    if settings.continuous_stt:
        try:
            local_engine = whisper._get_local()
            from .stt.continuous import ContinuousTranscriber

            transcriber = ContinuousTranscriber(
                buffer=buffer, engine=local_engine, samplerate=16000
            )
            transcriber.start()
            _say("continuous STT ON (background transcription via local model).")
        except Exception as e:
            settings.continuous_stt = False
            _say(
                f"continuous STT unavailable ({e}); using classic "
                f"transcribe-on-press path instead."
            )
    else:
        _say("continuous STT OFF (classic transcribe-on-press path).")

    # Bias Whisper toward the candidate's own vocabulary (names, tech,
    # company terms) extracted from resume / JD / about-me. Big accuracy
    # win on exactly the words interviews get wrong.
    from .stt.biasing import build_vocab_from_context

    def _refresh_whisper_bias() -> None:
        if not settings.stt_bias_enabled:
            whisper.set_bias([])
            if transcriber is not None:
                transcriber.set_bias([])
            _say("STT keyword biasing DISABLED (stt_bias_enabled=False).")
            return
        vocab = build_vocab_from_context(
            about=settings.about_me,
            resume=settings.resume_text,
            job_desc=settings.job_description,
            custom=settings.custom_system_prompt,
        )
        whisper.set_bias(vocab)
        # Keep the background transcriber's engine biased too (it holds
        # its own reference to the local engine).
        if transcriber is not None:
            transcriber.set_bias(vocab)

    _refresh_whisper_bias()

    # ---- Orchestration ----
    _say("wiring controller, prompts, providers...")
    from .core.controller import Controller
    from .core.history import ConversationHistory
    from .hotkeys.manager import HotkeyManager
    from .llm.base import LLMProvider
    from .llm.router import LLMRouter
    from .prompts.builder import ExampleScheduler, build_system_prompt
    from .stealth.windows import exclude_window_from_capture
    from .ui.overlay import OverlayWindow
    from .ui.settings_dialog import SettingsDialog

    def _llm_factory(s) -> LLMProvider:
        # Router picks Groq or DeepSeek per s.llm_backend and falls back
        # to the other automatically if the primary errors before the
        # first token (e.g. Groq free-tier tokens exhausted).
        return LLMRouter(s)

    def _prompt_for(s, include_example: bool) -> str:
        return build_system_prompt(
            resume=s.resume_text,
            job_desc=s.job_description,
            about=s.about_me,
            custom=s.custom_system_prompt,
            include_example=include_example,
        )

    scheduler = ExampleScheduler()
    history = ConversationHistory(max_turns=5)
    controller = Controller(
        settings=settings,
        audio_buffer=buffer,
        whisper=whisper,
        llm_factory=_llm_factory,
        scheduler=scheduler,
        prompt_builder=_prompt_for,
        history=history,
        transcriber=transcriber,
    )
    _say("controller ready (memory keeps last 5 Q+A turns).")

    # ---- UI ----
    _say("building overlay window...")
    hotkeys = HotkeyManager()
    overlay: OverlayWindow

    def open_settings_dialog() -> None:
        dlg = SettingsDialog(settings, parent=overlay)
        if dlg.exec():
            save_settings(settings)
            overlay.setWindowOpacity(settings.opacity)
            ok = exclude_window_from_capture(
                int(overlay.winId()), settings.exclude_from_capture
            )
            overlay.update_stealth_badge(settings.exclude_from_capture and ok)
            overlay.refresh_footer()
            apply_hotkeys()
            # Resume / JD / about-me may have changed -> rebuild the
            # Whisper biasing vocabulary so STT accuracy tracks the new
            # context immediately (no restart needed).
            # Also drop the cached cloud STT client so a new Groq key /
            # model / STT backend selection takes effect on the next press.
            whisper.invalidate()
            _refresh_whisper_bias()
            # Apply a live toggle of continuous transcription (start/stop
            # the background thread to match the new checkbox state).
            controller.apply_continuous_setting()

    overlay = OverlayWindow(
        settings,
        controller,
        on_open_settings=open_settings_dialog,
        on_quit=lambda: app.quit(),
        simple_mode=simple_mode,
    )
    _say("overlay constructed; calling show()...")
    overlay.show()
    _say("overlay.show() returned; placing on screen...")
    overlay.place_on_screen()

    stealth_active = False
    if settings.exclude_from_capture:
        ok = exclude_window_from_capture(int(overlay.winId()), True)
        if not ok and sys.platform == "win32":
            _say("WDA_EXCLUDEFROMCAPTURE failed - needs Windows 10 build 19041+.")
        else:
            stealth_active = bool(ok)
            _say(
                "stealth ON: window is hidden from screen-capture APIs. "
                "If you cannot see it locally either, run with --no-stealth."
            )
    else:
        _say("stealth OFF: window is visible to screen-capture too.")
    overlay.update_stealth_badge(stealth_active)

    # ---- Hotkeys (with cross-thread marshalling) ----
    dispatcher = HotkeyDispatcher()
    qc = Qt.ConnectionType.QueuedConnection
    # Two answer keys: short = no history, context = last 5 Q+A as memory.
    # The lambdas wrap controller.trigger_answer so it can be invoked with
    # the right mode argument from the queued-connection slot.
    dispatcher.answer_short_requested.connect(
        lambda: controller.trigger_answer("short"), qc
    )
    dispatcher.answer_context_requested.connect(
        lambda: controller.trigger_answer("context"), qc
    )
    dispatcher.toggle_requested.connect(lambda: overlay.toggle_visibility(), qc)
    dispatcher.clear_requested.connect(controller.clear, qc)
    dispatcher.settings_requested.connect(open_settings_dialog, qc)
    dispatcher.quit_requested.connect(app.quit, qc)

    def apply_hotkeys() -> None:
        hotkeys.set_hotkeys({
            settings.hotkey_answer_short: dispatcher.answer_short_requested.emit,
            settings.hotkey_answer_context: dispatcher.answer_context_requested.emit,
            settings.hotkey_toggle: dispatcher.toggle_requested.emit,
            settings.hotkey_clear: dispatcher.clear_requested.emit,
            settings.hotkey_settings: dispatcher.settings_requested.emit,
            settings.hotkey_quit: dispatcher.quit_requested.emit,
        })

    apply_hotkeys()
    _say("hotkeys registered.")

    # ---- System tray icon ----
    def _make_tray_icon() -> QIcon:
        pix = QPixmap(64, 64)
        pix.fill(Qt.GlobalColor.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(QColor("#7CC8FF")))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(6, 6, 52, 52)
        p.setPen(QColor("#0F0F16"))
        f = QFont(); f.setPointSize(28); f.setBold(True); p.setFont(f)
        p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "C")
        p.end()
        return QIcon(pix)

    tray = None
    if QSystemTrayIcon.isSystemTrayAvailable():
        tray = QSystemTrayIcon(_make_tray_icon())
        tray.setToolTip("cluely-killer")
        tray_menu = QMenu()
        tray_menu.addAction("Show / hide overlay", overlay.toggle_visibility)
        tray_menu.addAction("Settings...", open_settings_dialog)
        tray_menu.addSeparator()
        tray_menu.addAction("Quit", app.quit)
        tray.setContextMenu(tray_menu)

        def _on_tray_activated(reason):
            if reason == QSystemTrayIcon.ActivationReason.Trigger:
                overlay.toggle_visibility()

        tray.activated.connect(_on_tray_activated)
        tray.show()
        _say("system tray icon added.")
    else:
        _say("system tray not available on this platform.")

    if capture.last_error:
        controller.error.emit(f"Audio: {capture.last_error}")
        _say(f"audio thread reported: {capture.last_error}")

    def cleanup() -> None:
        hotkeys.stop()
        capture.stop()
        if transcriber is not None:
            transcriber.stop()

    app.aboutToQuit.connect(cleanup)

    _say("entering Qt event loop. The window should be visible now.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
