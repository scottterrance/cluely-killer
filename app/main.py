"""Application bootstrap.

Wires together: settings -> audio capture -> Whisper -> LLM factory ->
controller -> overlay -> hotkeys -> stealth.

Heavily-instrumented startup: every step prints a flushed marker so a
silent crash leaves an obvious last-known-good line in the terminal.
"""
from __future__ import annotations

import os
import sys


def _say(msg: str) -> None:
    print(f"[startup] {msg}", flush=True)


def _argv_value(name: str) -> str | None:
    """Return value for --name VAL or --name=VAL, or None."""
    for i, arg in enumerate(sys.argv):
        if arg.startswith(f"{name}="):
            return arg.split("=", 1)[1]
        if arg == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def main() -> None:
    # ----- Debug CLI flags -----
    no_stealth = "--no-stealth" in sys.argv
    reset_window = "--reset-window" in sys.argv
    simple_mode = "--simple" in sys.argv
    whisper_override = _argv_value("--whisper-model")

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
    if whisper_override:
        _say(f"--whisper-model: overriding to '{whisper_override}' for this session.")
        settings.whisper_model = whisper_override
    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key and not settings.groq_api_key:
        settings.groq_api_key = env_key
        save_settings(settings)
        _say("loaded GROQ_API_KEY from .env")
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if or_key and not settings.openrouter_api_key:
        settings.openrouter_api_key = or_key
        save_settings(settings)
        _say("loaded OPENROUTER_API_KEY from .env")
    ds_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if ds_key and not settings.deepseek_api_key:
        settings.deepseek_api_key = ds_key
        save_settings(settings)
        _say("loaded DEEPSEEK_API_KEY from .env")
    # Optional .env overrides for a custom DeepSeek endpoint / model.
    ds_base = os.getenv("DEEPSEEK_BASE_URL", "").strip()
    if ds_base:
        settings.deepseek_base_url = ds_base
        _say(f"DEEPSEEK_BASE_URL override = {ds_base!r}")
    ds_model_env = os.getenv("DEEPSEEK_MODEL", "").strip()
    if ds_model_env:
        settings.deepseek_model = ds_model_env
        _say(f"DEEPSEEK_MODEL override = {ds_model_env!r}")

    _say("creating QApplication...")
    from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPixmap
    from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

    # ----- Cross-thread bridge for hotkeys -----
    # pynput's GlobalHotKeys runs callbacks on its OWN listener thread.
    # Calling Qt UI methods (open dialog, show/hide window, ...) from a
    # non-GUI thread is undefined behavior and was causing the app to
    # freeze after a few hotkey presses. We route every hotkey through
    # this QObject's signals; because the QObject lives on the main
    # thread, Qt automatically uses a queued connection and the actual
    # handlers run on the GUI thread where they belong.
    class HotkeyDispatcher(QObject):
        answer_requested = pyqtSignal()
        toggle_requested = pyqtSignal()
        clear_requested = pyqtSignal()
        settings_requested = pyqtSignal()
        quit_requested = pyqtSignal()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    # Use the user-configurable display name everywhere it surfaces. By
    # default this is "cluely-killer" but the user can rename it to
    # something innocuous (e.g. "Notepad") so a peek at the taskbar /
    # Task Manager doesn't out them.
    app.setApplicationName(settings.app_display_name or "cluely-killer")

    # Make Ctrl+C in the terminal actually quit the app. Qt's C++ event
    # loop blocks Python signal delivery; the QTimer below wakes Python
    # every 200 ms so the SIGINT handler can run.
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

    buffer = RollingAudioBuffer(samplerate=16000, max_seconds=settings.buffer_seconds)
    capture = LoopbackCapture(buffer, samplerate=16000)
    capture.start()
    _say("audio capture started.")

    # ---- STT ----
    _say(
        f"loading faster-whisper '{settings.whisper_model}' "
        f"({settings.whisper_compute} on {settings.whisper_device})..."
    )
    _say("FIRST RUN downloads ~466 MB from Hugging Face. Wait 2-5 min.")
    _say("Do NOT press Ctrl+C during this step.")
    from .stt.whisper_engine import WhisperEngine

    whisper = WhisperEngine(
        model_size=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute,
    )
    _say("Whisper model loaded.")

    # ---- Orchestration ----
    _say("wiring controller, prompts, providers...")
    from .core.controller import Controller
    from .core.history import ConversationHistory
    from .hotkeys.manager import HotkeyManager
    from .llm.base import LLMProvider
    from .llm.deepseek_provider import DeepSeekProvider
    from .llm.groq_provider import GroqProvider
    from .llm.ollama_provider import OllamaProvider
    from .llm.openrouter_provider import OpenRouterProvider
    from .prompts.builder import ExampleScheduler, build_system_prompt
    from .stealth.windows import exclude_window_from_capture
    from .ui.overlay import OverlayWindow
    from .ui.settings_dialog import SettingsDialog

    def _llm_factory(s) -> LLMProvider:
        if s.provider == "ollama":
            return OllamaProvider(model=s.ollama_model, host=s.ollama_host)
        if s.provider == "openrouter":
            return OpenRouterProvider(api_key=s.openrouter_api_key, model=s.openrouter_model)
        if s.provider == "deepseek":
            return DeepSeekProvider(
                api_key=s.deepseek_api_key,
                model=s.deepseek_model,
                base_url=s.deepseek_base_url,
            )
        return GroqProvider(api_key=s.groq_api_key, model=s.groq_model)

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
    )
    _say("controller ready (memory keeps last 5 Q+A turns).")

    # ---- UI ----
    _say("building overlay window...")
    hotkeys = HotkeyManager()
    overlay: OverlayWindow

    # Re-entrance + hotkey-suspend guard for the Settings dialog.
    #
    # Win32 RegisterHotKey is GLOBAL: it consumes the keystroke even when
    # a modal dialog is on screen. While the user is typing into the
    # Settings dialog (API key, job description, etc.), any accidental
    # press of one of our hotkeys would still fire:
    #   - Ctrl+Shift+S -> opens ANOTHER nested settings dialog
    #   - Ctrl+Space   -> kicks off a 25-second AnswerWorker in the
    #                     background (and we've seen those rate-limit
    #                     and explode mid-stream)
    #   - Ctrl+Shift+Q -> quits the app outright while user is mid-edit
    #   - Ctrl+R       -> wipes the audio buffer + memory
    #   - Ctrl+\       -> hides the overlay, taking the dialog with it
    # Any of those leaves the app looking "stunned" and forces a kill.
    #
    # Fix: while a Settings dialog is on screen, fully UNREGISTER all
    # global hotkeys. Re-register only after the dialog is gone. The
    # _settings_open flag is a defensive second layer in case Qt ever
    # re-delivers a queued settings_requested signal while the previous
    # dialog hasn't finished tearing down.
    _settings_open = {"flag": False}

    def open_settings_dialog() -> None:
        if _settings_open["flag"]:
            # Already open (e.g. queued duplicate signal arrived during
            # the dialog's modal exec). Bring the existing one to front
            # if we can find it, otherwise just no-op so we don't stack
            # modals on top of modals.
            for w in QApplication.topLevelWidgets():
                if isinstance(w, SettingsDialog) and w.isVisible():
                    w.raise_()
                    w.activateWindow()
                    break
            return

        _settings_open["flag"] = True
        # Suspend ALL global hotkeys for the lifetime of the dialog so
        # accidental key combos while typing into a field don't fire
        # background actions.
        hotkeys.stop()
        try:
            dlg = SettingsDialog(settings, parent=overlay)
            try:
                result = dlg.exec()
            finally:
                # Don't rely on Python GC to tear the dialog down - on
                # PyQt6 this can leave the widget alive long enough that
                # a follow-up open lands on a half-destroyed instance.
                dlg.deleteLater()

            if result:
                save_settings(settings)
                overlay.setWindowOpacity(settings.opacity)
                ok = exclude_window_from_capture(
                    int(overlay.winId()), settings.exclude_from_capture
                )
                # If the user toggled stealth ON but the OS rejected it,
                # treat it as visible for the badge so they aren't lulled
                # into a false sense of security.
                overlay.update_stealth_badge(settings.exclude_from_capture and ok)
                overlay.refresh_footer()
        finally:
            # ALWAYS rebind hotkeys, even if the user cancelled or the
            # dialog raised. Otherwise the user is left with a running
            # app that doesn't respond to any of its hotkeys.
            apply_hotkeys()
            _settings_open["flag"] = False

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

    # Stealth must happen AFTER show() so the HWND is valid.
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
    # Force QueuedConnection on every link so no slot ever runs on the
    # pynput listener thread — even for plain Python callables where
    # Qt's auto-detection can pick DirectConnection.
    qc = Qt.ConnectionType.QueuedConnection
    dispatcher.answer_requested.connect(controller.trigger_answer, qc)
    dispatcher.toggle_requested.connect(lambda: overlay.toggle_visibility(), qc)
    dispatcher.clear_requested.connect(controller.clear, qc)
    dispatcher.settings_requested.connect(open_settings_dialog, qc)
    dispatcher.quit_requested.connect(app.quit, qc)

    def apply_hotkeys() -> None:
        # Bind hotkeys to signal.emit (thread-safe) instead of direct
        # callables that touch the UI.
        hotkeys.set_hotkeys({
            settings.hotkey_answer: dispatcher.answer_requested.emit,
            settings.hotkey_toggle: dispatcher.toggle_requested.emit,
            settings.hotkey_clear: dispatcher.clear_requested.emit,
            settings.hotkey_settings: dispatcher.settings_requested.emit,
            settings.hotkey_quit: dispatcher.quit_requested.emit,
        })

    apply_hotkeys()
    _say("hotkeys registered.")

    # ---- System tray icon ----
    # Even when the overlay is hidden, the tray icon stays so the user can
    # toggle it back, open settings, or quit cleanly.
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
        tray.setToolTip(settings.app_display_name or "cluely-killer")
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
        controller.stop_live_transcription()

    app.aboutToQuit.connect(cleanup)

    _say("entering Qt event loop. The window should be visible now.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
