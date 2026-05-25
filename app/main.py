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

    _say("creating QApplication...")
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("cluely-killer")
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
    from .hotkeys.manager import HotkeyManager
    from .llm.base import LLMProvider
    from .llm.groq_provider import GroqProvider
    from .llm.ollama_provider import OllamaProvider
    from .prompts.builder import ExampleScheduler, build_system_prompt
    from .stealth.windows import exclude_window_from_capture
    from .ui.overlay import OverlayWindow
    from .ui.settings_dialog import SettingsDialog

    def _llm_factory(s) -> LLMProvider:
        if s.provider == "ollama":
            return OllamaProvider(model=s.ollama_model, host=s.ollama_host)
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
    controller = Controller(
        settings=settings,
        audio_buffer=buffer,
        whisper=whisper,
        llm_factory=_llm_factory,
        scheduler=scheduler,
        prompt_builder=_prompt_for,
    )
    _say("controller ready.")

    # ---- UI ----
    _say("building overlay window...")
    hotkeys = HotkeyManager()
    overlay: OverlayWindow

    def open_settings_dialog() -> None:
        dlg = SettingsDialog(settings, parent=overlay)
        if dlg.exec():
            save_settings(settings)
            overlay.setWindowOpacity(settings.opacity)
            exclude_window_from_capture(int(overlay.winId()), settings.exclude_from_capture)
            overlay.refresh_footer()
            apply_hotkeys()

    overlay = OverlayWindow(
        settings,
        controller,
        on_open_settings=open_settings_dialog,
        simple_mode=simple_mode,
    )
    _say("overlay constructed; calling show()...")
    overlay.show()
    _say("overlay.show() returned; placing on screen...")
    overlay.place_on_screen()

    # Stealth must happen AFTER show() so the HWND is valid.
    if settings.exclude_from_capture:
        ok = exclude_window_from_capture(int(overlay.winId()), True)
        if not ok and sys.platform == "win32":
            _say("WDA_EXCLUDEFROMCAPTURE failed - needs Windows 10 build 19041+.")
        else:
            _say(
                "stealth ON: window is hidden from screen-capture APIs. "
                "If you cannot see it locally either, run with --no-stealth."
            )
    else:
        _say("stealth OFF: window is visible to screen-capture too.")

    # ---- Hotkeys ----
    def apply_hotkeys() -> None:
        hotkeys.set_hotkeys({
            settings.hotkey_answer: lambda: controller.trigger_answer(),
            settings.hotkey_toggle: lambda: overlay.toggle_visibility(),
            settings.hotkey_clear: lambda: controller.clear(),
            settings.hotkey_settings: lambda: open_settings_dialog(),
        })

    apply_hotkeys()
    _say("hotkeys registered.")

    if capture.last_error:
        controller.error.emit(f"Audio: {capture.last_error}")
        _say(f"audio thread reported: {capture.last_error}")

    def cleanup() -> None:
        hotkeys.stop()
        capture.stop()

    app.aboutToQuit.connect(cleanup)

    _say("entering Qt event loop. The window should be visible now.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
