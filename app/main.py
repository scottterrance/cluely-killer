"""Application bootstrap.

Wires together: settings -> audio capture -> Whisper -> LLM factory ->
controller -> overlay -> hotkeys -> stealth.
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from PyQt6.QtWidgets import QApplication

from .audio.buffer import RollingAudioBuffer
from .audio.loopback import LoopbackCapture
from .config import Settings, load_settings, save_settings
from .core.controller import Controller
from .hotkeys.manager import HotkeyManager
from .llm.base import LLMProvider
from .llm.groq_provider import GroqProvider
from .llm.ollama_provider import OllamaProvider
from .prompts.builder import ExampleScheduler, build_system_prompt
from .stealth.windows import exclude_window_from_capture
from .stt.whisper_engine import WhisperEngine
from .ui.overlay import OverlayWindow
from .ui.settings_dialog import SettingsDialog


def _llm_factory(settings: Settings) -> LLMProvider:
    if settings.provider == "ollama":
        return OllamaProvider(model=settings.ollama_model, host=settings.ollama_host)
    return GroqProvider(api_key=settings.groq_api_key, model=settings.groq_model)


def _prompt_for(settings: Settings, include_example: bool) -> str:
    return build_system_prompt(
        resume=settings.resume_text,
        job_desc=settings.job_description,
        about=settings.about_me,
        custom=settings.custom_system_prompt,
        include_example=include_example,
    )


def main() -> None:
    # On Windows, force COM into STA on the main thread BEFORE creating
    # QApplication. Some audio libs (soundcard / mediafoundation / comtypes)
    # initialize COM as MTA on first import, which causes Qt's OleInitialize
    # to fail with 0x80010106. Claiming STA here first avoids the conflict.
    if sys.platform == "win32":
        try:
            import ctypes
            COINIT_APARTMENTTHREADED = 0x2
            # S_OK (0) on success, S_FALSE (1) if already STA, RPC_E_CHANGED_MODE on conflict.
            ctypes.windll.ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
        except Exception:
            pass

    load_dotenv()
    settings = load_settings()
    # If the user dropped a key in .env, prefer it on first run.
    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key and not settings.groq_api_key:
        settings.groq_api_key = env_key
        save_settings(settings)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("cluely-killer")

    # ---- Audio pipeline (always on) ----
    buffer = RollingAudioBuffer(samplerate=16000, max_seconds=settings.buffer_seconds)
    capture = LoopbackCapture(buffer, samplerate=16000)
    capture.start()

    # ---- STT ----
    print(
        f"[startup] loading faster-whisper '{settings.whisper_model}' "
        f"({settings.whisper_compute} on {settings.whisper_device}) — "
        "first run downloads the model, this can take a few minutes."
    )
    whisper = WhisperEngine(
        model_size=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute,
    )

    # ---- Orchestration ----
    scheduler = ExampleScheduler()
    controller = Controller(
        settings=settings,
        audio_buffer=buffer,
        whisper=whisper,
        llm_factory=_llm_factory,
        scheduler=scheduler,
        prompt_builder=_prompt_for,
    )

    # ---- UI ----
    hotkeys = HotkeyManager()
    overlay: OverlayWindow  # forward declaration for closure

    def open_settings_dialog() -> None:
        dlg = SettingsDialog(settings, parent=overlay)
        if dlg.exec():
            save_settings(settings)
            overlay.setWindowOpacity(settings.opacity)
            exclude_window_from_capture(int(overlay.winId()), settings.exclude_from_capture)
            overlay.refresh_footer()
            apply_hotkeys()

    overlay = OverlayWindow(settings, controller, on_open_settings=open_settings_dialog)
    overlay.show()

    # Stealth must happen AFTER show() so the HWND is valid.
    if settings.exclude_from_capture:
        ok = exclude_window_from_capture(int(overlay.winId()), True)
        if not ok and sys.platform == "win32":
            print("[startup] WDA_EXCLUDEFROMCAPTURE failed — needs Windows 10 build 19041+.")

    # ---- Hotkeys ----
    def apply_hotkeys() -> None:
        hotkeys.set_hotkeys({
            settings.hotkey_answer: lambda: controller.trigger_answer(),
            settings.hotkey_toggle: lambda: overlay.toggle_visibility(),
            settings.hotkey_clear: lambda: controller.clear(),
            settings.hotkey_settings: lambda: open_settings_dialog(),
        })

    apply_hotkeys()

    # Surface any audio-thread error after the UI is up.
    if capture.last_error:
        controller.error.emit(f"Audio: {capture.last_error}")

    # ---- Cleanup on exit ----
    def cleanup() -> None:
        hotkeys.stop()
        capture.stop()

    app.aboutToQuit.connect(cleanup)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
