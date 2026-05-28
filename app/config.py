"""Persistent user settings.

Stored as JSON at ~/.cluely_killer/config.json so it survives reinstalls
and is not bundled with the repo.

This build is intentionally simple:
  - Whisper: 'small' only, loaded from the model files bundled next to
    the .exe (in models/hf-cache/hub/). Zero downloads, ever.
  - LLM: DeepSeek only. Cheap, fast, OpenAI-compatible API.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

CONFIG_DIR = Path.home() / ".cluely_killer"
CONFIG_FILE = CONFIG_DIR / "config.json"


@dataclass
class Settings:
    # ---- LLM provider (DeepSeek only) ----
    # Get a key at https://platform.deepseek.com/api_keys.
    # Pricing ~$0.14/M input + $0.28/M output for deepseek-chat,
    # so a typical interview Q+A is well under a tenth of a cent.
    deepseek_api_key: str = ""
    # 'deepseek-chat' (V3) is the right default - fast, OpenAI-class
    # quality. 'deepseek-reasoner' (R1) emits chain-of-thought first
    # so latency-to-first-token is higher; only use it if you actually
    # want stronger reasoning at the cost of speed.
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    # ---- Speech-to-Text (faster-whisper) ----
    # Locked to 'small' - the model files are bundled next to the .exe
    # so the friend never sees a download. ~466 MB on disk, ~3-5s
    # transcription per 25-second clip on a typical CPU at int8.
    whisper_model: str = "small"
    whisper_compute: str = "int8"
    whisper_device: str = "cpu"

    # ---- Audio ----
    # How many seconds of recent audio to send to whisper on each hotkey press.
    answer_window_seconds: float = 25.0
    # Total rolling buffer length kept in memory.
    buffer_seconds: float = 60.0

    # ---- Hotkeys (pynput GlobalHotKeys syntax) ----
    hotkey_answer: str = "<ctrl>+<space>"
    hotkey_toggle: str = "<ctrl>+\\"
    hotkey_clear: str = "<ctrl>+r"
    hotkey_settings: str = "<ctrl>+<shift>+s"
    hotkey_quit: str = "<ctrl>+<shift>+q"
    # Manual answer-mode overrides. Each forces a specific response style
    # instead of the default smart auto-classifier:
    #   <ctrl>+<shift>+1 -> SUMMARY: ties back to last 5 Q+A turns
    #   <ctrl>+<shift>+2 -> SIMPLE:  1-2 sentence standalone, no context
    #   <ctrl>+<shift>+3 -> DEEP:    technical deep-dive with implementation detail
    # Plain Ctrl+1/2/3 would conflict with browser tab switching, so we
    # use Ctrl+Shift+digit which is generally unclaimed by major apps.
    hotkey_answer_summary: str = "<ctrl>+<shift>+1"
    hotkey_answer_simple: str = "<ctrl>+<shift>+2"
    hotkey_answer_deep: str = "<ctrl>+<shift>+3"

    # ---- User context (injected into system prompt) ----
    about_me: str = ""
    resume_text: str = ""
    job_description: str = ""
    custom_system_prompt: str = ""

    # ---- Window / stealth ----
    exclude_from_capture: bool = True
    opacity: float = 0.95
    window_x: int = -1
    window_y: int = -1
    window_w: int = 580
    window_h: int = 360


def load_settings() -> Settings:
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            allowed = {f.name for f in fields(Settings)}
            return Settings(**{k: v for k, v in data.items() if k in allowed})
        except Exception as e:
            print(f"[config] failed to load, using defaults: {e}")
    return Settings()


def save_settings(s: Settings) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(asdict(s), indent=2), encoding="utf-8")
