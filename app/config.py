"""Persistent user settings.

Stored as JSON at ~/.cluely_killer/config.json so it survives reinstalls
and is not bundled with the repo.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

CONFIG_DIR = Path.home() / ".cluely_killer"
CONFIG_FILE = CONFIG_DIR / "config.json"


@dataclass
class Settings:
    # ---- LLM provider ----
    provider: str = "groq"  # "groq" | "openrouter" | "deepseek" | "ollama"

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    openrouter_api_key: str = ""
    # Comma-separated fallback chain. The provider tries each in order;
    # any non-200 response (rate-limit, model removed, server error)
    # silently falls through to the next model. The free model list on
    # OpenRouter rotates - check https://openrouter.ai/models?max_price=0
    # for the current set if you ever see "all N models failed".
    openrouter_model: str = (
        "qwen/qwen3-next-80b-a3b-instruct:free, "
        "deepseek/deepseek-v4-flash:free, "
        "meta-llama/llama-3.3-70b-instruct:free, "
        "openai/gpt-oss-120b:free, "
        "meta-llama/llama-3.2-3b-instruct:free"
    )

    # DeepSeek (cheap paid; key from https://platform.deepseek.com/api_keys).
    # 'deepseek-chat' (V3) is the default. 'deepseek-reasoner' (R1) is
    # available too but emits chain-of-thought first so latency is higher.
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    ollama_model: str = "llama3.1:8b"
    ollama_host: str = "http://localhost:11434"

    # ---- Speech-to-Text (faster-whisper) ----
    # Models in order of size/quality: tiny, base, small, medium, large-v3
    # "small" is the sweet spot for English on a CPU.
    whisper_model: str = "small"
    whisper_compute: str = "int8"   # int8 / int8_float16 / float16 / float32
    whisper_device: str = "cpu"     # cpu / cuda

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

    # ---- User context (injected into system prompt) ----
    about_me: str = ""
    resume_text: str = ""
    job_description: str = ""
    custom_system_prompt: str = ""

    # ---- Window / stealth ----
    # ON by default: the whole point of the app is that the interviewer
    # can't see the overlay during a screen share. We removed the
    # WA_TranslucentBackground attribute that used to break rendering on
    # Win11 24H2, so the stealth + visibility combination is now reliable.
    # Use --no-stealth at startup if you ever need to turn it off without
    # opening Settings (e.g. when running over RDP).
    exclude_from_capture: bool = True
    opacity: float = 0.95
    # Last known on-screen geometry; refreshed on close. -1 = "not set yet".
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
