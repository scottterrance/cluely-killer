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
    # 'large-v3-turbo' is a pruned large-v3 (decoder layers 32 -> 4):
    # markedly more accurate than 'small' - especially on names and
    # technical jargon - while staying fast enough for interview-length
    # clips at int8 on CPU. Model files are bundled next to the .exe
    # (in models/whisper-large-v3-turbo/), so the end user never sees a
    # download. ~1.6 GB on disk at int8.
    whisper_model: str = "large-v3-turbo"
    whisper_compute: str = "int8"
    whisper_device: str = "cpu"
    # SAFETY: when False (the default, and what every .exe ships with),
    # WhisperEngine refuses to download a missing model at runtime and
    # raises a clear "place the files here" error instead. Set True only
    # on a dev machine if you want faster-whisper to fetch the model from
    # HuggingFace once. An end user can NEVER trigger a mid-interview
    # multi-GB download.
    whisper_allow_auto_download: bool = False

    # ---- Audio ----
    # Fallback window for the FIRST press in a session, when no
    # "since-last-press" marker has been set yet. Also kept around as
    # a sane lower bound during interviews where the interviewer
    # asks very short questions.
    answer_window_seconds: float = 25.0
    # Hard ceiling on how much "since-last-press" audio gets sent to
    # Whisper + the LLM. If the interviewer rambles for longer than
    # this, only the most recent ``max_capture_seconds`` are used.
    # 120 s is the default; press the answer key more often to keep
    # transcripts tight.
    max_capture_seconds: float = 120.0
    # Total rolling buffer length kept in memory. Must be >=
    # ``max_capture_seconds`` or we'd silently drop audio before the
    # next press could read it. main.py enforces this at startup.
    buffer_seconds: float = 130.0

    # ---- Hotkeys (pynput-style GlobalHotKeys syntax) ----
    # Two answer modes:
    #   - SHORT: answer ONLY about whatever the interviewer said since
    #     the last press of either answer key. No prior Q+A context
    #     is sent to the LLM. Use this for self-contained questions.
    #   - CONTEXT: same fresh transcript, but ALSO sends the last 5
    #     Q+A pairs from the rolling history as conversation memory.
    #     Use this for follow-up questions ("can you elaborate on
    #     that?", "what about the edge case?").
    hotkey_answer_short: str = "1"
    hotkey_answer_context: str = "2"
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
    exclude_from_capture: bool = True
    opacity: float = 0.95
    window_x: int = -1
    window_y: int = -1
    # Default width is 70% of the original 580 px = 406 px. Narrower
    # column = less side-to-side eye scanning when reading on screen
    # share, which is harder for an interviewer's webcam to detect.
    window_w: int = 406
    # Default height bumped from 360 -> 540 (1.5x). Long answers from
    # DeepSeek were overflowing the old 360 px panel. The overlay is
    # still resizable manually if you want it shorter.
    window_h: int = 540


def load_settings() -> Settings:
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            allowed = {f.name for f in fields(Settings)}
            s = Settings(**{k: v for k, v in data.items() if k in allowed})
            # One-time migration: if the user's saved config still has
            # the old default height (360), bump to the new default
            # (540). Doesn't override anyone who customized their height.
            if s.window_h == 360:
                s.window_h = 540
                print("[config] migrated window_h 360 -> 540 (one-time bump)")
            # Same one-time migration for the width: 580 -> 406 (70%).
            if s.window_w == 580:
                s.window_w = 406
                print("[config] migrated window_w 580 -> 406 (one-time slim-down)")
            # buffer_seconds must always be >= max_capture_seconds, or
            # the rolling buffer would evict audio before the next
            # answer press could read it. Old configs with the prior
            # default of 60 s get bumped to the new 130 s default
            # automatically. Anyone who hand-tuned a higher value
            # keeps their value.
            # One-time STT upgrade: bump anyone still on the old bundled
            # 'small' default to 'large-v3-turbo'. Requires the turbo
            # model folder to be present next to the .exe (rebuild with
            # the new model staged - see setup-model.ps1). If you have a
            # reason to stay on 'small', set whisper_model in config.json
            # to something other than these two and it won't be touched.
            if s.whisper_model == "small":
                s.whisper_model = "large-v3-turbo"
                print("[config] migrated whisper_model 'small' -> 'large-v3-turbo'")
            return s
        except Exception as e:
            print(f"[config] failed to load, using defaults: {e}")
    return Settings()


def save_settings(s: Settings) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(asdict(s), indent=2), encoding="utf-8")
