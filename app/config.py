"""Persistent user settings.

Stored as JSON at ~/.cluely_killer/config.json so it survives reinstalls
and is not bundled with the repo.

This build is intentionally simple and fully offline-capable for STT:
  - Whisper: local 'large-v3-turbo' only, loaded from the model files
    bundled next to the .exe (models/whisper-large-v3-turbo/). Zero
    downloads at runtime.
  - LLM: DeepSeek only. Cheap, fast, OpenAI-compatible API.

load_settings() filters out any keys not present on the Settings
dataclass, so configs written by older builds (which had groq_* /
llm_backend / stt_backend fields) load cleanly - the obsolete keys are
simply dropped.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

CONFIG_DIR = Path.home() / ".cluely_killer"
CONFIG_FILE = CONFIG_DIR / "config.json"


@dataclass
class Settings:
    # ---- LLM (DeepSeek - the only provider) ----
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
    # Answer length - THE main lever for answer SPEED when STT is fast
    # (e.g. on a GPU). LLMs generate tokens sequentially, so answer
    # latency is ~proportional to output length. Options:
    #   "concise"  - 1-2 short sentences (~110 token cap). Fastest. Default.
    #   "normal"   - 2-3 sentences (~220 token cap).
    #   "detailed" - 3-5 sentences (~400 token cap). Slowest.
    answer_brevity: str = "concise"

    # ---- Speech-to-Text (local faster-whisper, offline) ----
    # 'large-v3-turbo' is a pruned large-v3 (decoder layers 32 -> 4):
    # markedly more accurate than 'small' - especially on names and
    # technical jargon. Model files are bundled next to the .exe (in
    # models/whisper-large-v3-turbo/), so the end user never sees a
    # download. ~1.5 GB on disk at int8.
    whisper_model: str = "large-v3-turbo"
    # Device: "auto" picks CUDA GPU if detected, else CPU. Force with
    # "cuda"/"gpu" or "cpu". GPU makes large-v3-turbo transcribe in
    # well under a second; CPU latency scales with audio length.
    whisper_device: str = "auto"
    # Compute type: "auto" -> float16 on GPU, int8 on CPU. Override with
    # an explicit CTranslate2 type (e.g. int8_float16, float32) if needed.
    whisper_compute: str = "auto"
    # CTranslate2 worker threads. 0 = auto (all cores minus one). Raising
    # this is the simplest local-STT speedup on a multi-core CPU.
    whisper_cpu_threads: int = 0
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

    # ---- Continuous STT (Phase 2) ----
    # When True, a background thread transcribes the audio buffer as the
    # interviewer talks (using the LOCAL whisper model only - never the
    # metered cloud STT). On a hotkey press the answer reads the
    # already-built transcript, so Whisper is no longer on the press
    # critical path and the perceived latency drops to ~the LLM call.
    # When False, the app uses the classic transcribe-on-press path
    # (whatever stt_backend is selected). Auto-disabled at runtime if
    # the local model isn't available.
    continuous_stt: bool = True
    # Feed a glossary of the candidate's resume/JD terms to Whisper to
    # bias proper-noun/jargon recognition. Helpful in theory, but on
    # short/quiet audio Whisper can ECHO the glossary into the transcript
    # ('Glossary, SIA, NJ, UI...'). We sanitize that echo, but if you'd
    # rather avoid the risk entirely set this False.
    stt_bias_enabled: bool = True
    # Speculative pre-generation: when the interviewer PAUSES (a likely
    # end-of-question), start generating the SHORT-mode DeepSeek answer
    # in the background BEFORE you press. If you then press '1', the
    # answer is already (partly) streamed, so it appears instantly -
    # hiding DeepSeek's generation time behind the natural pause. Costs a
    # few extra tokens on guesses you don't use. Only affects mode '1'
    # (short); mode '2' (context) always generates fresh. Requires
    # continuous_stt. Default on.
    speculative_enabled: bool = True

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
            # One-time device/compute upgrade: older builds hard-defaulted
            # to CPU/int8. Migrate those to 'auto' so the app now picks a
            # CUDA GPU automatically when present (and still uses CPU when
            # not). This is safe - 'auto' falls back to CPU/int8 if there's
            # no GPU. Anyone who explicitly set 'cuda' or a custom compute
            # type keeps it.
            if s.whisper_device == "cpu":
                s.whisper_device = "auto"
                print("[config] migrated whisper_device 'cpu' -> 'auto' (GPU auto-detect)")
            if s.whisper_compute == "int8":
                s.whisper_compute = "auto"
                print("[config] migrated whisper_compute 'int8' -> 'auto'")
            return s
        except Exception as e:
            print(f"[config] failed to load, using defaults: {e}")
    return Settings()


def save_settings(s: Settings) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(asdict(s), indent=2), encoding="utf-8")
