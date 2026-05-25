# cluely-killer

Real-time interview answer overlay for Windows. Listens to your meeting's
system audio, transcribes the interviewer's voice with `faster-whisper`,
and streams a concise, human-sounding answer onto a transparent overlay
that **does not appear in screen-share captures**.

- **Stealth.** `WDA_EXCLUDEFROMCAPTURE` hides the window from Zoom / Meet / Teams / OBS / Discord (Win 10 build 19041+).
- **System loopback audio.** No virtual cable required — captures whatever your speakers are playing via WASAPI.
- **Push-to-answer.** `Ctrl+Space` slices the last ~25 seconds, runs Whisper, streams an answer.
- **Pluggable LLM.** Groq (cloud, free tier, ~500+ tok/s) or Ollama (fully local).
- **Human-style output.** 3–5 sentences, **bolded** keywords, an example anchored to your resume every 3rd–4th answer.
- **All your context in one place.** About-me, resume, job description, and a custom system prompt — editable from the Settings dialog.

> Use this on yourself only. Recording or transcribing other people without consent may be illegal in your jurisdiction.

---

## Project layout

```
cluely-killer/
├── run.py                       # entry point
├── requirements.txt
├── .env.example
└── app/
    ├── main.py                  # bootstrap & wiring
    ├── config.py                # persistent JSON settings
    ├── audio/
    │   ├── buffer.py            # thread-safe rolling buffer
    │   └── loopback.py          # WASAPI loopback capture
    ├── stt/whisper_engine.py    # faster-whisper + Silero VAD
    ├── llm/
    │   ├── base.py              # provider interface
    │   ├── groq_provider.py     # cloud streaming
    │   └── ollama_provider.py   # local streaming
    ├── prompts/builder.py       # system prompt + example scheduler
    ├── core/controller.py       # orchestrator (signals to UI)
    ├── ui/
    │   ├── overlay.py           # frameless transparent window
    │   ├── settings_dialog.py   # tabbed settings
    │   └── styles.py            # QSS
    ├── stealth/windows.py       # SetWindowDisplayAffinity
    └── hotkeys/manager.py       # global hotkey listener
```

---

## Quick start (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip uninstall hf-xet -y      # avoid Rust downloader crashes on VMs
copy .env.example .env       # then paste your Groq key
python run.py
```

> The `pip uninstall hf-xet` step is only required on Windows VMs / hypervisor
> CPUs where Hugging Face's Rust download accelerator crashes with
> `STATUS_ILLEGAL_INSTRUCTION (0xc000001d)`. The app also sets
> `HF_HUB_DISABLE_XET=1` defensively, but uninstalling guarantees it.

Default hotkeys:

| Hotkey            | Action                                |
| ----------------- | ------------------------------------- |
| `Ctrl + Space`    | Answer the last question              |
| `Ctrl + \`        | Toggle overlay visibility             |
| `Ctrl + R`        | Clear the audio buffer                |
| `Ctrl + Shift + S`| Open Settings                         |

All of them are reconfigurable in the Settings → Hotkeys tab.

---

## Switching to local-only (Ollama)

1. Install Ollama from https://ollama.com.
2. Pull a model: `ollama pull llama3.1:8b` (or `qwen2.5:7b`, `phi3.5`).
3. In the app: `Ctrl+Shift+S` → AI Provider → switch to **ollama** → Save.

No keys, no cloud, no telemetry.
