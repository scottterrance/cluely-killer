# cluely-killer

Real-time interview answer overlay for Windows. Listens to your meeting's
system audio, transcribes the interviewer's voice with `faster-whisper`,
and streams a concise, human-sounding answer onto a transparent overlay
that **does not appear in screen-share captures**.

- **Stealth.** `WDA_EXCLUDEFROMCAPTURE` hides the window from Zoom / Meet / Teams / OBS / Discord (Win 10 build 19041+).
- **System loopback audio.** No virtual cable required — captures speakers via WASAPI.
- **Push-to-answer.** `Ctrl+Space` slices the last ~25 seconds, runs Whisper, streams an answer.
- **STT: bundled.** Whisper `small` ships inside the .exe folder. Zero downloads, ever, on any machine.
- **LLM: DeepSeek only.** OpenAI-compatible API, ~$0.14 per million tokens (cents per interview).
- **All your context in one place.** About-me, resume, job description, custom system prompt — editable from Settings.

> Use this on yourself only. Recording or transcribing other people without consent may be illegal in your jurisdiction.

---

## Quick start (Windows, dev)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip uninstall hf-xet -y
copy .env.example .env       # paste DEEPSEEK_API_KEY=sk-... inside
python run.py
```

First `python run.py` will download the Whisper `small` model (~466 MB)
to `models/hf-cache/`. After that, all subsequent runs (and the .exe
build) are fully offline.

Default hotkeys:

| Hotkey            | Action                                |
| ----------------- | ------------------------------------- |
| `Ctrl + Space`    | Answer the last question              |
| `Ctrl + \`        | Toggle overlay visibility             |
| `Ctrl + R`        | Clear the audio buffer                |
| `Ctrl + Shift + S`| Open Settings                         |
| `Ctrl + Shift + Q`| Quit                                  |

---

## Build a self-contained .exe and ship it to a friend

The friend gets a folder. They double-click `cluely-killer.exe`. It works.
No model download, no Python install, no setup.

```powershell
# 1. Make sure ./models/hf-cache/hub/models--*--faster-whisper-small/ is populated.
.\setup-model.ps1

# 2. Build (takes ~3-5 min the first time)
.\build.bat

# 3. Zip the dist folder (~750 MB)
Compress-Archive -Path .\dist\cluely-killer -DestinationPath .\cluely-killer-for-friend.zip -Force
```

Send the zip via Drive / WeTransfer.

### What your friend does

1. Extract the zip.
2. Double-click `cluely-killer.exe`.
   - SmartScreen: "More info" → "Run anyway" (it's unsigned).
   - Defender may flag PyInstaller .exes as a false positive — restore from quarantine if needed.
3. Press `Ctrl+Shift+S` → paste their DeepSeek API key (from https://platform.deepseek.com/api_keys) → Save.
4. (Optional) Paste their resume + the job description in Settings → Your Context.
5. In a Zoom / Meet / Teams call: press `Ctrl+Space` to get an answer for the last 25 seconds of audio.

That's it. The Whisper model is already inside the folder, so the friend
sees zero downloads on first launch — just `Whisper model loaded.` in
about 3 seconds.

---

## Project layout

```
cluely-killer/
├── run.py                       # entry point (sets up offline HF cache)
├── setup-model.ps1              # one-time helper: stage Whisper into ./models/
├── build.bat                    # one-click PyInstaller build
├── cluely-killer.spec           # PyInstaller config (bundles ./models/)
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
    │   └── deepseek_provider.py # DeepSeek streaming chat
    ├── prompts/builder.py       # system prompt + example scheduler
    ├── core/controller.py       # orchestrator (signals to UI)
    ├── ui/
    │   ├── overlay.py           # frameless transparent window
    │   ├── settings_dialog.py   # tabbed settings
    │   └── styles.py            # QSS
    ├── stealth/windows.py       # SetWindowDisplayAffinity
    └── hotkeys/manager.py       # global hotkey listener
```
