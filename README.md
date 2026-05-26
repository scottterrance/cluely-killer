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

---

## Build a standalone `.exe` and ship it to a friend

Bundle the entire app (Python runtime + Qt + faster-whisper + ctranslate2
+ soundcard + all your code) into a single folder you can copy to any
Windows machine with no Python installed.

### On YOUR machine (build once)

```powershell
.\build.bat
```

Output: `dist\cluely-killer\cluely-killer.exe`. First build takes ~3-5
minutes (PyInstaller statically scans every dependency); subsequent
builds are faster. The whole `dist\cluely-killer\` folder is ~250-300 MB
before the Whisper model gets downloaded on the target machine.

### Hand-off to a friend (zero-Python install on their side)

1. Zip the entire `dist\cluely-killer\` folder.
2. Send it (Drive / Dropbox / WeTransfer — it's bigger than the email
   limit on most providers).
3. Tell them to:
   - Unzip anywhere (Desktop, Documents, wherever).
   - Double-click `cluely-killer.exe` inside the unzipped folder.
   - Click "More info" -> "Run anyway" on the SmartScreen warning
     (the .exe is unsigned; signing costs $$$ and is out of scope).
4. **First run downloads the Whisper model.** Default is
   `large-v3-turbo` (~1.5 GB) which downloads to
   `%USERPROFILE%\.cache\huggingface\` automatically on first launch.
   They need:
   - A reasonable internet connection (the download takes 2-10 min).
   - ~3-4 GB free RAM during transcription.
   - A CPU with AVX2 (anything from ~2014 onward — basically any
     non-ancient laptop).
   If the machine is older / weaker, tell them to open Settings
   (`Ctrl+Shift+S`) -> Speech-to-Text tab -> change Whisper model to
   `small`. The download shrinks to ~466 MB and CPU usage drops a lot;
   English accuracy is still very good.
5. They should set their own LLM provider key in Settings -> AI Provider
   (Groq free tier, OpenRouter free tier, DeepSeek paid tier, or
   Ollama if they want fully local). Keys are stored in their own
   `%USERPROFILE%\.config\cluely-killer\` directory; nothing leaves
   the machine except the LLM API call itself.
6. Tell them to (optionally) rename the app via Settings -> Window ->
   "App display name" so the tray tooltip / Task Manager process name
   reads something innocuous like "Notepad" instead of "cluely-killer".

### Notes / gotchas

- **Whisper model is NOT bundled.** Legal redistribution of the model
  weights is murky, and 1.5 GB would balloon the zip. The first-run
  download fetches it from Hugging Face directly.
- **Ollama is NOT bundled either.** If your friend wants the
  fully-local-LLM path, they install Ollama separately from
  https://ollama.com and `ollama pull llama3.1:8b`.
- **Microsoft Defender** sometimes flags PyInstaller-built .exes as
  generic-trojan. This is a well-known false positive; the real fix is
  code-signing, which is out of scope. If Defender quarantines the .exe,
  your friend can right-click -> Restore, or add the folder to
  Defender's exclusion list.
- **Hugging Face Rust downloader (`hf_xet`) crashes on hypervisor CPUs**
  with `STATUS_ILLEGAL_INSTRUCTION (0xc000001d)`. The app sets
  `HF_HUB_DISABLE_XET=1` defensively so the model download falls back
  to the pure-Python path. No action needed on the friend's side.
- **Console window.** First builds keep `console=True` in
  `cluely-killer.spec` so any crash leaves a readable trace. For a
  release build with no terminal window, edit `console=True` to
  `console=False` in the spec and rebuild.
- **Renaming the .exe** for stronger stealth: change `name="cluely-killer"`
  in `cluely-killer.spec` (two places — `EXE` and `COLLECT`) and
  rebuild. Combine with the in-app `App display name` setting for a
  fully-rebranded process.
