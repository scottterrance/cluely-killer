
# cluely-killer

Real-time interview answer overlay for Windows. Listens to your meeting's
system audio, transcribes the interviewer's voice with `faster-whisper`,
and streams a concise, human-sounding answer onto a transparent overlay
that **does not appear in screen-share captures**.

## Design philosophy

> **The only objective is to maximize the candidate's probability of being hired.**

Not beautiful English. Not the smartest answer. Not maximum technical detail.
Only hiring probability. Every prompt, UI decision, and optimization serves
this single goal.

### How it works

- **One LLM call per answer.** No multi-agent pipelines, no second-pass
  rewriting, no iterative scoring. Streaming and prefix-cache efficiency
  are fully preserved.
- **Optimize before generation.** The prompt is a loss function, not a
  workflow. Silent planning (interviewer intent → three strongest facts →
  one metric → remove everything else) shapes the model's output before
  the first token is generated.
- **PRIMARY sentence first.** Every answer opens with one gold, large,
  self-complete sentence. If the candidate is interrupted after speaking
  it, the answer still sounds finished.
- **Information budget.** Hard maximum per answer: 3 major ideas · 2
  technologies · 1 metric · 1 achievement. Interviewers remember almost
  nothing — choose the three things deliberately.
- **Spoken English.** 10–18 words per sentence, never more than 25.
  No academic, corporate, or marketing tone. Sounds like an experienced
  engineer talking naturally.

---

## Features

- **Stealth.** `WDA_EXCLUDEFROMCAPTURE` hides the window from Zoom / Meet / Teams / OBS / Discord (Win 10 build 19041+).
- **System loopback audio.** No virtual cable required — captures speakers via WASAPI.
- **Push-to-answer.** `1` slices the last ~25 seconds, runs Whisper, streams an answer.
- **STT: bundled.** Whisper `small` ships inside the .exe folder. Zero downloads on any machine.
- **LLM: DeepSeek only.** OpenAI-compatible API, ~$0.14 per million tokens (cents per interview).
- **Hiring-probability prompt.** Every answer opens with one self-complete `PRIMARY` sentence (gold, large), obeys a strict information budget, and is tuned to spoken (not written) English — all in a single LLM call.
- **Interview modes.** Four modes re-weight each answer to the interviewer's true intent:
  - **Balanced** (default) — auto-adapts to each question
  - **Recruiter** — communication, confidence, business value
  - **Hiring Manager** — ownership, execution, delivery
  - **Technical** — engineering depth, architecture, trade-offs
- **Mode badge in header.** The active interview mode is always visible in the overlay header. Color-coded: grey (balanced), blue (recruiter), orange (hiring manager), purple (technical).
- **Keyword highlighting.** `==word==` renders RED (stress when speaking), `**word**` renders YELLOW (secondary emphasis). Technologies, metrics, outcomes, and action verbs are automatically highlighted.
- **Question classifier.** Auto-detects question type (Behavioural / Technical / System Design / Culture Fit / Salary) and shows a depth hint aligned with the hiring-probability framework.
- **Filler word detector.** Scores the interviewer's speech clarity and shows a badge after each answer.
- **Speculative pre-generation.** Starts answering on the interviewer's pause, before the key press, so the answer appears instantly.
- **Semantic cache.** Repeated questions answered instantly from cache. Auto-invalidates when resume/JD changes.
- **Smart resume retrieval (RAG).** For large resumes, injects only the chunks relevant to each question.
- **Live transcription.** Displays the interviewer's speech in real-time so you can read along.
- **Persona presets.** Switch between job applications (e.g. "Stripe Senior PM" / "Junior Dev") with one click.
- **Rephrase (key `3`).** Generates a different version of the last answer: different opening, different example, same facts.
- **Context mode (key `2`).** Includes the last 5 Q+A turns for follow-up questions.

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

| Hotkey             | Action                                |
| ------------------ | ------------------------------------- |
| `1`                | Answer the last question              |
| `2`                | Answer with conversation context      |
| `3`                | Rephrase the last answer              |
| `Ctrl + \`         | Toggle overlay visibility             |
| `Ctrl + R`         | Clear the audio buffer + forget       |
| `Ctrl + Shift + S` | Open Settings                         |
| `Ctrl + Shift + Q` | Quit                                  |

---

## Answer format

Every answer uses tagged sections rendered as colored blocks:

| Tag         | Color  | Meaning                                         |
| ----------- | ------ | ----------------------------------------------- |
| `[PRIMARY]` | Gold   | The one self-complete sentence — speak this first |
| `[S]`       | Blue   | STAR: Situation                                 |
| `[T]`       | Purple | STAR: Task                                      |
| `[A]`       | Orange | STAR: Action (include the one metric here)      |
| `[R]`       | Green  | STAR: Result                                    |
| `[POINT]`   | Blue   | Technical: core answer                          |
| `[HOW]`     | Grey   | Technical: implementation                       |
| `[WHY]`     | Purple | Technical / General: reasoning                  |
| `[RESULT]`  | Green  | Technical: real-world outcome                   |
| `[NEED]`    | Red    | System Design: constraint                       |
| `[OPT]`     | Amber  | System Design: options                          |
| `[PICK]`    | Green  | System Design: chosen approach                  |
| `[TRADE]`   | Orange | System Design: trade-off                        |
| `[CLOSE]`   | Green  | General / Culture: connecting to role           |
| `[CONT]`    | Amber  | Follow-up: link to previous answer              |

---

## Build a self-contained .exe

```powershell
# 1. Stage the Whisper model
.\setup-model.ps1

# 2. Build (takes ~3-5 min the first time)
.\build.bat

# 3. Zip and send
Compress-Archive -Path .\dist\cluely-killer -DestinationPath .\cluely-killer-for-friend.zip -Force
```

The friend gets a folder. They double-click `cluely-killer.exe`. No model
download, no Python install, no setup. SmartScreen: "More info" → "Run anyway".

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
    ├── prompts/builder.py       # hiring-probability system prompt + example scheduler
    ├── core/
    │   ├── controller.py        # orchestrator (signals to UI)
    │   ├── analysis.py          # question classifier + filler word detector
    │   ├── history.py           # conversation history
    │   ├── personas.py          # persona presets
    │   ├── rag.py               # lexical resume retrieval
    │   └── semantic_cache.py    # repeated-question cache
    ├── ui/
    │   ├── overlay.py           # frameless transparent window + mode badge
    │   ├── settings_dialog.py   # tabbed settings
    │   └── styles.py            # QSS (dark theme, gold PRIMARY emphasis)
    ├── stealth/windows.py       # SetWindowDisplayAffinity
    └── hotkeys/manager.py       # global hotkey listener
```

---

## Legal

Use this on yourself only. Recording or transcribing other people without
consent may be illegal in your jurisdiction.
