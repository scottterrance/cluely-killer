"""Entry point.

Run from the project root:
    python run.py
    python run.py --simple       # normal titled window
    python run.py --no-stealth   # disable WDA_EXCLUDEFROMCAPTURE
    python run.py --reset-window # ignore saved position
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Pre-import setup. Must run BEFORE faster-whisper / huggingface_hub
# import anything.
#
# Strategy: skip HuggingFace entirely. The Whisper model lives in a flat
# directory at <app_dir>/models/whisper-<size>/ (populated by
# setup-model.ps1 in dev, bundled into dist/ at build time). The
# WhisperEngine reads CLUELY_APP_DIR to find that folder and passes the
# directory path directly to faster-whisper, which loads model.bin /
# config.json / tokenizer.json without ever calling huggingface.co.
#
# We still set HF_HUB_OFFLINE=1 + the no-progress-bars / no-telemetry
# flags as defence-in-depth in case anything inside faster-whisper or
# its deps tries an opportunistic HF lookup.
# ---------------------------------------------------------------------------
import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    _APP_DIR = Path(sys.executable).parent
else:
    _APP_DIR = Path(__file__).resolve().parent

os.environ["CLUELY_APP_DIR"] = str(_APP_DIR)

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

print(f"[startup] app dir: {_APP_DIR}", flush=True)
print(f"[startup] looking for bundled model under: {_APP_DIR / 'models'}", flush=True)

import faulthandler
import traceback

# faulthandler catches NATIVE crashes (segfault, illegal instruction,
# abort) that ordinary Python try/except cannot.
faulthandler.enable()

try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass


def _save_crash(exc_text: str) -> Path | None:
    try:
        log = Path.home() / ".cluely_killer" / "crash.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write("\n=== crash ===\n")
            f.write(exc_text)
        return log
    except Exception:
        return None


if __name__ == "__main__":
    try:
        from app.main import main
        main()
    except SystemExit:
        raise
    except BaseException:
        tb = traceback.format_exc()
        print("\n" + "=" * 60, flush=True)
        print("UNHANDLED EXCEPTION (the app would silently exit otherwise):", flush=True)
        print("=" * 60, flush=True)
        print(tb, flush=True)
        log = _save_crash(tb)
        if log is not None:
            print(f"[crash] full traceback also saved to: {log}", flush=True)
        try:
            input("\nPress Enter to close this window...")
        except (EOFError, KeyboardInterrupt):
            pass
        sys.exit(1)
