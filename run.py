"""Entry point.

Run from the project root:
    python run.py
    python run.py --simple       # normal titled window
    python run.py --no-stealth   # disable WDA_EXCLUDEFROMCAPTURE
    python run.py --reset-window # ignore saved position
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# CRITICAL: configure Hugging Face Hub BEFORE faster-whisper imports it.
#
# This build ships the Whisper 'small' model INSIDE the project folder
# (at models/hf-cache/). On launch we point HF_HOME at that bundled
# folder and force HF_HUB_OFFLINE=1 so faster-whisper:
#   - never makes a network call to huggingface.co
#   - never shows a "downloading" progress bar
#   - never fails on flaky WiFi or DNS
# It just loads the local model and starts.
#
# Also disables hf_xet (Rust accelerator that crashes on hypervisor CPUs)
# and the metadata progress bars.
# ---------------------------------------------------------------------------
import os
import sys
from pathlib import Path

# Where the .exe (or run.py in dev) lives. The bundled model sits next
# to it under models/hf-cache/.
if getattr(sys, "frozen", False):
    _APP_DIR = Path(sys.executable).parent
else:
    _APP_DIR = Path(__file__).resolve().parent

_BUNDLED_HF_CACHE = _APP_DIR / "models" / "hf-cache"
_BUNDLED_HF_HUB = _BUNDLED_HF_CACHE / "hub"

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ["HF_HOME"] = str(_BUNDLED_HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(_BUNDLED_HF_HUB)
os.environ["HF_HUB_OFFLINE"] = "1"

print(f"[startup] using bundled HF cache at: {_BUNDLED_HF_HUB}", flush=True)
print("[startup] HF offline mode forced - no network, no downloads.", flush=True)

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
