"""Entry point.

Run from the project root:
    python run.py
    python run.py --simple                  # normal titled window
    python run.py --no-stealth              # disable WDA_EXCLUDEFROMCAPTURE
    python run.py --reset-window            # ignore saved position
    python run.py --whisper-model tiny      # try a smaller model
"""
from __future__ import annotations

import faulthandler
import sys
import traceback
from pathlib import Path

# faulthandler catches NATIVE crashes (segfault, illegal instruction,
# abort) that ordinary Python try/except cannot. Without it,
# ctranslate2 / numpy / torch crashes look like the process silently
# evaporated. With it, we get a Python-level stack trace pointing at
# the exact line where the C++ code died.
faulthandler.enable()

# Disable Python's stdout/stderr buffering so we see every print() the
# instant it happens. PowerShell can buffer aggressively otherwise.
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
