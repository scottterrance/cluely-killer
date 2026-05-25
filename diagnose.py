"""diagnose.py - visibility diagnostic for cluely-killer.

Run:  python diagnose.py

What this does:
  1. Prints your Windows session type (Console / RDP / Service).
  2. Lists every connected screen with its coordinates.
  3. Pops a plain, normal, fully-opaque PyQt6 window centered on your
     primary screen with NO frameless / transparent / Tool /
     always-on-top flags.

Interpretation:
  - If you SEE the diagnostic window  -> PyQt + your display work; the
    issue is specifically the overlay's window flags. Tell me.
  - If you DON'T see it either        -> the issue is environmental
    (RDP capture protection, VM rendering, antivirus, etc.).
    Send me the terminal output.
"""
from __future__ import annotations

import os
import platform
import sys


def banner(title: str) -> None:
    print("=" * 64)
    print(title)
    print("=" * 64)


def main() -> None:
    banner("cluely-killer - visibility diagnostic")
    print(f"Python      : {sys.version.split()[0]}  ({sys.executable})")
    print(f"Platform    : {platform.platform()}")
    print(f"Username    : {os.environ.get('USERNAME', '?')}")

    session = os.environ.get("SESSIONNAME", "?")
    print(f"SESSIONNAME : {session}")
    if session.upper().startswith("RDP") or session.upper().startswith("ICA"):
        print("  >> You are connected via Remote Desktop / Citrix.")
        print("  >> WDA_EXCLUDEFROMCAPTURE makes the window invisible to your remote view.")
        print("  >> Stealth must stay OFF on RDP. Run the app with --no-stealth.")
    elif session.lower() == "console":
        print("  >> You are on the physical console session. Stealth flag should work.")
    else:
        print("  >> Unknown session type. Treat with caution.")

    print()
    try:
        from PyQt6.QtCore import Qt  # noqa: F401
        from PyQt6.QtGui import QGuiApplication
        from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
    except Exception as e:
        print(f"PyQt6 import FAILED: {type(e).__name__}: {e}")
        print("Run:  pip install PyQt6")
        return

    app = QApplication(sys.argv)
    screens = QGuiApplication.screens()
    print(f"Detected {len(screens)} screen(s):")
    for i, s in enumerate(screens):
        g = s.geometry()
        avail = s.availableGeometry()
        primary_marker = " (PRIMARY)" if s == QGuiApplication.primaryScreen() else ""
        print(
            f"  [{i}] '{s.name()}'{primary_marker}\n"
            f"      full     : x={g.x()} y={g.y()} {g.width()}x{g.height()}\n"
            f"      available: x={avail.x()} y={avail.y()} {avail.width()}x{avail.height()}\n"
            f"      DPR={s.devicePixelRatio()}  refresh={s.refreshRate()}Hz"
        )

    pg = QGuiApplication.primaryScreen().availableGeometry()
    w = QWidget()
    w.setWindowTitle("cluely-killer DIAGNOSTIC - close this window when done")
    w.setMinimumSize(560, 320)

    layout = QVBoxLayout(w)
    label = QLabel(
        "<h2 style='color:#7CC8FF;'>You can see this window.</h2>"
        "<p>Good news: <b>PyQt6 rendering works on your setup</b>. The original overlay's "
        "frameless / transparent / Tool / always-on-top flag combination is what "
        "made the real overlay invisible.</p>"
        "<p><b>Tell Kiro:</b></p>"
        "<ul>"
        f"<li>SESSIONNAME = <code>{session}</code></li>"
        f"<li>Number of screens = {len(screens)}</li>"
        "<li>Whether you connect to this PC via RDP / VPS / VM / direct keyboard</li>"
        "</ul>"
        "<p>Close this window when done.</p>"
    )
    label.setStyleSheet(
        "font-size: 13px; padding: 18px; color: #e6e9ef;"
        " background: #1a1d24; border: 2px solid #7CC8FF;"
    )
    label.setWordWrap(True)
    layout.addWidget(label)

    w.resize(620, 380)
    cx = pg.x() + (pg.width() - w.width()) // 2
    cy = pg.y() + (pg.height() - w.height()) // 3
    w.move(cx, cy)
    w.show()
    w.raise_()
    w.activateWindow()

    print()
    print(f"Diagnostic window placed at x={w.x()} y={w.y()} size={w.width()}x{w.height()}")
    print("If you do NOT see a blue-bordered window after 5 seconds,")
    print("the issue is environmental (RDP, VM, antivirus). Send Kiro this terminal output.")
    print()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
