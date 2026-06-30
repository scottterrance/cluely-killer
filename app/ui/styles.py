"""Qt stylesheet for the overlay.

Solid (non-translucent) dark background. We rely on setWindowOpacity
for the soft "see through" feel, NOT WA_TranslucentBackground, because
the translucent attribute breaks frameless rendering on Win 11 24H2+.

Design philosophy: the PRIMARY sentence must be instantly visible at a
glance. Everything else is secondary. The candidate is speaking while
reading — the UI must not compete for attention.

Color system:
  Gold  (#FFD166) — PRIMARY sentence, highest emphasis
  Blue  (#7CC8FF) — POINT / SITUATION / status
  Green (#4CAF50) — RESULT / PICK / CLOSE / positive outcomes
  Red   (#FF6B6B) — NEED / inline ==keyword== stress marks
  Amber (#FFC107) — OPTIONS / CONT / follow-up links
  Purple(#a78bfa) — TASK / WHY / reasoning
  Orange(#FF9F43) — ACTION / TRADE-OFF / decisions
"""

APP_QSS = """
QWidget#OverlayRoot {
    background-color: #0A0A12;
    border: 1px solid rgba(255, 255, 255, 25);
}

#container {
    background-color: #0A0A12;
}

#status {
    color: #7CC8FF;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 4px;
}

/* ── Interview mode badge ─────────────────────────────────────────────── */
/* Color is set dynamically in Python (update_mode_badge) based on mode.  */
/* This is the base/fallback style only. */
#modeBadge {
    color: #c8ced9;
    background-color: rgba(200, 206, 217, 20);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 2px 7px;
    border-radius: 8px;
    border: 1px solid rgba(200, 206, 217, 55);
}

/* ── Stealth badge ────────────────────────────────────────────────────── */
#stealthBadge {
    color: #4ade80;
    background-color: rgba(74, 222, 128, 28);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 2px 7px;
    border-radius: 8px;
    border: 1px solid rgba(74, 222, 128, 60);
}

#stealthBadge[alarm="true"] {
    color: #fca5a5;
    background-color: rgba(248, 113, 113, 38);
    border: 1px solid rgba(248, 113, 113, 90);
}

/* ── Hidden internal badges (mem, backend) ────────────────────────────── */
#memBadge {
    color: #c8ced9;
    font-size: 9px;
    padding: 0 6px;
}

#backendBadge {
    color: #7CC8FF;
    background-color: rgba(124, 200, 255, 24);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.3px;
    padding: 2px 7px;
    border-radius: 8px;
    border: 1px solid rgba(124, 200, 255, 55);
}

#backendBadge[alarm="true"] {
    color: #fcd34d;
    background-color: rgba(251, 191, 36, 36);
    border: 1px solid rgba(251, 191, 36, 90);
}

/* ── Question type auto-classifier badge ──────────────────────────────── */
#qtypeBadge {
    color: #a78bfa;
    background-color: rgba(167, 139, 250, 22);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.3px;
    padding: 2px 7px;
    border-radius: 8px;
    border: 1px solid rgba(167, 139, 250, 55);
}

/* ── Filler word / confidence badge — color set dynamically in Python ─── */
#fillerBadge {
    font-size: 9px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 3px;
}

/* ── Question display ─────────────────────────────────────────────────── */
#question {
    color: #7a8494;
    font-size: 12px;
    font-style: italic;
    padding: 2px 0;
}

/* ── Live transcription panel ─────────────────────────────────────────── */
/* Same font size as answer for easy reading while the interviewer speaks. */
#liveTranscript {
    background-color: rgba(255, 255, 255, 5);
    color: #b0b8c8;
    border: none;
    border-left: 2px solid rgba(124, 200, 255, 70);
    font-size: 14px;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    padding: 4px 8px;
    border-radius: 0 4px 4px 0;
}

#liveTranscriptLabel {
    color: #7CC8FF;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 2px 0;
}

/* ── Answer display ───────────────────────────────────────────────────── */
/* The PRIMARY section is rendered inline as HTML with gold background.    */
/* This base style applies to the QTextBrowser container only.             */
#answer {
    background-color: transparent;
    color: #f1f3f5;
    border: none;
    font-size: 14px;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    selection-background-color: rgba(124, 200, 255, 60);
}

#answer b, #answer strong {
    color: #FFD166;
    font-weight: 700;
}

/* ── Footer ───────────────────────────────────────────────────────────── */
#footer {
    color: #444c5c;
    font-size: 10px;
    padding-top: 2px;
}

/* ── Icon buttons ─────────────────────────────────────────────────────── */
#iconBtn {
    background-color: transparent;
    color: #c8ced9;
    border: none;
    font-size: 14px;
    border-radius: 4px;
}

#iconBtn:hover {
    background-color: rgba(255, 255, 255, 22);
    color: white;
}

#iconBtn[text="\u00d7"]:hover {
    background-color: rgba(232, 80, 80, 80);
    color: white;
}

/* ── Settings dialog ──────────────────────────────────────────────────── */
QDialog {
    background-color: #14171f;
    color: #e6e9ef;
}

QLabel { color: #c8ced9; }

QLineEdit, QTextEdit, QComboBox, QDoubleSpinBox {
    background-color: #0d1018;
    color: #f1f3f5;
    border: 1px solid #252b38;
    border-radius: 4px;
    padding: 4px 6px;
}

QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
    border: 1px solid rgba(124, 200, 255, 80);
}

QPushButton {
    background-color: #252b38;
    color: #f1f3f5;
    border: none;
    padding: 6px 14px;
    border-radius: 4px;
}
QPushButton:hover { background-color: #333b4e; }

QTabWidget::pane { border: 1px solid #252b38; }
QTabBar::tab {
    background: #0d1018;
    color: #9aa3b2;
    padding: 6px 12px;
    border: 1px solid #252b38;
    border-bottom: none;
}
QTabBar::tab:selected { background: #252b38; color: #f1f3f5; }

QCheckBox { color: #c8ced9; }
QCheckBox::indicator { border: 1px solid #3a4150; border-radius: 3px; }
QCheckBox::indicator:checked { background-color: #7CC8FF; border-color: #7CC8FF; }
"""
