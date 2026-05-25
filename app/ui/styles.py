"""Qt stylesheet for the overlay.

Solid (non-translucent) dark background. We rely on setWindowOpacity
for the soft "see through" feel, NOT WA_TranslucentBackground, because
the translucent attribute breaks frameless rendering on Win 11 24H2+.
"""

APP_QSS = """
QWidget#OverlayRoot {
    background-color: #0F0F16;
    border: 1px solid rgba(255, 255, 255, 30);
}

#container {
    background-color: #0F0F16;
}

#status {
    color: #7CC8FF;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 4px;
}

#question {
    color: #9aa3b2;
    font-size: 12px;
    font-style: italic;
    padding: 2px 0;
}

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

#footer {
    color: #555c6b;
    font-size: 10px;
    padding-top: 2px;
}

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

QDialog {
    background-color: #1a1d24;
    color: #e6e9ef;
}

QLabel { color: #c8ced9; }

QLineEdit, QTextEdit, QComboBox, QDoubleSpinBox {
    background-color: #11141a;
    color: #f1f3f5;
    border: 1px solid #2a2f3a;
    border-radius: 4px;
    padding: 4px 6px;
}

QPushButton {
    background-color: #2a2f3a;
    color: #f1f3f5;
    border: none;
    padding: 6px 14px;
    border-radius: 4px;
}
QPushButton:hover { background-color: #3a4150; }

QTabWidget::pane { border: 1px solid #2a2f3a; }
QTabBar::tab {
    background: #11141a;
    color: #9aa3b2;
    padding: 6px 12px;
    border: 1px solid #2a2f3a;
    border-bottom: none;
}
QTabBar::tab:selected { background: #2a2f3a; color: #f1f3f5; }

QCheckBox { color: #c8ced9; }
"""
