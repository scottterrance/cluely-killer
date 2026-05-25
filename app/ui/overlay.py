"""Frameless, draggable, always-on-top overlay window.

Renders the LLM stream as it arrives. **bold** is converted to <b>
on every chunk so keyword highlighting appears live.

Window flags:
  - FramelessWindowHint     : no title bar (we draw our own header)
  - WindowStaysOnTopHint    : always above the meeting window
  - NO WA_TranslucentBackground: this attribute combined with frameless
    causes the window to render zero pixels on Windows 11 24H2 (build
    26100+) under newer DWM compositors. We use a solid dark background
    instead. Stealth-via-WDA_EXCLUDEFROMCAPTURE still works without it.
  - NO Qt.WindowType.Tool   : keeping the window in the taskbar makes it
    findable if it ever ends up off-screen. Stealth users can opt it
    back via Settings later if desired.

Pass simple_mode=True for a fully-normal titled window (debug fallback).
"""
from __future__ import annotations

import re
from typing import Callable

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings
from ..core.controller import Controller
from .styles import APP_QSS

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", flags=re.DOTALL)


def _md_to_html(text: str) -> str:
    text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    return text.replace("\n", "<br>")


class OverlayWindow(QWidget):
    def __init__(
        self,
        settings: Settings,
        controller: Controller,
        on_open_settings: Callable[[], None],
        simple_mode: bool = False,
    ):
        super().__init__()
        self.settings = settings
        self.controller = controller
        self.on_open_settings = on_open_settings
        self.simple_mode = simple_mode
        self._answer_text = ""
        self._drag_offset = None

        self._setup_window()
        self._build_ui()
        self._wire_signals()

    # ------------------------------------------------------------------
    def _setup_window(self) -> None:
        self.setObjectName("OverlayRoot")
        if self.simple_mode:
            # Plain decorated window — guaranteed visible everywhere.
            self.setWindowTitle("cluely-killer")
            self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
            )
        # NO WA_TranslucentBackground: it interacts badly with Qt6 +
        # frameless on Windows 11 24H2 and produces a 0-pixel window.
        self.setWindowOpacity(self.settings.opacity)
        self.setMinimumSize(420, 240)
        self.resize(
            max(self.settings.window_w, 420),
            max(self.settings.window_h, 240),
        )

    def place_on_screen(self) -> None:
        """Center on the primary screen, or restore last position if it lies
        on a currently-attached monitor. Always called *after* show().
        """
        from PyQt6.QtGui import QGuiApplication

        screens = QGuiApplication.screens()
        target = None
        sx, sy = self.settings.window_x, self.settings.window_y
        if sx >= 0 and sy >= 0:
            for s in screens:
                g = s.geometry()
                if g.contains(sx + 20, sy + 20):
                    target = (sx, sy)
                    break
        if target is None:
            primary = QGuiApplication.primaryScreen().availableGeometry()
            target = (
                primary.x() + (primary.width() - self.width()) // 2,
                primary.y() + (primary.height() - self.height()) // 3,
            )
        self.move(*target)
        self.raise_()
        self.activateWindow()
        print(
            f"[overlay] placed at x={self.x()} y={self.y()} "
            f"size={self.width()}x{self.height()} "
            f"on a {len(screens)}-screen setup. "
            f"simple_mode={self.simple_mode}."
        )

    def closeEvent(self, e):  # noqa: N802 (Qt API)
        self.settings.window_x = self.x()
        self.settings.window_y = self.y()
        self.settings.window_w = self.width()
        self.settings.window_h = self.height()
        super().closeEvent(e)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        container = QFrame(self)
        container.setObjectName("container")
        outer.addWidget(container)

        v = QVBoxLayout(container)
        v.setContentsMargins(14, 10, 14, 12)
        v.setSpacing(8)

        # --- Header bar ---
        header = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("status")
        header.addWidget(self.status_label)
        header.addStretch()

        self.settings_btn = QPushButton("\u2699")
        self.settings_btn.setObjectName("iconBtn")
        self.settings_btn.setFixedSize(26, 26)
        self.settings_btn.setToolTip("Settings")
        self.settings_btn.clicked.connect(lambda: self.on_open_settings())
        header.addWidget(self.settings_btn)

        self.hide_btn = QPushButton("\u2013")
        self.hide_btn.setObjectName("iconBtn")
        self.hide_btn.setFixedSize(26, 26)
        self.hide_btn.setToolTip("Hide")
        self.hide_btn.clicked.connect(self.hide)
        header.addWidget(self.hide_btn)

        v.addLayout(header)

        self.question_label = QLabel("Press Ctrl+Space to answer the last question.")
        self.question_label.setObjectName("question")
        self.question_label.setWordWrap(True)
        self.question_label.setMaximumHeight(60)
        v.addWidget(self.question_label)

        self.answer_view = QTextBrowser()
        self.answer_view.setObjectName("answer")
        self.answer_view.setOpenExternalLinks(False)
        self.answer_view.setFrameShape(QFrame.Shape.NoFrame)
        v.addWidget(self.answer_view, stretch=1)

        self.footer = QLabel(self._footer_text())
        self.footer.setObjectName("footer")
        v.addWidget(self.footer)

        self.setStyleSheet(APP_QSS)

    def _footer_text(self) -> str:
        return (
            f"{self.settings.hotkey_answer} answer  \u00b7  "
            f"{self.settings.hotkey_toggle} hide  \u00b7  "
            f"{self.settings.hotkey_clear} clear  \u00b7  "
            f"{self.settings.hotkey_settings} settings"
        )

    def refresh_footer(self) -> None:
        self.footer.setText(self._footer_text())

    # ------------------------------------------------------------------
    def _wire_signals(self) -> None:
        c = self.controller
        c.transcript_ready.connect(self._on_transcript)
        c.answer_started.connect(self._on_answer_start)
        c.answer_chunk.connect(self._on_answer_chunk)
        c.answer_finished.connect(self._on_answer_finished)
        c.error.connect(self._on_error)
        c.status.connect(self._on_status)

    @pyqtSlot(str)
    def _on_transcript(self, text: str) -> None:
        display = text if len(text) <= 220 else "..." + text[-220:]
        self.question_label.setText(f"Q: {display}")

    @pyqtSlot()
    def _on_answer_start(self) -> None:
        self._answer_text = ""
        self.answer_view.setHtml("")
        self.status_label.setText("Answering...")

    @pyqtSlot(str)
    def _on_answer_chunk(self, chunk: str) -> None:
        self._answer_text += chunk
        self.answer_view.setHtml(_md_to_html(self._answer_text))
        cursor = self.answer_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.answer_view.setTextCursor(cursor)

    @pyqtSlot()
    def _on_answer_finished(self) -> None:
        self.status_label.setText("Ready")

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self.status_label.setText(msg[:80])

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self.status_label.setText(msg)

    # ------------------------------------------------------------------
    # Custom drag (only meaningful in frameless mode; harmless otherwise)
    def mousePressEvent(self, e):
        if self.simple_mode:
            return super().mousePressEvent(e)
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e):
        if self.simple_mode:
            return super().mouseMoveEvent(e)
        if self._drag_offset is not None and e.buttons() & Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_offset)
            e.accept()

    def mouseReleaseEvent(self, e):
        self._drag_offset = None

    # ------------------------------------------------------------------
    def toggle_visibility(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()
