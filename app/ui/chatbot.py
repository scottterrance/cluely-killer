"""Standalone chatbot overlay window.

A separate, independently hideable frameless window that lets the user
type free-form questions and receive streaming DeepSeek answers.
It is completely independent of the main cheating overlay — no audio,
no hotkey-triggered answers — just a plain chat interface.

Hotkey default: <ctrl>+<shift>+c  (toggle show/hide)
"""
from __future__ import annotations

import threading
from typing import Callable

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QKeyEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollBar,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings
from ..llm.deepseek_provider import DeepSeekProvider

# ── Minimal QSS for the chatbot window ────────────────────────────────────
_CHATBOT_QSS = """
#ChatbotRoot {
    background-color: transparent;
}
#chatContainer {
    background-color: rgba(13, 16, 24, 230);
    border: 1px solid rgba(124, 200, 255, 55);
    border-radius: 10px;
}
#chatHeader {
    color: #7CC8FF;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    padding: 0 4px;
}
#chatHistory {
    background-color: transparent;
    color: #f1f3f5;
    border: none;
    font-size: 13px;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    selection-background-color: rgba(124, 200, 255, 60);
}
#chatInput {
    background-color: rgba(255, 255, 255, 8);
    color: #f1f3f5;
    border: 1px solid rgba(124, 200, 255, 50);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
    font-family: 'Segoe UI', 'Inter', sans-serif;
}
#chatInput:focus {
    border: 1px solid rgba(124, 200, 255, 140);
}
#sendBtn {
    background-color: rgba(124, 200, 255, 30);
    color: #7CC8FF;
    border: 1px solid rgba(124, 200, 255, 60);
    border-radius: 6px;
    font-size: 13px;
    font-weight: 700;
    padding: 6px 14px;
}
#sendBtn:hover {
    background-color: rgba(124, 200, 255, 60);
}
#sendBtn:disabled {
    color: #444c5c;
    border-color: rgba(68, 76, 92, 60);
    background-color: transparent;
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
#statusLabel {
    color: #444c5c;
    font-size: 10px;
    padding: 0 4px;
}
"""

# ── Internal Qt signals for thread-safe UI updates ────────────────────────
class _ChatSignals(QObject):
    chunk_received = pyqtSignal(str)
    response_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)


class ChatbotWindow(QWidget):
    """Standalone chatbot overlay.

    Parameters
    ----------
    settings:
        Shared Settings object (reads deepseek_api_key / model / base_url).
    simple_mode:
        When True, use a normal titled window instead of frameless.
    """

    def __init__(
        self,
        settings: Settings,
        simple_mode: bool = False,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.simple_mode = simple_mode
        self._drag_offset = None
        self._is_streaming = False
        self._current_response = ""
        # Chat history as list of (role, text) tuples for context
        self._history: list[dict] = []
        # Signals bridge (worker thread -> Qt main thread)
        self._signals = _ChatSignals()
        self._signals.chunk_received.connect(self._on_chunk)
        self._signals.response_finished.connect(self._on_finished)
        self._signals.error_occurred.connect(self._on_error)
        self._setup_window()
        self._build_ui()

    # ------------------------------------------------------------------
    def _setup_window(self) -> None:
        self.setObjectName("ChatbotRoot")
        if self.simple_mode:
            self.setWindowTitle("cluely-killer — Chatbot")
            self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
            )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(self.settings.opacity)
        self.setMinimumSize(380, 420)
        self.resize(420, 520)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        container = QFrame(self)
        container.setObjectName("chatContainer")
        outer.addWidget(container)

        v = QVBoxLayout(container)
        v.setContentsMargins(10, 8, 10, 10)
        v.setSpacing(6)

        # ── Header row ────────────────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(4)

        title = QLabel("CHATBOT")
        title.setObjectName("chatHeader")
        header.addWidget(title)
        header.addStretch()

        # Clear chat button
        clear_btn = QPushButton("⌫")
        clear_btn.setObjectName("iconBtn")
        clear_btn.setFixedSize(26, 26)
        clear_btn.setToolTip("Clear chat history")
        clear_btn.clicked.connect(self._clear_chat)
        header.addWidget(clear_btn)

        # Hide button
        hide_btn = QPushButton("×")
        hide_btn.setObjectName("iconBtn")
        hide_btn.setFixedSize(26, 26)
        hide_btn.setToolTip("Hide chatbot (hotkey to show again)")
        hide_btn.clicked.connect(self.hide)
        header.addWidget(hide_btn)

        v.addLayout(header)

        # ── Chat history display ───────────────────────────────────────
        self._history_view = QTextBrowser()
        self._history_view.setObjectName("chatHistory")
        self._history_view.setOpenExternalLinks(False)
        self._history_view.setFrameShape(QFrame.Shape.NoFrame)
        v.addWidget(self._history_view, stretch=1)

        # ── Status label ──────────────────────────────────────────────
        self._status_label = QLabel("Type a message and press Enter or Send")
        self._status_label.setObjectName("statusLabel")
        v.addWidget(self._status_label)

        # ── Input row ─────────────────────────────────────────────────
        input_row = QHBoxLayout()
        input_row.setSpacing(6)

        self._input = QLineEdit()
        self._input.setObjectName("chatInput")
        self._input.setPlaceholderText("Ask anything...")
        self._input.returnPressed.connect(self._send_message)
        input_row.addWidget(self._input, stretch=1)

        self._send_btn = QPushButton("Send")
        self._send_btn.setObjectName("sendBtn")
        self._send_btn.setFixedWidth(60)
        self._send_btn.clicked.connect(self._send_message)
        input_row.addWidget(self._send_btn)

        v.addLayout(input_row)

        self.setStyleSheet(_CHATBOT_QSS)

    # ------------------------------------------------------------------
    # Chat logic
    # ------------------------------------------------------------------
    def _send_message(self) -> None:
        text = self._input.text().strip()
        if not text or self._is_streaming:
            return
        self._input.clear()
        self._append_message("user", text)
        self._history.append({"role": "user", "content": text})
        self._start_streaming(text)

    def _start_streaming(self, user_text: str) -> None:
        """Kick off a background thread that streams the DeepSeek response."""
        api_key = self.settings.deepseek_api_key
        if not api_key:
            self._signals.error_occurred.emit(
                "No DeepSeek API key — set one in Settings → AI Provider."
            )
            return

        self._is_streaming = True
        self._current_response = ""
        self._send_btn.setEnabled(False)
        self._status_label.setText("Thinking…")

        # Build prior messages for multi-turn context (last 10 turns)
        prior = self._history[:-1]  # exclude the message we just added
        prior = prior[-10:] if len(prior) > 10 else prior

        def _worker() -> None:
            try:
                provider = DeepSeekProvider(
                    api_key=api_key,
                    model=self.settings.deepseek_model,
                    base_url=self.settings.deepseek_base_url,
                    max_tokens=800,
                )
                system_prompt = (
                    "You are a helpful, concise assistant. "
                    "Answer the user's questions clearly and directly. "
                    "Use markdown formatting where it helps readability."
                )
                for chunk in provider.stream_chat(
                    system_prompt=system_prompt,
                    user_message=user_text,
                    prior_messages=prior if prior else None,
                ):
                    self._signals.chunk_received.emit(chunk)
                self._signals.response_finished.emit()
            except Exception as exc:
                self._signals.error_occurred.emit(str(exc)[:200])

        threading.Thread(target=_worker, daemon=True).start()

    @pyqtSlot(str)
    def _on_chunk(self, chunk: str) -> None:
        self._current_response += chunk
        self._refresh_assistant_bubble()

    @pyqtSlot()
    def _on_finished(self) -> None:
        self._is_streaming = False
        self._send_btn.setEnabled(True)
        self._status_label.setText("Ready")
        if self._current_response:
            self._history.append({"role": "assistant", "content": self._current_response})
        self._current_response = ""

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._is_streaming = False
        self._send_btn.setEnabled(True)
        self._status_label.setText(f"Error: {msg[:80]}")
        self._append_message("error", msg)
        self._current_response = ""

    def _clear_chat(self) -> None:
        self._history.clear()
        self._history_view.setHtml("")
        self._status_label.setText("Chat cleared")

    # ------------------------------------------------------------------
    # HTML rendering helpers
    # ------------------------------------------------------------------
    def _append_message(self, role: str, text: str) -> None:
        """Append a fully-formed user or error bubble to the history view."""
        if role == "user":
            bubble_html = self._user_bubble_html(text)
        else:
            bubble_html = self._error_bubble_html(text)
        # Append to existing HTML
        current = self._history_view.toHtml()
        # Strip the trailing </body></html> so we can append
        if "</body>" in current:
            current = current[: current.rfind("</body>")]
        else:
            current = ""
        self._history_view.setHtml(current + bubble_html + "</body></html>")
        self._scroll_to_bottom()

    def _refresh_assistant_bubble(self) -> None:
        """Re-render the entire history with the current streaming response."""
        html = self._build_full_html()
        self._history_view.setHtml(html)
        self._scroll_to_bottom()

    def _build_full_html(self) -> str:
        parts: list[str] = []
        for msg in self._history:
            if msg["role"] == "user":
                parts.append(self._user_bubble_html(msg["content"]))
            elif msg["role"] == "assistant":
                parts.append(self._assistant_bubble_html(msg["content"]))
        # Append streaming partial response
        if self._current_response:
            parts.append(self._assistant_bubble_html(self._current_response, streaming=True))
        return "".join(parts)

    @staticmethod
    def _escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
        )

    def _user_bubble_html(self, text: str) -> str:
        escaped = self._escape(text)
        return (
            f'<div style="margin:6px 0 6px 40px;padding:8px 12px;'
            f'background:rgba(124,200,255,0.15);'
            f'border-radius:10px 10px 2px 10px;'
            f'color:#e0f0ff;font-size:13px;line-height:1.5;">'
            f'{escaped}</div>'
        )

    def _assistant_bubble_html(self, text: str, streaming: bool = False) -> str:
        escaped = self._escape(text)
        cursor = '<span style="color:#7CC8FF;">▌</span>' if streaming else ""
        return (
            f'<div style="margin:6px 40px 6px 0;padding:8px 12px;'
            f'background:rgba(255,255,255,0.05);'
            f'border-left:3px solid rgba(124,200,255,0.5);'
            f'border-radius:0 10px 10px 0;'
            f'color:#f1f3f5;font-size:13px;line-height:1.5;">'
            f'{escaped}{cursor}</div>'
        )

    def _error_bubble_html(self, text: str) -> str:
        escaped = self._escape(text)
        return (
            f'<div style="margin:6px 0;padding:8px 12px;'
            f'background:rgba(255,107,107,0.12);'
            f'border-left:3px solid rgba(255,107,107,0.6);'
            f'border-radius:0 6px 6px 0;'
            f'color:#fca5a5;font-size:12px;">'
            f'Error: {escaped}</div>'
        )

    def _scroll_to_bottom(self) -> None:
        QTimer.singleShot(
            0,
            lambda: self._history_view.verticalScrollBar().setValue(
                self._history_view.verticalScrollBar().maximum()
            ),
        )

    # ------------------------------------------------------------------
    # Drag support (frameless mode)
    # ------------------------------------------------------------------
    def mousePressEvent(self, e: QKeyEvent) -> None:
        if self.simple_mode:
            return super().mousePressEvent(e)
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = (
                e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            e.accept()

    def mouseMoveEvent(self, e: QKeyEvent) -> None:
        if self.simple_mode:
            return super().mouseMoveEvent(e)
        if self._drag_offset is not None and e.buttons() & Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_offset)
            e.accept()

    def mouseReleaseEvent(self, e: QKeyEvent) -> None:
        self._drag_offset = None

    # ------------------------------------------------------------------
    def toggle_visibility(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()
            self._input.setFocus()
