"""Frameless, draggable, always-on-top overlay window.

Renders the LLM stream as it arrives. Two highlight conventions are
recognized:

  ==word==   -> RED   (most stressed keywords, 2-3 per sentence)
  **word**   -> YELLOW (softer secondary emphasis, sparing)

The header carries a STEALTH / VISIBLE badge so the candidate can verify
at a glance that the overlay is hidden from screen capture before
starting the interview, plus a small "mem N" badge showing how many
prior Q+A turns the LLM is remembering.
"""
from __future__ import annotations

import re
from typing import Callable

from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from ..config import Settings
from ..core.controller import Controller
from ..core.analysis import FillerReport, ClassificationResult
from .styles import APP_QSS

# Order matters: parse ==red== BEFORE **bold** so the regexes don't fight
# over '=' / '*' boundaries on partial streams.
_RED_RE = re.compile(r"==(.+?)==", flags=re.DOTALL)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", flags=re.DOTALL)

_RED_STYLE = 'color:#FF6B6B;font-weight:700;'


def _md_to_html(text: str) -> str:
    text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    text = _RED_RE.sub(rf'<span style="{_RED_STYLE}">\1</span>', text)
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    return text.replace("\n", "<br>")


class OverlayWindow(QWidget):
    def __init__(
        self,
        settings: Settings,
        controller: Controller,
        on_open_settings: Callable[[], None],
        on_quit: Callable[[], None],
        simple_mode: bool = False,
    ):
        super().__init__()
        self.settings = settings
        self.controller = controller
        self.on_open_settings = on_open_settings
        self.on_quit = on_quit
        self.simple_mode = simple_mode
        self._answer_text = ""
        self._drag_offset = None
        self._user_scrolled_down = False  # tracks if user scrolled down during streaming
        # Accumulated live transcript text (rolling window of last ~800 chars)
        self._live_transcript_text = ""

        self._setup_window()
        self._build_ui()
        self._wire_signals()
        self.update_stealth_badge(settings.exclude_from_capture)

    # ------------------------------------------------------------------
    def _setup_window(self) -> None:
        self.setObjectName("OverlayRoot")
        if self.simple_mode:
            self.setWindowTitle("cluely-killer")
            self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
            )
        self.setWindowOpacity(self.settings.opacity)
        # Taller minimum so long answers are always fully visible.
        self.setMinimumSize(420, 520)
        self.resize(
            max(self.settings.window_w, 420),
            max(self.settings.window_h, 600),
        )

    def place_on_screen(self) -> None:
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

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        container = QFrame(self)
        container.setObjectName("container")
        outer.addWidget(container)

        v = QVBoxLayout(container)
        v.setContentsMargins(14, 10, 14, 12)
        v.setSpacing(8)

        # --- Header ---
        header = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("status")
        header.addWidget(self.status_label)

        # Question type badge: shows auto-classified type of last question.
        self.qtype_label = QLabel("")
        self.qtype_label.setObjectName("qtypeBadge")
        self.qtype_label.setToolTip("Auto-detected question type and recommended depth")
        self.qtype_label.setVisible(False)
        header.addWidget(self.qtype_label)

        # Filler word / confidence badge: shows clarity score of last question.
        self.filler_label = QLabel("")
        self.filler_label.setObjectName("fillerBadge")
        self.filler_label.setToolTip("Interviewer speech clarity (filler word count)")
        self.filler_label.setVisible(False)
        header.addWidget(self.filler_label)

        # Hidden labels kept for internal signal wiring but not shown in UI.
        # mem and backend info is available via status_label tooltip.
        self.mem_label = QLabel("")
        self.mem_label.setVisible(False)
        self.backend_label = QLabel("")
        self.backend_label.setProperty("alarm", "false")
        self.backend_label.setVisible(False)

        header.addStretch()

        self.stealth_label = QLabel("STEALTH")
        self.stealth_label.setObjectName("stealthBadge")
        self.stealth_label.setProperty("alarm", "false")
        self.stealth_label.setToolTip("Hidden from Zoom / Teams / Meet / OBS screen capture.")
        header.addWidget(self.stealth_label)

        self.settings_btn = QPushButton("\u2699")
        self.settings_btn.setObjectName("iconBtn")
        self.settings_btn.setFixedSize(26, 26)
        self.settings_btn.setToolTip("Settings")
        self.settings_btn.clicked.connect(lambda: self.on_open_settings())
        header.addWidget(self.settings_btn)

        self.hide_btn = QPushButton("\u2013")
        self.hide_btn.setObjectName("iconBtn")
        self.hide_btn.setFixedSize(26, 26)
        self.hide_btn.setToolTip("Hide (still running in tray)")
        self.hide_btn.clicked.connect(self.hide)
        header.addWidget(self.hide_btn)

        self.quit_btn = QPushButton("\u00d7")
        self.quit_btn.setObjectName("iconBtn")
        self.quit_btn.setFixedSize(26, 26)
        self.quit_btn.setToolTip("Quit cluely-killer")
        self.quit_btn.clicked.connect(lambda: self.on_quit())
        header.addWidget(self.quit_btn)

        v.addLayout(header)

        # --- Question / transcript (last confirmed question) ---
        self.question_label = QLabel(
            "Press '1' for a quick answer to the last thing said, "
            "or '2' to also use the last 5 Q+A as context."
        )
        self.question_label.setObjectName("question")
        self.question_label.setWordWrap(True)
        self.question_label.setMaximumHeight(50)
        v.addWidget(self.question_label)

        # --- Live transcription panel (optional, same font size as answer) ---
        self._live_transcript_header = QLabel("LIVE")
        self._live_transcript_header.setObjectName("liveTranscriptLabel")
        self._live_transcript_view = QTextBrowser()
        self._live_transcript_view.setObjectName("liveTranscript")
        self._live_transcript_view.setOpenExternalLinks(False)
        self._live_transcript_view.setFrameShape(QFrame.Shape.NoFrame)
        # Allow up to ~5 lines (~110px) so longer sentences are readable.
        # No hard max - the panel will grow with content up to this soft cap.
        self._live_transcript_view.setMaximumHeight(110)
        self._live_transcript_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )
        v.addWidget(self._live_transcript_header)
        v.addWidget(self._live_transcript_view, stretch=0)
        # Show/hide based on setting
        self._apply_live_transcript_visibility()

        # --- Answer ---
        self.answer_view = QTextBrowser()
        self.answer_view.setObjectName("answer")
        self.answer_view.setOpenExternalLinks(False)
        self.answer_view.setFrameShape(QFrame.Shape.NoFrame)
        v.addWidget(self.answer_view, stretch=1)

        # --- Footer ---
        self.footer = QLabel(self._footer_text())
        self.footer.setObjectName("footer")
        v.addWidget(self.footer)

        self.setStyleSheet(APP_QSS)

    def _footer_text(self) -> str:
        rephrase_key = getattr(self.settings, "hotkey_rephrase", "3")
        return (
            f"{self.settings.hotkey_answer_short} answer-only  \u00b7  "
            f"{self.settings.hotkey_answer_context} answer+context  \u00b7  "
            f"{rephrase_key} rephrase  \u00b7  "
            f"{self.settings.hotkey_toggle} hide  \u00b7  "
            f"{self.settings.hotkey_clear} clear+forget  \u00b7  "
            f"{self.settings.hotkey_settings} settings"
        )

    def refresh_footer(self) -> None:
        self.footer.setText(self._footer_text())

    # ------------------------------------------------------------------
    def update_stealth_badge(self, enabled: bool) -> None:
        if enabled:
            self.stealth_label.setText("STEALTH")
            self.stealth_label.setToolTip(
                "STEALTH ON - hidden from Zoom / Teams / Meet / OBS screen capture."
            )
            self.stealth_label.setProperty("alarm", "false")
        else:
            self.stealth_label.setText("VISIBLE")
            self.stealth_label.setToolTip(
                "STEALTH OFF - the interviewer WILL see this overlay if you share your screen!"
            )
            self.stealth_label.setProperty("alarm", "true")
        # Re-evaluate the [alarm] style selector.
        self.stealth_label.style().unpolish(self.stealth_label)
        self.stealth_label.style().polish(self.stealth_label)

    # ------------------------------------------------------------------
    def _apply_live_transcript_visibility(self) -> None:
        """Show or hide the live transcript panel based on settings."""
        enabled = getattr(self.settings, "live_transcription_enabled", True)
        self._live_transcript_header.setVisible(enabled)
        self._live_transcript_view.setVisible(enabled)

    def refresh_live_transcript_setting(self) -> None:
        """Called after settings save to apply the live transcription toggle."""
        self._apply_live_transcript_visibility()
        if not getattr(self.settings, "live_transcription_enabled", True):
            self._live_transcript_text = ""
            self._live_transcript_view.setHtml("")

    def _wire_signals(self) -> None:
        c = self.controller
        c.transcript_ready.connect(self._on_transcript)
        c.answer_started.connect(self._on_answer_start)
        c.answer_chunk.connect(self._on_answer_chunk)
        c.answer_finished.connect(self._on_answer_finished)
        c.error.connect(self._on_error)
        c.status.connect(self._on_status)
        c.history_changed.connect(self._on_history_changed)
        c.backend_used.connect(self._on_backend_used)
        c.live_transcript_segment.connect(self._on_live_segment)
        c.filler_report.connect(self._on_filler_report)
        c.question_classified.connect(self._on_question_classified)

    @pyqtSlot(str)
    def _on_live_segment(self, text: str) -> None:
        """Append a new Whisper segment to the live transcript display.
        NEVER clears - text accumulates permanently so the user can always
        read back what was said. A rolling 800-char window keeps the panel
        from growing unbounded while preserving recent context.
        """
        if not getattr(self.settings, "live_transcription_enabled", True):
            return
        self._live_transcript_text += (" " if self._live_transcript_text else "") + text.strip()
        # Rolling window: keep only the last 800 chars so the panel
        # stays readable but never loses recent speech.
        if len(self._live_transcript_text) > 800:
            self._live_transcript_text = "\u2026 " + self._live_transcript_text[-760:]
        self._live_transcript_view.setHtml(
            f'<span style="color:#c8ced9;font-size:14px;">{self._live_transcript_text}</span>'
        )
        cursor = self._live_transcript_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._live_transcript_view.setTextCursor(cursor)

    @pyqtSlot(str)
    def _on_transcript(self, text: str) -> None:
        """Called when a full answer-ready transcript is confirmed.
        Does NOT clear the live panel - the user can keep reading the
        accumulated transcript. Just updates the confirmed Q label.
        """
        # Do NOT clear _live_transcript_text or the live panel here.
        # The user explicitly wants to keep reading what was said.
        display = text if len(text) <= 220 else "\u2026" + text[-220:]
        self.question_label.setText(f"Q: {display}")

    @pyqtSlot()
    def _on_answer_start(self) -> None:
        self._answer_text = ""
        self._user_scrolled_down = False  # reset: user hasn't scrolled yet
        self.answer_view.setHtml("")
        self.status_label.setText("Answering...")
        # Pin to top immediately.
        self.answer_view.verticalScrollBar().setValue(0)

    @pyqtSlot(str)
    def _on_answer_chunk(self, chunk: str) -> None:
        self._answer_text += chunk
        # Snapshot scroll state BEFORE setHtml() wipes it.
        sb = self.answer_view.verticalScrollBar()
        # If the user has scrolled down at all, track that intent.
        if sb.value() > 4:
            self._user_scrolled_down = True
        self.answer_view.setHtml(_md_to_html(self._answer_text))
        # setHtml() always resets the scrollbar to 0 internally.
        # We use a zero-delay timer so Qt finishes layout BEFORE we
        # restore the position - otherwise maximum() is still 0.
        if self._user_scrolled_down:
            # User scrolled down intentionally: follow the bottom.
            QTimer.singleShot(0, lambda: self.answer_view.verticalScrollBar().setValue(
                self.answer_view.verticalScrollBar().maximum()
            ))
        # else: leave at 0 (top) - first line stays visible.

    @pyqtSlot()
    def _on_answer_finished(self) -> None:
        self.status_label.setText("Ready")

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self.status_label.setText(msg[:80])

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self.status_label.setText(msg)

    @pyqtSlot(int)
    def _on_history_changed(self, n: int) -> None:
        self.mem_label.setText(f"mem {n}")

    @pyqtSlot(str, str, bool)
    def _on_backend_used(self, stt_label: str, llm_label: str, fell_back: bool) -> None:
        # Compact ground-truth readout, e.g. "local (continuous) | DeepSeek".
        self.backend_label.setText(f"{stt_label}  |  {llm_label}")
        self.backend_label.setToolTip(
            f"Engines for the last answer:\n  STT: {stt_label}\n  LLM: {llm_label}"
        )
        self.backend_label.setProperty("alarm", "true" if fell_back else "false")
        self.backend_label.style().unpolish(self.backend_label)
        self.backend_label.style().polish(self.backend_label)

    @pyqtSlot(object)
    def _on_filler_report(self, report: FillerReport) -> None:
        """Show the filler word / confidence badge after each answer."""
        score = report.confidence_score
        if score >= 90:
            color = "#4CAF50"  # green - very clear
            icon = "Clear"
        elif score >= 70:
            color = "#FFC107"  # amber - some fillers
            icon = report.label
        else:
            color = "#FF6B6B"  # red - many fillers
            icon = report.label
        self.filler_label.setText(icon)
        self.filler_label.setStyleSheet(
            f"color:{color};font-size:9px;font-weight:600;"
            "background:rgba(255,255,255,0.06);border-radius:3px;"
            "padding:1px 4px;"
        )
        tip_lines = [f"Clarity score: {score}/100"]
        if report.per_filler:
            tip_lines.append("Filler words detected:")
            for word, cnt in sorted(report.per_filler.items(), key=lambda x: -x[1]):
                tip_lines.append(f"  '{word}': {cnt}x")
        else:
            tip_lines.append("No filler words - very clear speech.")
        self.filler_label.setToolTip("\n".join(tip_lines))
        self.filler_label.setVisible(True)

    @pyqtSlot(object)
    def _on_question_classified(self, result: ClassificationResult) -> None:
        """Show the question type badge and depth hint after classification."""
        self.qtype_label.setText(result.display_label)
        tip = f"Type: {result.question_type}\nRecommended depth: {result.recommended_brevity}"
        if result.depth_hint:
            tip += f"\nTip: {result.depth_hint}"
        self.qtype_label.setToolTip(tip)
        self.qtype_label.setVisible(True)
        # Show the depth hint briefly in the status bar.
        if result.depth_hint:
            self.status_label.setText(result.depth_hint)
            QTimer.singleShot(3500, lambda: self.status_label.setText("Answering..."))

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
