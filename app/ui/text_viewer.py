"""Text file viewer overlay window.

A separate, independently hideable frameless window that lets the user
open multiple .txt files as tabs — similar to how a text editor shows
multiple documents. Each tab displays the file content in a scrollable
read-only view, styled to match the rest of the overlay.

The user can:
  • Open one or more .txt files via a file-picker button (adds a new tab
    for each file selected, or focuses an existing tab if already open).
  • Close individual tabs with the × on the tab label.
  • Drag the window by its header (frameless mode).
  • Hide / show with the dedicated hotkey (default: <ctrl>+<shift>+t).

Hotkey default: <ctrl>+<shift>+t  (toggle show/hide)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabBar,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings

# ── Minimal QSS for the text viewer window ────────────────────────────────
_VIEWER_QSS = """
#TextViewerRoot {
    background-color: transparent;
}
#viewerContainer {
    background-color: rgba(13, 16, 24, 230);
    border: 1px solid rgba(255, 209, 102, 55);
    border-radius: 10px;
}
#viewerHeader {
    color: #FFD166;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    padding: 0 4px;
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
#openBtn {
    background-color: rgba(255, 209, 102, 25);
    color: #FFD166;
    border: 1px solid rgba(255, 209, 102, 60);
    border-radius: 5px;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 10px;
}
#openBtn:hover {
    background-color: rgba(255, 209, 102, 55);
}
QTabWidget::pane {
    border: none;
    background: transparent;
}
QTabBar::tab {
    background: rgba(255, 255, 255, 6);
    color: #9aa3b2;
    padding: 4px 10px;
    border: 1px solid rgba(255, 255, 255, 12);
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    font-size: 11px;
    max-width: 160px;
}
QTabBar::tab:selected {
    background: rgba(255, 209, 102, 18);
    color: #FFD166;
    border-color: rgba(255, 209, 102, 55);
}
QTabBar::tab:hover:!selected {
    background: rgba(255, 255, 255, 12);
    color: #c8ced9;
}
#fileContent {
    background-color: transparent;
    color: #f1f3f5;
    border: none;
    font-size: 13px;
    font-family: 'Consolas', 'Courier New', monospace;
    selection-background-color: rgba(255, 209, 102, 60);
    line-height: 1.6;
}
#emptyLabel {
    color: #444c5c;
    font-size: 13px;
    font-style: italic;
}
#statusLabel {
    color: #444c5c;
    font-size: 10px;
    padding: 0 4px;
}
"""


class _TabLabel(QWidget):
    """Custom tab label widget with a filename and a close (×) button."""

    def __init__(self, filename: str, on_close: callable) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._label = QLabel(filename)
        self._label.setStyleSheet("color: inherit; font-size: 11px;")
        layout.addWidget(self._label)

        close_btn = QPushButton("×")
        close_btn.setFixedSize(16, 16)
        close_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #666; border: none; "
            "font-size: 13px; font-weight: bold; padding: 0; }"
            "QPushButton:hover { color: #ff6b6b; }"
        )
        close_btn.clicked.connect(on_close)
        layout.addWidget(close_btn)


class TextViewerWindow(QWidget):
    """Tabbed text file viewer overlay.

    Parameters
    ----------
    settings:
        Shared Settings object (reads opacity).
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
        # Map from absolute file path -> tab index
        self._open_files: dict[str, int] = {}
        self._setup_window()
        self._build_ui()

    # ------------------------------------------------------------------
    def _setup_window(self) -> None:
        self.setObjectName("TextViewerRoot")
        if self.simple_mode:
            self.setWindowTitle("cluely-killer — Text Viewer")
            self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
            )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(self.settings.opacity)
        self.setMinimumSize(460, 400)
        self.resize(600, 520)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        container = QFrame(self)
        container.setObjectName("viewerContainer")
        outer.addWidget(container)

        v = QVBoxLayout(container)
        v.setContentsMargins(10, 8, 10, 10)
        v.setSpacing(6)

        # ── Header row ────────────────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(4)

        title = QLabel("TEXT VIEWER")
        title.setObjectName("viewerHeader")
        header.addWidget(title)
        header.addStretch()

        # Open file(s) button
        open_btn = QPushButton("+ Open")
        open_btn.setObjectName("openBtn")
        open_btn.setToolTip("Open one or more .txt files as tabs")
        open_btn.clicked.connect(self._open_files_dialog)
        header.addWidget(open_btn)

        # Hide button
        hide_btn = QPushButton("×")
        hide_btn.setObjectName("iconBtn")
        hide_btn.setFixedSize(26, 26)
        hide_btn.setToolTip("Hide text viewer (hotkey to show again)")
        hide_btn.clicked.connect(self.hide)
        header.addWidget(hide_btn)

        v.addLayout(header)

        # ── Tab widget ────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(False)  # we handle close ourselves
        self._tabs.setMovable(True)
        self._tabs.setDocumentMode(True)
        v.addWidget(self._tabs, stretch=1)

        # ── Empty-state placeholder ───────────────────────────────────
        self._empty_widget = QWidget()
        empty_layout = QVBoxLayout(self._empty_widget)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_lbl = QLabel('Click "+ Open" to load .txt files as tabs')
        empty_lbl.setObjectName("emptyLabel")
        empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(empty_lbl)
        self._tabs.addTab(self._empty_widget, "  —  ")
        self._tabs.tabBar().setTabButton(0, QTabBar.ButtonPosition.RightSide, None)

        # ── Status label ──────────────────────────────────────────────
        self._status_label = QLabel("No files open")
        self._status_label.setObjectName("statusLabel")
        v.addWidget(self._status_label)

        self.setStyleSheet(_VIEWER_QSS)

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------
    def _open_files_dialog(self) -> None:
        """Show a file picker and open each selected file as a tab."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open text files",
            "",
            "Text files (*.txt);;All files (*.*)",
        )
        for path in paths:
            self.open_file(path)

    def open_file(self, path: str) -> None:
        """Open a file in a new tab, or focus its existing tab."""
        abs_path = str(Path(path).resolve())

        # Already open? Just switch to that tab.
        if abs_path in self._open_files:
            idx = self._open_files[abs_path]
            self._tabs.setCurrentIndex(idx)
            self._status_label.setText(f"Already open: {Path(abs_path).name}")
            return

        try:
            text = Path(abs_path).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            self._status_label.setText(f"Error reading file: {exc}")
            return

        # Remove the empty placeholder tab if it's still the only tab
        if self._tabs.count() == 1 and self._tabs.widget(0) is self._empty_widget:
            self._tabs.removeTab(0)
            self._open_files.clear()

        # Build the content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)

        viewer = QTextBrowser()
        viewer.setObjectName("fileContent")
        viewer.setOpenExternalLinks(False)
        viewer.setFrameShape(QFrame.Shape.NoFrame)
        viewer.setPlainText(text)
        content_layout.addWidget(viewer)

        filename = Path(abs_path).name
        idx = self._tabs.addTab(content_widget, "")

        # Custom tab label with close button
        tab_label = _TabLabel(filename, on_close=lambda checked=False, p=abs_path: self._close_tab(p))
        self._tabs.tabBar().setTabButton(idx, QTabBar.ButtonPosition.RightSide, tab_label)

        self._open_files[abs_path] = idx
        self._tabs.setCurrentIndex(idx)
        self._status_label.setText(f"Opened: {filename}  ({len(text):,} chars)")

    def _close_tab(self, abs_path: str) -> None:
        """Close the tab for the given file path."""
        if abs_path not in self._open_files:
            return
        idx = self._open_files.pop(abs_path)
        self._tabs.removeTab(idx)

        # Re-index remaining open files (tab indices shift after removal)
        updated: dict[str, int] = {}
        for p, old_idx in self._open_files.items():
            new_idx = old_idx - 1 if old_idx > idx else old_idx
            updated[p] = new_idx
        self._open_files = updated

        # Show empty placeholder if no tabs remain
        if self._tabs.count() == 0:
            self._tabs.addTab(self._empty_widget, "  —  ")
            self._tabs.tabBar().setTabButton(0, QTabBar.ButtonPosition.RightSide, None)
            self._status_label.setText("No files open")
        else:
            self._status_label.setText(f"{self._tabs.count()} file(s) open")

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
