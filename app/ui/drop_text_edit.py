"""QTextEdit that accepts dropped .pdf / .docx / .txt / .md files.

Drop a file onto the widget -> the text is extracted and replaces the
current contents. If extraction fails (corrupted PDF, missing optional
dep) we surface it as a friendly QMessageBox instead of swallowing it.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PyQt6.QtWidgets import QMessageBox, QTextEdit

from ..utils.extract import SUPPORTED_SUFFIXES, extract_text


class DropZoneTextEdit(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    @staticmethod
    def _supported(path: str) -> bool:
        return Path(path).suffix.lower() in SUPPORTED_SUFFIXES

    def _supported_url(self, e) -> bool:
        if not e.mimeData().hasUrls():
            return False
        return any(
            self._supported(u.toLocalFile())
            for u in e.mimeData().urls()
            if u.isLocalFile()
        )

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:  # type: ignore[override]
        if self._supported_url(e):
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e: QDragMoveEvent) -> None:  # type: ignore[override]
        if self._supported_url(e):
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e: QDropEvent) -> None:  # type: ignore[override]
        if not e.mimeData().hasUrls():
            return super().dropEvent(e)
        for url in e.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            if not self._supported(path):
                continue
            try:
                text = extract_text(path)
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Import failed",
                    f"Could not read {Path(path).name}:\n\n"
                    f"{type(exc).__name__}: {exc}",
                )
                e.ignore()
                return
            if text:
                self.setPlainText(text)
                e.acceptProposedAction()
                return
        # Fall back to default behaviour (insert text-as-text) if no
        # supported file was actually consumed above.
        super().dropEvent(e)
