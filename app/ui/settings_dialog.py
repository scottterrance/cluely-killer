"""Settings dialog. Tabbed: Provider, Context, Audio/STT, Hotkeys, Window."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings
from ..utils.extract import extract_text
from .drop_text_edit import DropZoneTextEdit


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("cluely-killer — Settings")
        self.resize(660, 580)

        tabs = QTabWidget()
        tabs.addTab(self._provider_tab(), "AI Provider")
        tabs.addTab(self._context_tab(), "Your Context")
        tabs.addTab(self._audio_tab(), "Audio / STT")
        tabs.addTab(self._hotkeys_tab(), "Hotkeys")
        tabs.addTab(self._window_tab(), "Window")

        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        save_btn.clicked.connect(self._save)
        cancel_btn.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(cancel_btn)
        buttons.addWidget(save_btn)

        outer = QVBoxLayout(self)
        outer.addWidget(tabs)
        outer.addLayout(buttons)

    # ------------------------------------------------------------------
    def _provider_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["groq", "openrouter", "ollama"])
        self.provider_combo.setCurrentText(self.settings.provider)

        self.groq_key = QLineEdit(self.settings.groq_api_key)
        self.groq_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.groq_model = QLineEdit(self.settings.groq_model)

        self.openrouter_key = QLineEdit(self.settings.openrouter_api_key)
        self.openrouter_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.openrouter_model = QLineEdit(self.settings.openrouter_model)

        self.ollama_model = QLineEdit(self.settings.ollama_model)
        self.ollama_host = QLineEdit(self.settings.ollama_host)

        f.addRow("Provider:", self.provider_combo)
        f.addRow(QLabel("<b>Groq (cloud, fast, free tier)</b>"))
        f.addRow("API key:", self.groq_key)
        f.addRow("Model:", self.groq_model)
        f.addRow(QLabel(
            "<b>OpenRouter (cloud, fewer IP blocks than Groq)</b>"
            "<br><i>Free key at <code>https://openrouter.ai/keys</code>. "
            "Free models end in <code>:free</code>.</i>"
        ))
        f.addRow("API key:", self.openrouter_key)
        f.addRow("Model:", self.openrouter_model)
        f.addRow(QLabel("<b>Ollama (local model)</b>"))
        f.addRow("Model:", self.ollama_model)
        f.addRow("Host:", self.ollama_host)
        return w

    def _context_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel("About me (1-3 sentences):"))
        self.about_edit = QTextEdit(self.settings.about_me)
        self.about_edit.setMaximumHeight(70)
        v.addWidget(self.about_edit)

        # Resume row: label + Import button on the right.
        resume_row = QHBoxLayout()
        resume_row.addWidget(QLabel("Resume:"))
        resume_row.addStretch()
        resume_import_btn = QPushButton("Import .pdf / .docx / .txt...")
        resume_import_btn.setToolTip("Pick a file to extract its text into this box.")
        resume_import_btn.clicked.connect(lambda: self._import_into(self.resume_edit, "resume"))
        resume_row.addWidget(resume_import_btn)
        v.addLayout(resume_row)
        self.resume_edit = DropZoneTextEdit()
        self.resume_edit.setPlainText(self.settings.resume_text)
        self.resume_edit.setPlaceholderText(
            "Drag a .pdf / .docx / .txt onto this box, or click Import. "
            "You can also paste plain text."
        )
        v.addWidget(self.resume_edit)

        # Job description row: same pattern.
        jd_row = QHBoxLayout()
        jd_row.addWidget(QLabel("Job description:"))
        jd_row.addStretch()
        jd_import_btn = QPushButton("Import .pdf / .docx / .txt...")
        jd_import_btn.clicked.connect(lambda: self._import_into(self.job_edit, "job description"))
        jd_row.addWidget(jd_import_btn)
        v.addLayout(jd_row)
        self.job_edit = DropZoneTextEdit()
        self.job_edit.setPlainText(self.settings.job_description)
        self.job_edit.setPlaceholderText(
            "Drag a .pdf / .docx / .txt onto this box, or click Import."
        )
        self.job_edit.setMaximumHeight(120)
        v.addWidget(self.job_edit)

        v.addWidget(QLabel("Custom system prompt (advanced - appended to base rules):"))
        self.custom_edit = QTextEdit(self.settings.custom_system_prompt)
        self.custom_edit.setMaximumHeight(80)
        v.addWidget(self.custom_edit)
        return w

    def _import_into(self, target_edit: QTextEdit, label: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Import {label}",
            "",
            "Documents (*.pdf *.docx *.txt *.md);;All files (*)",
        )
        if not path:
            return
        try:
            text = extract_text(path)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Import failed",
                f"Could not read {Path(path).name}:\n\n"
                f"{type(exc).__name__}: {exc}",
            )
            return
        if not text:
            QMessageBox.information(
                self,
                "Empty file",
                f"{Path(path).name} appears to contain no extractable text.",
            )
            return
        target_edit.setPlainText(text)

    def _audio_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.whisper_model_combo.setCurrentText(self.settings.whisper_model)

        self.whisper_compute_combo = QComboBox()
        self.whisper_compute_combo.addItems(["int8", "int8_float16", "float16", "float32"])
        self.whisper_compute_combo.setCurrentText(self.settings.whisper_compute)

        self.whisper_device_combo = QComboBox()
        self.whisper_device_combo.addItems(["cpu", "cuda"])
        self.whisper_device_combo.setCurrentText(self.settings.whisper_device)

        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(5.0, 60.0)
        self.window_spin.setSingleStep(1.0)
        self.window_spin.setValue(self.settings.answer_window_seconds)

        f.addRow("Whisper model:", self.whisper_model_combo)
        f.addRow("Compute type:", self.whisper_compute_combo)
        f.addRow("Device:", self.whisper_device_combo)
        f.addRow("Audio window (sec):", self.window_spin)
        f.addRow(QLabel("<i>Whisper changes apply on next app restart.</i>"))
        return w

    def _hotkeys_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.hk_answer = QLineEdit(self.settings.hotkey_answer)
        self.hk_toggle = QLineEdit(self.settings.hotkey_toggle)
        self.hk_clear = QLineEdit(self.settings.hotkey_clear)
        self.hk_settings = QLineEdit(self.settings.hotkey_settings)
        self.hk_quit = QLineEdit(self.settings.hotkey_quit)
        f.addRow("Answer:", self.hk_answer)
        f.addRow("Toggle overlay:", self.hk_toggle)
        f.addRow("Clear buffer:", self.hk_clear)
        f.addRow("Open settings:", self.hk_settings)
        f.addRow("Quit app:", self.hk_quit)
        f.addRow(
            QLabel(
                "<i>pynput syntax — e.g. &lt;ctrl&gt;+&lt;space&gt;, "
                "&lt;ctrl&gt;+&lt;shift&gt;+s, &lt;alt&gt;+a</i>"
            )
        )
        return w

    def _window_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.exclude_check = QCheckBox("Hide window from screen capture (Windows 10 build 19041+)")
        self.exclude_check.setChecked(self.settings.exclude_from_capture)

        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.4, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setDecimals(2)
        self.opacity_spin.setValue(self.settings.opacity)

        f.addRow(self.exclude_check)
        f.addRow("Opacity:", self.opacity_spin)
        return w

    # ------------------------------------------------------------------
    def _save(self) -> None:
        s = self.settings
        s.provider = self.provider_combo.currentText()
        s.groq_api_key = self.groq_key.text().strip()
        s.groq_model = self.groq_model.text().strip()
        s.openrouter_api_key = self.openrouter_key.text().strip()
        s.openrouter_model = self.openrouter_model.text().strip()
        s.ollama_model = self.ollama_model.text().strip()
        s.ollama_host = self.ollama_host.text().strip()

        s.about_me = self.about_edit.toPlainText()
        s.resume_text = self.resume_edit.toPlainText()
        s.job_description = self.job_edit.toPlainText()
        s.custom_system_prompt = self.custom_edit.toPlainText()

        s.whisper_model = self.whisper_model_combo.currentText()
        s.whisper_compute = self.whisper_compute_combo.currentText()
        s.whisper_device = self.whisper_device_combo.currentText()
        s.answer_window_seconds = float(self.window_spin.value())

        s.hotkey_answer = self.hk_answer.text().strip()
        s.hotkey_toggle = self.hk_toggle.text().strip()
        s.hotkey_clear = self.hk_clear.text().strip()
        s.hotkey_settings = self.hk_settings.text().strip()
        s.hotkey_quit = self.hk_quit.text().strip()

        s.exclude_from_capture = self.exclude_check.isChecked()
        s.opacity = float(self.opacity_spin.value())
        self.accept()
