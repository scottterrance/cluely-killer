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
    QInputDialog,
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
from ..core.personas import DEFAULT_NAME, Persona, PersonaStore
from ..utils.extract import extract_text
from .drop_text_edit import DropZoneTextEdit


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.persona_store = PersonaStore()
        # First-time use: seed Default from whatever's currently in Settings
        # so the user never sees an empty dropdown.
        self.persona_store.ensure_seeded(
            Persona(
                name=DEFAULT_NAME,
                about_me=settings.about_me,
                resume_text=settings.resume_text,
                job_description=settings.job_description,
                custom_system_prompt=settings.custom_system_prompt,
            )
        )
        # Suppress the persona-changed signal during programmatic
        # repopulation so we don't trigger spurious field overwrites.
        self._suppress_persona_signal = False

        self.setWindowTitle("cluely-killer - Settings")
        self.resize(680, 620)

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

        # Build now that all widgets exist.
        self._populate_persona_dropdown()
        self._load_active_persona_into_fields()

    # ------------------------------------------------------------------
    def _provider_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)

        self.deepseek_key = QLineEdit(self.settings.deepseek_api_key)
        self.deepseek_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.deepseek_model = QLineEdit(self.settings.deepseek_model)
        self.deepseek_base_url = QLineEdit(self.settings.deepseek_base_url)

        f.addRow(QLabel(
            "<b>DeepSeek (cloud, OpenAI-compatible, ~$0.14/M tokens)</b>"
            "<br><i>Get a key at <code>https://platform.deepseek.com/api_keys</code>. "
            "Models: <code>deepseek-chat</code> (V3, fast - recommended) or "
            "<code>deepseek-reasoner</code> (R1, slower / stronger reasoning).</i>"
        ))
        f.addRow("API key:", self.deepseek_key)
        f.addRow("Model:", self.deepseek_model)
        f.addRow("Base URL:", self.deepseek_base_url)
        return w

    def _context_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # ----- Persona row at the very top of the tab -----
        persona_row = QHBoxLayout()
        persona_row.addWidget(QLabel("Persona:"))
        self.persona_combo = QComboBox()
        self.persona_combo.setToolTip(
            "Switch between saved persona presets. "
            "Clicking the main Save button below also updates the active persona."
        )
        self.persona_combo.currentTextChanged.connect(self._persona_changed)
        persona_row.addWidget(self.persona_combo, stretch=1)

        self.persona_save_as_btn = QPushButton("Save As...")
        self.persona_save_as_btn.setToolTip("Create a new persona with the current Context fields")
        self.persona_save_as_btn.clicked.connect(self._persona_save_as)
        persona_row.addWidget(self.persona_save_as_btn)

        self.persona_rename_btn = QPushButton("Rename...")
        self.persona_rename_btn.clicked.connect(self._persona_rename)
        persona_row.addWidget(self.persona_rename_btn)

        self.persona_delete_btn = QPushButton("Delete")
        self.persona_delete_btn.clicked.connect(self._persona_delete)
        persona_row.addWidget(self.persona_delete_btn)
        v.addLayout(persona_row)

        v.addWidget(QLabel(
            "<i>Personas let you switch between job applications "
            "(e.g. 'Stripe Senior PM' / 'Junior Dev') with one click.</i>"
        ))

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

    # ---- Persona helpers --------------------------------------------------
    def _populate_persona_dropdown(self) -> None:
        self._suppress_persona_signal = True
        try:
            self.persona_combo.clear()
            self.persona_combo.addItems(self.persona_store.names())
            active = self.persona_store.active_name()
            idx = self.persona_combo.findText(active)
            if idx >= 0:
                self.persona_combo.setCurrentIndex(idx)
        finally:
            self._suppress_persona_signal = False

    def _load_active_persona_into_fields(self) -> None:
        p = self.persona_store.get_active()
        if p is None:
            return
        self.about_edit.setPlainText(p.about_me)
        self.resume_edit.setPlainText(p.resume_text)
        self.job_edit.setPlainText(p.job_description)
        self.custom_edit.setPlainText(p.custom_system_prompt)

    def _current_persona_from_fields(self, name: str) -> Persona:
        return Persona(
            name=name,
            about_me=self.about_edit.toPlainText(),
            resume_text=self.resume_edit.toPlainText(),
            job_description=self.job_edit.toPlainText(),
            custom_system_prompt=self.custom_edit.toPlainText(),
        )

    def _persona_changed(self, name: str) -> None:
        if self._suppress_persona_signal or not name:
            return
        # Picking a different persona overwrites the boxes with that
        # persona's content. Persisted immediately so the choice survives
        # a Cancel.
        self.persona_store.set_active(name)
        self._load_active_persona_into_fields()

    def _persona_save_as(self) -> None:
        name, ok = QInputDialog.getText(self, "Save persona", "Name for new persona:")
        if not ok:
            return
        name = name.strip()
        if not name:
            QMessageBox.warning(self, "Save persona", "Name cannot be empty.")
            return
        if name in self.persona_store.names():
            QMessageBox.warning(
                self, "Save persona",
                f"A persona named {name!r} already exists. Use Rename or pick a different name."
            )
            return
        self.persona_store.upsert(self._current_persona_from_fields(name))
        self.persona_store.set_active(name)
        self._populate_persona_dropdown()

    def _persona_rename(self) -> None:
        old = self.persona_store.active_name()
        new, ok = QInputDialog.getText(
            self, "Rename persona", "New name:", text=old
        )
        if not ok:
            return
        new = new.strip()
        if not self.persona_store.rename(old, new):
            QMessageBox.warning(
                self, "Rename persona",
                "Rename failed. The new name must be non-empty, different from "
                "the current one, and not already in use."
            )
            return
        self._populate_persona_dropdown()

    def _persona_delete(self) -> None:
        active = self.persona_store.active_name()
        if len(self.persona_store.names()) <= 1:
            QMessageBox.information(
                self, "Delete persona",
                "Can't delete the last persona. Create another one first."
            )
            return
        ans = QMessageBox.question(
            self, "Delete persona",
            f"Delete persona {active!r}? This can't be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        if not self.persona_store.delete(active):
            QMessageBox.warning(self, "Delete persona", "Delete failed.")
            return
        self._populate_persona_dropdown()
        self._load_active_persona_into_fields()

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

        # Whisper model is locked to 'small' - the model files are
        # bundled inside the .exe folder and that's the only one
        # available offline. Showing a dropdown that lets the user pick
        # 'medium' or 'large-v3' would just trigger a 1-3 GB download
        # attempt that fails because we're locked offline.
        self.whisper_model_label = QLabel(f"<code>{self.settings.whisper_model}</code> (bundled, offline-only)")

        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(5.0, 60.0)
        self.window_spin.setSingleStep(1.0)
        self.window_spin.setValue(self.settings.answer_window_seconds)

        f.addRow("Whisper model:", self.whisper_model_label)
        f.addRow("Audio window (sec):", self.window_spin)
        f.addRow(QLabel(
            "<i>Whisper is bundled offline. No downloads, ever.</i>"
        ))
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
        s.deepseek_api_key = self.deepseek_key.text().strip()
        s.deepseek_model = self.deepseek_model.text().strip() or "deepseek-chat"
        s.deepseek_base_url = self.deepseek_base_url.text().strip() or "https://api.deepseek.com/v1"

        s.about_me = self.about_edit.toPlainText()
        s.resume_text = self.resume_edit.toPlainText()
        s.job_description = self.job_edit.toPlainText()
        s.custom_system_prompt = self.custom_edit.toPlainText()

        # Whisper model is hard-pinned to 'small' (bundled). Don't let
        # anyone overwrite it from the UI.
        s.answer_window_seconds = float(self.window_spin.value())

        s.hotkey_answer = self.hk_answer.text().strip()
        s.hotkey_toggle = self.hk_toggle.text().strip()
        s.hotkey_clear = self.hk_clear.text().strip()
        s.hotkey_settings = self.hk_settings.text().strip()
        s.hotkey_quit = self.hk_quit.text().strip()

        s.exclude_from_capture = self.exclude_check.isChecked()
        s.opacity = float(self.opacity_spin.value())

        # Sync the active persona with whatever's now in the boxes so
        # personas always reflect what the user just committed.
        active = self.persona_store.active_name()
        self.persona_store.upsert(
            Persona(
                name=active,
                about_me=s.about_me,
                resume_text=s.resume_text,
                job_description=s.job_description,
                custom_system_prompt=s.custom_system_prompt,
            )
        )

        self.accept()
