"""Settings dialog. Tabbed: Provider, Context, Audio/STT, Hotkeys, Window."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
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

from ..config import Settings, save_settings
from ..core.personas import DEFAULT_NAME, Persona, PersonaStore
from ..utils.extract import extract_text
from .drop_text_edit import DropZoneTextEdit


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        # CRITICAL: pass parent=None to super().__init__, NOT the overlay.
        #
        # The overlay has Qt.WindowType.WindowStaysOnTopHint. On Windows,
        # any child HWND of a topmost window inherits the topmost z-order
        # at the OS level - regardless of which Qt flags we set on the
        # child. Just clearing WindowStaysOnTopHint on the dialog isn't
        # enough; Windows still places it above all non-topmost windows
        # because its parent is topmost.
        #
        # Detaching by passing None makes the dialog a fully independent
        # top-level window. Browsers, PDF readers, etc. can now cover it
        # normally when the user clicks them. The `parent` argument is
        # kept in the signature for API compatibility (callers still pass
        # `parent=overlay`) but is intentionally ignored.
        super().__init__(None)
        # Reset window flags to a clean Dialog window (titlebar + close
        # button, no inherited Frameless/StayOnTop from the overlay).
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowCloseButtonHint
        )

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

        # ---- DeepSeek (the only LLM) ----
        self.deepseek_key = QLineEdit(self.settings.deepseek_api_key)
        self.deepseek_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.deepseek_model = QLineEdit(self.settings.deepseek_model)
        self.deepseek_base_url = QLineEdit(self.settings.deepseek_base_url)
        f.addRow(QLabel(
            "<b>DeepSeek (cloud LLM, ~$0.14/M tokens)</b>"
            "<br><i>Get a key at <code>https://platform.deepseek.com/api_keys</code>. "
            "Models: <code>deepseek-chat</code> (V3, fast - recommended) or "
            "<code>deepseek-reasoner</code> (R1, slower / stronger reasoning).</i>"
        ))
        f.addRow("DeepSeek API key:", self.deepseek_key)
        f.addRow("DeepSeek model:", self.deepseek_model)
        f.addRow("DeepSeek base URL:", self.deepseek_base_url)

        # Answer length = THE answer-SPEED lever. LLMs generate tokens
        # one at a time, so a shorter answer streams faster. On a fast
        # (GPU) STT setup the LLM is the bottleneck, so 'concise' is the
        # difference between a ~4s and a ~1.5s answer.
        self.brevity_combo = QComboBox()
        for label, val in [
            ("Concise - 1-2 sentences (FASTEST)", "concise"),
            ("Normal - 2-3 sentences", "normal"),
            ("Detailed - 3-5 sentences (slowest)", "detailed"),
        ]:
            self.brevity_combo.addItem(label, val)
        bi = self.brevity_combo.findData(self.settings.answer_brevity)
        self.brevity_combo.setCurrentIndex(bi if bi >= 0 else 0)
        f.addRow("Answer length (speed):", self.brevity_combo)
        f.addRow(QLabel(
            "<i>Answer length is the main lever for <b>answer speed</b>: the "
            "model writes one word at a time, so a shorter answer appears "
            "faster. Pick <b>Concise</b> for the fastest replies.</i>"
        ))

        f.addRow(QLabel(
            "<hr><i>Transcription is done by the bundled <b>local</b> Whisper "
            "model (offline). Configure it on the <b>Audio / STT</b> tab.</i>"
        ))
        return w

    def _context_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # ----- Persona row at the very top of the tab -----
        persona_row = QHBoxLayout()
        persona_row.addWidget(QLabel("Persona:"))
        self.persona_combo = QComboBox()
        self.persona_combo.setToolTip(
            "Switch between saved persona presets. Picking a persona "
            "INSTANTLY swaps the about-me / resume / JD / custom prompt "
            "the LLM will see for the next Ctrl+Space - no Save needed."
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

    def _apply_active_persona_to_runtime(self) -> None:
        """Mirror the active persona's content into self.settings + persist.

        This is what makes a persona swap take effect immediately without
        the user having to click Save. The Controller's prompt_builder
        reads from settings.about_me / resume_text / job_description /
        custom_system_prompt, so updating those + writing config.json is
        all it takes for the very next Ctrl+Space to use the new context.
        """
        p = self.persona_store.get_active()
        if p is None:
            return
        self.settings.about_me = p.about_me
        self.settings.resume_text = p.resume_text
        self.settings.job_description = p.job_description
        self.settings.custom_system_prompt = p.custom_system_prompt
        save_settings(self.settings)
        print(
            f"[settings] active persona -> {p.name!r}: "
            f"about={len(p.about_me)} chars, resume={len(p.resume_text)} chars, "
            f"jd={len(p.job_description)} chars, custom={len(p.custom_system_prompt)} chars",
            flush=True,
        )

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
        # persona's content AND immediately copies that content into
        # settings + writes config.json. The runtime (Controller's
        # prompt_builder) reads from settings every Ctrl+Space, so
        # the next answer uses the new persona instantly. No Save
        # button required - even closing the dialog with Cancel
        # preserves the swap.
        self.persona_store.set_active(name)
        self._load_active_persona_into_fields()
        self._apply_active_persona_to_runtime()

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
        # Newly-created persona becomes active -> push to runtime.
        self._apply_active_persona_to_runtime()

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
        # Deletion auto-promotes another persona to active -> push it.
        self._apply_active_persona_to_runtime()

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

        # Model selector. large-v3-turbo = best accuracy; small/base =
        # much faster on CPU (3-4x) at some accuracy cost. Each requires
        # the matching model folder bundled next to the .exe
        # (models/whisper-<name>/). Changing this needs a restart.
        self.model_combo = QComboBox()
        for label, val in [
            ("large-v3-turbo (best accuracy)", "large-v3-turbo"),
            ("small (3-4x faster on CPU)", "small"),
            ("base (fastest on CPU)", "base"),
        ]:
            self.model_combo.addItem(label, val)
        mi = self.model_combo.findData(self.settings.whisper_model)
        if mi < 0:
            # Unknown/custom model name - add it so it's not lost.
            self.model_combo.addItem(f"{self.settings.whisper_model} (custom)", self.settings.whisper_model)
            mi = self.model_combo.count() - 1
        self.model_combo.setCurrentIndex(mi)

        # Device selector. Auto = GPU if present else CPU. This is THE
        # speed lever: GPU transcribes large-v3-turbo in well under 1s.
        self.device_combo = QComboBox()
        for label, val in [
            ("Auto (GPU if available, else CPU)", "auto"),
            ("GPU (CUDA)", "cuda"),
            ("CPU", "cpu"),
        ]:
            self.device_combo.addItem(label, val)
        di = self.device_combo.findData(self.settings.whisper_device)
        self.device_combo.setCurrentIndex(di if di >= 0 else 0)

        # CTranslate2 worker threads (CPU mode only). Higher = faster
        # local STT on a multi-core CPU. 0 = auto (all cores minus one).
        import os as _os
        self.cpu_threads_spin = QDoubleSpinBox()
        self.cpu_threads_spin.setDecimals(0)
        self.cpu_threads_spin.setRange(0, float(max(2, (_os.cpu_count() or 8))))
        self.cpu_threads_spin.setSingleStep(1)
        self.cpu_threads_spin.setValue(float(self.settings.whisper_cpu_threads))

        # First-press fallback window. Only used on the very first
        # answer of a session, before the since-last-press marker has
        # been set. After that, every press uses the marker-based
        # capture (capped at "Max capture" below).
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(5.0, 60.0)
        self.window_spin.setSingleStep(1.0)
        self.window_spin.setValue(self.settings.answer_window_seconds)

        # Hard ceiling on since-last-press audio.
        self.max_capture_spin = QDoubleSpinBox()
        self.max_capture_spin.setRange(15.0, 600.0)
        self.max_capture_spin.setSingleStep(5.0)
        self.max_capture_spin.setValue(self.settings.max_capture_seconds)

        # Continuous STT toggle. When on, a background thread transcribes
        # as the interviewer talks so the press path is just the LLM call.
        self.continuous_check = QCheckBox(
            "Continuous transcription (background, near-instant answers)"
        )
        self.continuous_check.setChecked(self.settings.continuous_stt)

        # Keyword biasing toggle. Helps recognize names/jargon but can
        # leak the glossary into transcripts on quiet audio.
        self.bias_check = QCheckBox(
            "Bias transcription toward my resume/JD keywords"
        )
        self.bias_check.setChecked(self.settings.stt_bias_enabled)

        f.addRow("Whisper model:", self.model_combo)
        f.addRow("Device:", self.device_combo)
        f.addRow("CPU threads (0 = auto):", self.cpu_threads_spin)
        f.addRow("First-press window (sec):", self.window_spin)
        f.addRow("Max capture per press (sec):", self.max_capture_spin)
        f.addRow(self.continuous_check)
        f.addRow(self.bias_check)
        f.addRow(QLabel(
            "<i><b>Device = GPU</b> is the big speed win: it transcribes "
            "large-v3-turbo in well under a second. Needs an NVIDIA GPU + the "
            "CUDA/cuDNN runtime DLLs. If GPU init fails, the app falls back to "
            "CPU automatically.<br><br>"
            "<b>No GPU?</b> Switch <b>Whisper model</b> to <b>small</b> or "
            "<b>base</b> - 3-4x faster than turbo on CPU.<br><br>"
            "<b>Changing model or device requires an app restart.</b> Other "
            "options apply immediately.<br><br>"
            "On each press, only the interviewer's <b>last question</b> is "
            "transcribed (not the whole window). Each press of '1' or '2' covers "
            "everything said since the previous press, capped at <b>Max capture</b>.</i>"
        ))
        return w

    def _hotkeys_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.hk_answer_short = QLineEdit(self.settings.hotkey_answer_short)
        self.hk_answer_context = QLineEdit(self.settings.hotkey_answer_context)
        self.hk_toggle = QLineEdit(self.settings.hotkey_toggle)
        self.hk_clear = QLineEdit(self.settings.hotkey_clear)
        self.hk_settings = QLineEdit(self.settings.hotkey_settings)
        self.hk_quit = QLineEdit(self.settings.hotkey_quit)
        f.addRow("Answer (no context):", self.hk_answer_short)
        f.addRow("Answer (last 5 Q+A as context):", self.hk_answer_context)
        f.addRow("Toggle overlay:", self.hk_toggle)
        f.addRow("Clear buffer:", self.hk_clear)
        f.addRow("Open settings:", self.hk_settings)
        f.addRow("Quit app:", self.hk_quit)
        f.addRow(
            QLabel(
                "<i>pynput syntax - e.g. <b>1</b>, <b>2</b>, &lt;ctrl&gt;+&lt;space&gt;, "
                "&lt;ctrl&gt;+&lt;shift&gt;+s, &lt;alt&gt;+a.<br>"
                "Bare digits like <b>1</b> / <b>2</b> are <i>global</i>: while the app "
                "is running they will be intercepted everywhere on the OS, so don't "
                "set them to keys you also need for typing.</i>"
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

        # DeepSeek (the only LLM provider)
        s.deepseek_api_key = self.deepseek_key.text().strip()
        s.deepseek_model = self.deepseek_model.text().strip() or "deepseek-chat"
        s.deepseek_base_url = self.deepseek_base_url.text().strip() or "https://api.deepseek.com/v1"
        s.answer_brevity = self.brevity_combo.currentData() or "concise"

        s.about_me = self.about_edit.toPlainText()
        s.resume_text = self.resume_edit.toPlainText()
        s.job_description = self.job_edit.toPlainText()
        s.custom_system_prompt = self.custom_edit.toPlainText()

        s.whisper_model = self.model_combo.currentData() or "large-v3-turbo"
        s.whisper_device = self.device_combo.currentData() or "auto"
        s.whisper_cpu_threads = int(self.cpu_threads_spin.value())
        s.answer_window_seconds = float(self.window_spin.value())
        s.max_capture_seconds = float(self.max_capture_spin.value())
        s.continuous_stt = self.continuous_check.isChecked()
        s.stt_bias_enabled = self.bias_check.isChecked()
        # buffer_seconds must always exceed max_capture_seconds. Bump
        # it here so the Audio tab can't get persisted into a state
        # where the next app start would silently drop audio.
        if s.buffer_seconds < s.max_capture_seconds + 5:
            s.buffer_seconds = s.max_capture_seconds + 10

        s.hotkey_answer_short = self.hk_answer_short.text().strip() or "1"
        s.hotkey_answer_context = self.hk_answer_context.text().strip() or "2"
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
