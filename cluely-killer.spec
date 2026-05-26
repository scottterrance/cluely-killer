# PyInstaller spec for cluely-killer.
#
# Build with:        pyinstaller --noconfirm cluely-killer.spec
# Or just run:       build.bat
#
# Output: dist/cluely-killer/cluely-killer.exe  (one-folder bundle).
# To distribute, zip the entire dist/cluely-killer/ folder.
#
# Whisper model files are NOT bundled (they're 466+ MB and would fail
# legal redistribution policies). They download to ~/.cache/huggingface
# on first run, same as in dev.

# ruff: noqa
from PyInstaller.utils.hooks import collect_data_files, collect_submodules


# ---- Native-binary libs that PyInstaller can miss ----
# faster-whisper bundles tokenizer assets next to its package; ctranslate2
# ships the actual inference DLLs; soundcard loads its mediafoundation
# backend dynamically on Windows.
fwhisper_data    = collect_data_files("faster_whisper")
ct2_data         = collect_data_files("ctranslate2")
soundcard_data   = collect_data_files("soundcard")

fwhisper_hidden  = collect_submodules("faster_whisper")
ct2_hidden       = collect_submodules("ctranslate2")
soundcard_hidden = collect_submodules("soundcard")

# Lazy-imported inside app/utils/extract.py - PyInstaller won't see them
# during static analysis, so list them explicitly.
lazy_hidden = [
    "pypdf",
    "docx",
    "lxml",       # pulled in by python-docx
    "lxml.etree",
]

# Modules we definitely don't ship - shaves ~30 MB off the bundle.
excludes = [
    "tkinter",
    "matplotlib",
    "scipy",
    "pandas",
    "PIL",
    "PySide2",
    "PySide6",
    "PyQt5",
    "test",
    "tests",
    "pytest",
    "unittest",
]


a = Analysis(
    ["run.py"],
    pathex=["."],
    binaries=[],
    datas=[
        *fwhisper_data,
        *ct2_data,
        *soundcard_data,
    ],
    hiddenimports=[
        *fwhisper_hidden,
        *ct2_hidden,
        *soundcard_hidden,
        *lazy_hidden,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="cluely-killer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX is known to break PyQt6 plugin DLLs - leave compression off.
    upx=False,
    # Keep console=True for the first builds so users see startup logs and
    # any crash trace. Flip to False once the app is stable for a "release"
    # build with no terminal window.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="cluely-killer",
)
