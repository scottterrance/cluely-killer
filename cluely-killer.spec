# PyInstaller spec for cluely-killer.
#
# Build with:        pyinstaller --noconfirm cluely-killer.spec
# Or just run:       build.bat
#
# Output: dist/cluely-killer/cluely-killer.exe  (one-folder bundle).
# To distribute, zip the entire dist/cluely-killer/ folder.
#
# THIS BUILD BUNDLES THE WHISPER 'large-v3-turbo' MODEL (local STT).
# Before running build.bat, stage the model with `setup-model.ps1` once
# (it downloads large-v3-turbo into ./models/whisper-large-v3-turbo/).
# build.bat then copies ./models/ next to the .exe so the end user's
# machine never downloads anything. Transcription is fully local/offline;
# only the DeepSeek answer call needs the network. Final dist size ~2 GB
# (the model alone is ~1.5 GB at int8).

# ruff: noqa
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules


# ---- Native-binary libs that PyInstaller can miss ----
fwhisper_data    = collect_data_files("faster_whisper")
ct2_data         = collect_data_files("ctranslate2")
soundcard_data   = collect_data_files("soundcard")

fwhisper_hidden  = collect_submodules("faster_whisper")
ct2_hidden       = collect_submodules("ctranslate2")
soundcard_hidden = collect_submodules("soundcard")

# Lazy-imported inside app/utils/extract.py - PyInstaller won't see them.
lazy_hidden = [
    "pypdf",
    "docx",
    "lxml",
    "lxml.etree",
    # httpx is imported by the DeepSeek provider. Normally picked up via
    # analysis, but list it explicitly to be safe.
    "httpx",
]

excludes = [
    "tkinter", "matplotlib", "scipy", "pandas", "PIL",
    "PySide2", "PySide6", "PyQt5",
    "test", "tests", "pytest", "unittest",
]


a = Analysis(
    ["run.py"],
    pathex=["."],
    binaries=[],
    datas=[
        *fwhisper_data,
        *ct2_data,
        *soundcard_data,
        # NOTE: ./models/ is intentionally NOT bundled here. PyInstaller
        # 6.x stuffs `datas` into _internal/ which breaks our flat path
        # lookup, and large binary files (model.bin is ~461 MB) sometimes
        # get silently dropped by COLLECT. build.bat copies ./models/
        # next to the .exe via xcopy AFTER PyInstaller finishes - simpler
        # and 100% predictable.
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
    upx=False,
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
