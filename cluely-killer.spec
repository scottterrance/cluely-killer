# PyInstaller spec for cluely-killer.
#
# Build with:        pyinstaller --noconfirm cluely-killer.spec
# Or just run:       build.bat
#
# Output: dist/cluely-killer/cluely-killer.exe  (one-folder bundle).
# To distribute, zip the entire dist/cluely-killer/ folder.
#
# THIS BUILD BUNDLES THE WHISPER 'small' MODEL.
# Before running build.bat, populate ./models/hf-cache/hub/ with the
# Whisper small model (run setup-model.ps1 once - it copies from your
# existing HF cache). The spec walks ./models/ and ships every file
# next to the .exe so the friend's machine never downloads anything.
# Final dist size: ~750 MB (250 MB app + 466 MB model).

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
]

# Walk ./models/ and emit a (src, dst) tuple per file. The destination
# path is relative to the .exe's folder, so models/ ends up next to
# cluely-killer.exe in the dist folder. run.py points HF cache at that
# location.
def _bundle_models():
    out = []
    if not os.path.isdir("models"):
        print("[spec] WARNING: ./models/ not found - the .exe will not have a")
        print("[spec]          bundled model and will fail at startup. Run")
        print("[spec]          setup-model.ps1 first to populate ./models/.")
        return out
    for root, _dirs, files in os.walk("models"):
        for f in files:
            src = os.path.join(root, f)
            dst = os.path.dirname(src)  # preserves models/.../ structure
            out.append((src, dst))
    print(f"[spec] bundling {len(out)} model files from ./models/")
    return out


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
        *_bundle_models(),
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
