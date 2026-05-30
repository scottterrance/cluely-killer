# PyInstaller spec for cluely-killer.
#
# Build with:        pyinstaller --noconfirm cluely-killer.spec
# Or just run:       build.bat
#
# Output: dist/cluely-killer/cluely-killer.exe  (one-folder bundle).
# To distribute, zip the entire dist/cluely-killer/ folder.
#
# THIS BUILD BUNDLES THE WHISPER MODEL(S) (local STT) + the NVIDIA
# CUDA/cuDNN runtime DLLs (for GPU acceleration on machines that have an
# NVIDIA GPU).
#
# Before running build.bat:
#   1. Stage one or more models with setup-model.ps1, e.g.:
#        .\setup-model.ps1 -Model large-v3-turbo
#        .\setup-model.ps1 -Model small
#      Each lands in .\models\whisper-<name>\ and build.bat copies the
#      whole .\models\ folder next to the .exe.
#   2. (For GPU support) install the CUDA DLLs into the build venv:
#        pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
#      These are just files - they bundle fine even on an Intel-only
#      build machine; they only need a real NVIDIA GPU at RUNTIME.
#
# Transcription is fully local/offline; only the DeepSeek answer call
# needs the network. Dist size depends on which models you stage
# (~1.5 GB per large-v3-turbo, ~0.5 GB per small) plus ~0.5 GB of CUDA
# DLLs if bundled.

# ruff: noqa
import glob
import os
import site
from PyInstaller.utils.hooks import collect_data_files, collect_submodules


# ---- Native-binary libs that PyInstaller can miss ----
fwhisper_data    = collect_data_files("faster_whisper")
ct2_data         = collect_data_files("ctranslate2")
soundcard_data   = collect_data_files("soundcard")

fwhisper_hidden  = collect_submodules("faster_whisper")
ct2_hidden       = collect_submodules("ctranslate2")
soundcard_hidden = collect_submodules("soundcard")


def _collect_nvidia_dlls():
    """Bundle the NVIDIA CUDA/cuDNN runtime DLLs from the pip
    'nvidia-*-cu12' packages so GPU transcription works on the end-user's
    machine WITHOUT them installing a separate CUDA toolkit.

    KEY POINT: these are just DLL files. They install + bundle fine even
    on a build machine with NO NVIDIA GPU (e.g. an Intel-only laptop).
    They only need a real NVIDIA GPU at RUNTIME, on the user's machine.

    Best-effort: returns [] if the packages aren't installed, which yields
    a CPU-only build (still works; just no GPU acceleration). Install them
    on the build machine with:
        pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

    DLLs are placed at the bundle ROOT (dest ".") so CTranslate2's loader
    finds them on the Windows DLL search path.
    """
    roots = set()
    try:
        import nvidia
        for p in list(getattr(nvidia, "__path__", [])):
            roots.add(p)
    except Exception:
        pass
    candidates = []
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass
    for sp in candidates:
        cand = os.path.join(sp, "nvidia")
        if os.path.isdir(cand):
            roots.add(cand)

    pairs = []
    seen = set()
    for root in roots:
        for dll in glob.glob(os.path.join(root, "**", "*.dll"), recursive=True):
            name = os.path.basename(dll).lower()
            if name in seen:
                continue
            seen.add(name)
            pairs.append((dll, "."))
    print(f"[spec] bundling {len(pairs)} NVIDIA CUDA/cuDNN DLL(s) for GPU support")
    return pairs


nvidia_binaries = _collect_nvidia_dlls()

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
    binaries=[
        # NVIDIA CUDA/cuDNN runtime DLLs (empty list -> CPU-only build).
        *nvidia_binaries,
    ],
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
