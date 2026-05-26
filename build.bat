@echo off
REM Build cluely-killer.exe via PyInstaller.
REM
REM Output: dist\cluely-killer\cluely-killer.exe
REM Distribute: zip the entire dist\cluely-killer\ folder.
REM
REM First build takes ~3-5 min (PyInstaller scans every dependency).
REM Subsequent builds with --noconfirm are faster.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No .venv found. Run setup first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

REM Install PyInstaller into the venv if it's not already there.
python -c "import PyInstaller" 2>NUL
if errorlevel 1 (
    echo [build] PyInstaller not installed. Installing...
    pip install --quiet "pyinstaller>=6.0"
    if errorlevel 1 (
        echo [build] Failed to install PyInstaller. See output above.
        pause
        exit /b 1
    )
)

echo [build] Cleaning previous build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [build] Running PyInstaller...
pyinstaller --noconfirm cluely-killer.spec
if errorlevel 1 (
    echo.
    echo [build] BUILD FAILED. See output above.
    pause
    exit /b 1
)

REM Copy the Whisper model next to the .exe.
REM We do this OUTSIDE PyInstaller because PyInstaller 6.x stuffs `datas`
REM into a `_internal\` subfolder that breaks our path lookup, and large
REM binary files (model.bin is ~461 MB) sometimes get silently dropped
REM during COLLECT. xcopy is dumb and reliable.
if exist "models" (
    echo [build] Copying models\ next to the .exe...
    xcopy /E /I /Y /Q "models" "dist\cluely-killer\models" >NUL
    if errorlevel 1 (
        echo [build] Model copy FAILED.
        pause
        exit /b 1
    )
    echo [build] Models copied.
) else (
    echo [build] WARNING: .\models\ not found. The .exe will fail at startup.
    echo [build]          Run setup-model.ps1 first to populate .\models\.
)

echo.
echo ============================================================
echo  Build complete.
echo  Output: dist\cluely-killer\cluely-killer.exe
echo  Distribute: zip the entire dist\cluely-killer\ folder.
echo ============================================================
echo.
pause
