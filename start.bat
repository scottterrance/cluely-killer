@echo off
REM cluely-killer launcher.
REM Double-click this file from File Explorer to start the app.
REM No PowerShell, no remembering venv activation.

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

REM Pass through any args (e.g. --no-stealth, --whisper-model tiny)
python run.py %*

REM Keep window open if app died, so we can read any error
if errorlevel 1 pause
