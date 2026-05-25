@echo off
REM Same as start.bat but always keeps the console open and runs without stealth.
REM Use this if anything ever feels off — you'll see all logs.

cd /d "%~dp0"
call ".venv\Scripts\activate.bat"
python run.py --no-stealth --reset-window
pause
