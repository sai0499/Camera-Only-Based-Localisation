@echo off
setlocal
cd /d "%~dp0"
call setup.cmd --no-pause
if errorlevel 1 (
  echo Setup failed. Resolve the error above and retry.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -u -B scripts\build_submission_assets.py
if errorlevel 1 (
  echo Map preparation failed. Read the error above and README.md.
  pause
  exit /b 1
)
start "" /b ".venv\Scripts\pythonw.exe" -B -c "import time,webbrowser; time.sleep(2); webbrowser.open('http://127.0.0.1:8765')"
call launch_ui.cmd
exit /b %ERRORLEVEL%
