@echo off
setlocal
cd /d "%~dp0"
for %%F in (manifest.json map.npz retrieval.json retrieval.npz) do if not exist "artifacts\fit-seven-rescue-v2\%%F" (
  echo Missing frozen map file: artifacts\fit-seven-rescue-v2\%%F
  echo Copy the supplied frozen map folder into artifacts before starting. See README_SUBMISSION.md.
  pause
  exit /b 1
)
call setup.cmd --no-pause
if errorlevel 1 (
  echo Setup failed. Resolve the error above and retry.
  pause
  exit /b 1
)
start "" /b ".venv\Scripts\pythonw.exe" -B -c "import time,webbrowser; time.sleep(2); webbrowser.open('http://127.0.0.1:8765')"
call launch_ui.cmd
exit /b %ERRORLEVEL%
