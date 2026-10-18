@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.cmd first to create the project environment.
  exit /b 1
)
echo Open http://127.0.0.1:8765 in your browser. Ctrl+C stops the UI server.
".venv\Scripts\python.exe" -B scripts\serve_replay.py --port 8765
