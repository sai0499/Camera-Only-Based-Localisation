@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1"
set "setup_exit=%ERRORLEVEL%"
if not "%setup_exit%"=="0" echo Setup failed. Read the error above; rerunning setup is safe.
if /I not "%~1"=="--no-pause" pause
exit /b %setup_exit%
