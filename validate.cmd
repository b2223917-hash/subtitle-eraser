@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m subtitle_eraser.validate
if errorlevel 1 (
  pause
  exit /b 1
)
start "" "outputs\validation\report.html"
