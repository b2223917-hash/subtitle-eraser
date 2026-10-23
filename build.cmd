@echo off
setlocal
cd /d "%~dp0"
rem Optional arguments: build.cmd [dist-directory] [work-directory]
set "SUBTITLE_DIST=%~1"
if not defined SUBTITLE_DIST set "SUBTITLE_DIST=dist-calibrated"
set "SUBTITLE_BUILD=%~2"
if not defined SUBTITLE_BUILD set "SUBTITLE_BUILD=build-calibrated"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.cmd first.
  exit /b 1
)
".venv\Scripts\python.exe" -m PyInstaller --version >nul 2>nul
if errorlevel 1 (
  echo Build dependencies are missing. Run setup.cmd first.
  exit /b 1
)
".venv\Scripts\python.exe" -m PyInstaller --clean --noconfirm --workpath "%SUBTITLE_BUILD%" --distpath "%SUBTITLE_DIST%" SubtitleEraser.spec
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" tools\package_reports.py --dist-root "%SUBTITLE_DIST%"
if errorlevel 1 exit /b 1
echo Build complete: %SUBTITLE_DIST%\SubtitleEraser\SubtitleEraser.exe
exit /b 0
