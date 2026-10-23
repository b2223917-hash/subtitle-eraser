@echo off
setlocal
cd /d "%~dp0"
rem /quiet suppresses the final pause for automation. Existing environments are reused.
set "NO_PAUSE="
if /I "%~1"=="/quiet" set "NO_PAUSE=1"
if defined CI set "NO_PAUSE=1"
where uv >nul 2>nul
if errorlevel 1 goto standard_python
if not exist ".venv\Scripts\python.exe" (
  uv venv --python 3.11 .venv
  if errorlevel 1 goto fail
)
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt
if errorlevel 1 goto fail
goto verify

:standard_python
if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 (
    echo Install Python 3.11 x64 from https://www.python.org/downloads/windows/
    echo or uv from https://docs.astral.sh/uv/getting-started/installation/ then retry.
    goto fail
  )
)
".venv\Scripts\python.exe" -m ensurepip --upgrade
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt
if errorlevel 1 goto fail

:verify
".venv\Scripts\python.exe" -c "import sys,struct; assert sys.platform == 'win32' and (3,11) <= sys.version_info[:2] < (3,13) and struct.calcsize('P') == 8, 'Use 64-bit Python 3.11 or 3.12 on Windows'"
if errorlevel 1 goto fail
echo Setup complete. Open start.cmd, or run .venv\Scripts\python.exe -m pytest -q
if not defined NO_PAUSE pause
exit /b 0

:fail
echo Setup failed. See the error above.
if not defined NO_PAUSE pause
exit /b 1
