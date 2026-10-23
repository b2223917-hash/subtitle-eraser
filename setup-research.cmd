@echo off
cd /d "%~dp0"
uv venv --python 3.11 .research-venv
if errorlevel 1 goto fail
uv pip install --python .research-venv\Scripts\python.exe torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto fail
uv pip install --python .research-venv\Scripts\python.exe -r research-requirements.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" tools\setup_research.py
if errorlevel 1 goto fail
echo Research environment ready.
pause
exit /b 0
:fail
echo Research setup failed.
pause
exit /b 1
