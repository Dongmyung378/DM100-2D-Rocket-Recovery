@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m powered_landing_guidance.local_app %*
) else (
    py -3.12 -m powered_landing_guidance.local_app %*
)
if errorlevel 1 (
    echo.
    echo Install Python 3.12 and the UI dependencies first:
    echo python -m pip install -e ".[ui,video]"
    pause
)
