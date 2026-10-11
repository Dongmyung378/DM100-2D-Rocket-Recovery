@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"

if defined DM100_PYTHON goto custom_python
if "%DM100_USE_MANAGED_PYTHON%"=="1" goto managed_python

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -I -X utf8 -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1
    if not errorlevel 1 (
        ".venv\Scripts\python.exe" -I -X utf8 "scripts\launch_dm100.py" %*
        goto result
    )
)

py -3.12 -I -X utf8 -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1
if not errorlevel 1 (
    py -3.12 -I -X utf8 "scripts\launch_dm100.py" %*
    goto result
)

python -I -X utf8 -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1
if not errorlevel 1 (
    python -I -X utf8 "scripts\launch_dm100.py" %*
    goto result
)

python3 -I -X utf8 -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1
if not errorlevel 1 (
    python3 -I -X utf8 "scripts\launch_dm100.py" %*
    goto result
)

rem An existing local uv-managed runtime can repair a moved developer checkout.
rem A fresh clone does not need uv or this directory.
for /d %%D in (".local\uv-python\cpython-3.12*-windows-*") do (
    if exist "%%~fD\python.exe" (
        "%%~fD\python.exe" -I -X utf8 -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1
        if not errorlevel 1 (
            "%%~fD\python.exe" -I -X utf8 "scripts\launch_dm100.py" %*
            goto result
        )
    )
)

:managed_python
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap_windows.ps1" %*
goto result

:custom_python
"%DM100_PYTHON%" -I -X utf8 -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1
if errorlevel 1 (
    echo DM100_PYTHON must point to a working Python 3.12.x executable.
    goto failed
)
"%DM100_PYTHON%" -I -X utf8 "scripts\launch_dm100.py" %*

:result
set "DM100_EXIT_CODE=%errorlevel%"
if "%DM100_EXIT_CODE%"=="0" exit /b 0
echo.
echo DM100 could not finish. See the error message above.
if not defined DM100_NO_PAUSE pause
exit /b %DM100_EXIT_CODE%

:failed
if not defined DM100_NO_PAUSE pause
exit /b 1
