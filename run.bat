@echo off
REM ============================================================
REM  vision-gesture-control - double-click launcher (Windows)
REM  - creates a local virtual environment on first run
REM  - installs dependencies
REM  - starts the webcam gesture app
REM ============================================================

setlocal
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "PYEXE=%VENV_DIR%\Scripts\python.exe"

REM --- pick a Python interpreter for creating the venv ---------
REM  Try, in order: py -3.13, py -3.12, py, python
set "BOOTSTRAP_PY="
for %%P in ("py -3.13" "py -3.12" "py" "python") do (
    if not defined BOOTSTRAP_PY (
        %%~P --version >nul 2>&1 && set "BOOTSTRAP_PY=%%~P"
    )
)

if not exist "%PYEXE%" (
    if not defined BOOTSTRAP_PY (
        echo [ERROR] No Python found. Install Python 3.12 or 3.13 from python.org and retry.
        pause
        exit /b 1
    )
    echo [setup] Creating virtual environment in %VENV_DIR% ...
    %BOOTSTRAP_PY% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

echo [setup] Installing / updating dependencies ...
"%PYEXE%" -m pip install --upgrade pip >nul
"%PYEXE%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency install failed.
    echo         MediaPipe needs Python 3.9-3.12. If you are on 3.13/3.14,
    echo         install Python 3.12 and delete the .venv folder, then retry.
    pause
    exit /b 1
)

echo.
echo [run] Starting vision-gesture-control  (press Q or ESC in the window to quit)
echo.
set "PYTHONPATH=%~dp0src"
"%PYEXE%" -m vision_gesture_control.main %*

echo.
echo [done] Application closed.
pause
endlocal
