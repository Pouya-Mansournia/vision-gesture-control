@echo off
REM Double-click to run the unit test suite (no webcam needed).
setlocal
cd /d "%~dp0"

set "PYEXE=.venv\Scripts\python.exe"
if not exist "%PYEXE%" (
    where py >nul 2>&1 && (py -3.12 -m venv .venv) || (python -m venv .venv)
)
"%PYEXE%" -m pip install --upgrade pip >nul
"%PYEXE%" -m pip install pytest numpy >nul
"%PYEXE%" -m pytest -q
pause
endlocal
