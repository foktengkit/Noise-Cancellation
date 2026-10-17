@echo off
setlocal
pushd "%~dp0"

set "NOISE_PYTHON=.venv-secure\Scripts\python.exe"
if not exist "%NOISE_PYTHON%" set "NOISE_PYTHON=.venv\Scripts\python.exe"
if not exist "%NOISE_PYTHON%" (
    echo Project Python environment is missing.
    echo Run these commands in this folder first:
    echo   py -3.14 -m venv .venv-secure
    echo   .venv-secure\Scripts\python.exe -m pip install -r requirements.txt
    pause
    popd
    exit /b 1
)

echo Noise Control: http://127.0.0.1:8000
echo Keep this window open while using the app. Press Ctrl+C to stop.
echo.
"%NOISE_PYTHON%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"
if errorlevel 1 (
    echo Python 3.12 or newer is required. Use the secure environment.
    pause
    popd
    exit /b 1
)
"%NOISE_PYTHON%" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --workers 1 --no-proxy-headers --ws-max-size 131072 --ws-max-queue 4
set "NOISE_LIFT_EXIT=%errorlevel%"

if not "%NOISE_LIFT_EXIT%"=="0" (
    echo.
    echo The server could not start. Check the error above.
    echo If port 8000 is already in use, try opening http://127.0.0.1:8000
    pause
)

popd
exit /b %NOISE_LIFT_EXIT%
