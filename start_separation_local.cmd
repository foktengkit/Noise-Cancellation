@echo off
setlocal
pushd "%~dp0"
if not exist ".venv-secure\Scripts\python.exe" (
    echo The secure Python environment is missing. See README.md.
    pause
    popd
    exit /b 1
)
".venv-secure\Scripts\python.exe" -m scripts.run_separation_local
if errorlevel 1 pause
popd
endlocal
