@echo off
setlocal
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Run uv sync --locked --cache-dir .uv-cache in the project directory first.
    exit /b 1
)
pushd "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m catanzero.watch %*
set "watchExitCode=%errorlevel%"
popd
exit /b %watchExitCode%
