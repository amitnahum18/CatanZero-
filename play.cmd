@echo off
setlocal
pushd "%~dp0"
if exist ".ml-venv\Scripts\python.exe" goto native
echo Using the project environment. Install uv and run: uv sync
uv run --with-requirements requirements-play.txt python -m catanzero.play_server --open %*
set "playExitCode=%errorlevel%"
goto finished
:native
set "PYTHONPATH=%~dp0src;%~dp0.venv\Lib\site-packages"
set "PYTHONHASHSEED=0"
".ml-venv\Scripts\python.exe" -m catanzero.play_server --open %*
set "playExitCode=%errorlevel%"
:finished
popd
exit /b %playExitCode%
