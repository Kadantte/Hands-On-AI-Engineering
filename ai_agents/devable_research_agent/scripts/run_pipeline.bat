@echo off
setlocal EnableExtensions

REM AI Research Agent — daily data collection (Task Scheduler entry point)
REM Runs at 7:45am Lagos time; OpenClaw reads digest_input_today.json at 8am.

set "PROJECT_ROOT=%~dp0.."
set "DEVABLE_PROJECT_ROOT=%PROJECT_ROOT%"
cd /d "%PROJECT_ROOT%" || (
    echo ERROR: Could not change to project directory: %PROJECT_ROOT%
    exit /b 1
)

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

REM Prefer the Python launcher; fall back to python on PATH.
where py >nul 2>&1
if %ERRORLEVEL%==0 (
    py -3 "%PROJECT_ROOT%\scripts\run_pipeline.py"
    exit /b %ERRORLEVEL%
)

python "%PROJECT_ROOT%\scripts\run_pipeline.py"
exit /b %ERRORLEVEL%
