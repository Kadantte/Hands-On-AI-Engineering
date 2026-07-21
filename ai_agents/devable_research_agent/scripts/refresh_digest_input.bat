@echo off
REM Refresh digest_input_today.json from existing data_today.json (no re-collection).
REM Run this before a manual OpenClaw cron digest if the filter rules changed.

cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

if not exist data_today.json (
    echo ERROR: data_today.json missing. Run scripts\run_pipeline.bat first.
    exit /b 1
)

py -3 scripts\filter_top_items.py data_today.json digest_input_today.json --top 30
if errorlevel 1 (
    python scripts\filter_top_items.py data_today.json digest_input_today.json --top 30
)
exit /b %ERRORLEVEL%
