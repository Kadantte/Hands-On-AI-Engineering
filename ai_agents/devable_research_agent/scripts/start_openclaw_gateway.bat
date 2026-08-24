@echo off
setlocal EnableExtensions

REM Start OpenClaw gateway if not already listening (scheduled at 7:30am Lagos).
set "GATEWAY_PORT=18789"
set "GATEWAY_CMD=%USERPROFILE%\.openclaw\gateway.cmd"
set "LOG_DIR=%USERPROFILE%\.openclaw"
set "LOG_FILE=%LOG_DIR%\gateway-scheduled-start.log"

if not exist "%GATEWAY_CMD%" (
    echo [%date% %time%] ERROR gateway.cmd not found: %GATEWAY_CMD%>> "%LOG_FILE%"
    exit /b 1
)

powershell -NoProfile -Command ^
  "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:%GATEWAY_PORT%/' -UseBasicParsing -TimeoutSec 3; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) { exit 0 } } catch { exit 1 }"
if %ERRORLEVEL%==0 (
    echo [%date% %time%] Gateway already running on port %GATEWAY_PORT%>> "%LOG_FILE%"
    exit /b 0
)

echo [%date% %time%] Starting OpenClaw gateway>> "%LOG_FILE%"
start "" /MIN cmd /c ""%GATEWAY_CMD%" >> "%LOG_FILE%" 2>&1"
exit /b 0
