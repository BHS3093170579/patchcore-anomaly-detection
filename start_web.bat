@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM Start the unsupervised defect-detection web server on Windows.
REM Override the interpreter with: set PY=C:\path\to\python.exe
REM before running this script.

if not defined PY set "PY=python"
set "URL=http://127.0.0.1:8000/"

REM 1) Already running? Just open the browser.
powershell -NoProfile -Command "try { if ((Invoke-WebRequest '%URL%health' -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { exit 0 } } catch {}; exit 1"
if %errorlevel%==0 (
  echo [INFO] Service is already running: %URL%
  start "" "%URL%"
  exit /b 0
)

REM 2) Check the interpreter.
"%PY%" -c "import sys" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python not found: %PY%
  echo Install Python 3.10+ and add it to PATH, or run:
  echo   set PY=C:\path\to\python.exe
  echo   start_web.bat
  echo.
  pause
  exit /b 1
)

REM 3) Start the server and open the browser once it is up.
echo Starting service: %URL%
echo Browser opens automatically in about 8 seconds. Keep this window open.
start "" /min powershell -NoProfile -Command "Start-Sleep -Seconds 8; Start-Process '%URL%'"

"%PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

echo.
echo [INFO] Service stopped.
echo If port 8000 was busy, close the process using it and run again.
pause
