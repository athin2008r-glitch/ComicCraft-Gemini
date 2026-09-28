@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo ComicCraft-Gemini
echo ==========================================
echo.

REM Use an existing venv in the parent folder if present.
if exist "..\venv\Scripts\python.exe" (
    set "PYTHON=%~dp0..\venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON=%~dp0venv\Scripts\python.exe"
) else (
    echo No virtual environment found.
    echo Run setup.bat first.
    pause
    exit /b 1
)

echo Starting ComicCraft...
echo.
echo Open: http://127.0.0.1:8000
echo.
echo Keep this window open while using the application.
echo Press Ctrl+C here to stop the server.
echo.

start "" http://127.0.0.1:8000
"%PYTHON%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

echo.
echo ComicCraft has stopped.
pause
