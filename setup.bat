@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo ComicCraft-Gemini - First Time Setup
echo ==========================================
echo.

REM Use an existing venv in the parent folder if present.
if exist "..\venv\Scripts\python.exe" (
    set "PYTHON=%~dp0..\venv\Scripts\python.exe"
    echo Using existing virtual environment: ..\venv
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON=%~dp0venv\Scripts\python.exe"
    echo Using existing virtual environment: venv
) else (
    echo Creating a virtual environment...
    py -m venv venv
    if errorlevel 1 (
        echo.
        echo ERROR: Could not create the virtual environment.
        echo Make sure Python is installed and the "py" command works.
        pause
        exit /b 1
    )
    set "PYTHON=%~dp0venv\Scripts\python.exe"
)

echo.
echo Installing project dependencies...
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Dependency installation failed.
    pause
    exit /b 1
)

if not exist ".env" if exist ".env.example" (
    copy /Y ".env.example" ".env" >nul
    echo.
    echo Created .env from .env.example.
    echo Add your API keys/settings to .env before using the app.
) else (
    echo.
    echo Existing .env found. It was not changed.
)

echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo Double-click run.bat to start ComicCraft.
echo.
pause
