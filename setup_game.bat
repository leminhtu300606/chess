@echo off
echo ==================================================
echo   Chess AI App Setup & Run
echo ==================================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.x from python.org or Microsoft Store.
    echo.
    pause
    exit /b
)

REM Check if venv exists, if not create it
if not exist "venv" (
    echo [INFO] Creating virtual environment 'venv'...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create venv. Make sure 'python-venv' is installed if on Linux, or check permissions.
        pause
        exit /b
    )
    echo [INFO] Virtual environment created.
) else (
    echo [INFO] 'venv' already exists.
)

REM Activate venv
echo [INFO] Activating virtual environment...
call venv\Scripts\activate

REM Upgrade build tools
echo [INFO] Upgrading pip and setuptools...
python -m pip install --upgrade pip setuptools wheel

REM Install dependencies (Using pygame-ce for better compatibility)
echo [INFO] Installing pygame library...
pip uninstall -y pygame
pip install pygame-ce >nul

REM Run the game
echo [INFO] Starting Chess AI...
python main.py

REM Deactivate (optional, as script ends)
deactivate
pause
