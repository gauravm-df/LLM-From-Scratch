@echo off
REM ============================================================
REM AUTOMATIC VIRTUAL ENVIRONMENT SETUP (WINDOWS)
REM ============================================================

echo ============================================================
echo SETTING UP ISOLATED PYTHON VIRTUAL ENVIRONMENT (WINDOWS)
echo ============================================================

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    pause
    exit /b 1
)

python --version

if not exist "venv" (
    echo Creating virtual environment in .\venv...
    python -m venv venv
) else (
    echo Virtual environment .\venv already exists.
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing project dependencies...
pip install -r requirements.txt

echo ============================================================
echo SETUP COMPLETED SUCCESSFULLY!
echo To activate this environment anytime, run:
echo     venv\Scripts\activate
echo ============================================================
pause

