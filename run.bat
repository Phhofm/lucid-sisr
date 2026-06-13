@echo off
rem LUCID - Windows One-Click Installer & Launcher
rem Automatically manages Python virtual environment, system dependencies, and GUI boot.

echo =============================================
echo    LUCID: SISR Dataset Curator Launcher
echo =============================================

cd /d "%~dp0"

rem 1. Detect Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python 3.10+ is required but was not found on your system.
    echo.
    echo [1] Open Microsoft Store to install Python 3.11 (Recommended - One-click install)
    echo [2] Open Python.org Downloads page (Manual install)
    echo [3] Abort launcher
    echo.
    set /p choice="Please select an option (1-3): "
    
    if "%choice%"=="1" (
        echo Opening Microsoft Store...
        start ms-windows-store://search?query=Python
        echo Please install Python from the store and restart this launcher script.
    )
    if "%choice%"=="2" (
        echo Opening python.org...
        start https://www.python.org/downloads/
        echo Please run the installer, check "Add Python to PATH", and restart this launcher script.
    )
    pause
    exit /b 1
)

rem Ensure Python version is compatible (3.10+)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python 3.10+ is required. Your installed Python version is too old.
    echo Opening python.org downloads...
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

rem 2. Virtual Environment setup
set VENV_DIR=.venv
if not exist "%VENV_DIR%" (
    echo Creating local Python virtual environment in %VENV_DIR%...
    python -m venv "%VENV_DIR%"
)

rem 3. Activate venv
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

rem 4. Install / Update dependencies in venv
echo Checking dependencies...
pip show lucid-sisr >nul 2>nul
if %errorlevel% neq 0 (
    echo First time setup: Installing dependencies...
    python -m pip install --upgrade pip
    pip install -e .
) else (
    rem Quick incremental install check to ensure everything is up to date
    pip install -e . --no-deps >nul 2>nul
)

rem 5. Launch GUI
echo Launching LUCID GUI...
python -m lucid.gui

echo =============================================
pause
