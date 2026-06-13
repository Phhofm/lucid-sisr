#!/usr/bin/env bash
# LUCID - Linux/macOS One-Click Installer & Launcher
# Automatically manages Python virtual environment, system dependencies, and GUI boot.

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "============================================="
echo "   LUCID: SISR Dataset Curator Launcher   "
echo "============================================="

# Function to offer Python installation
install_python_prompt() {
    echo "Python 3.10+ is required but was not found on your system."
    read -p "Would you like to attempt to install Python automatically now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command -v brew &>/dev/null; then
                echo "Installing Python via Homebrew..."
                brew install python
            else
                echo "Homebrew not found. Launching standard macOS Command Line Tools installer..."
                echo "Follow the on-screen prompt to install command-line developer tools (includes Python)."
                xcode-select --install
            fi
        elif [ -f /etc/debian_version ]; then
            # Ubuntu / Debian
            echo "Installing Python 3 & Virtual Environment tools via APT (requires sudo)..."
            sudo apt update
            sudo apt install -y python3 python3-pip python3-venv python3-tk
        else
            echo "Your operating system is not automatically supported for Python auto-install."
            echo "Please install Python 3.10+ manually from https://www.python.org/downloads/"
            exit 1
        fi
    else
        echo "Aborting launcher. Please install Python 3.10+ manually."
        exit 1
    fi
}

# 1. Detect Python
PYTHON_CMD=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    install_python_prompt
    # Re-check after installation attempt
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        echo "Failed to verify Python installation. Please restart the script."
        exit 1
    fi
fi

# Ensure Python version is compatible (3.10+)
VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
MAJOR=$(echo "$VERSION" | cut -d. -f1)
MINOR=$(echo "$VERSION" | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]; }; then
    echo "Your system Python version ($VERSION) is older than 3.10."
    install_python_prompt
    # Use the newly installed command
    PYTHON_CMD="python3"
fi

# Ensure python3-venv is present (especially on Debian/Ubuntu where it's split from python core)
if ! $PYTHON_CMD -c "import venv" &>/dev/null; then
    echo "Python 'venv' module is missing."
    if [ -f /etc/debian_version ]; then
        read -p "Would you like to install python3-venv via APT? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo apt update && sudo apt install -y python3-venv python3-tk
        fi
    else
        echo "Please install the Python virtual environment package (python3-venv) for your distribution."
        exit 1
    fi
fi

# Ensure tkinter is available for the GUI (needed on first-time Linux setups even when venv exists)
if [ ! -d "$VENV_DIR" ] || ! $PYTHON_CMD -c "import tkinter" &>/dev/null; then
    if [ -f /etc/debian_version ] && [ -d "$VENV_DIR" ]; then
        if ! python3 -c "import tkinter" &>/dev/null 2>&1; then
            echo "Installing tkinter support (required for GUI)..."
            sudo apt install -y python3-tk
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS Homebrew Python ships tkinter by default; verify via brew prefix
        if command -v brew &>/dev/null; then
            brew_prefix=$(brew --prefix python3 2>/dev/null || true)
            if [ -n "$brew_prefix" ] && [ ! -f "$brew_prefix/lib/python3."*/"tkinter"*.so 2>/dev/null ]; then
                echo "Warning: tkinter may not be available. Install via: brew install python-tk"
            fi
        fi
    fi
fi

# 2. Virtual Environment setup
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating local Python virtual environment in $VENV_DIR..."
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

# 3. Activate venv
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# 4. Install / Update dependencies in venv
echo "Checking dependencies..."
if ! pip show lucid-sisr &>/dev/null; then
    echo "First time setup: Installing dependencies..."
    pip install --upgrade pip
    pip install -e .
else
    pip install -e . --no-deps &>/dev/null || true
fi

# 5. Launch GUI
echo "Launching LUCID GUI..."
python -m lucid.gui

echo "============================================="
