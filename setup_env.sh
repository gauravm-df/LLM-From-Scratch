#!/bin/bash
# ============================================================
# AUTOMATIC VIRTUAL ENVIRONMENT SETUP (LINUX / MACOS)
# This keeps your main system 100% clean by isolating dependencies!
# ============================================================

set -e

echo "============================================================"
echo "SETTING UP ISOLATED PYTHON VIRTUAL ENVIRONMENT"
echo "============================================================"

# 1. Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH."
    exit 1
fi

echo "Using Python: $(python3 --version)"

# 2. Create isolated venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment in ./venv..."
    python3 -m venv venv
else
    echo "Virtual environment ./venv already exists."
fi

# 3. Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# 4. Upgrade pip inside venv
echo "Upgrading pip inside venv..."
pip install --upgrade pip

# 5. Install dependencies inside venv
echo "Installing project dependencies..."
pip install -r requirements.txt

echo "============================================================"
echo "SETUP COMPLETED SUCCESSFULLY!"
echo "To activate this environment anytime on this system, run:"
echo "    source venv/bin/activate"
echo "============================================================"

