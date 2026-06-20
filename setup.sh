#!/bin/bash
# Mirage VLM Dashboard — one-command setup for new users.
#
# Usage: bash setup.sh
#
# 1. Creates Python venv (if missing)
# 2. Sources it, installs dependencies
# 3. Downloads data files from GitHub Releases (~83 MB)
# 4. Optionally starts the dashboard

set -euo pipefail

echo "========================================"
echo " Mirage VLM — Dashboard Setup"
echo "========================================"
echo ""

# 1. Virtual environment
if [ ! -d "env" ]; then
    echo "[1/3] Creating Python virtual environment..."
    python3 -m venv env
else
    echo "[1/3] Virtual environment already exists, skipping."
fi

source env/bin/activate

# 2. Dependencies
echo "[2/3] Installing dependencies..."
pip install -r requirements.txt --quiet

# 3. Data files
echo "[3/3] Downloading data..."
bash scripts/setup_data.sh

echo ""
echo "========================================"
echo " Setup complete!"
echo ""
echo " To start the dashboard next time:"
echo "   source env/bin/activate"
echo "   python3 run_dashboard.py"
echo "========================================"
echo ""

read -p "Start the dashboard now? (Y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    python3 run_dashboard.py
fi
