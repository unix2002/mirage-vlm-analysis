#!/bin/bash

# Mirage VLM Analysis Setup Script
# This script loads modules, ensures dependencies, and configures PYTHONPATH.
# Usage: source setup.sh

echo "--- Initializing Mirage VLM Environment ---"

# 1. Load System Modules
echo "[1/3] Loading modules..."
module load 2023 PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# 2. Ensure Dependencies (User Space)
echo "[2/3] Checking dependencies..."
# Upgrade typing_extensions first to avoid Dash compatibility issues
pip install --user --upgrade typing_extensions --quiet
# Install project requirements and missing analytics tools
pip install --user -r requirements.txt scikit-learn umap-learn scipy --quiet

# 3. Configure PYTHONPATH
echo "[3/3] Configuring PYTHONPATH..."
# Get the absolute path of the project directory
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [[ ":$PYTHONPATH:" != *":$PROJECT_ROOT:"* ]]; then
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    echo "Added $PROJECT_ROOT to PYTHONPATH"
else
    echo "Project root already in PYTHONPATH"
fi

echo "--- Setup Complete ---"
echo "To start the dashboard, run: python3 run_dashboard.py"
echo "To run tests, run: python3 -m pytest tests/dashboard"
