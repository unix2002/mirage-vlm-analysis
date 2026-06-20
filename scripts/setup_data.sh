#!/bin/bash
# Download data files for Mirage VLM dashboard.
# Usage: bash scripts/setup_data.sh
#
# Downloads ~83 MB from GitHub Releases. Run once after cloning.
# All files are skipped if already present (idempotent).
#
# Set MIRAGE_DATA_URL to override the base URL.

set -euo pipefail

BASE_URL="${MIRAGE_DATA_URL:-https://github.com/unix2002/mirage-vlm-analysis/releases/download/data-v1}"
DATA_DIR="${1:-data}"

mkdir -p "$DATA_DIR/processed"

# 1. Attention cache — enables per-layer heatmap slider & spatial focus
if [ ! -f "$DATA_DIR/processed/attn_full.npz" ]; then
    echo "Downloading attention cache (38 MB)..."
    curl -L -o "$DATA_DIR/processed/attn_full.npz" "$BASE_URL/attn_full.npz"
    echo "Done."
else
    echo "Attention cache already present, skipping."
fi

# 2. Sample metadata — required for dashboard to load any data
if [ ! -f "$DATA_DIR/metadata.json" ]; then
    echo "Downloading metadata.json (2.7 MB)..."
    curl -L -o "$DATA_DIR/metadata.json" "$BASE_URL/metadata.json"
    echo "Done."
else
    echo "metadata.json already present, skipping."
fi

# 3. Maze images — enables Level 1 micro-view maze overlays
if [ ! -f "$DATA_DIR/vsp_spatial_planning.tar.gz" ]; then
    echo "Downloading maze images (42 MB)..."
    curl -L -o "$DATA_DIR/vsp_spatial_planning.tar.gz" "$BASE_URL/vsp_spatial_planning.tar.gz"
    echo "Done."
else
    echo "Maze images already present, skipping."
fi

echo ""
echo "All data ready. Run: python3 run_dashboard.py"
