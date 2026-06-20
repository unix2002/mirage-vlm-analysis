#!/bin/bash
# Download data files for Mirage VLM dashboard.
# Usage: bash scripts/setup_data.sh
#
# Downloads from GitHub Releases:
#   data/processed/attn_full.npz       (~38 MB, per-layer attention cache)
#   data/metadata.json                 (~2.7 MB, sample index)
#
# Optional (for maze overlays in Level 1 micro view):
#   data/vsp_spatial_planning.tar.gz   (~42 MB)
#
# Set MIRAGE_DATA_URL to override the base URL.

set -euo pipefail

BASE_URL="${MIRAGE_DATA_URL:-https://github.com/unix2002/mirage-vlm-analysis/releases/download/data-v1}"
DATA_DIR="${1:-data}"

mkdir -p "$DATA_DIR/processed"

echo "Downloading attention cache (38 MB)..."
curl -L -o "$DATA_DIR/processed/attn_full.npz" "$BASE_URL/attn_full.npz" --progress-bar
echo "Done."

if [ ! -f "$DATA_DIR/metadata.json" ]; then
    echo "Downloading metadata.json (2.7 MB)..."
    curl -L -o "$DATA_DIR/metadata.json" "$BASE_URL/metadata.json" --progress-bar
    echo "Done."
else
    echo "metadata.json already present, skipping."
fi

echo ""
echo "Setup complete. Run: python3 run_dashboard.py"
