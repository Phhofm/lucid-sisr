#!/bin/bash
# Upload lucid_cc0 to HuggingFace
# PREREQUISITE: Create the dataset on huggingface.co first:
#   https://huggingface.co/new-dataset -> name: lucid_cc0, license: CC0
#
# Usage:
#   cd /home/phips/Documents/GitHub/KiloApp2/kiloapp/lucid
#   bash huggingface/upload_lucid_cc0.sh

set -euo pipefail

VENV="/home/phips/Documents/GitHub/KiloApp2/kiloapp/lucid/.venv"
STAGING="/home/phips/Documents/GitHub/KiloApp2/kiloapp/lucid/huggingface/lucid_cc0"
TILES="/home/phips/Documents/dataset/lucidv2_cc0_06"
REPO_ID="Phips/lucid_cc0"
REPO_TYPE="dataset"

source "$VENV/bin/activate"

echo "=== Step 1: Upload metadata (README, LICENSE, provenance.csv, configs, logs) ==="
huggingface-cli upload "$REPO_ID" "$STAGING" "/" \
  --repo-type "$REPO_TYPE" \
  --include "*.md,*.csv,*.yml,*.yaml,*.png,*.txt" \
  --no-exclude "tiles/*" \
  --commit-message "Add metadata: README, LICENSE, provenance.csv, training configs and logs"

echo ""
echo "=== Step 2: Upload tiles (231,788 PNGs, ~32 GB) ==="
echo "This will take a while. Using LFS for files > 10 MB."
huggingface-cli upload "$REPO_ID" "$TILES" "/tiles" \
  --repo-type "$REPO_TYPE" \
  --include "*.png" \
  --commit-message "Add 231,788 CC0 tiles (pd12m + uhdiqa, complexity >= 0.6)"

echo ""
echo "=== Upload complete! ==="
echo "Dataset URL: https://huggingface.co/datasets/$REPO_ID"
