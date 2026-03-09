#!/usr/bin/env bash
set -euo pipefail

# 3-layer grid search (same as 2026-03-04 run), but with train/test=10000.
# Run this script from anywhere; paths are resolved relative to columnar_ed_ann.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_SCRIPT="$ROOT_DIR/scripts/grid_search_3layer_target6h.sh"

if [[ ! -f "$BASE_SCRIPT" ]]; then
  echo "Base script not found: $BASE_SCRIPT"
  exit 2
fi

cd "$ROOT_DIR"

echo "============================================================"
echo "3-layer grid search (same params as yesterday, 10k/10k)"
echo "root: $ROOT_DIR"
echo "============================================================"
echo "- mode: full"
echo "- epochs: 30"
echo "- train: 10000"
echo "- test: 10000"
echo "- grid definition: scripts/grid_search_3layer_target6h.sh"
echo

bash "$BASE_SCRIPT" --full --epochs 30 --train 10000 --test 10000
