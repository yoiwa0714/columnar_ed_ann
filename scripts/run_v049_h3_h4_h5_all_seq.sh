#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

bash "$ROOT_DIR/scripts/run_v049_h3_experiments.sh"
bash "$ROOT_DIR/scripts/run_v049_h4_experiments.sh"
bash "$ROOT_DIR/scripts/run_v049_h5_experiments.sh"

echo "All H3/H4/H5 sequential experiments completed."