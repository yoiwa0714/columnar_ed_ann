#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="logs/v049_fashion5_clf_layerwise_grid_${TS}.log"

BASE_HIDDEN="1024,1024,1024,1024,1024"
BASE_ARGS=(
  --dataset fashion
  --hidden "$BASE_HIDDEN"
  --train 10000
  --test 10000
  --epochs 10
  --seed 42
  --column_neurons 20
  --init_method he
  --gabor_features
  --lr 0.02
  --gradient_clip 0.03
  --init_scales 1.6,1.6,1.8,1.6,1.6,0.8
  --viz 2
  --heatmap
)

echo "[$(date '+%F %T')] START fashion5 layerwise clf grid" | tee -a "$MASTER_LOG"
echo "[$(date '+%F %T')] Base clf: 0.002,0.002,0.002,0.002,0.002" | tee -a "$MASTER_LOG"

run_case() {
  local l1="$1"
  local l2="$2"
  local l3="$3"
  local l4="$4"
  local l5="$5"
  local tag="$6"
  local clf="${l1},${l2},${l3},${l4},${l5}"
  local logfile="logs/v049_fashion_hid1024:1024:1024:1024:1024_tr10000_te10000_epo10_seed42_cn20_im:he_gabor_lr0.02_clf${clf//,/:}_gc0.03_is1.6:1.6:1.8:1.6:1.6:0.8_${tag}.log"

  echo "[$(date '+%F %T')] RUN ${tag} clf=${clf}" | tee -a "$MASTER_LOG"
  uv run python columnar_ed_ann.py "${BASE_ARGS[@]}" --column_lr_factors "$clf" 2>&1 | tee "$logfile"

  local best_line final_line
  best_line="$(grep 'Best Test Accuracy:' "$logfile" | tail -n 1 || true)"
  final_line="$(grep 'Final Test Accuracy:' "$logfile" | tail -n 1 || true)"
  echo "[$(date '+%F %T')] DONE ${tag} ${final_line} ${best_line}" | tee -a "$MASTER_LOG"
}

# Layer1..Layer5: 0.002 -> 0.0015 and 0.0025
run_case 0.0015 0.002  0.002  0.002  0.002  L1_DN
run_case 0.0025 0.002  0.002  0.002  0.002  L1_UP
run_case 0.002  0.0015 0.002  0.002  0.002  L2_DN
run_case 0.002  0.0025 0.002  0.002  0.002  L2_UP
run_case 0.002  0.002  0.0015 0.002  0.002  L3_DN
run_case 0.002  0.002  0.0025 0.002  0.002  L3_UP
run_case 0.002  0.002  0.002  0.0015 0.002  L4_DN
run_case 0.002  0.002  0.002  0.0025 0.002  L4_UP
run_case 0.002  0.002  0.002  0.002  0.0015 L5_DN
run_case 0.002  0.002  0.002  0.002  0.0025 L5_UP

echo "[$(date '+%F %T')] ALL DONE" | tee -a "$MASTER_LOG"
echo "MASTER_LOG=$MASTER_LOG"
