#!/usr/bin/env bash
set -euo pipefail

# H4/H5 選抜4条件を 20epoch で逐次実行
# 条件は v049_h45_refine_grid と同一、差分は epochs=20 と --viz 2 --heatmap の追加のみ

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs/v049"

TRAIN=10000
TEST=10000
EPOCHS=20
SEED=42
CN=10
INIT_METHOD="he"
DATASET="fashion"
GC="0.03"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

run_case() {
  local label="$1"
  local hidden="$2"
  local lr="$3"
  local clf="$4"
  local isv="$5"

  local hidden_tag="${hidden//,/:}"
  local clf_tag="${clf//,/:}"
  local is_tag="${isv//,/:}"

  local log_file="$LOG_DIR/v049_${DATASET}_hid${hidden_tag}_tr${TRAIN}_te${TEST}_epo${EPOCHS}_seed${SEED}_cn${CN}_im:${INIT_METHOD}_gabor_lr${lr}_clf${clf_tag}_gc${GC}_is${is_tag}_viz2_heatmap.log"

  echo "[START] ${label}"
  echo "[LOG]   ${log_file}"

  uv run python columnar_ed_ann.py \
    --dataset "$DATASET" \
    --hidden "$hidden" \
    --train "$TRAIN" --test "$TEST" --epochs "$EPOCHS" \
    --seed "$SEED" --column_neurons "$CN" \
    --init_method "$INIT_METHOD" --gabor_features \
    --lr "$lr" --column_lr_factors "$clf" --gradient_clip "$GC" \
    --init_scales "$isv" \
    --viz 2 --heatmap > "$log_file" 2>&1

  echo "[DONE]  ${label}"
}

# 1) H4_B_lr0.03_tail1.2
run_case "H4_B_lr0.03_tail1.2" \
  "1024,1024,1024,1024" \
  "0.03" \
  "0.005,0.003,0.002,0.0015" \
  "0.9,0.9,1.8,1.2,0.8"

# 2) H4_A_lr0.03_tail1.2
run_case "H4_A_lr0.03_tail1.2" \
  "1024,1024,1024,1024" \
  "0.03" \
  "0.005,0.003,0.002" \
  "0.9,0.9,1.8,1.2,0.8"

# 3) H5_A_lr0.07_tail1.4
run_case "H5_A_lr0.07_tail1.4" \
  "1024,1024,1024,1024,1024" \
  "0.07" \
  "0.005,0.003,0.002" \
  "0.9,0.9,1.8,1.4,1.4,0.8"

# 4) H5_A_lr0.07_tail1.2
run_case "H5_A_lr0.07_tail1.2" \
  "1024,1024,1024,1024,1024" \
  "0.07" \
  "0.005,0.003,0.002" \
  "0.9,0.9,1.8,1.2,1.2,0.8"

echo "All selected H4/H5 20-epoch runs completed."
