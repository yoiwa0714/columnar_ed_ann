#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs/v049"

TRAIN=10000
TEST=10000
EPOCHS=10
SEED=42
CN=10
INIT_METHOD="he"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

run_case() {
  local label="$1"
  local dataset="$2"
  local hidden="$3"
  local lr="$4"
  local clf="$5"
  local gc="$6"
  local isv="$7"

  local hidden_tag="${hidden//,/:}"
  local clf_tag="${clf//,/:}"
  local is_tag="${isv//,/:}"
  local log_file="$LOG_DIR/v049_${dataset}_hid${hidden_tag}_tr${TRAIN}_te${TEST}_epo${EPOCHS}_seed${SEED}_cn${CN}_im:${INIT_METHOD}_gabor_lr${lr}_clf${clf_tag}_gc${gc}_is${is_tag}.log"

  echo "[START] ${label}"
  echo "[LOG]   ${log_file}"

  uv run python columnar_ed_ann.py \
    --dataset "$dataset" \
    --hidden "$hidden" \
    --train "$TRAIN" --test "$TEST" --epochs "$EPOCHS" \
    --seed "$SEED" --column_neurons "$CN" \
    --init_method "$INIT_METHOD" --gabor_features \
    --lr "$lr" --column_lr_factors "$clf" --gradient_clip "$gc" \
    --init_scales "$isv" > "$log_file" 2>&1

  echo "[DONE]  ${label}"
}

# 4層: v049で高精度だったFashion条件を基準
run_case "H4_FASHION_IS09" "fashion" "1024,1024,1024,1024" "0.05" "0.005,0.003,0.002,0.0015" "0.03" "0.9,0.9,1.8,1.6,0.8"
run_case "H4_FASHION_IS07" "fashion" "1024,1024,1024,1024" "0.05" "0.005,0.003,0.002,0.0015" "0.03" "0.7,0.7,1.8,1.6,0.8"

echo "All H4 experiments completed."