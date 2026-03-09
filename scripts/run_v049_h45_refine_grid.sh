#!/usr/bin/env bash
set -euo pipefail

# 4層/5層の追加探索（逐次実行）
# 方針:
#  - is0=0.9固定
#  - 後段1.6スケールを {1.2, 1.4, 1.8} で探索
#  - column_lr_factors は A/B 比較
#    A: 3要素  (0.005,0.003,0.002)
#    B: 層数要素（4層=0.005,0.003,0.002,0.0015 / 5層=0.005,0.003,0.002,0.0015,0.001）
#  - lr は {0.03, 0.07}
# 合計: 4層12条件 + 5層12条件 = 24条件

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs/v049"

TRAIN=10000
TEST=10000
EPOCHS=10
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

  local log_file="$LOG_DIR/v049_${DATASET}_hid${hidden_tag}_tr${TRAIN}_te${TEST}_epo${EPOCHS}_seed${SEED}_cn${CN}_im:${INIT_METHOD}_gabor_lr${lr}_clf${clf_tag}_gc${GC}_is${is_tag}.log"

  echo "[START] ${label}"
  echo "[LOG]   ${log_file}"

  uv run python columnar_ed_ann.py \
    --dataset "$DATASET" \
    --hidden "$hidden" \
    --train "$TRAIN" --test "$TEST" --epochs "$EPOCHS" \
    --seed "$SEED" --column_neurons "$CN" \
    --init_method "$INIT_METHOD" --gabor_features \
    --lr "$lr" --column_lr_factors "$clf" --gradient_clip "$GC" \
    --init_scales "$isv" > "$log_file" 2>&1

  echo "[DONE]  ${label}"
}

# 4層探索
H4_HIDDEN="1024,1024,1024,1024"
for lr in 0.03 0.07; do
  for tail in 1.2 1.4 1.8; do
    # A: 3要素clf
    run_case "H4_A_lr${lr}_tail${tail}" "$H4_HIDDEN" "$lr" "0.005,0.003,0.002" "0.9,0.9,1.8,${tail},0.8"
    # B: 4要素clf
    run_case "H4_B_lr${lr}_tail${tail}" "$H4_HIDDEN" "$lr" "0.005,0.003,0.002,0.0015" "0.9,0.9,1.8,${tail},0.8"
  done
done

echo "All H4 experiments completed."

# 5層探索
# 注: 5層は後段1.6が2箇所あるため、両方を同じ候補値にそろえて探索
H5_HIDDEN="1024,1024,1024,1024,1024"
for lr in 0.03 0.07; do
  for tail in 1.2 1.4 1.8; do
    # A: 3要素clf
    run_case "H5_A_lr${lr}_tail${tail}" "$H5_HIDDEN" "$lr" "0.005,0.003,0.002" "0.9,0.9,1.8,${tail},${tail},0.8"
    # B: 5要素clf
    run_case "H5_B_lr${lr}_tail${tail}" "$H5_HIDDEN" "$lr" "0.005,0.003,0.002,0.0015,0.001" "0.9,0.9,1.8,${tail},${tail},0.8"
  done
done

echo "All H5 experiments completed."
echo "All H4/H5 refine-grid experiments completed."
