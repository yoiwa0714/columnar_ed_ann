#!/usr/bin/env bash
set -euo pipefail

# 5層パラメータ探索（lr / init_scales / column_lr_factors）
#
# 目的:
# - 4層最適設定（共有済み: lr=0.04, clf=0.005,0.004,0.003,0.002, is=0.9,1.6,1.8,1.6,0.8）を
#   5層に拡張したベースから、3軸の有効方向を短時間で把握する。
# - 実行時間を約2時間以内に抑えるため、全組合せではなく8ケースの準直交セットを使用。
#
# 使い方:
#   bash scripts/run_v049_h5_lr_is_clf_quick_grid.sh
#   bash scripts/run_v049_h5_lr_is_clf_quick_grid.sh --dry_run
#   bash scripts/run_v049_h5_lr_is_clf_quick_grid.sh --train 5000 --test 5000 --epochs 5

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs/v049"

# 5層固定条件（4層ベストを5層へ拡張）
DATASET="mnist"
HIDDEN="1024,1024,1024,1024,1024"
TRAIN=10000
TEST=10000
EPOCHS=10
SEED=42
CN=20
INIT_METHOD="he"
GC="0.03"

DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"; shift 2 ;;
    --hidden)
      HIDDEN="$2"; shift 2 ;;
    --train)
      TRAIN="$2"; shift 2 ;;
    --test)
      TEST="$2"; shift 2 ;;
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --cn|--column_neurons)
      CN="$2"; shift 2 ;;
    --init_method)
      INIT_METHOD="$2"; shift 2 ;;
    --gc|--gradient_clip)
      GC="$2"; shift 2 ;;
    --dry_run)
      DRY_RUN=1; shift ;;
    *)
      echo "Unknown argument: $1"
      exit 2 ;;
  esac
done

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

# ケース定義:
# label|lr|column_lr_factors|init_scales
# init_scales は5層なので6要素（L0..L4 + out）
CASES=(
  "B00|0.04|0.005,0.004,0.003,0.002,0.0015|0.9,1.6,1.8,1.6,1.6,0.8"
  "L03|0.03|0.005,0.004,0.003,0.002,0.0015|0.9,1.6,1.8,1.6,1.6,0.8"
  "L05|0.05|0.005,0.004,0.003,0.002,0.0015|0.9,1.6,1.8,1.6,1.6,0.8"
  "I14|0.04|0.005,0.004,0.003,0.002,0.0015|0.9,1.6,1.8,1.4,1.4,0.8"
  "I18|0.04|0.005,0.004,0.003,0.002,0.0015|0.9,1.6,1.8,1.8,1.8,0.8"
  "CLO|0.04|0.005,0.004,0.003,0.002,0.0010|0.9,1.6,1.8,1.6,1.6,0.8"
  "CHI|0.04|0.005,0.004,0.003,0.0025,0.0020|0.9,1.6,1.8,1.6,1.6,0.8"
  "X1 |0.03|0.005,0.004,0.003,0.0025,0.0020|0.9,1.6,1.8,1.8,1.8,0.8"
)

TOTAL=${#CASES[@]}

echo "===== H5 quick grid: lr/is/clf ====="
echo "dataset=$DATASET hidden=$HIDDEN train=$TRAIN test=$TEST epochs=$EPOCHS seed=$SEED"
echo "cn=$CN init_method=$INIT_METHOD gc=$GC"
echo "cases=$TOTAL"
echo "dry_run=$DRY_RUN"
echo

run_case() {
  local label="$1"
  local lr="$2"
  local clf="$3"
  local isv="$4"

  local hidden_tag="${HIDDEN//,/:}"
  local clf_tag="${clf//,/:}"
  local is_tag="${isv//,/:}"

  local log_file="$LOG_DIR/v049_${DATASET}_hid${hidden_tag}_tr${TRAIN}_te${TEST}_epo${EPOCHS}_seed${SEED}_cn${CN}_im:${INIT_METHOD}_gabor_lr${lr}_clf${clf_tag}_gc${GC}_is${is_tag}_h5quick_${label}.log"

  echo "[START] $label"
  echo "  lr=$lr"
  echo "  clf=$clf"
  echo "  is =$isv"
  echo "  log=$log_file"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY] uv run python columnar_ed_ann_v049.py --dataset $DATASET --hidden $HIDDEN --train $TRAIN --test $TEST --epochs $EPOCHS --seed $SEED --column_neurons $CN --init_method $INIT_METHOD --gabor_features --lr $lr --column_lr_factors $clf --gradient_clip $GC --init_scales $isv"
    echo
    return 0
  fi

  local t0 t1 dt
  t0=$(date +%s)
  uv run python columnar_ed_ann_v049.py \
    --dataset "$DATASET" \
    --hidden "$HIDDEN" \
    --train "$TRAIN" --test "$TEST" --epochs "$EPOCHS" \
    --seed "$SEED" --column_neurons "$CN" \
    --init_method "$INIT_METHOD" --gabor_features \
    --lr "$lr" --column_lr_factors "$clf" --gradient_clip "$GC" \
    --init_scales "$isv" > "$log_file" 2>&1
  t1=$(date +%s)
  dt=$((t1 - t0))

  echo "[DONE]  $label elapsed=${dt}s"
  echo
}

idx=0
global_start=$(date +%s)

for c in "${CASES[@]}"; do
  idx=$((idx + 1))
  IFS='|' read -r label lr clf isv <<< "$c"
  label="${label// /}"
  echo "[$idx/$TOTAL]"
  run_case "$label" "$lr" "$clf" "$isv"
done

global_end=$(date +%s)
global_elapsed=$((global_end - global_start))
echo "All cases completed. total_elapsed=${global_elapsed}s"
