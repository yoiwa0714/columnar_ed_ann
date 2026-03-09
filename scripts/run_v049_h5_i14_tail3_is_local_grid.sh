#!/usr/bin/env bash
set -euo pipefail

# I14固定の局所探索: init_scales 末尾3要素（L3, L4, out）をそれぞれ ±0.2
#
# 固定条件（ユーザー指定）:
# - --hidden 1024,1024,1024,1024,1024
# - --train 10000 --test 10000 --epochs 10
# - --viz 2 --heatmap
#
# 固定条件（I14ベース）:
# - dataset=mnist, seed=42, column_neurons=20, init_method=he, gabor_features
# - lr=0.04, column_lr_factors=0.005,0.004,0.003,0.002,0.0015, gradient_clip=0.03
# - init_scales(base)=0.9,1.6,1.8,1.4,1.4,0.8
#
# 探索対象（計6ケース）:
# - L3: 1.4 -> 1.2, 1.6
# - L4: 1.4 -> 1.2, 1.6
# - out: 0.8 -> 0.6, 1.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs/v049"

DATASET="mnist"
HIDDEN="1024,1024,1024,1024,1024"
TRAIN=10000
TEST=10000
EPOCHS=10
SEED=42
CN=20
INIT_METHOD="he"
GC="0.03"
LR="0.04"
CLF="0.005,0.004,0.003,0.002,0.0015"
VIZ="2"

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
    --lr)
      LR="$2"; shift 2 ;;
    --clf|--column_lr_factors)
      CLF="$2"; shift 2 ;;
    --viz)
      VIZ="$2"; shift 2 ;;
    --dry_run)
      DRY_RUN=1; shift ;;
    *)
      echo "Unknown argument: $1"
      exit 2 ;;
  esac
done

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

# label|init_scales
CASES=(
  "T3M|0.9,1.6,1.8,1.2,1.4,0.8"
  "T3P|0.9,1.6,1.8,1.6,1.4,0.8"
  "T4M|0.9,1.6,1.8,1.4,1.2,0.8"
  "T4P|0.9,1.6,1.8,1.4,1.6,0.8"
  "TOUTM|0.9,1.6,1.8,1.4,1.4,0.6"
  "TOUTP|0.9,1.6,1.8,1.4,1.4,1.0"
)

TOTAL=${#CASES[@]}

echo "===== H5 I14 tail3 local is grid ====="
echo "dataset=$DATASET hidden=$HIDDEN train=$TRAIN test=$TEST epochs=$EPOCHS seed=$SEED"
echo "cn=$CN init_method=$INIT_METHOD gc=$GC lr=$LR clf=$CLF viz=$VIZ heatmap=on"
echo "base_is=0.9,1.6,1.8,1.4,1.4,0.8"
echo "cases=$TOTAL"
echo "dry_run=$DRY_RUN"
echo

run_case() {
  local label="$1"
  local isv="$2"

  local hidden_tag="${HIDDEN//,/:}"
  local clf_tag="${CLF//,/:}"
  local is_tag="${isv//,/:}"

  local log_file="$LOG_DIR/v049_${DATASET}_hid${hidden_tag}_tr${TRAIN}_te${TEST}_epo${EPOCHS}_seed${SEED}_cn${CN}_im:${INIT_METHOD}_gabor_lr${LR}_clf${clf_tag}_gc${GC}_is${is_tag}_h5i14tail3_${label}.log"

  echo "[START] $label"
  echo "  is=$isv"
  echo "  log=$log_file"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY] uv run python columnar_ed_ann_v049.py --dataset $DATASET --hidden $HIDDEN --train $TRAIN --test $TEST --epochs $EPOCHS --seed $SEED --column_neurons $CN --init_method $INIT_METHOD --gabor_features --lr $LR --column_lr_factors $CLF --gradient_clip $GC --init_scales $isv --viz $VIZ --heatmap"
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
    --lr "$LR" --column_lr_factors "$CLF" --gradient_clip "$GC" \
    --init_scales "$isv" \
    --viz "$VIZ" --heatmap > "$log_file" 2>&1
  t1=$(date +%s)
  dt=$((t1 - t0))

  echo "[DONE]  $label elapsed=${dt}s"
  echo
}

idx=0
global_start=$(date +%s)

for c in "${CASES[@]}"; do
  idx=$((idx + 1))
  IFS='|' read -r label isv <<< "$c"
  echo "[$idx/$TOTAL]"
  run_case "$label" "$isv"
done

global_end=$(date +%s)
global_elapsed=$((global_end - global_start))
echo "All cases completed. total_elapsed=${global_elapsed}s"
