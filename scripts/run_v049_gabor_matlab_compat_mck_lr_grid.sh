#!/usr/bin/env bash
set -euo pipefail

# MATLAB互換Gabor: mc_smooth_k × lr の2軸グリッド（逐次実行）
#
# 固定条件は最新共有の4層高精度設定をベースにし、
# --matlab_compat の Gabor 側のみを探索する。
#
# 可変軸:
#   - lr:         0.02,0.03,0.04,0.05,0.06
#   - mc_smooth_k: 1.2,1.4,1.5,1.6,1.8,2.1,2.4
#
# 使い方:
#   bash scripts/run_v049_gabor_matlab_compat_mck_lr_grid.sh
#   bash scripts/run_v049_gabor_matlab_compat_mck_lr_grid.sh --dry_run
#   bash scripts/run_v049_gabor_matlab_compat_mck_lr_grid.sh --epochs 5 --train 5000 --test 5000
#   bash scripts/run_v049_gabor_matlab_compat_mck_lr_grid.sh \
#       --lr_list 0.03,0.04,0.05 --mck_list 1.4,1.5,1.6,1.8

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXP_SCRIPT="$ROOT_DIR/columnar_ed_ann_v049_gabor_experiment.py"
LOG_DIR="$ROOT_DIR/logs/v049/MATLAB_compat"

# 最新共有の4層ベスト設定をデフォルト採用
DATASET="mnist"
HIDDEN="1024,1024,1024,1024"
TRAIN=10000
TEST=10000
EPOCHS=10
SEED=42
CN=20
INIT_METHOD="he"
GC="0.03"
CLF="0.005,0.004,0.003,0.002"
INIT_SCALES="0.9,1.6,1.8,1.6,0.8"

# MATLAB互換Gabor固定パラメータ（既探索で良好だった中心）
MC_WL="2.8,5.6,11.3"
MC_BW="1.0"
MC_AR="0.5"

# 2軸グリッド
LR_LIST="0.02,0.03,0.04,0.05,0.06"
MCK_LIST="1.2,1.4,1.5,1.6,1.8,2.1,2.4"

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
    --clf|--column_lr_factors)
      CLF="$2"; shift 2 ;;
    --init_scales)
      INIT_SCALES="$2"; shift 2 ;;
    --mc_wavelengths)
      MC_WL="$2"; shift 2 ;;
    --mc_bandwidth)
      MC_BW="$2"; shift 2 ;;
    --mc_aspect_ratio)
      MC_AR="$2"; shift 2 ;;
    --lr_list)
      LR_LIST="$2"; shift 2 ;;
    --mck_list)
      MCK_LIST="$2"; shift 2 ;;
    --dry_run)
      DRY_RUN=1; shift ;;
    *)
      echo "Unknown argument: $1"
      exit 2 ;;
  esac
done

if [[ ! -f "$EXP_SCRIPT" ]]; then
  echo "Experiment script not found: $EXP_SCRIPT"
  exit 2
fi

IFS=',' read -r -a LR_ARR <<< "$LR_LIST"
IFS=',' read -r -a MCK_ARR <<< "$MCK_LIST"

if [[ ${#LR_ARR[@]} -eq 0 || ${#MCK_ARR[@]} -eq 0 ]]; then
  echo "lr_list and mck_list must not be empty"
  exit 2
fi

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

TOTAL=$(( ${#LR_ARR[@]} * ${#MCK_ARR[@]} ))
echo "===== MATLAB-compat mck x lr grid ====="
echo "dataset=$DATASET hidden=$HIDDEN train=$TRAIN test=$TEST epochs=$EPOCHS seed=$SEED"
echo "cn=$CN init_method=$INIT_METHOD gc=$GC"
echo "clf=$CLF init_scales=$INIT_SCALES"
echo "mc_wl=$MC_WL mc_bw=$MC_BW mc_ar=$MC_AR"
echo "lr_list=$LR_LIST"
echo "mck_list=$MCK_LIST"
echo "total_cases=$TOTAL"
echo "dry_run=$DRY_RUN"
echo

case_idx=0
global_start=$(date +%s)

for lr in "${LR_ARR[@]}"; do
  for mck in "${MCK_ARR[@]}"; do
    case_idx=$((case_idx + 1))

    hidden_tag="${HIDDEN//,/:}"
    clf_tag="${CLF//,/:}"
    is_tag="${INIT_SCALES//,/:}"
    mc_wl_tag="${MC_WL//,/:}"

    log_file="$LOG_DIR/v049_${DATASET}_hid${hidden_tag}_tr${TRAIN}_te${TEST}_epo${EPOCHS}_seed${SEED}_cn${CN}_im:${INIT_METHOD}_gabor_matlabcompat_lr${lr}_clf${clf_tag}_gc${GC}_is${is_tag}_mcwl${mc_wl_tag}_mcbw${MC_BW}_mcar${MC_AR}_mck${mck}.log"

    echo "[$case_idx/$TOTAL] START lr=$lr mck=$mck"
    echo "[LOG] $log_file"

    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[DRY] uv run python $EXP_SCRIPT --matlab_compat --mc_wavelengths $MC_WL --mc_bandwidth $MC_BW --mc_aspect_ratio $MC_AR --mc_smooth_k $mck --dataset $DATASET --hidden $HIDDEN --train $TRAIN --test $TEST --epochs $EPOCHS --seed $SEED --column_neurons $CN --init_method $INIT_METHOD --gabor_features --lr $lr --column_lr_factors $CLF --gradient_clip $GC --init_scales $INIT_SCALES"
      echo
      continue
    fi

    case_start=$(date +%s)
    uv run python "$EXP_SCRIPT" \
      --matlab_compat \
      --mc_wavelengths "$MC_WL" \
      --mc_bandwidth "$MC_BW" \
      --mc_aspect_ratio "$MC_AR" \
      --mc_smooth_k "$mck" \
      --dataset "$DATASET" \
      --hidden "$HIDDEN" \
      --train "$TRAIN" --test "$TEST" --epochs "$EPOCHS" \
      --seed "$SEED" --column_neurons "$CN" \
      --init_method "$INIT_METHOD" --gabor_features \
      --lr "$lr" --column_lr_factors "$CLF" --gradient_clip "$GC" \
      --init_scales "$INIT_SCALES" > "$log_file" 2>&1
    case_end=$(date +%s)
    elapsed=$((case_end - case_start))
    echo "[$case_idx/$TOTAL] DONE  lr=$lr mck=$mck elapsed=${elapsed}s"
    echo
  done
done

global_end=$(date +%s)
global_elapsed=$((global_end - global_start))
echo "All requested cases completed. total_elapsed=${global_elapsed}s"
