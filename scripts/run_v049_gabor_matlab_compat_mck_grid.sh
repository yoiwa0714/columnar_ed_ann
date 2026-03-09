#!/usr/bin/env bash
set -euo pipefail

# MATLAB互換Gabor: mck専用グリッドサーチ（逐次実行）
# - 実験スクリプト: columnar_ed_ann_v049_gabor_experiment.py
# - 固定: wl=2.8,5.6,11.3 / bw=1.0 / ar=0.5
# - 可変: mck={1.2,1.5,1.8,2.1}
#
# 使い方:
#   bash scripts/run_v049_gabor_matlab_compat_mck_grid.sh
#   bash scripts/run_v049_gabor_matlab_compat_mck_grid.sh --mode h4
#   bash scripts/run_v049_gabor_matlab_compat_mck_grid.sh --mode h5
#   bash scripts/run_v049_gabor_matlab_compat_mck_grid.sh --mode both --epochs 10

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXP_SCRIPT="$ROOT_DIR/columnar_ed_ann_v049_gabor_experiment.py"
LOG_DIR="$ROOT_DIR/logs/v049/MATLAB_compat"

MODE="h4"   # h4 | h5 | both
TRAIN=10000
TEST=10000
EPOCHS=10
SEED=42
DATASET="fashion"
CN=10
INIT_METHOD="he"
LR="0.03"
GC="0.03"

# ネットワーク側は既存探索と同条件
H4_HIDDEN="1024,1024,1024,1024"
H5_HIDDEN="1024,1024,1024,1024,1024"
H4_CLF="0.005,0.003,0.002,0.0015"
H5_CLF="0.005,0.003,0.002,0.0015,0.001"
H4_IS="0.9,0.9,1.8,1.4,0.8"
H5_IS="0.9,0.9,1.8,1.4,1.4,0.8"

# MATLAB互換Gabor固定パラメータ
MC_WL="2.8,5.6,11.3"
MC_BW="1.0"
MC_AR="0.5"

MCKS=("1.2" "1.5" "1.8" "2.1")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"; shift 2 ;;
    --train)
      TRAIN="$2"; shift 2 ;;
    --test)
      TEST="$2"; shift 2 ;;
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --lr)
      LR="$2"; shift 2 ;;
    --gc)
      GC="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"
      exit 2 ;;
  esac
done

if [[ "$MODE" != "h4" && "$MODE" != "h5" && "$MODE" != "both" ]]; then
  echo "--mode must be one of: h4, h5, both"
  exit 2
fi

if [[ ! -f "$EXP_SCRIPT" ]]; then
  echo "Experiment script not found: $EXP_SCRIPT"
  exit 2
fi

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

run_case() {
  local label="$1"
  local hidden="$2"
  local clf="$3"
  local isv="$4"
  local mc_k="$5"

  local hidden_tag="${hidden//,/:}"
  local clf_tag="${clf//,/:}"
  local is_tag="${isv//,/:}"
  local mc_wl_tag="${MC_WL//,/:}"

  local log_file="$LOG_DIR/v049_${DATASET}_hid${hidden_tag}_tr${TRAIN}_te${TEST}_epo${EPOCHS}_seed${SEED}_cn${CN}_im:${INIT_METHOD}_gabor_matlabcompat_lr${LR}_clf${clf_tag}_gc${GC}_is${is_tag}_mcwl${mc_wl_tag}_mcbw${MC_BW}_mcar${MC_AR}_mck${mc_k}_viz2_heatmap.log"

  echo "[START] ${label}"
  echo "[LOG]   ${log_file}"

  uv run python "$EXP_SCRIPT" \
    --matlab_compat \
    --mc_wavelengths "$MC_WL" \
    --mc_bandwidth "$MC_BW" \
    --mc_aspect_ratio "$MC_AR" \
    --mc_smooth_k "$mc_k" \
    --dataset "$DATASET" \
    --hidden "$hidden" \
    --train "$TRAIN" --test "$TEST" --epochs "$EPOCHS" \
    --seed "$SEED" --column_neurons "$CN" \
    --init_method "$INIT_METHOD" --gabor_features \
    --lr "$LR" --column_lr_factors "$clf" --gradient_clip "$GC" \
    --init_scales "$isv" \
    --viz 2 --heatmap > "$log_file" 2>&1

  echo "[DONE]  ${label}"
}

if [[ "$MODE" == "h4" || "$MODE" == "both" ]]; then
  for k in "${MCKS[@]}"; do
    run_case "H4_mck${k}" "$H4_HIDDEN" "$H4_CLF" "$H4_IS" "$k"
  done
  echo "All H4 mck-grid cases completed."
fi

if [[ "$MODE" == "h5" || "$MODE" == "both" ]]; then
  for k in "${MCKS[@]}"; do
    run_case "H5_mck${k}" "$H5_HIDDEN" "$H5_CLF" "$H5_IS" "$k"
  done
  echo "All H5 mck-grid cases completed."
fi

echo "All requested MATLAB-compat mck-grid cases completed."
