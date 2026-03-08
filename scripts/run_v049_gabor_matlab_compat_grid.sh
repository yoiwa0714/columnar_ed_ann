#!/usr/bin/env bash
set -euo pipefail

# MATLAB互換寄りGabor探索（逐次実行専用）
# - 実験スクリプト: columnar_ed_ann_v049_gabor_experiment.py
# - 切替: --matlab_compat
# - 9条件（Gaborパラメータ）を、4層/5層に対して順次実行
#
# 使い方:
#   bash scripts/run_v049_gabor_matlab_compat_grid.sh
#   bash scripts/run_v049_gabor_matlab_compat_grid.sh --mode h4
#   bash scripts/run_v049_gabor_matlab_compat_grid.sh --mode h5
#   bash scripts/run_v049_gabor_matlab_compat_grid.sh --mode both --train 10000 --test 10000 --epochs 10

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXP_SCRIPT="$ROOT_DIR/columnar_ed_ann_v049_gabor_experiment.py"
LOG_DIR="$ROOT_DIR/logs/v049"

MODE="both"  # h4 | h5 | both
TRAIN=10000
TEST=10000
EPOCHS=10
SEED=42
DATASET="fashion"
CN=10
INIT_METHOD="he"
LR="0.03"
GC="0.03"

# ネットワーク側は固定（Gabor条件のみ評価）
H4_HIDDEN="1024,1024,1024,1024"
H5_HIDDEN="1024,1024,1024,1024,1024"
H4_CLF="0.005,0.003,0.002,0.0015"
H5_CLF="0.005,0.003,0.002,0.0015,0.001"
H4_IS="0.9,0.9,1.8,1.4,0.8"
H5_IS="0.9,0.9,1.8,1.4,1.4,0.8"

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

# label|wavelengths|bandwidth|aspect_ratio|smooth_k
CASES=(
  "MC01|2.8,5.6,11.3|1.0|0.5|3.0"
  "MC02|4,8,16|1.0|0.5|3.0"
  "MC03|3,6,12|1.0|0.5|3.0"
  "MC04|2.8,5.6,11.3|0.8|0.5|3.0"
  "MC05|2.8,5.6,11.3|1.4|0.5|3.0"
  "MC06|2.8,5.6,11.3|1.0|0.4|3.0"
  "MC07|2.8,5.6,11.3|1.0|0.7|3.0"
  "MC08|2.8,5.6,11.3|1.0|0.5|1.5"
  "MC09|2.8,5.6,11.3|1.0|0.5|4.5"
)

run_case() {
  local label="$1"
  local hidden="$2"
  local clf="$3"
  local isv="$4"
  local mc_wl="$5"
  local mc_bw="$6"
  local mc_ar="$7"
  local mc_k="$8"

  local hidden_tag="${hidden//,/:}"
  local clf_tag="${clf//,/:}"
  local is_tag="${isv//,/:}"
  local mc_wl_tag="${mc_wl//,/:}"

  local log_file="$LOG_DIR/v049_${DATASET}_hid${hidden_tag}_tr${TRAIN}_te${TEST}_epo${EPOCHS}_seed${SEED}_cn${CN}_im:${INIT_METHOD}_gabor_matlabcompat_lr${LR}_clf${clf_tag}_gc${GC}_is${is_tag}_mcwl${mc_wl_tag}_mcbw${mc_bw}_mcar${mc_ar}_mck${mc_k}.log"

  echo "[START] ${label}"
  echo "[LOG]   ${log_file}"

  uv run python "$EXP_SCRIPT" \
    --matlab_compat \
    --mc_wavelengths "$mc_wl" \
    --mc_bandwidth "$mc_bw" \
    --mc_aspect_ratio "$mc_ar" \
    --mc_smooth_k "$mc_k" \
    --dataset "$DATASET" \
    --hidden "$hidden" \
    --train "$TRAIN" --test "$TEST" --epochs "$EPOCHS" \
    --seed "$SEED" --column_neurons "$CN" \
    --init_method "$INIT_METHOD" --gabor_features \
    --lr "$LR" --column_lr_factors "$clf" --gradient_clip "$GC" \
    --init_scales "$isv" > "$log_file" 2>&1

  echo "[DONE]  ${label}"
}

if [[ "$MODE" == "h4" || "$MODE" == "both" ]]; then
  for c in "${CASES[@]}"; do
    IFS='|' read -r tag mc_wl mc_bw mc_ar mc_k <<< "$c"
    run_case "H4_${tag}" "$H4_HIDDEN" "$H4_CLF" "$H4_IS" "$mc_wl" "$mc_bw" "$mc_ar" "$mc_k"
  done
  echo "All H4 MATLAB-compat Gabor cases completed."
fi

if [[ "$MODE" == "h5" || "$MODE" == "both" ]]; then
  for c in "${CASES[@]}"; do
    IFS='|' read -r tag mc_wl mc_bw mc_ar mc_k <<< "$c"
    run_case "H5_${tag}" "$H5_HIDDEN" "$H5_CLF" "$H5_IS" "$mc_wl" "$mc_bw" "$mc_ar" "$mc_k"
  done
  echo "All H5 MATLAB-compat Gabor cases completed."
fi

echo "All requested MATLAB-compat Gabor grid cases completed."
