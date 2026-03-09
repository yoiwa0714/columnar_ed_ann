#!/usr/bin/env bash
set -u

MODE=""
EPOCHS=""
TRAIN=5000
TEST=5000
TARGET_HOURS=6
AUTO_FROM_SMOKE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE="smoke"; shift ;;
    --full) MODE="full"; shift ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --train) TRAIN="$2"; shift 2 ;;
    --test) TEST="$2"; shift 2 ;;
    --target-hours) TARGET_HOURS="$2"; shift 2 ;;
    --auto-epochs-from-smoke) AUTO_FROM_SMOKE="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "Specify --smoke or --full"
  exit 2
fi

if [[ "$MODE" == "smoke" ]]; then
  EPOCHS=1
fi

if [[ "$MODE" == "full" && -n "$AUTO_FROM_SMOKE" ]]; then
  if [[ ! -f "$AUTO_FROM_SMOKE" ]]; then
    echo "Smoke summary csv not found: $AUTO_FROM_SMOKE"
    exit 2
  fi
  AVG_EPOCH_SEC=$(awk -F'\t' 'NR>1 && $14>0 {sum+=$14; n++} END{if(n>0) printf "%.6f", sum/n; else print "0"}' "$AUTO_FROM_SMOKE")
  if awk "BEGIN{exit !($AVG_EPOCH_SEC > 0)}"; then
    RAW_EPOCHS=$(awk -v h="$TARGET_HOURS" -v a="$AVG_EPOCH_SEC" 'BEGIN{printf "%.0f", (h*3600)/(20*a)}')
    if [[ "$RAW_EPOCHS" -lt 5 ]]; then
      EPOCHS=5
    elif [[ "$RAW_EPOCHS" -gt 30 ]]; then
      EPOCHS=30
    else
      EPOCHS="$RAW_EPOCHS"
    fi
  else
    echo "Failed to compute epochs from smoke CSV: avg_epoch_sec=$AVG_EPOCH_SEC"
    exit 2
  fi
fi

if [[ -z "$EPOCHS" ]]; then
  EPOCHS=20
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

# Prefer project-local venv, but allow external active environment.
if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="./.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN"
  echo "Hint: create .venv in this repo or set PYTHON_BIN=/path/to/python"
  exit 2
fi

if [[ ! -f "columnar_ed_ann.py" ]]; then
  echo "columnar_ed_ann.py not found in $ROOT_DIR"
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="logs/grid_search_3layer_${MODE}_${TS}"
SUMMARY_CSV="results/grid_search_3layer_${MODE}_${TS}.tsv"
mkdir -p "$RUN_DIR" results

echo "mode=$MODE"
echo "epochs=$EPOCHS"
echo "train=$TRAIN test=$TEST"
echo "run_dir=$RUN_DIR"
echo "summary_csv=$SUMMARY_CSV"

printf "idx\ttag\thidden\tinit_scales\tlr\tcolumn_lr_factors\tgradient_clip\tepochs\ttrain\ttest\tstart_ts\tend_ts\tduration_sec\tsec_per_epoch\tbest_test_pct\tfinal_test_pct\tstatus\tlog_file\n" > "$SUMMARY_CSV"

EXPERIMENTS=(
  "B00|2048,1024,1024|0.7,1.8,1.8,0.8|0.20|0.005,0.003,0.002|0.03"
  "S01|2048,1024,1024|0.7,1.8,1.6,0.8|0.20|0.005,0.003,0.002|0.03"
  "S02|2048,1024,1024|0.7,1.8,2.0,0.8|0.20|0.005,0.003,0.002|0.03"
  "S03|2048,1024,1024|0.7,1.6,1.8,0.8|0.20|0.005,0.003,0.002|0.03"
  "S04|2048,1024,1024|0.7,2.0,1.8,0.8|0.20|0.005,0.003,0.002|0.03"
  "L01|2048,1024,1024|0.7,1.8,1.8,0.8|0.15|0.005,0.003,0.002|0.03"
  "L02|2048,1024,1024|0.7,1.8,1.8,0.8|0.22|0.005,0.003,0.002|0.03"
  "L03|2048,1024,1024|0.7,1.8,1.6,0.8|0.15|0.005,0.003,0.002|0.03"
  "L04|2048,1024,1024|0.7,1.8,1.6,0.8|0.22|0.005,0.003,0.002|0.03"
  "L05|2048,1024,1024|0.7,1.8,2.0,0.8|0.15|0.005,0.003,0.002|0.03"
  "L06|2048,1024,1024|0.7,1.8,2.0,0.8|0.22|0.005,0.003,0.002|0.03"
  "C01|2048,1024,1024|0.7,1.8,1.8,0.8|0.20|0.004,0.0025,0.0015|0.03"
  "C02|2048,1024,1024|0.7,1.8,1.8,0.8|0.20|0.006,0.004,0.003|0.03"
  "C03|2048,1024,1024|0.7,1.8,1.6,0.8|0.20|0.004,0.0025,0.0015|0.03"
  "C04|2048,1024,1024|0.7,1.8,2.0,0.8|0.20|0.004,0.0025,0.0015|0.03"
  "G01|2048,1024,1024|0.7,1.8,1.8,0.8|0.20|0.005,0.003,0.002|0.025"
  "G02|2048,1024,1024|0.7,1.8,1.8,0.8|0.20|0.005,0.003,0.002|0.035"
  "H01|2048,1536,1024|0.7,1.8,1.8,0.8|0.20|0.005,0.003,0.002|0.03"
  "H02|2048,1024,768|0.7,1.8,1.8,0.8|0.20|0.005,0.003,0.002|0.03"
  "H03|2048,1536,1024|0.7,1.8,1.6,0.8|0.20|0.005,0.003,0.002|0.03"
)

IDX=0
for exp in "${EXPERIMENTS[@]}"; do
  IDX=$((IDX+1))
  IFS='|' read -r TAG HIDDEN INIT_SCALES LR_VALUE CLRF_VALUE GCLIP <<< "$exp"

  LOG_FILE="$RUN_DIR/$(printf '%02d' "$IDX")_${TAG}.log"
  START_TS="$(date +%Y-%m-%dT%H:%M:%S)"
  START_SEC="$(date +%s)"

  echo "[$IDX/${#EXPERIMENTS[@]}] $TAG hidden=$HIDDEN is=$INIT_SCALES lr=$LR_VALUE clrf=$CLRF_VALUE gc=$GCLIP"

  CMD=(
    "$PYTHON_BIN" columnar_ed_ann.py
    --dataset mnist
    --hidden "$HIDDEN"
    --train "$TRAIN"
    --test "$TEST"
    --epochs "$EPOCHS"
    --seed 42
    --column_neurons 10
    --init_method he
    --gabor_features
    --lr "$LR_VALUE"
    --column_lr_factors "$CLRF_VALUE"
    --gradient_clip "$GCLIP"
    --init_scales "$INIT_SCALES"
  )

  "${CMD[@]}" > "$LOG_FILE" 2>&1
  EXIT_CODE=$?

  END_TS="$(date +%Y-%m-%dT%H:%M:%S)"
  END_SEC="$(date +%s)"
  DURATION_SEC=$((END_SEC - START_SEC))
  SEC_PER_EPOCH=$(awk -v d="$DURATION_SEC" -v e="$EPOCHS" 'BEGIN{if(e>0) printf "%.3f", d/e; else print "0"}')

  STATUS="OK"
  if [[ $EXIT_CODE -ne 0 ]]; then
    STATUS="ERR($EXIT_CODE)"
  fi

  BEST_TEST=""
  FINAL_TEST=""

  if grep -Eq 'Best Test=[0-9.]+%' "$LOG_FILE"; then
    BEST_TEST=$(grep -Eo 'Best Test=[0-9.]+%' "$LOG_FILE" | tail -n 1 | sed -E 's/Best Test=([0-9.]+)%/\1/')
  fi
  if [[ -z "$BEST_TEST" ]] && grep -Eq 'ベスト精度: Test=[0-9.]+' "$LOG_FILE"; then
    BEST_TEST=$(grep -Eo 'ベスト精度: Test=[0-9.]+' "$LOG_FILE" | tail -n 1 | sed -E 's/.*Test=([0-9.]+)/\1/' | awk '{printf "%.4f", $1*100}')
  fi
  if grep -Eq '最終精度: Train=[0-9.]+, Test=[0-9.]+' "$LOG_FILE"; then
    FINAL_TEST=$(grep -Eo '最終精度: Train=[0-9.]+, Test=[0-9.]+' "$LOG_FILE" | tail -n 1 | sed -E 's/.*Test=([0-9.]+)/\1/' | awk '{printf "%.4f", $1*100}')
  fi

  [[ -z "$BEST_TEST" ]] && BEST_TEST="NA"
  [[ -z "$FINAL_TEST" ]] && FINAL_TEST="NA"

  printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$IDX" "$TAG" "$HIDDEN" "$INIT_SCALES" "$LR_VALUE" "$CLRF_VALUE" "$GCLIP" \
    "$EPOCHS" "$TRAIN" "$TEST" "$START_TS" "$END_TS" "$DURATION_SEC" "$SEC_PER_EPOCH" \
    "$BEST_TEST" "$FINAL_TEST" "$STATUS" "$LOG_FILE" >> "$SUMMARY_CSV"
done

echo "Done: $SUMMARY_CSV"

echo "--- Status count ---"
awk -F'\t' 'NR>1{c[$17]++} END{for(k in c) print k ":" c[k]}' "$SUMMARY_CSV" | sort

echo "--- Top by best_test_pct (OK only) ---"
awk -F'\t' 'NR==1 || ($17=="OK" && $15!="NA")' "$SUMMARY_CSV" | { head -n 1; tail -n +2 | sort -t$'\t' -k15,15nr | head -n 10; }

echo "--- Runtime estimate ---"
AVG_EPOCH=$(awk -F'\t' 'NR>1 && $14>0 {sum+=$14; n++} END{if(n>0) printf "%.3f", sum/n; else print "NA"}' "$SUMMARY_CSV")
if [[ "$AVG_EPOCH" != "NA" ]]; then
  EST_HOURS=$(awk -v a="$AVG_EPOCH" -v e="$EPOCHS" 'BEGIN{printf "%.2f", (20*a*e)/3600}')
  echo "avg_sec_per_epoch=$AVG_EPOCH"
  echo "estimated_total_hours_for_20_runs_at_${EPOCHS}ep=$EST_HOURS"
fi

# Fail loudly if all runs failed quickly (common env/path issue on another PC).
OK_COUNT=$(awk -F'\t' 'NR>1 && $17=="OK"{c++} END{print c+0}' "$SUMMARY_CSV")
if [[ "$OK_COUNT" -eq 0 ]]; then
  echo "All runs failed. Check first log:"
  FIRST_LOG=$(awk -F'\t' 'NR==2{print $18}' "$SUMMARY_CSV")
  echo "  $FIRST_LOG"
  exit 3
fi
