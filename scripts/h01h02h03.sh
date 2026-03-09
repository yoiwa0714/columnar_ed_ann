bash -lc '
set -eu
cd /home/yoichi/develop/ai/columnar_ed_ann
PYTHON_BIN="/home/yoichi/develop/ai/column_ed_snn/.venv/bin/python"
RUN_DIR="logs/grid_search_3layer_full_20260305_201950"
SUMMARY_TSV="results/grid_search_3layer_full_20260305_201950.tsv"
EPOCHS=30
TRAIN=10000
TEST=10000

run_case() {
  IDX="$1"; TAG="$2"; HIDDEN="$3"; INIT_SCALES="$4"; LR_VALUE="$5"; CLRF_VALUE="$6"; GCLIP="$7"
  LOG_FILE="$RUN_DIR/$(printf "%02d" "$IDX")_${TAG}.log"
  START_TS="$(date +%Y-%m-%dT%H:%M:%S)"
  START_SEC="$(date +%s)"

  "$PYTHON_BIN" columnar_ed_ann.py \
    --dataset mnist \
    --hidden "$HIDDEN" \
    --train "$TRAIN" --test "$TEST" \
    --epochs "$EPOCHS" --seed 42 \
    --column_neurons 10 --init_method he --gabor_features \
    --lr "$LR_VALUE" \
    --column_lr_factors "$CLRF_VALUE" \
    --gradient_clip "$GCLIP" \
    --init_scales "$INIT_SCALES" > "$LOG_FILE" 2>&1

  EXIT_CODE=$?
  END_TS="$(date +%Y-%m-%dT%H:%M:%S)"
  END_SEC="$(date +%s)"
  DURATION_SEC=$((END_SEC - START_SEC))
  SEC_PER_EPOCH=$(awk -v d="$DURATION_SEC" -v e="$EPOCHS" '"'"'BEGIN{if(e>0) printf "%.3f", d/e; else print "0"}'"'"')

  STATUS="OK"; [ "$EXIT_CODE" -ne 0 ] && STATUS="ERR($EXIT_CODE)"
  BEST_TEST=""
  FINAL_TEST=""

  if grep -Eq "Best Test=[0-9.]+%" "$LOG_FILE"; then
    BEST_TEST=$(grep -Eo "Best Test=[0-9.]+%" "$LOG_FILE" | tail -n 1 | sed -E "s/Best Test=([0-9.]+)%/\\1/")
  fi
  if [ -z "$BEST_TEST" ] && grep -Eq "ベスト精度: Test=[0-9.]+" "$LOG_FILE"; then
    BEST_TEST=$(grep -Eo "ベスト精度: Test=[0-9.]+" "$LOG_FILE" | tail -n 1 | sed -E "s/.*Test=([0-9.]+)/\\1/" | awk "{printf \"%.4f\", \$1*100}")
  fi
  if grep -Eq "最終精度: Train=[0-9.]+, Test=[0-9.]+" "$LOG_FILE"; then
    FINAL_TEST=$(grep -Eo "最終精度: Train=[0-9.]+, Test=[0-9.]+" "$LOG_FILE" | tail -n 1 | sed -E "s/.*Test=([0-9.]+)/\\1/" | awk "{printf \"%.4f\", \$1*100}")
  fi
  [ -z "$BEST_TEST" ] && BEST_TEST="NA"
  [ -z "$FINAL_TEST" ] && FINAL_TEST="NA"

  TMP="tmp/summary_rebuild_20260306.tsv"
  awk -F "\t" -v idx="$IDX" '"'"'NR==1 || $1 != idx'"'"' "$SUMMARY_TSV" > "$TMP"
  mv "$TMP" "$SUMMARY_TSV"

  printf "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$IDX" "$TAG" "$HIDDEN" "$INIT_SCALES" "$LR_VALUE" "$CLRF_VALUE" "$GCLIP" \
    "$EPOCHS" "$TRAIN" "$TEST" "$START_TS" "$END_TS" "$DURATION_SEC" "$SEC_PER_EPOCH" \
    "$BEST_TEST" "$FINAL_TEST" "$STATUS" "$LOG_FILE" >> "$SUMMARY_TSV"

  echo "$TAG done: status=$STATUS best=$BEST_TEST final=$FINAL_TEST"
}

run_case 18 H01 "2048,1536,1024" "0.7,1.8,1.8,0.8" "0.20" "0.005,0.003,0.002" "0.03"
run_case 19 H02 "2048,1024,768"  "0.7,1.8,1.8,0.8" "0.20" "0.005,0.003,0.002" "0.03"
run_case 20 H03 "2048,1536,1024" "0.7,1.8,1.6,0.8" "0.20" "0.005,0.003,0.002" "0.03"

echo "DONE"
wc -l "$SUMMARY_TSV"
tail -n 5 "$SUMMARY_TSV"
'
