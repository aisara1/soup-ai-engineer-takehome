#!/usr/bin/env bash
set -uo pipefail

RUN_DIR="logs/train_final_v3"
mkdir -p "$RUN_DIR"

echo "=== FINAL RUN START ===" | tee "$RUN_DIR/run_info.txt"
date -u | tee -a "$RUN_DIR/run_info.txt"

echo -e "\n=== INPUT HASHES ===" | tee -a "$RUN_DIR/run_info.txt"
sha256sum \
  soup.yaml \
  data/train.jsonl \
  data/eval.jsonl \
  prepare_data.py \
  verify_training.py \
  /content/models/qwen2.5-1.5b-pinned/model.safetensors \
  | tee -a "$RUN_DIR/run_info.txt"

echo -e "\n=== GPU BEFORE ==="
nvidia-smi | tee "$RUN_DIR/nvidia_smi_before.txt"

nvidia-smi \
  --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
  --format=csv \
  -l 1 > "$RUN_DIR/nvidia_smi_sampling.csv" 2>&1 &
MON_PID=$!

cleanup() {
    kill "$MON_PID" 2>/dev/null || true
    wait "$MON_PID" 2>/dev/null || true
}
trap cleanup EXIT

set +e
/content/venv312/bin/soup --log-level verbose train -c soup.yaml -y \
  2>&1 | tee "$RUN_DIR/training.log"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

cleanup
trap - EXIT

echo -e "\n=== GPU AFTER ==="
nvidia-smi | tee "$RUN_DIR/nvidia_smi_after.txt"

echo -e "\n=== FINAL RUN END ===" | tee -a "$RUN_DIR/run_info.txt"
date -u | tee -a "$RUN_DIR/run_info.txt"
echo "training exit code: $TRAIN_STATUS" | tee -a "$RUN_DIR/run_info.txt"

exit "$TRAIN_STATUS"
