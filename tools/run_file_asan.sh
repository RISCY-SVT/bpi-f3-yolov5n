#!/usr/bin/env bash
# Run file playback scenario under ASan/LSan to surface leaks.
set -euo pipefail

ASAN_OPTIONS_DEFAULT=${ASAN_OPTIONS_DEFAULT:-detect_leaks=1,abort_on_error=0,alloc_dealloc_mismatch=1,handle_segv=1,log_path=/data/bpi-f3-yolov5n/asan}
ASAN_LOG_DIR=${ASAN_LOG_DIR:-/data/bpi-f3-yolov5n/asan}
PROJECT_DIR=$(pwd)

mkdir -p artifacts/asan_logs
mkdir -p "$ASAN_LOG_DIR"
rm -f "$ASAN_LOG_DIR".* artifacts/asan_logs/run-file-asan.log

OUT=artifacts/run-file-asan.mp4
MET=artifacts/run-file-asan.jsonl
LOG=artifacts/asan_logs/run-file-asan.log
rm -f "$OUT" "$MET"

timeout 120s env ASAN_OPTIONS="$ASAN_OPTIONS_DEFAULT" ./build/yolov5n_pipeline \
  --src "file:${PROJECT_DIR}/input_video.mp4" \
  --out "${PROJECT_DIR}/$OUT" \
  --enc h264 \
  --display off \
  --weights cpu_model/hhb.bm \
  --nn-cpus auto --io-cpus auto \
  --perf-json "${PROJECT_DIR}/$MET" \
  > "$LOG" 2>&1 || true

echo "LOG=$LOG"
echo "METRICS=$MET"
echo "ASAN_DIR=$ASAN_LOG_DIR"
