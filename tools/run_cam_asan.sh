#!/usr/bin/env bash
# Run the camera SDL pipeline under ASan/LSan for a short leak-detection window.
set -euo pipefail

ASAN_OPTIONS_DEFAULT=${ASAN_OPTIONS_DEFAULT:-detect_leaks=1,abort_on_error=0,alloc_dealloc_mismatch=1,handle_segv=1,log_path=/data/bpi-f3-yolov5n/asan}
ASAN_LOG_DIR=${ASAN_LOG_DIR:-/data/bpi-f3-yolov5n/asan}

find_camera() {
  local cam=""
  for link in /dev/v4l/by-id/*-video-index0; do
    [[ -e "$link" ]] || continue
    local real
    real=$(readlink -f "$link" 2>/dev/null || true)
    [[ -n "$real" ]] || continue
    cam="$real"
    break
  done
  if [[ -z "$cam" ]]; then
    for dev in /dev/video{0..63}; do
      [[ -e "$dev" ]] || continue
      cam="$dev"
      break
    done
  fi
  echo "$cam"
}

if ! command -v v4l2-ctl >/dev/null 2>&1; then
  echo NO_CAMERA
  exit 0
fi

CAMERA=$(find_camera)
if [[ -z "$CAMERA" ]]; then
  echo NO_CAMERA
  exit 0
fi

mkdir -p artifacts/asan_logs
mkdir -p "$ASAN_LOG_DIR"
rm -f "$ASAN_LOG_DIR".* artifacts/asan_logs/run-cam-yuyv-asan.log

OUT=artifacts/run-cam-yuyv-asan.avi
MET=artifacts/run-cam-yuyv-asan.jsonl
PROBE=artifacts/display_probe_last_cam_yuyv_asan.ppm
LOG=artifacts/asan_logs/run-cam-yuyv-asan.log
rm -f "$OUT" "$MET" "$PROBE"

timeout 120s env ASAN_OPTIONS="$ASAN_OPTIONS_DEFAULT" ./build/yolov5n_pipeline \
  --src "v4l2:${CAMERA}?fmt=yuyv" \
  --out "$OUT" \
  --enc raw \
  --display sdl \
  --sdl-driver auto \
  --display-probe "$PROBE" \
  --weights cpu_model/hhb.bm \
  --nn-cpus auto --io-cpus auto \
  --pp rvv \
  --perf-json "$MET" \
  > "$LOG" 2>&1 || true

echo "LOG=$LOG"
echo "METRICS=$MET"
echo "ASAN_DIR=$ASAN_LOG_DIR"
