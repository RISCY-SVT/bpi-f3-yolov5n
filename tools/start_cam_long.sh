#!/usr/bin/env bash
# Launch the YUYV SDL pipeline without a frame limit and report the process PID.
set -euo pipefail

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

mkdir -p artifacts
OUT=artifacts/run-cam-yuyv-sdl-rvv-long.avi
MET=artifacts/run-cam-yuyv-sdl-rvv-long.jsonl
PROBE=artifacts/display_probe_last_cam_yuyv_sdl_rvv_long.ppm
LOG=artifacts/run-cam-yuyv-sdl-rvv-long.log
PID_FILE=artifacts/run-cam-yuyv-sdl-rvv-long.pid
rm -f "$OUT" "$MET" "$PROBE" "$LOG" "$PID_FILE"

nohup ./build/yolov5n_pipeline \
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
  --max-frames 0 \
  > "$LOG" 2>&1 &
PID=$!
printf 'PID=%s\n' "$PID" | tee "$PID_FILE"
printf 'LOG=%s\n' "$LOG"
printf 'METRICS=%s\n' "$MET"
