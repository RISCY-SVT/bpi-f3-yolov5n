#!/usr/bin/env bash
# Stop long-running pipeline instances launched via start_cam_long.sh.
set -euo pipefail

PID_FILE=artifacts/run-cam-yuyv-sdl-rvv-long.pid
if [[ -f "$PID_FILE" ]]; then
  PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
  if [[ -n "$PID" ]]; then
    kill "$PID" >/dev/null 2>&1 || true
    sleep 1
    kill -KILL "$PID" >/dev/null 2>&1 || true
  fi
  rm -f "$PID_FILE"
fi
pkill -TERM -f yolov5n_pipeline >/dev/null 2>&1 || true
pkill -KILL -f yolov5n_pipeline >/dev/null 2>&1 || true
