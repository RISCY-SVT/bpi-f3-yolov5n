#!/usr/bin/env zsh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}"/env.sh

# file or cam
runmode=$1

# run with file input:
if [[ $runmode == "file" ]]; then
  echo "Running with file input"
  set -x
  ./build/yolov5n_pipeline \
    --src file:${SCRIPT_DIR}/input_video.mp4 \
    --out ${SCRIPT_DIR}/out.mp4 \
    --display off \
    --weights ${SCRIPT_DIR}/cpu_model/hhb.bm \
    --nn-cpus auto \
    --io-cpus auto \
    --perf-json ${SCRIPT_DIR}/metrics.jsonl
    set +x
    exit 0
fi

if [[ $runmode == "cam" ]]; then
  echo "Running with camera input"
  set -x
  ./build/yolov5n_pipeline \
    --src v4l2:/dev/video0 \
    --out ${SCRIPT_DIR}/out.mp4 \
    --display off \
    --weights ${SCRIPT_DIR}/cpu_model/hhb.bm \
    --nn-cpus auto \
    --io-cpus auto \
    --perf-json ${SCRIPT_DIR}/metrics.jsonl
    set +x
    exit 0
fi

if [[ $runmode == "raw-sw" ]]; then
  echo "Running with SW raw output (no encoding)"
  set -x
  ./build/yolov5n_pipeline \
    --src file:${SCRIPT_DIR}/input_video.mp4 \
    --out ${SCRIPT_DIR}/out_raw-sw.avi --enc raw --display off \
    --weights ${SCRIPT_DIR}/cpu_model/hhb.bm \
    --nn-cpus auto \
    --io-cpus auto \
    --max-frames 120 \
    --pp sw \
    --perf-json ${SCRIPT_DIR}/metrics_raw-sw.jsonl
    set +x
    exit 0
fi

if [[ $runmode == "raw-rvv" ]]; then
  echo "Running with RVV raw output (no encoding)"
  set -x
  ./build/yolov5n_pipeline \
    --src file:${SCRIPT_DIR}/input_video.mp4 \
    --out ${SCRIPT_DIR}/out_raw-rvv.avi --enc raw --display off \
    --weights ${SCRIPT_DIR}/cpu_model/hhb.bm \
    --nn-cpus auto \
    --io-cpus auto \
    --max-frames 120 \
    --pp rvv \
    --perf-json ${SCRIPT_DIR}/metrics_raw-rvv.jsonl
    set +x
    exit 0
fi

CAM=""
for link in /dev/v4l/by-id/*-video-index0; do
  [ -e "$link" ] || continue
  real_path=$(readlink -f "$link" 2>/dev/null || true)
  [ -n "$real_path" ] || continue
  CAM="$real_path"
  break
done
if [ -z "$CAM" ]; then \
  for dev in /dev/video{0..63}; do \
    [ -e "$dev" ] || continue
    CAM="$dev"
    break
  done
fi
echo "Using camera device: $CAM"
if [ -z "$CAM" ]; then
  echo "No camera device found!"
  exit 1
fi

# run-cam-yuyv-sdl-sw: camera → SDL window, SW preprocess, raw AVI for reference
if [[ $runmode == "run-cam-yuyv-sdl-sw" ]]; then
  echo "Running camera → SDL (SW preprocess)"
  # --- DISPLAY ENVs: prefer Wayland, fallback to KMSDRM when TTY (no DE) ---
  # Note: comments/messages in English only!
#    --sdl-driver "${SDL_VIDEODRIVER}" \

  set -x
  ./build/yolov5n_pipeline \
    --src "v4l2:$CAM?fmt=yuyv" \
    --display sdl \
    --display-probe "${SCRIPT_DIR}/display_probe_last_cam_yuyv_sdl_sw.ppm" \
    --watchdog-sec 10 \
    --out "${SCRIPT_DIR}/out_cam_yuyv_sdl_sw.avi" --enc raw \
    --weights "${SCRIPT_DIR}/cpu_model/hhb.bm" \
    --nn-cpus auto --io-cpus auto \
    --max-frames 0 \
    --pp sw \
    --perf-json "${SCRIPT_DIR}/metrics_cam_yuyv_sdl_sw.jsonl"
  set +x
  exit 0
fi

# run-cam-yuyv-sdl-rvv: camera → SDL window, RVV preprocess, raw AVI for reference
if [[ $runmode == "run-cam-yuyv-sdl-rvv" ]]; then
  echo "Running camera → SDL (RVV preprocess)"
  # --- DISPLAY ENVs: prefer Wayland, fallback to KMSDRM when TTY (no DE) ---
  set -x
  ./build/yolov5n_pipeline \
    --src "v4l2:$CAM?fmt=yuyv" \
    --display sdl \
    --display-probe "${SCRIPT_DIR}/display_probe_last_cam_yuyv_sdl_rvv.ppm" \
    --watchdog-sec 10 \
    --out "${SCRIPT_DIR}/out_cam_yuyv_sdl_rvv.avi" --enc raw \
    --weights "${SCRIPT_DIR}/cpu_model/hhb.bm" \
    --nn-cpus auto --io-cpus auto \
    --max-frames 0 \
    --pp rvv \
    --perf-json "${SCRIPT_DIR}/metrics_cam_yuyv_sdl_rvv.jsonl"
  set +x
  exit 0
fi
