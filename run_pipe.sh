#!/usr/bin/env zsh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}"/env.sh


function usage() {
  echo "Usage: $0 {file|cam|raw-sw|raw-rvv|run-cam-yuyv-sdl-sw|run-cam-yuyv-sdl-rvv}"
  echo "  file: run with input video file"
  echo "  cam: run with camera input"
  echo "  raw-sw: run with SW raw output (no encoding)"
  echo "  raw-rvv: run with RVV raw output (no encoding)"
  echo "  run-cam-yuyv-sdl-sw: camera → SDL window, SW preprocess, raw AVI for reference"
  echo "  run-cam-yuyv-sdl-rvv: camera → SDL window, RVV preprocess, raw AVI for reference"
  echo "  run-cam-live-rvv: camera live → SDL window, RVV preprocess, raw AVI for reference"
  exit 1
}
if [ $# -ne 1 ]; then
  usage
fi
function get_cam() {
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
echo "$CAM"
}

# file or cam
runmode=$1

case $runmode in

# run with file input:
"file")
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
;;

"cam")
  CAM=$(get_cam)
  if [ -z "$CAM" ]; then
    echo "No camera device found!"
    exit 1
  fi
  echo "Running with camera input: $CAM"
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
;;

"raw-sw")
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
;;

"raw-rvv")
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
;;


# run-cam-yuyv-sdl-sw: camera → SDL window, SW preprocess, raw AVI for reference
"run-cam-yuyv-sdl-sw")
  echo "Running camera → SDL (SW preprocess)"
  CAM=$(get_cam)
  if [ -z "$CAM" ]; then
    echo "No camera device found!"
    exit 1
  fi
  echo "Using camera device: $CAM"
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
;;

# run-cam-yuyv-sdl-rvv: camera → SDL window, RVV preprocess, raw AVI for reference
"run-cam-yuyv-sdl-rvv")
  echo "Running camera → SDL (RVV preprocess)"
  # --- DISPLAY ENVs: prefer Wayland, fallback to KMSDRM when TTY (no DE) ---
  CAM=$(get_cam)
  if [ -z "$CAM" ]; then
    echo "No camera device found!"
    exit 1
  fi
  echo "Using camera device: $CAM"
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
;;
"run-cam-live-rvv")
  echo "Running camera live → SDL (RVV preprocess)"
  CAM=$(get_cam)
  if [ -z "$CAM" ]; then
    echo "No camera device found!"
    exit 1
  fi
  echo "Detected camera device: $CAM"
  mkdir -p artifacts
  OUT="${SCRIPT_DIR}/artifacts/out_cam_live_rvv.avi"
  MET="${SCRIPT_DIR}/artifacts/metrics_cam_live_rvv.jsonl"
  LOG="${SCRIPT_DIR}/artifacts/run-cam-live-rvv.log"
  PROBE="${SCRIPT_DIR}/artifacts/probe_cam_live_rvv.ppm"
  rm -f "$OUT" "$MET" "$LOG" "$PROBE"
  set -x
  ./build/yolov5n_pipeline \
    --src 'v4l2:auto?fmt=yuyv' \
    --display sdl \
    --sdl-driver wayland \
    --display-vsync off \
    --display-allow-null on \
    --display-probe "$PROBE" \
    --live --latency-mode live \
    --pp sw \
    --weights cpu_model/hhb.bm \
    --nn-cpus auto --io-cpus auto \
    --perf-json "$MET" \
    --watchdog-sec 0 \
    --enc raw --out /dev/null
  set +x
;;
"run-cam-live-sw")
  echo "Running camera live → SDL (SW preprocess)"
  CAM=$(get_cam)
  if [ -z "$CAM" ]; then
    echo "No camera device found!"
    exit 1
  fi
  echo "Detected camera device: $CAM"
  mkdir -p artifacts
  OUT="${SCRIPT_DIR}/artifacts/out_cam_live_sw.avi"
  MET="${SCRIPT_DIR}/artifacts/metrics_cam_live_sw.jsonl"
  LOG="${SCRIPT_DIR}/artifacts/run-cam-live-sw.log"
  PROBE="${SCRIPT_DIR}/artifacts/probe_cam_live_sw.ppm"
  rm -f "$OUT" "$MET" "$LOG" "$PROBE"
  set -x
  ./build/yolov5n_pipeline \
    --src 'v4l2:auto?fmt=yuyv' \
    --display sdl \
    --sdl-driver wayland \
    --display-vsync off \
    --display-allow-null on \
    --display-probe "$PROBE" \
    --live --latency-mode live \
    --pp sw \
    --weights cpu_model/hhb.bm \
    --nn-cpus auto --io-cpus auto \
    --perf-json "$MET" \
    --watchdog-sec 0 \
    --enc raw --out /dev/null
  set +x
;;
*)
  usage
  ;;
esac 
exit 0
