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
fi
