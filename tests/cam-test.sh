#!/usr/bin/env zsh

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

CAM=$(get_cam)
if [ -z "$CAM" ]; then
  echo "No camera device found!"
  exit 1
fi
echo "Detected camera device: $CAM"

v4l2-ctl -d "$CAM" --stream-mmap=3 --stream-count=300 --stream-to=/dev/null -v width=1280,height=720,pixelformat=YUYV
