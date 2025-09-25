#!/usr/bin/env bash
set -Eeuo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <sw|rvv>" >&2
  exit 1
fi

MODE="$1"
if [[ "$MODE" != "sw" && "$MODE" != "rvv" ]]; then
  echo "[ERROR] mode must be sw or rvv" >&2
  exit 1
fi

PROJECT_ROOT=$(pwd)
ART_DIR="$PROJECT_ROOT/artifacts"
RUN_LABEL=${LIVE_RUN_LABEL:-cam_live_${MODE}}
RUN_TAG="$RUN_LABEL"
LOG_TAG=$(echo "$RUN_LABEL" | tr '_' '-')
MET="$PROJECT_ROOT/metrics_${RUN_TAG}.jsonl"
PROBE="$PROJECT_ROOT/probe_${RUN_TAG}.ppm"
LOG="$ART_DIR/run-${LOG_TAG}.log"
RING_JSON="$ART_DIR/ring_${RUN_TAG}.jsonl"
RING_LOG="$ART_DIR/ringdump_${RUN_TAG}.log"
RUN_SECONDS=${LIVE_DURATION:-45}
DEBUG_FLAG=${LIVE_DEBUG:-0}
ENCODER=${LIVE_ENCODER:-raw}

if [[ "$ENCODER" == "null" ]]; then
  OUT=""
else
  OUT="$PROJECT_ROOT/out_${RUN_TAG}.avi"
fi

export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-wayland}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/1000}
export WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}

mkdir -p "$ART_DIR"
rm -f "$MET" "$PROBE" "$LOG" "$RING_JSON" "$RING_LOG"
if [[ -n "$OUT" ]]; then
  rm -f "$OUT"
fi

echo "[INFO] artifacts dir: $ART_DIR" | tee "$LOG"

echo "[INFO] run_label=$RUN_LABEL" | tee -a "$LOG"
echo "[INFO] run_seconds=$RUN_SECONDS debug_flag=$DEBUG_FLAG" | tee -a "$LOG"
if [[ -n "$OUT" ]]; then
  echo "[INFO] encoder=$ENCODER out=$OUT" | tee -a "$LOG"
else
  echo "[INFO] encoder=$ENCODER out=disabled" | tee -a "$LOG"
fi

# Discover camera device (prefer by-id)
cam=""
for link in /dev/v4l/by-id/*-video-index0; do
  [ -e "$link" ] || continue
  real_path=$(readlink -f "$link" 2>/dev/null || true)
  [ -n "$real_path" ] || continue
  cam="$real_path"
  break
done
if [ -z "$cam" ]; then
  for dev in /dev/video{0..63}; do
    [ -e "$dev" ] || continue
    cam="$dev"
    break
  done
fi
if [ -z "$cam" ]; then
  echo "[ERROR] no camera device found" | tee -a "$LOG" >&2
  exit 2
fi

echo "[INFO] camera=$cam" | tee -a "$LOG"

cmd=(
  ./build/yolov5n_pipeline
  --src "v4l2:auto?fmt=yuyv"
  --display sdl
  --display-vsync off
  --display-allow-null on
  --display-probe "$PROBE"
  --live
  --latency-mode live
  --live-ttl-ms "${LIVE_TTL_MS:-1000}"
  --pp "$MODE"
  --enc "$ENCODER"
  --nn-workers "${LIVE_NN_WORKERS:-4}"
  --weights cpu_model/hhb.bm
  --nn-cpus auto
  --io-cpus auto
  --watchdog-sec 0
  --perf-json "$MET"
)

if [[ -n "$OUT" ]]; then
  cmd+=(--out "$OUT")
fi

echo "[INFO] launching pipeline" | tee -a "$LOG"
set +e
("${cmd[@]}") >>"$LOG" 2>&1 &
PIPE_PID=$!
set -e

echo "[INFO] pipeline pid=$PIPE_PID" | tee -a "$LOG"
trap 'kill -TERM $PIPE_PID 2>/dev/null || true' INT TERM

elapsed=0
while kill -0 "$PIPE_PID" 2>/dev/null && [ "$elapsed" -lt "$RUN_SECONDS" ]; do
  sleep 1
  elapsed=$((elapsed + 1))
done

echo "[INFO] elapsed=${elapsed}s" | tee -a "$LOG"

if kill -0 "$PIPE_PID" 2>/dev/null; then
  if [ "$DEBUG_FLAG" = "1" ]; then
    echo "[INFO] sending SIGUSR1 to pid=$PIPE_PID" | tee -a "$LOG"
    kill -USR1 "$PIPE_PID" 2>/dev/null || true
    sleep 5
  fi
  echo "[INFO] stopping pipeline pid=$PIPE_PID" | tee -a "$LOG"
  kill -INT "$PIPE_PID" 2>/dev/null || true
  sleep 3
  if kill -0 "$PIPE_PID" 2>/dev/null; then
    kill -TERM "$PIPE_PID" 2>/dev/null || true
  fi
fi

set +e
wait "$PIPE_PID"
status=$?
set -e
trap - INT TERM

echo "[INFO] pipeline exit=$status" | tee -a "$LOG"
if [ "$status" -ne 0 ] && [ "$status" -ne 130 ] && [ "$status" -ne 143 ]; then
  echo "[ERROR] pipeline exit code=$status" | tee -a "$LOG" >&2
  exit "$status"
fi

required_artifacts=("$MET" "$PROBE")
if [[ -n "$OUT" ]]; then
  required_artifacts+=("$OUT")
fi
for path in "${required_artifacts[@]}"; do
  if [ ! -f "$path" ]; then
    echo "[ERROR] missing artifact $path" | tee -a "$LOG" >&2
    exit 3
  fi
done

lines=$(wc -l < "$MET")
if [ "$lines" -lt 50 ]; then
  echo "[ERROR] metrics lines=$lines (<50)" | tee -a "$LOG" >&2
  exit 4
fi

probe_size=$(stat -c %s "$PROBE")
if [ "$probe_size" -le 2048 ]; then
  echo "[ERROR] probe size $probe_size (<=2048)" | tee -a "$LOG" >&2
  exit 4
fi

if [[ -n "$OUT" ]]; then
  avi_size=$(stat -c %s "$OUT")
  if [ "$avi_size" -le 1000000 ]; then
    echo "[ERROR] avi size $avi_size (<=1000000)" | tee -a "$LOG" >&2
    exit 4
  fi

  format=$(ffprobe -v error -show_entries format=format_name -of default=nk=1:nw=1 "$OUT")
  if [ "$format" != "avi" ]; then
    echo "[ERROR] unexpected container $format" | tee -a "$LOG" >&2
    exit 4
  fi
fi

python3 - "$MET" "$LOG" <<'PY'
import json
import statistics
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
log_path = Path(sys.argv[2])
content = metrics_path.read_text().strip()
if not content:
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("[ERROR] metrics file empty\n")
    print("[ERROR] metrics file empty", file=sys.stderr)
    sys.exit(5)

records = []
for raw in content.splitlines():
    try:
        records.append(json.loads(raw))
    except json.JSONDecodeError as exc:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"[ERROR] invalid JSON: {exc}\n")
        print(f"[ERROR] invalid JSON: {exc}", file=sys.stderr)
        sys.exit(5)

last = records[-10:]
positive_tail = sum(1 for rec in last if float(rec.get("out_fps", 0) or 0) > 0.0)
if positive_tail == 0:
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("[ERROR] out_fps <= 0 detected in last metrics window\n")
    print("[ERROR] out_fps <= 0 detected in last metrics window", file=sys.stderr)
    sys.exit(5)

trim_start = 5 if len(records) > 10 else 0
core = records[trim_start:]
if not core:
    core = records

out_values = [float(rec.get("out_fps", 0) or 0) for rec in core]
if not out_values:
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("[ERROR] missing out_fps samples\n")
    print("[ERROR] missing out_fps samples", file=sys.stderr)
    sys.exit(6)

e2e_samples = []
for rec in core:
    val = rec.get("e2e_ms", {}).get("p95") if isinstance(rec.get("e2e_ms"), dict) else None
    if val is None:
        continue
    try:
        e2e_samples.append(float(val))
    except (TypeError, ValueError):
        continue

if not e2e_samples:
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("[ERROR] missing e2e_p95 samples\n")
    print("[ERROR] missing e2e_p95 samples", file=sys.stderr)
    sys.exit(6)

out_p50 = statistics.median_high(sorted(out_values))
e2e_p95_p50 = statistics.median_high(sorted(e2e_samples))

if out_p50 < 4.0:
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[ERROR] out_fps_p50={out_p50:.2f} (<4.0)\n")
    print(f"[ERROR] out_fps_p50={out_p50:.2f} (<4.0)", file=sys.stderr)
    sys.exit(6)

warn_e2e = e2e_p95_p50 > 500.0

with log_path.open("a", encoding="utf-8") as fh:
    fh.write(f"[INFO] out_fps_p50={out_p50:.2f}\n")
    if warn_e2e:
        fh.write(f"[WARN] e2e_p95_p50={e2e_p95_p50:.2f} ms (>500)\n")
    else:
        fh.write(f"[INFO] e2e_p95_p50={e2e_p95_p50:.2f} ms\n")

if warn_e2e:
    print(f"[WARN] e2e_p95_p50={e2e_p95_p50:.2f} ms (>500)")
PY

latest_ring_json=$(ls -1t "$PROJECT_ROOT"/artifacts/ring_* 2>/dev/null | head -n1 || true)
if [ -n "$latest_ring_json" ]; then
  cp "$latest_ring_json" "$RING_JSON"
fi
latest_ring_log=$(ls -1t /data/Work_Logs/ringdump_*_stop.log 2>/dev/null | head -n1 || true)
if [ -n "$latest_ring_log" ]; then
  cp "$latest_ring_log" "$RING_LOG"
fi

if [[ -n "$OUT" ]]; then
  cp "$OUT" "$ART_DIR/out_${RUN_TAG}.avi"
fi
cp "$MET" "$ART_DIR/metrics_${RUN_TAG}.jsonl"
cp "$PROBE" "$ART_DIR/probe_${RUN_TAG}.ppm"

echo "[PASS] metrics_lines=$lines" | tee -a "$LOG"
echo "[PASS] probe_size=$probe_size" | tee -a "$LOG"
if [[ -n "$OUT" ]]; then
  echo "[PASS] avi_size=$avi_size" | tee -a "$LOG"
  echo "[PASS] format=$format" | tee -a "$LOG"
fi
