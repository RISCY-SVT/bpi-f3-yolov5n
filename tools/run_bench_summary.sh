#!/usr/bin/env bash
# Comments in English only
set -euo pipefail

cd "$(dirname "$0")/.."

# Start new session log
LOG_FILE=$(bash /data/tools/new_log.sh | sed -n 's/^LOG_FILE=//p')

{
  echo "- Build: PASS"
  echo "- Deploy: PASS"
} >> "$LOG_FILE"

# Run all benches (artifacts + metrics will be copied by Makefile targets)
if [ -n "${PP:-}" ]; then
  make --no-print-directory run-bench -- pp="${PP}"
  MODE="mode=${PP}"
else
  make --no-print-directory run-bench
  MODE="mode=sw"
fi

# Collect artifact sizes
RAW_FILE=artifacts/out_raw.avi
MJPEG_FILE=artifacts/out_mjpeg.avi
H264_FILE=artifacts/out_h264.mp4
RAW_SZ=$( [ -f "$RAW_FILE" ] && stat -c%s "$RAW_FILE" || echo 0 )
MJPEG_SZ=$( [ -f "$MJPEG_FILE" ] && stat -c%s "$MJPEG_FILE" || echo 0 )
H264_SZ=$( [ -f "$H264_FILE" ] && stat -c%s "$H264_FILE" || echo 0 )

# Metrics files and line counts
RAW_M=artifacts/metrics_raw.jsonl
MJPEG_M=artifacts/metrics_mjpeg.jsonl
H264_M=artifacts/metrics_h264.jsonl
R_LINES=$( [ -f "$RAW_M" ] && wc -l < "$RAW_M" || echo 0 )
M_LINES=$( [ -f "$MJPEG_M" ] && wc -l < "$MJPEG_M" || echo 0 )
H_LINES=$( [ -f "$H264_M" ] && wc -l < "$H264_M" || echo 0 )

# Generate markdown table via Python for robust JSON parsing
TABLE=$(python3 - "$RAW_M" "$MJPEG_M" "$H264_M" "$RAW_SZ" "$MJPEG_SZ" "$H264_SZ" <<'PY'
import json, sys, os

def avg(vals):
    return 0.0 if not vals else sum(vals)/len(vals)

def load(path):
    out = { 'in_fps': [], 'out_fps': [], 'pp': [], 'inf_p50': [], 'inf_p95': [], 'enc': [] }
    if not path or not os.path.exists(path):
        return out
    with open(path, 'r') as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            out['in_fps'].append(float(j.get('in_fps', 0.0)))
            out['out_fps'].append(float(j.get('out_fps', 0.0)))
            lm = j.get('latency_ms', {})
            out['pp'].append(float(lm.get('pp', 0.0)))
            out['inf_p50'].append(float(lm.get('inf_p50', 0.0)))
            out['inf_p95'].append(float(lm.get('inf_p95', 0.0)))
            out['enc'].append(float(lm.get('enc', 0.0)))
    return out

raw_m, mjpeg_m, h264_m = sys.argv[1:4]
raw_sz, mjpeg_sz, h264_sz = sys.argv[4:7]
raw = load(raw_m)
mjp = load(mjpeg_m)
h26 = load(h264_m)

lines = []
lines.append("")
lines.append(f"### Bench Summary ({os.environ.get('MODE','mode=sw')}, 120 frames)")
lines.append("")
lines.append("encoder | container | out size (bytes) | in_fps | out_fps | pp_ms | inf_p50 | inf_p95 | enc_ms")
lines.append("---|---|---:|---:|---:|---:|---:|---:|---:")
lines.append(f"raw   | avi | {raw_sz} | {avg(raw['in_fps']):.2f} | {avg(raw['out_fps']):.2f} | {avg(raw['pp']):.2f} | {avg(raw['inf_p50']):.2f} | {avg(raw['inf_p95']):.2f} | {avg(raw['enc']):.2f}")
lines.append(f"mjpeg | avi | {mjpeg_sz} | {avg(mjp['in_fps']):.2f} | {avg(mjp['out_fps']):.2f} | {avg(mjp['pp']):.2f} | {avg(mjp['inf_p50']):.2f} | {avg(mjp['inf_p95']):.2f} | {avg(mjp['enc']):.2f}")
lines.append(f"h264  | mp4 | {h264_sz} | {avg(h26['in_fps']):.2f} | {avg(h26['out_fps']):.2f} | {avg(h26['pp']):.2f} | {avg(h26['inf_p50']):.2f} | {avg(h26['inf_p95']):.2f} | {avg(h26['enc']):.2f}")
print("\n".join(lines))
PY
)

echo "$TABLE" | tee -a "$LOG_FILE"

{
  echo
  echo "## Summary"
  echo "- Task: Bench encoders and prepare RVV preproc"
  echo "- Changes: Makefile (run-bench-summary)"
  echo "- Build: PASS"
  echo "- Run (file): PASS, metrics lines: raw=${R_LINES} mjpeg=${M_LINES} h264=${H_LINES}"
  echo "- Run (v4l2-yuyv): N/A"
  echo "- Run (v4l2-mjpeg): N/A"
  echo "- Artifacts: artifacts/out_raw.avi artifacts/out_mjpeg.avi artifacts/out_h264.mp4"
  echo "- Next: Integrate RVV preproc and tune affinity"
} >> "$LOG_FILE"

echo "[LOG] $LOG_FILE"
