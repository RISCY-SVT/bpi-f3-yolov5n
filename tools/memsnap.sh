#!/usr/bin/env bash
# Record RSS (kB) of yolov5n_pipeline and system meminfo every interval.
# Usage: ./tools/memsnap.sh <pid> <interval_sec> <duration_sec> <outfile>
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <pid> [interval_sec] [duration_sec] [outfile]" >&2
  exit 1
fi

PID="$1"
INTERVAL="${2:-1}"
DURATION="${3:-600}"
OUT="${4:-/tmp/memsnap.csv}"

if ! [[ "$PID" =~ ^[0-9]+$ ]]; then
  echo "PID must be numeric" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
printf "ts_sec,rss_kb,vm_kb,pgfault,pgmajfault\n" > "$OUT"
END=$(( $(date +%s) + DURATION ))
while [ "$(date +%s)" -lt "$END" ]; do
  if [ ! -d "/proc/$PID" ]; then
    printf "# process %s exited\n" "$PID" >> "$OUT"
    break
  fi
  RSS=$(awk '/VmRSS:/ {print $2}' /proc/"$PID"/status 2>/dev/null || echo 0)
  VM=$(awk '/VmSize:/ {print $2}' /proc/"$PID"/status 2>/dev/null || echo 0)
  MIN_FAULT=$(awk '{print $10}' /proc/"$PID"/stat 2>/dev/null || echo 0)
  MAJ_FAULT=$(awk '{print $12}' /proc/"$PID"/stat 2>/dev/null || echo 0)
  TS=$(date +%s)
  printf "%s,%s,%s,%s,%s\n" "$TS" "${RSS:-0}" "${VM:-0}" "${MIN_FAULT:-0}" "${MAJ_FAULT:-0}" >> "$OUT"
  sleep "$INTERVAL"
done
