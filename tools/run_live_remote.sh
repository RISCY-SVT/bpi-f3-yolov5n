#!/usr/bin/env bash
set -Eeuo pipefail

if [ $# -ne 3 ]; then
  echo "Usage: $0 <sw|rvv> <ssh_target> <remote_project_dir>" >&2
  exit 1
fi

MODE="$1"
SSH_TARGET="$2"
PROJECT_ROOT="$3"

ssh "$SSH_TARGET" "cd $PROJECT_ROOT && LIVE_DURATION=${LIVE_DURATION:-} LIVE_DEBUG=${LIVE_DEBUG:-} LIVE_RUN_LABEL=${LIVE_RUN_LABEL:-} LIVE_TTL_MS=${LIVE_TTL_MS:-} LIVE_NN_WORKERS=${LIVE_NN_WORKERS:-} LIVE_ENCODER=${LIVE_ENCODER:-} bash tools/run_live_remote_inner.sh $MODE"
