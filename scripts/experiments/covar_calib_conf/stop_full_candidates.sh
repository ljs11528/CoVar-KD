#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
cd "$ROOT_DIR"

MAX_ITERATIONS=${MAX_ITERATIONS:-80000}
RUN_LOG_DIR=${RUN_LOG_DIR:-"$ROOT_DIR/runs/covar_calib_conf_full"}
PID_FILE="$RUN_LOG_DIR/full_candidates_${MAX_ITERATIONS}.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No pid file: $PID_FILE"
  exit 0
fi

pid=$(cat "$PID_FILE")
if [[ -z "$pid" ]]; then
  echo "Empty pid file: $PID_FILE"
  exit 0
fi

if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "Sent SIGTERM to full candidate pid=$pid"
else
  echo "Process not running: pid=$pid"
fi
