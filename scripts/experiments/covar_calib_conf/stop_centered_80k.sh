#!/usr/bin/env bash
# Stop the centered calibration 80k queue.
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
cd "$ROOT_DIR"

MAX_ITERATIONS=${MAX_ITERATIONS:-80000}
RUN_LOG_DIR=${RUN_LOG_DIR:-"$ROOT_DIR/runs/covar_calib_conf_centered_80k"}
PID_FILE="$RUN_LOG_DIR/centered_80k_${MAX_ITERATIONS}.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No pid file: $PID_FILE"
  exit 0
fi

pid=$(cat "$PID_FILE")
if [[ -z "$pid" ]]; then
  echo "Empty pid file: $PID_FILE"
  rm -f "$PID_FILE"
  exit 0
fi

if kill -0 "$pid" 2>/dev/null; then
  # Kill the whole process group (the runner + any active training)
  kill -TERM -- -"$(ps -o pgid= -p "$pid" | tr -d ' ')" 2>/dev/null || kill "$pid"
  echo "Sent SIGTERM to centered 80k queue (pid=$pid, pgid=$(ps -o pgid= -p "$pid" | tr -d ' '))"
else
  echo "Process not running: pid=$pid"
  rm -f "$PID_FILE"
fi
