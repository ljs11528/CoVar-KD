#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
RUN_LOG_DIR=${RUN_LOG_DIR:-"$ROOT_DIR/runs/covar_npu_phaseC"}
MAX_ITERATIONS=${MAX_ITERATIONS:-20000}
PID_FILE="$RUN_LOG_DIR/phaseC_triage_${MAX_ITERATIONS}.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No pid file found: $PID_FILE"
  exit 0
fi

pid=$(cat "$PID_FILE")
if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
  echo "No running process for pid=$pid"
  exit 0
fi

echo "Stopping Phase C triage process group: pid=$pid"
kill "-$pid" 2>/dev/null || kill "$pid"
