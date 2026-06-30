#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
cd "$ROOT_DIR"

MAX_ITERATIONS=${MAX_ITERATIONS:-80000}
RUN_LOG_DIR=${RUN_LOG_DIR:-"$ROOT_DIR/runs/covar_npu_phaseC_80k"}
RUN_LOG="$RUN_LOG_DIR/phaseC_80k_pair_${MAX_ITERATIONS}.log"
PID_FILE="$RUN_LOG_DIR/phaseC_80k_pair_${MAX_ITERATIONS}.pid"
SAVE_ROOT=${SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseC_80k"}

echo "== PID =="
if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "running pid=$pid"
  else
    echo "pid file exists but process is not running: $pid"
  fi
else
  echo "no pid file: $PID_FILE"
fi

echo "== NPU =="
npu-smi info || true

echo "== Queue log tail =="
if [[ -f "$RUN_LOG" ]]; then
  tail -80 "$RUN_LOG"
else
  echo "no run log: $RUN_LOG"
fi

echo "== Latest validation summaries =="
find "$SAVE_ROOT" -maxdepth 2 -name '*_log.txt' -print 2>/dev/null | sort | while read -r log_file; do
  echo "-- $log_file"
  grep -E 'Iters:|Overall validation' "$log_file" | tail -5 || true
done
