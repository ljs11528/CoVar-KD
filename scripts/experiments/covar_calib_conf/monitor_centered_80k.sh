#!/usr/bin/env bash
# Monitor the centered calibration 80k queue.
# Shows GPU status, latest log tail, and checkpoint summary.
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
cd "$ROOT_DIR"

MAX_ITERATIONS=${MAX_ITERATIONS:-80000}
SAVE_ROOT=${SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_calib_conf_centered_80k"}
RUN_LOG_DIR=${RUN_LOG_DIR:-"$ROOT_DIR/runs/covar_calib_conf_centered_80k"}
RUN_LOG="$RUN_LOG_DIR/centered_80k_${MAX_ITERATIONS}.log"
PID_FILE="$RUN_LOG_DIR/centered_80k_${MAX_ITERATIONS}.pid"
SUMMARY="$RUN_LOG_DIR/centered_80k_summary_${MAX_ITERATIONS}.csv"
PYTHON=${PYTHON:-"$ROOT_DIR/.venv/bin/python"}

echo "=== Centered 80k Queue Status ==="
if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Status: RUNNING (pid=$pid)"
  else
    echo "Status: FINISHED or DEAD (pid=$pid)"
  fi
else
  echo "Status: not started (no pid file)"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo ""
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null || true
fi

if [[ -f "$RUN_LOG" ]]; then
  echo ""
  echo "=== Recent Log (last 40 lines) ==="
  tail -n 40 "$RUN_LOG"
fi

if [[ -d "$SAVE_ROOT" ]]; then
  echo ""
  echo "=== Checkpoints ==="
  find "$SAVE_ROOT" -name "*miou-*.pth" -printf "%T+ %f\n" 2>/dev/null | sort -r | head -20 || true
fi
