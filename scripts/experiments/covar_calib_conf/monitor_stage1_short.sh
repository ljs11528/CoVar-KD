#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
cd "$ROOT_DIR"

MAX_ITERATIONS=${MAX_ITERATIONS:-10000}
SAVE_ROOT=${SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_calib_conf"}
RUN_LOG_DIR=${RUN_LOG_DIR:-"$ROOT_DIR/runs/covar_calib_conf_stage1"}
RUN_LOG="$RUN_LOG_DIR/stage1_short_${MAX_ITERATIONS}.log"
PID_FILE="$RUN_LOG_DIR/stage1_short_${MAX_ITERATIONS}.pid"
SUMMARY="$RUN_LOG_DIR/stage1_summary_${MAX_ITERATIONS}.csv"
PYTHON=${PYTHON:-"$ROOT_DIR/.venv/bin/python"}

if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Stage 1 status: RUNNING pid=$pid"
  else
    echo "Stage 1 status: NOT RUNNING pid=$pid"
  fi
else
  echo "Stage 1 status: no pid file for MAX_ITERATIONS=$MAX_ITERATIONS"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
fi

if [[ -f "$RUN_LOG" ]]; then
  echo
  echo "--- queue log tail: $RUN_LOG ---"
  tail -n 40 "$RUN_LOG"
fi

if [[ -d "$SAVE_ROOT" ]]; then
  "$PYTHON" scripts/experiments/covar_calib_conf/summarize_runs.py --root "$SAVE_ROOT" --output "$SUMMARY" || true
  if [[ -f "$SUMMARY" ]]; then
    echo
    echo "--- summary: $SUMMARY ---"
    cat "$SUMMARY"
  fi
fi
