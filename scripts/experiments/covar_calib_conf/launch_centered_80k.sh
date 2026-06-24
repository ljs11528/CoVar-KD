#!/usr/bin/env bash
# Launch the centered calibration 80k queue in background.
# Usage: bash scripts/experiments/covar_calib_conf/launch_centered_80k.sh
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
cd "$ROOT_DIR"

MAX_ITERATIONS=${MAX_ITERATIONS:-80000}
GPU_ID=${GPU_ID:-0}
SAVE_ROOT=${SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_calib_conf_centered_80k"}
RUN_LOG_DIR=${RUN_LOG_DIR:-"$ROOT_DIR/runs/covar_calib_conf_centered_80k"}
RUN_LOG="$RUN_LOG_DIR/centered_80k_${MAX_ITERATIONS}.log"
PID_FILE="$RUN_LOG_DIR/centered_80k_${MAX_ITERATIONS}.pid"

mkdir -p "$RUN_LOG_DIR" "$SAVE_ROOT"

if [[ -f "$PID_FILE" ]]; then
  old_pid=$(cat "$PID_FILE")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Centered 80k queue appears to be running already: pid=$old_pid"
    echo "log=$RUN_LOG"
    exit 1
  fi
fi

export MAX_ITERATIONS GPU_ID SAVE_ROOT
setsid nohup bash scripts/experiments/covar_calib_conf/run_centered_80k.sh > "$RUN_LOG" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"

echo "Started centered 80k queue"
echo "pid=$pid"
echo "max_iterations=$MAX_ITERATIONS"
echo "gpu_id=$GPU_ID"
echo "save_root=$SAVE_ROOT"
echo "log=$RUN_LOG"
echo "pid_file=$PID_FILE"
echo ""
echo "Monitor:  bash scripts/experiments/covar_calib_conf/monitor_centered_80k.sh"
echo "Stop:     bash scripts/experiments/covar_calib_conf/stop_centered_80k.sh"
