#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
RUN_DIR="$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseM_cwd_covar"
PID_FILE="$RUN_DIR/phaseM_after_phaseL.pid"
LOG_FILE="$RUN_DIR/phaseM_after_phaseL.nohup.log"
RUN_SCRIPT="$ROOT_DIR/scripts/experiments/kd_baselines_npu/run_phaseM_after_phaseL.sh"
mkdir -p "$RUN_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "[kd-baselines-npu] Phase M handoff already running pid=$old_pid"
    exit 0
  fi
fi

echo "[kd-baselines-npu] launching Phase M handoff at $(date -Is)" >> "$LOG_FILE"
setsid bash "$RUN_SCRIPT" >> "$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
echo "[kd-baselines-npu] launched Phase M handoff pid=$pid log=$LOG_FILE"
