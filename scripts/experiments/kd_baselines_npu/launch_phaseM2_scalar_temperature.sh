#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
RUN_DIR=${PHASEM2_SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseM2_scalar_temperature"}
PID_FILE="$RUN_DIR/phaseM2_scalar_temperature.pid"
QUEUE_LOG="$RUN_DIR/phaseM2_scalar_temperature.nohup.log"
RUN_SCRIPT="$ROOT_DIR/scripts/experiments/kd_baselines_npu/run_phaseM2_scalar_temperature.sh"
mkdir -p "$RUN_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if [[ "$old_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    old_command="$(ps -o args= -p "$old_pid")"
    if [[ "$old_command" == *"run_phaseM2_scalar_temperature.sh"* ]]; then
      echo "[kd-baselines-npu] Phase M2 already running pid=$old_pid"
      exit 0
    fi
    echo "[kd-baselines-npu] refusing launch: live unrelated pid=$old_pid from $PID_FILE command=$old_command" >&2
    exit 1
  fi
fi

echo "[kd-baselines-npu] launching Phase M2 scalar-temperature controls at $(date -Is)" >> "$QUEUE_LOG"
setsid env \
  ROOT_DIR="$ROOT_DIR" \
  PHASEM2_SAVE_ROOT="$RUN_DIR" \
  PHASEM2_LOG_ROOT="${PHASEM2_LOG_ROOT:-$ROOT_DIR/runs/logs/kd_baselines_npu/phaseM2_scalar_temperature}" \
  bash "$RUN_SCRIPT" >> "$QUEUE_LOG" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
echo "[kd-baselines-npu] launched Phase M2 pid=$pid log=$QUEUE_LOG"
