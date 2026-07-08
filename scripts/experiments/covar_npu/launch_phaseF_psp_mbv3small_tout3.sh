#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
cd "$ROOT_DIR"

MAX_ITERATIONS=${MAX_ITERATIONS:-80000}
ASCEND_DEVICES=${ASCEND_DEVICES:-"0,1"}
SAVE_ROOT=${SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseF_psp_mbv3small"}
RUN_LOG_DIR=${RUN_LOG_DIR:-"$ROOT_DIR/runs/covar_npu_phaseF_psp_mbv3small"}
RUN_LOG="$RUN_LOG_DIR/phaseF_psp_mbv3small_tout3_${MAX_ITERATIONS}.log"
PID_FILE="$RUN_LOG_DIR/phaseF_psp_mbv3small_tout3_${MAX_ITERATIONS}.pid"
MASTER_PORT=${MASTER_PORT:-29660}
SEED=${SEED:-1234}

mkdir -p "$RUN_LOG_DIR" "$SAVE_ROOT"

if [[ -f "$PID_FILE" ]]; then
  old_pid=$(cat "$PID_FILE")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Phase F PSP-MobileNetV3-Small already appears to be running: pid=$old_pid"
    echo "log=$RUN_LOG"
    exit 1
  fi
fi

export MAX_ITERATIONS ASCEND_DEVICES SAVE_ROOT MASTER_PORT SEED
setsid nohup bash scripts/experiments/covar_npu/run_phaseF_psp_mbv3small_tout3.sh >> "$RUN_LOG" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"

echo "Started Phase F PSP-MobileNetV3-Small Tout=3.0"
echo "pid=$pid"
echo "seed=$SEED"
echo "max_iterations=$MAX_ITERATIONS"
echo "ascend_devices=$ASCEND_DEVICES"
echo "save_root=$SAVE_ROOT"
echo "log=$RUN_LOG"
echo "pid_file=$PID_FILE"

