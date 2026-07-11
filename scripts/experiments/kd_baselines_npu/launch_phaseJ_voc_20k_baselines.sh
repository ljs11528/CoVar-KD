#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
RUN_DIR="$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseJ_voc_20k"
mkdir -p "$RUN_DIR"

LOG_FILE="$RUN_DIR/phaseJ_voc_20k_baselines.nohup.log"
PID_FILE="$RUN_DIR/phaseJ_voc_20k_baselines.pid"

export ASCEND_DEVICES=${ASCEND_DEVICES:-"0"}
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}
export BATCH_SIZE=${BATCH_SIZE:-16}

setsid bash -c "cd '$ROOT_DIR' && exec bash scripts/experiments/kd_baselines_npu/run_phaseJ_voc_20k_baselines.sh" \
  > "$LOG_FILE" 2>&1 &
echo "$!" > "$PID_FILE"

echo "[kd-baselines-npu] launched pid=$(cat "$PID_FILE")"
echo "[kd-baselines-npu] log=$LOG_FILE"
