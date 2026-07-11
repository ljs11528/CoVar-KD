#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
RUN_DIR="$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseL_cwd_seed_stability"
VARIANT_LOG_ROOT="$ROOT_DIR/runs/logs/kd_baselines_npu/phaseL_cwd_seed_stability"
RUN_SCRIPT="$ROOT_DIR/scripts/experiments/kd_baselines_npu/run_phaseL_cwd_seed.sh"
mkdir -p "$RUN_DIR" "$VARIANT_LOG_ROOT"

launch_seed() {
  local seed="$1"
  local device="$2"
  local master_port="$3"
  local name="phaseL_cwd_seed${seed}"
  local pid_file="$RUN_DIR/${name}.pid"
  local queue_log="$RUN_DIR/${name}.nohup.log"

  if [[ -f "$pid_file" ]]; then
    local old_pid
    old_pid="$(cat "$pid_file")"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "[kd-baselines-npu] $name already running pid=$old_pid"
      return 0
    fi
  fi

  echo "[kd-baselines-npu] launching $name device=$device at $(date -Is)" >> "$queue_log"
  setsid env \
    ASCEND_DEVICES="$device" \
    NPROC_PER_NODE=1 \
    BATCH_SIZE=16 \
    MAX_ITERATIONS=80000 \
    SAVE_PER_ITERS=800 \
    VAL_PER_ITERS=800 \
    MASTER_PORT="$master_port" \
    SEED="$seed" \
    SAVE_ROOT="$RUN_DIR" \
    LOG_ROOT="$VARIANT_LOG_ROOT" \
    bash "$RUN_SCRIPT" >> "$queue_log" 2>&1 < /dev/null &

  local pid=$!
  echo "$pid" > "$pid_file"
  echo "[kd-baselines-npu] launched $name pid=$pid device=$device log=$queue_log"
}

launch_seed 2025 0 29760
launch_seed 3407 1 29761
