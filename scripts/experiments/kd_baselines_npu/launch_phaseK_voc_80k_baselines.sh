#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
RUN_DIR="$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseK_voc_80k"
VARIANT_LOG_ROOT="$ROOT_DIR/runs/logs/kd_baselines_npu/phaseK_voc_80k"
mkdir -p "$RUN_DIR" "$VARIANT_LOG_ROOT"

launch_queue() {
  local queue_name="$1"
  local device="$2"
  local master_port="$3"
  local queue_script="$4"
  local pid_file="$RUN_DIR/${queue_name}.pid"
  local queue_log="$RUN_DIR/${queue_name}.nohup.log"

  if [[ -f "$pid_file" ]]; then
    local old_pid
    old_pid="$(cat "$pid_file")"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "[kd-baselines-npu] $queue_name already running pid=$old_pid"
      return 0
    fi
  fi

  echo "[kd-baselines-npu] launching $queue_name device=$device at $(date -Is)" >> "$queue_log"
  setsid env \
    ASCEND_DEVICES="$device" \
    NPROC_PER_NODE=1 \
    BATCH_SIZE=16 \
    MAX_ITERATIONS=80000 \
    SAVE_PER_ITERS=800 \
    VAL_PER_ITERS=800 \
    MASTER_PORT="$master_port" \
    SAVE_ROOT="$RUN_DIR" \
    LOG_ROOT="$VARIANT_LOG_ROOT" \
    bash "$queue_script" >> "$queue_log" 2>&1 < /dev/null &

  local pid=$!
  echo "$pid" > "$pid_file"
  echo "[kd-baselines-npu] launched $queue_name pid=$pid device=$device"
  echo "[kd-baselines-npu] log=$queue_log"
}

launch_queue \
  "phaseK_npu0_cwd_ifvd" \
  "0" \
  "29750" \
  "$ROOT_DIR/scripts/experiments/kd_baselines_npu/run_phaseK_voc_80k_npu0.sh"

launch_queue \
  "phaseK_npu1_skd_kdonly" \
  "1" \
  "29751" \
  "$ROOT_DIR/scripts/experiments/kd_baselines_npu/run_phaseK_voc_80k_npu1.sh"
