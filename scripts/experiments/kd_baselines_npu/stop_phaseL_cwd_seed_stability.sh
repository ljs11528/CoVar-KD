#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
RUN_DIR="$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseL_cwd_seed_stability"

stop_seed() {
  local seed="$1"
  local name="phaseL_cwd_seed${seed}"
  local pid_file="$RUN_DIR/${name}.pid"

  if [[ ! -f "$pid_file" ]]; then
    echo "[kd-baselines-npu] no pid file for seed=$seed"
    return 0
  fi

  local pid command pgid
  pid="$(cat "$pid_file")"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[kd-baselines-npu] seed=$seed is not running pid=$pid"
    return 0
  fi

  command="$(ps -o args= -p "$pid")"
  if [[ "$command" != *"run_phaseL_cwd_seed.sh"* ]]; then
    echo "[kd-baselines-npu] refusing to stop unexpected pid=$pid command=$command" >&2
    return 1
  fi

  pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
  kill -TERM -- "-$pgid"
  echo "[kd-baselines-npu] stopped seed=$seed pid=$pid pgid=$pgid"
}

stop_seed 2025
stop_seed 3407
