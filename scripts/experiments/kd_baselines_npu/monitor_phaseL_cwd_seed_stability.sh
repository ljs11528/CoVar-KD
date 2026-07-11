#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
RUN_DIR="$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseL_cwd_seed_stability"

show_seed() {
  local seed="$1"
  local name="phaseL_cwd_seed${seed}"
  local pid_file="$RUN_DIR/${name}.pid"
  local queue_log="$RUN_DIR/${name}.nohup.log"

  echo "[kd-baselines-npu] seed=$seed"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "[kd-baselines-npu] state=running pid=$pid"
    else
      echo "[kd-baselines-npu] state=stopped pid=$pid"
    fi
  else
    echo "[kd-baselines-npu] state=not-launched"
  fi

  if [[ -f "$queue_log" ]]; then
    grep '\[kd-baselines-npu\].*variant=' "$queue_log" | tail -n 1 || true
    grep 'Iters:' "$queue_log" | tail -n 1 || true
    grep 'Sample: 1449,' "$queue_log" | tail -n 1 || true
    grep 'Total training time:' "$queue_log" | tail -n 1 || true
  else
    echo "[kd-baselines-npu] missing log=$queue_log"
  fi
}

show_seed 2025
show_seed 3407
npu-smi info
