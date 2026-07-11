#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
RUN_DIR="$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseK_voc_80k"

stop_queue() {
  local queue_name="$1"
  local pid_file="$RUN_DIR/${queue_name}.pid"

  if [[ ! -f "$pid_file" ]]; then
    echo "[kd-baselines-npu] no pid file for $queue_name"
    return 0
  fi

  local pid command pgid
  pid="$(cat "$pid_file")"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[kd-baselines-npu] $queue_name is not running pid=$pid"
    return 0
  fi

  command="$(ps -o args= -p "$pid")"
  if [[ "$command" != *"run_phaseK_voc_80k"* ]]; then
    echo "[kd-baselines-npu] refusing to stop unexpected pid=$pid command=$command" >&2
    return 1
  fi

  pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
  kill -TERM -- "-$pgid"
  echo "[kd-baselines-npu] stopped $queue_name pid=$pid pgid=$pgid"
}

stop_queue "phaseK_npu0_cwd_ifvd"
stop_queue "phaseK_npu1_skd_kdonly"
