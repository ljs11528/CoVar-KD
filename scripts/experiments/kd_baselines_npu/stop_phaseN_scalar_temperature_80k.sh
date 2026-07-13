#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
RUN_DIR=${PHASEN_SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseN_scalar_temperature_80k"}
PID_FILE="$RUN_DIR/phaseN_scalar_temperature_80k.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "[kd-baselines-npu] Phase N has no controller pid file"
  exit 0
fi

pid="$(cat "$PID_FILE")"
if [[ ! "$pid" =~ ^[1-9][0-9]*$ ]]; then
  echo "[kd-baselines-npu] refusing to stop invalid pid from $PID_FILE: $pid" >&2
  exit 1
fi

if kill -0 "$pid" 2>/dev/null; then
  command="$(ps -o args= -p "$pid")"
  if [[ "$command" != *"run_phaseN_scalar_temperature_80k.sh"* ]]; then
    echo "[kd-baselines-npu] refusing to stop unexpected pid=$pid command=$command" >&2
    exit 1
  fi
  pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
  if [[ ! "$pgid" =~ ^[1-9][0-9]*$ ]] || [[ "$pgid" != "$pid" ]]; then
    echo "[kd-baselines-npu] refusing to stop unsafe process group pid=$pid pgid=$pgid" >&2
    exit 1
  fi
  kill -TERM -- "-$pgid"
  echo "[kd-baselines-npu] stopped Phase N pid=$pid pgid=$pgid"
  exit 0
fi

# A controller can exit while a worker from its setsid process group survives.
# Only target that group when a recorded Phase N worker still belongs to it.
for worker_pid_file in "$RUN_DIR"/runtime/*/*.pid; do
  [[ -e "$worker_pid_file" ]] || continue
  worker_pid="$(cat "$worker_pid_file")"
  if [[ "$worker_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$worker_pid" 2>/dev/null; then
    worker_command="$(ps -o args= -p "$worker_pid")"
    worker_pgid="$(ps -o pgid= -p "$worker_pid" | tr -d '[:space:]')"
    if [[ "$worker_command" == *"run_phaseN_cwd_scalar_variant.sh"* && "$worker_pgid" == "$pid" ]]; then
      kill -TERM -- "-$worker_pgid"
      echo "[kd-baselines-npu] stopped orphaned Phase N process group pgid=$worker_pgid via worker pid=$worker_pid"
      exit 0
    fi
    echo "[kd-baselines-npu] refusing to stop unexpected worker pid=$worker_pid pgid=$worker_pgid command=$worker_command" >&2
    exit 1
  fi
done

echo "[kd-baselines-npu] Phase N is not running pid=$pid"
