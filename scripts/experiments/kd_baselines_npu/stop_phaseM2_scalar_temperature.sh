#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
RUN_DIR=${PHASEM2_SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseM2_scalar_temperature"}
PID_FILE="$RUN_DIR/phaseM2_scalar_temperature.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "[kd-baselines-npu] Phase M2 has no pid file"
  exit 0
fi

pid="$(cat "$PID_FILE")"
if [[ ! "$pid" =~ ^[1-9][0-9]*$ ]]; then
  echo "[kd-baselines-npu] refusing to stop invalid pid from $PID_FILE: $pid" >&2
  exit 1
fi
if ! kill -0 "$pid" 2>/dev/null; then
  echo "[kd-baselines-npu] Phase M2 is not running pid=$pid"
  exit 0
fi

command="$(ps -o args= -p "$pid")"
if [[ "$command" != *"run_phaseM2_scalar_temperature.sh"* ]]; then
  echo "[kd-baselines-npu] refusing to stop unexpected pid=$pid command=$command" >&2
  exit 1
fi

pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
if [[ ! "$pgid" =~ ^[1-9][0-9]*$ ]] || [[ "$pgid" != "$pid" ]]; then
  echo "[kd-baselines-npu] refusing to stop unsafe process group pid=$pid pgid=$pgid" >&2
  exit 1
fi

kill -TERM -- "-$pgid"
echo "[kd-baselines-npu] stopped Phase M2 pid=$pid pgid=$pgid"
