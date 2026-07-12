#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
RUN_DIR="$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseM_cwd_covar"
PID_FILE="$RUN_DIR/phaseM_after_phaseL.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "[kd-baselines-npu] Phase M handoff has no pid file"
  exit 0
fi

pid="$(cat "$PID_FILE")"
if ! kill -0 "$pid" 2>/dev/null; then
  echo "[kd-baselines-npu] Phase M handoff is not running pid=$pid"
  exit 0
fi

command="$(ps -o args= -p "$pid")"
if [[ "$command" != *"run_phaseM_after_phaseL.sh"* ]]; then
  echo "[kd-baselines-npu] refusing to stop unexpected pid=$pid command=$command" >&2
  exit 1
fi

pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
kill -TERM -- "-$pgid"
echo "[kd-baselines-npu] stopped Phase M handoff pid=$pid pgid=$pgid"
