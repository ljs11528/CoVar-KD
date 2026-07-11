#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
RUN_DIR="$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseJ_voc_20k"
PID_FILE="$RUN_DIR/phaseJ_voc_20k_baselines.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "[kd-baselines-npu] no pid file: $PID_FILE"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "[kd-baselines-npu] stopped pid=$PID"
else
  echo "[kd-baselines-npu] pid=$PID is not running"
fi
