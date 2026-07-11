#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
RUN_DIR="$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseJ_voc_20k"
LOG_FILE="$RUN_DIR/phaseJ_voc_20k_baselines.nohup.log"
PID_FILE="$RUN_DIR/phaseJ_voc_20k_baselines.pid"

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  if kill -0 "$PID" 2>/dev/null; then
    echo "[kd-baselines-npu] queue running pid=$PID"
  else
    echo "[kd-baselines-npu] queue pid=$PID is not running"
  fi
else
  echo "[kd-baselines-npu] no pid file: $PID_FILE"
fi

echo "[kd-baselines-npu] latest queue log:"
if [[ -f "$LOG_FILE" ]]; then
  tail -n 80 "$LOG_FILE"
else
  echo "missing log: $LOG_FILE"
fi

echo "[kd-baselines-npu] validation summaries:"
find "$RUN_DIR" -path '*/kd_deeplabv3_mobilenet_ssseg_voc_best_model.pth' -printf '%h\n' 2>/dev/null | sort || true
