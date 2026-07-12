#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
RUN_DIR="$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseM_cwd_covar"
PID_FILE="$RUN_DIR/phaseM_after_phaseL.pid"

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "[kd-baselines-npu] Phase M handoff running pid=$pid"
  else
    echo "[kd-baselines-npu] Phase M handoff stopped pid=$pid"
  fi
else
  echo "[kd-baselines-npu] Phase M handoff not launched"
fi

for stage in smoke triage_20k; do
  for variant in fixed covar; do
    log_file="$RUN_DIR/${stage}_${variant}.nohup.log"
    echo "[kd-baselines-npu] stage=$stage variant=$variant"
    if [[ -f "$log_file" ]]; then
      grep '\[kd-baselines-npu\].*variant=' "$log_file" | tail -n 1 || true
      grep 'Iters:' "$log_file" | tail -n 1 || true
      grep 'Sample: 1449,' "$log_file" | tail -n 1 || true
      grep 'Total training time:' "$log_file" | tail -n 1 || true
      rg 'Traceback|RuntimeError|ERROR|Killed|OOM|out of memory' "$log_file" | tail -n 3 || true
    else
      echo "[kd-baselines-npu] pending"
    fi
  done
done

tail -n 8 "$RUN_DIR/phaseM_after_phaseL.nohup.log" 2>/dev/null || true
