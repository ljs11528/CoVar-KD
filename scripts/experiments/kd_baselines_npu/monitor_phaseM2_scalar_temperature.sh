#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
RUN_DIR=${PHASEM2_SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseM2_scalar_temperature"}
LOG_ROOT=${PHASEM2_LOG_ROOT:-"$ROOT_DIR/runs/logs/kd_baselines_npu/phaseM2_scalar_temperature"}
PID_FILE="$RUN_DIR/phaseM2_scalar_temperature.pid"
QUEUE_LOG="$RUN_DIR/phaseM2_scalar_temperature.nohup.log"
LOG_NAME="deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$pid" 2>/dev/null; then
    command="$(ps -o args= -p "$pid")"
    if [[ "$command" == *"run_phaseM2_scalar_temperature.sh"* ]]; then
      echo "[kd-baselines-npu] Phase M2 state=running pid=$pid"
    else
      echo "[kd-baselines-npu] Phase M2 state=pid-reused pid=$pid command=$command"
    fi
  else
    echo "[kd-baselines-npu] Phase M2 state=stopped pid=$pid"
  fi
else
  echo "[kd-baselines-npu] Phase M2 state=not-launched"
fi

show_variant() {
  local stage="$1"
  local budget_label="$2"
  local max_iterations="$3"
  local temp_label="$4"
  local variant="cwd_tout3_kdtemp${temp_label}_${budget_label}_seed1234"
  local training_log="$LOG_ROOT/$stage/$variant/$LOG_NAME"
  local runtime_log="$RUN_DIR/runtime/$stage/kdtemp${temp_label}.nohup.log"

  echo "[kd-baselines-npu] stage=$stage variant=$variant"
  if [[ -f "$training_log" ]]; then
    grep 'Iters:' "$training_log" | tail -n 1 || true
    grep 'Sample: 1449,' "$training_log" | tail -n 1 || true
    grep 'Total training time:' "$training_log" | tail -n 1 || true
    if grep -Fq "Iters: ${max_iterations}/${max_iterations}" "$training_log" \
      && grep -Fq "Total training time:" "$training_log"; then
      echo "[kd-baselines-npu] complete=true"
    else
      echo "[kd-baselines-npu] complete=false"
    fi
  else
    echo "[kd-baselines-npu] pending log=$training_log"
  fi

  if [[ -f "$runtime_log" ]]; then
    rg 'Traceback|RuntimeError|ERROR|Killed|OOM|out of memory' "$runtime_log" | tail -n 3 || true
  fi
}

show_variant smoke smoke 20 0p5
show_variant smoke smoke 20 0p6
show_variant triage_20k 20k 20000 0p5
show_variant triage_20k 20k 20000 0p6

if [[ -f "$QUEUE_LOG" ]]; then
  tail -n 8 "$QUEUE_LOG" || true
fi

if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info || true
fi
