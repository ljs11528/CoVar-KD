#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
RUN_DIR=${PHASEN_SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseN_scalar_temperature_80k"}
LOG_ROOT=${PHASEN_LOG_ROOT:-"$ROOT_DIR/runs/logs/kd_baselines_npu/phaseN_scalar_temperature_80k"}
PID_FILE="$RUN_DIR/phaseN_scalar_temperature_80k.pid"
QUEUE_LOG="$RUN_DIR/phaseN_scalar_temperature_80k.nohup.log"
LOG_NAME="deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"
PHASEN_COMMON="$ROOT_DIR/scripts/experiments/kd_baselines_npu/phaseN_scalar_temperature_common.sh"
STUDENT_PRETRAINED_BASE="$ROOT_DIR/data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth"
# shellcheck source=/dev/null
source "$PHASEN_COMMON"

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$pid" 2>/dev/null; then
    command="$(ps -o args= -p "$pid")"
    if [[ "$command" == *"run_phaseN_scalar_temperature_80k.sh"* ]]; then
      echo "[kd-baselines-npu] Phase N state=running pid=$pid"
    else
      echo "[kd-baselines-npu] Phase N state=pid-reused pid=$pid command=$command"
    fi
  else
    echo "[kd-baselines-npu] Phase N state=stopped pid=$pid"
  fi
else
  echo "[kd-baselines-npu] Phase N state=not-launched"
fi

show_variant() {
  local stage="$1"
  local budget_label="$2"
  local max_iterations="$3"
  local temp_label="$4"
  local kd_temperature
  case "$temp_label" in
    1p0) kd_temperature=1.0 ;;
    0p6) kd_temperature=0.6 ;;
    *) echo "[kd-baselines-npu] invalid temperature label=$temp_label" >&2; return 2 ;;
  esac
  local save_per_iters=800
  local val_per_iters=800
  local skip_val=False
  local require_best=True
  if [[ "$stage" == "smoke" ]]; then
    save_per_iters=100
    val_per_iters=100
    skip_val=True
    require_best=False
  fi
  local variant="cwd_tout3_kdtemp${temp_label}_${budget_label}_seed1234"
  local training_log="$LOG_ROOT/$stage/$variant/$LOG_NAME"
  local save_dir="$RUN_DIR/$stage/$variant"
  local runtime_log="$RUN_DIR/runtime/$stage/kdtemp${temp_label}.nohup.log"
  local worker_pid_file="$RUN_DIR/runtime/$stage/kdtemp${temp_label}.pid"

  echo "[kd-baselines-npu] stage=$stage variant=$variant"
  if [[ -f "$worker_pid_file" ]]; then
    local worker_pid
    worker_pid="$(cat "$worker_pid_file")"
    if [[ "$worker_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$worker_pid" 2>/dev/null; then
      echo "[kd-baselines-npu] worker=running pid=$worker_pid"
    else
      echo "[kd-baselines-npu] worker=stopped pid=$worker_pid"
    fi
  else
    echo "[kd-baselines-npu] worker=not-launched"
  fi

  if [[ -f "$training_log" ]]; then
    grep 'Iters:' "$training_log" | tail -n 1 || true
    grep 'Sample: 1449,' "$training_log" | tail -n 1 || true
    grep 'Total training time:' "$training_log" | tail -n 1 || true
    if phaseN_log_session_complete \
      "$training_log" "$max_iterations" "$kd_temperature" "$save_per_iters" \
      "$val_per_iters" "$skip_val" "$STUDENT_PRETRAINED_BASE" \
      && phaseN_artifacts_complete "$save_dir" "$require_best"; then
      echo "[kd-baselines-npu] complete=true"
    else
      echo "[kd-baselines-npu] complete=false"
    fi
  else
    echo "[kd-baselines-npu] pending log=$training_log"
  fi

  if [[ -f "$runtime_log" ]]; then
    grep -E 'Traceback|RuntimeError|ERROR|Killed|OOM|out of memory' "$runtime_log" | tail -n 3 || true
  fi
}

show_variant smoke smoke 20 1p0
show_variant smoke smoke 20 0p6
show_variant main_80k 80k 80000 1p0
show_variant main_80k 80k 80000 0p6

if [[ -f "$QUEUE_LOG" ]]; then
  tail -n 10 "$QUEUE_LOG" || true
fi

if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info || true
fi
