#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
RUN_DIR=${PHASEN_SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseN_scalar_temperature_80k"}
LOG_ROOT=${PHASEN_LOG_ROOT:-"$ROOT_DIR/runs/logs/kd_baselines_npu/phaseN_scalar_temperature_80k"}
PID_FILE="$RUN_DIR/phaseN_scalar_temperature_80k.pid"
QUEUE_LOG="$RUN_DIR/phaseN_scalar_temperature_80k.nohup.log"
RUN_SCRIPT="$ROOT_DIR/scripts/experiments/kd_baselines_npu/run_phaseN_scalar_temperature_80k.sh"
PHASEN_COMMON="$ROOT_DIR/scripts/experiments/kd_baselines_npu/phaseN_scalar_temperature_common.sh"
STUDENT_PRETRAINED_BASE="$ROOT_DIR/data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth"
LOG_NAME="deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"
# shellcheck source=/dev/null
source "$PHASEN_COMMON"
mkdir -p "$RUN_DIR" "$LOG_ROOT"

variant_complete() {
  local variant="$1"
  local kd_temperature="$2"
  local training_log="$LOG_ROOT/main_80k/$variant/$LOG_NAME"
  phaseN_log_session_complete \
    "$training_log" 80000 "$kd_temperature" 800 800 False \
    "$STUDENT_PRETRAINED_BASE" || return 1
  phaseN_artifacts_complete "$RUN_DIR/main_80k/$variant" True
}

if variant_complete "cwd_tout3_kdtemp1p0_80k_seed1234" 1.0 \
  && variant_complete "cwd_tout3_kdtemp0p6_80k_seed1234" 0.6; then
  echo "[kd-baselines-npu] Phase N already complete; nothing to launch"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if [[ "$old_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    old_command="$(ps -o args= -p "$old_pid")"
    if [[ "$old_command" == *"run_phaseN_scalar_temperature_80k.sh"* ]]; then
      echo "[kd-baselines-npu] Phase N already running pid=$old_pid"
      exit 0
    fi
    echo "[kd-baselines-npu] refusing launch: live unrelated pid=$old_pid from $PID_FILE command=$old_command" >&2
    exit 1
  fi
fi

# Do not duplicate an orphaned worker if the controller exited unexpectedly.
for worker_pid_file in "$RUN_DIR"/runtime/*/*.pid; do
  [[ -e "$worker_pid_file" ]] || continue
  worker_pid="$(cat "$worker_pid_file")"
  if [[ "$worker_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$worker_pid" 2>/dev/null; then
    worker_command="$(ps -o args= -p "$worker_pid")"
    if [[ "$worker_command" == *"run_phaseN_cwd_scalar_variant.sh"* ]]; then
      echo "[kd-baselines-npu] Phase N worker already running pid=$worker_pid file=$worker_pid_file; refusing duplicate controller" >&2
      exit 1
    fi
    echo "[kd-baselines-npu] live unrelated worker pid=$worker_pid from $worker_pid_file command=$worker_command" >&2
    exit 1
  fi
done

echo "[kd-baselines-npu] launching Phase N scalar-temperature 80k comparison at $(date -Is)" >> "$QUEUE_LOG"
setsid env \
  ROOT_DIR="$ROOT_DIR" \
  PHASEN_SAVE_ROOT="$RUN_DIR" \
  PHASEN_LOG_ROOT="$LOG_ROOT" \
  bash "$RUN_SCRIPT" >> "$QUEUE_LOG" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
echo "[kd-baselines-npu] launched Phase N pid=$pid log=$QUEUE_LOG"
