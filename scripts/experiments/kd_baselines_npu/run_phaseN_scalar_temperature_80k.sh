#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
CHECKPOINT_ROOT=${PHASEN_SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseN_scalar_temperature_80k"}
LOG_ROOT_BASE=${PHASEN_LOG_ROOT:-"$ROOT_DIR/runs/logs/kd_baselines_npu/phaseN_scalar_temperature_80k"}
VARIANT_SCRIPT="$ROOT_DIR/scripts/experiments/kd_baselines_npu/run_phaseN_cwd_scalar_variant.sh"
PHASEN_COMMON="$ROOT_DIR/scripts/experiments/kd_baselines_npu/phaseN_scalar_temperature_common.sh"
DATA_DIR="$ROOT_DIR/dataset/VOCAug/"
TEACHER_PRETRAINED="$ROOT_DIR/data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth"
STUDENT_PRETRAINED_BASE="$ROOT_DIR/data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth"
PYTHON="/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python"
LOG_NAME="deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"
# shellcheck source=/dev/null
source "$PHASEN_COMMON"

variant_complete() {
  local log_root="$1"
  local save_root="$2"
  local variant="$3"
  local max_iterations="$4"
  local kd_temperature="$5"
  local save_per_iters="$6"
  local val_per_iters="$7"
  local skip_val="$8"
  local training_log="$log_root/$variant/$LOG_NAME"

  phaseN_log_session_complete \
    "$training_log" "$max_iterations" "$kd_temperature" "$save_per_iters" \
    "$val_per_iters" "$skip_val" "$STUDENT_PRETRAINED_BASE" || return 1
  local require_best=True
  if [[ "$skip_val" == "True" ]]; then
    require_best=False
  fi
  phaseN_artifacts_complete "$save_root/$variant" "$require_best"
}

run_pair() {
  local stage="$1"
  local budget_label="$2"
  local max_iterations="$3"
  local skip_val="$4"
  local log_iter="$5"
  local save_per_iters="$6"
  local val_per_iters="$7"
  local port_base="$8"
  local save_root="$CHECKPOINT_ROOT/$stage"
  local log_root="$LOG_ROOT_BASE/$stage"
  local runtime_root="$CHECKPOINT_ROOT/runtime/$stage"
  local runtime_t10="$runtime_root/kdtemp1p0.nohup.log"
  local runtime_t06="$runtime_root/kdtemp0p6.nohup.log"
  local pid_t10_file="$runtime_root/kdtemp1p0.pid"
  local pid_t06_file="$runtime_root/kdtemp0p6.pid"
  local variant_t10="cwd_tout3_kdtemp1p0_${budget_label}_seed1234"
  local variant_t06="cwd_tout3_kdtemp0p6_${budget_label}_seed1234"

  mkdir -p "$save_root" "$log_root" "$runtime_root"
  echo "[kd-baselines-npu] Phase N stage=$stage pair starting at $(date -Is)"

  if [[ -f "$pid_t10_file" ]]; then
    local old_pid_t10
    old_pid_t10="$(cat "$pid_t10_file")"
    if [[ "$old_pid_t10" =~ ^[1-9][0-9]*$ ]] && kill -0 "$old_pid_t10" 2>/dev/null; then
      echo "[kd-baselines-npu] refusing duplicate Phase N stage=$stage kd_temperature=1.0 pid=$old_pid_t10" >&2
      return 1
    fi
  fi
  if [[ -f "$pid_t06_file" ]]; then
    local old_pid_t06
    old_pid_t06="$(cat "$pid_t06_file")"
    if [[ "$old_pid_t06" =~ ^[1-9][0-9]*$ ]] && kill -0 "$old_pid_t06" 2>/dev/null; then
      echo "[kd-baselines-npu] refusing duplicate Phase N stage=$stage kd_temperature=0.6 pid=$old_pid_t06" >&2
      return 1
    fi
  fi

  env \
    ASCEND_DEVICES=0 \
    NPROC_PER_NODE=1 \
    BATCH_SIZE=16 \
    WORKERS=8 \
    DATA_DIR="$DATA_DIR" \
    TEACHER_PRETRAINED="$TEACHER_PRETRAINED" \
    STUDENT_PRETRAINED_BASE="$STUDENT_PRETRAINED_BASE" \
    PYTHON="$PYTHON" \
    MAX_ITERATIONS="$max_iterations" \
    LOG_ITER="$log_iter" \
    SAVE_PER_ITERS="$save_per_iters" \
    VAL_PER_ITERS="$val_per_iters" \
    MASTER_PORT="$port_base" \
    SEED=1234 \
    SKIP_VAL="$skip_val" \
    PHASEN_KD_TEMP=1.0 \
    SAVE_ROOT="$save_root" \
    LOG_ROOT="$log_root" \
    bash "$VARIANT_SCRIPT" >> "$runtime_t10" 2>&1 &
  local pid_t10=$!
  echo "$pid_t10" > "$pid_t10_file"

  env \
    ASCEND_DEVICES=1 \
    NPROC_PER_NODE=1 \
    BATCH_SIZE=16 \
    MAX_ITERATIONS="$max_iterations" \
    LOG_ITER="$log_iter" \
    SAVE_PER_ITERS="$save_per_iters" \
    VAL_PER_ITERS="$val_per_iters" \
    WORKERS=8 \
    DATA_DIR="$DATA_DIR" \
    TEACHER_PRETRAINED="$TEACHER_PRETRAINED" \
    STUDENT_PRETRAINED_BASE="$STUDENT_PRETRAINED_BASE" \
    MASTER_PORT="$((port_base + 1))" \
    PYTHON="$PYTHON" \
    SEED=1234 \
    SKIP_VAL="$skip_val" \
    PHASEN_KD_TEMP=0.6 \
    SAVE_ROOT="$save_root" \
    LOG_ROOT="$log_root" \
    bash "$VARIANT_SCRIPT" >> "$runtime_t06" 2>&1 &
  local pid_t06=$!
  echo "$pid_t06" > "$pid_t06_file"

  local failed=0
  if ! wait "$pid_t10"; then
    echo "[kd-baselines-npu] Phase N stage=$stage kd_temperature=1.0 failed; see $runtime_t10" >&2
    failed=1
  fi
  if ! wait "$pid_t06"; then
    echo "[kd-baselines-npu] Phase N stage=$stage kd_temperature=0.6 failed; see $runtime_t06" >&2
    failed=1
  fi

  local expected_skip_val=False
  if [[ "$skip_val" == "1" ]]; then
    expected_skip_val=True
  fi
  if ! variant_complete "$log_root" "$save_root" "$variant_t10" "$max_iterations" 1.0 \
    "$save_per_iters" "$val_per_iters" "$expected_skip_val"; then
    echo "[kd-baselines-npu] Phase N stage=$stage kd_temperature=1.0 did not produce a complete log" >&2
    failed=1
  fi
  if ! variant_complete "$log_root" "$save_root" "$variant_t06" "$max_iterations" 0.6 \
    "$save_per_iters" "$val_per_iters" "$expected_skip_val"; then
    echo "[kd-baselines-npu] Phase N stage=$stage kd_temperature=0.6 did not produce a complete log" >&2
    failed=1
  fi
  if [[ "$failed" -ne 0 ]]; then
    return 1
  fi

  echo "[kd-baselines-npu] Phase N stage=$stage pair finished at $(date -Is)"
}

mkdir -p "$CHECKPOINT_ROOT" "$LOG_ROOT_BASE"
echo "[kd-baselines-npu] Phase N scalar-temperature 80k comparison started at $(date -Is)"

# Both smoke runs must complete successfully before either 80k run is launched.
run_pair smoke smoke 20 1 5 100 100 29840
run_pair main_80k 80k 80000 0 20 800 800 29850

echo "[kd-baselines-npu] Phase N scalar-temperature 80k comparison finished at $(date -Is)"
