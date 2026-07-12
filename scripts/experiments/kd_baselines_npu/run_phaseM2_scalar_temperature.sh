#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
CHECKPOINT_ROOT=${PHASEM2_SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseM2_scalar_temperature"}
LOG_ROOT_BASE=${PHASEM2_LOG_ROOT:-"$ROOT_DIR/runs/logs/kd_baselines_npu/phaseM2_scalar_temperature"}
VARIANT_SCRIPT="$ROOT_DIR/scripts/experiments/kd_baselines_npu/run_phaseM2_cwd_scalar_variant.sh"

run_pair() {
  local stage="$1"
  local max_iterations="$2"
  local skip_val="$3"
  local log_iter="$4"
  local save_per_iters="$5"
  local val_per_iters="$6"
  local port_base="$7"
  local save_root="$CHECKPOINT_ROOT/$stage"
  local log_root="$LOG_ROOT_BASE/$stage"
  local runtime_root="$CHECKPOINT_ROOT/runtime/$stage"
  local log_t05="$runtime_root/kdtemp0p5.nohup.log"
  local log_t06="$runtime_root/kdtemp0p6.nohup.log"

  mkdir -p "$save_root" "$log_root" "$runtime_root"
  echo "[kd-baselines-npu] Phase M2 stage=$stage pair starting at $(date -Is)"

  env \
    ASCEND_DEVICES=0 \
    NPROC_PER_NODE=1 \
    BATCH_SIZE=16 \
    MAX_ITERATIONS="$max_iterations" \
    LOG_ITER="$log_iter" \
    SAVE_PER_ITERS="$save_per_iters" \
    VAL_PER_ITERS="$val_per_iters" \
    MASTER_PORT="$port_base" \
    SEED=1234 \
    SKIP_VAL="$skip_val" \
    PHASEM2_KD_TEMP=0.5 \
    SAVE_ROOT="$save_root" \
    LOG_ROOT="$log_root" \
    bash "$VARIANT_SCRIPT" >> "$log_t05" 2>&1 &
  local pid_t05=$!

  env \
    ASCEND_DEVICES=1 \
    NPROC_PER_NODE=1 \
    BATCH_SIZE=16 \
    MAX_ITERATIONS="$max_iterations" \
    LOG_ITER="$log_iter" \
    SAVE_PER_ITERS="$save_per_iters" \
    VAL_PER_ITERS="$val_per_iters" \
    MASTER_PORT="$((port_base + 1))" \
    SEED=1234 \
    SKIP_VAL="$skip_val" \
    PHASEM2_KD_TEMP=0.6 \
    SAVE_ROOT="$save_root" \
    LOG_ROOT="$log_root" \
    bash "$VARIANT_SCRIPT" >> "$log_t06" 2>&1 &
  local pid_t06=$!

  local failed=0
  if ! wait "$pid_t05"; then
    echo "[kd-baselines-npu] Phase M2 stage=$stage kd_temperature=0.5 failed; see $log_t05" >&2
    failed=1
  fi
  if ! wait "$pid_t06"; then
    echo "[kd-baselines-npu] Phase M2 stage=$stage kd_temperature=0.6 failed; see $log_t06" >&2
    failed=1
  fi
  if [[ "$failed" -ne 0 ]]; then
    return 1
  fi

  echo "[kd-baselines-npu] Phase M2 stage=$stage pair finished at $(date -Is)"
}

mkdir -p "$CHECKPOINT_ROOT" "$LOG_ROOT_BASE"
echo "[kd-baselines-npu] Phase M2 scalar-temperature controls started at $(date -Is)"
run_pair smoke 20 1 5 100 100 29820
run_pair triage_20k 20000 0 20 800 800 29830
echo "[kd-baselines-npu] Phase M2 scalar-temperature controls finished at $(date -Is)"
