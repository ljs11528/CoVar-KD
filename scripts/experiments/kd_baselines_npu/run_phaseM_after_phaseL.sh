#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
PHASEL_RUN_DIR="$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseL_cwd_seed_stability"
PHASEL_LOG_ROOT="$ROOT_DIR/runs/logs/kd_baselines_npu/phaseL_cwd_seed_stability"
PHASEM_ROOT="$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseM_cwd_covar"
RUN_SCRIPT="$ROOT_DIR/scripts/experiments/kd_baselines_npu/run_phaseM_cwd_variant.sh"
SUMMARY_SCRIPT="$ROOT_DIR/scripts/experiments/kd_baselines_npu/summarize_phaseL_phaseM.py"
PYTHON=${PYTHON:-"/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python"}

phase_l_running() {
  local seed pid_file pid
  for seed in 2025 3407; do
    pid_file="$PHASEL_RUN_DIR/phaseL_cwd_seed${seed}.pid"
    if [[ -f "$pid_file" ]]; then
      pid="$(cat "$pid_file")"
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        return 0
      fi
    fi
  done
  return 1
}

verify_phase_l_complete() {
  local seed log_file
  for seed in 2025 3407; do
    log_file="$PHASEL_LOG_ROOT/cwd_80k_seed${seed}/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"
    if [[ ! -f "$log_file" ]] \
      || ! grep -q 'Iters: 80000/80000' "$log_file" \
      || ! grep -q 'Total training time:' "$log_file"; then
      echo "[kd-baselines-npu] Phase L seed=$seed did not complete cleanly: $log_file" >&2
      return 1
    fi
  done
}

run_pair() {
  local stage="$1"
  local max_iterations="$2"
  local skip_val="$3"
  local log_iter="$4"
  local save_per_iters="$5"
  local val_per_iters="$6"
  local save_root="$PHASEM_ROOT/$stage"
  local variant_log_root="$ROOT_DIR/runs/logs/kd_baselines_npu/phaseM_cwd_covar/$stage"
  local fixed_log="$PHASEM_ROOT/${stage}_fixed.nohup.log"
  local covar_log="$PHASEM_ROOT/${stage}_covar.nohup.log"

  mkdir -p "$save_root" "$variant_log_root"
  echo "[kd-baselines-npu] Phase M $stage pair starting at $(date -Is)"

  env \
    ASCEND_DEVICES=0 \
    NPROC_PER_NODE=1 \
    BATCH_SIZE=16 \
    MAX_ITERATIONS="$max_iterations" \
    LOG_ITER="$log_iter" \
    SAVE_PER_ITERS="$save_per_iters" \
    VAL_PER_ITERS="$val_per_iters" \
    MASTER_PORT=29770 \
    SEED=1234 \
    SKIP_VAL="$skip_val" \
    PHASEM_VARIANT=fixed \
    SAVE_ROOT="$save_root" \
    LOG_ROOT="$variant_log_root" \
    bash "$RUN_SCRIPT" >> "$fixed_log" 2>&1 &
  local fixed_pid=$!

  env \
    ASCEND_DEVICES=1 \
    NPROC_PER_NODE=1 \
    BATCH_SIZE=16 \
    MAX_ITERATIONS="$max_iterations" \
    LOG_ITER="$log_iter" \
    SAVE_PER_ITERS="$save_per_iters" \
    VAL_PER_ITERS="$val_per_iters" \
    MASTER_PORT=29771 \
    SEED=1234 \
    SKIP_VAL="$skip_val" \
    PHASEM_VARIANT=covar \
    SAVE_ROOT="$save_root" \
    LOG_ROOT="$variant_log_root" \
    bash "$RUN_SCRIPT" >> "$covar_log" 2>&1 &
  local covar_pid=$!

  local failed=0
  if ! wait "$fixed_pid"; then
    echo "[kd-baselines-npu] Phase M $stage fixed variant failed" >&2
    failed=1
  fi
  if ! wait "$covar_pid"; then
    echo "[kd-baselines-npu] Phase M $stage CoVar variant failed" >&2
    failed=1
  fi
  if [[ "$failed" -ne 0 ]]; then
    return 1
  fi

  echo "[kd-baselines-npu] Phase M $stage pair finished at $(date -Is)"
}

mkdir -p "$PHASEM_ROOT"
echo "[kd-baselines-npu] Phase M handoff waiting for Phase L at $(date -Is)"
while phase_l_running; do
  sleep 30
done

verify_phase_l_complete
phase_l_report="$ROOT_DIR/reports/$(date +%F)_phaseL_cwd_seed_stability.md"
if ! "$PYTHON" "$SUMMARY_SCRIPT" phase-l --root "$ROOT_DIR" --report "$phase_l_report"; then
  echo "[kd-baselines-npu] warning: failed to write Phase L report" >&2
else
  echo "[kd-baselines-npu] wrote Phase L report: $phase_l_report"
fi
echo "[kd-baselines-npu] Phase L complete; starting Phase M at $(date -Is)"
run_pair smoke 20 1 5 100 100
run_pair triage_20k 20000 0 20 800 800
phase_m_report="$ROOT_DIR/reports/$(date +%F)_phaseM_cwd_covar_triage.md"
if ! "$PYTHON" "$SUMMARY_SCRIPT" phase-m --root "$ROOT_DIR" --report "$phase_m_report"; then
  echo "[kd-baselines-npu] warning: failed to write Phase M report" >&2
else
  echo "[kd-baselines-npu] wrote Phase M report: $phase_m_report"
fi
echo "[kd-baselines-npu] Phase M handoff finished at $(date -Is)"
