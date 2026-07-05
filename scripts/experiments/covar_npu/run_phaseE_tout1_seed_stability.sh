#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${ROOT_DIR:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
MAX_ITERATIONS=${MAX_ITERATIONS:-80000}
SAVE_ROOT=${SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseE_seed_stability"}
PHASED_ROOT=${PHASED_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseD_main_table"}
MASTER_PORT=${MASTER_PORT:-29650}
PHASEE_SEEDS=${PHASEE_SEEDS:-"2025 3407"}
export ROOT_DIR MAX_ITERATIONS SAVE_ROOT MASTER_PORT

LOG_NAME="deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"

variant_complete() {
  local variant="$1"
  local log_file="$SAVE_ROOT/$variant/$LOG_NAME"
  [[ -f "$log_file" ]] && grep -q "Iters: ${MAX_ITERATIONS}/${MAX_ITERATIONS}" "$log_file" && grep -q "Total training time" "$log_file"
}

run_no_covar() {
  local seed="$1"
  local variant="phaseE_seed${seed}_cirkd_no_covar_tout1"
  if variant_complete "$variant"; then
    echo "[covar-npu] skip complete $variant"
    return 0
  fi
  (
    export SEED="$seed"
    export SAVE_ROOT
    export MASTER_PORT
    source "$SCRIPT_DIR/common_voc_cirkdv2_npu.sh"
    run_variant "$variant" \
      --teacher-output-temp 1.0 \
      --no-covar
  )
}

run_covar() {
  local seed="$1"
  local variant="phaseE_seed${seed}_covar_newton_gamma2_tout1"
  if variant_complete "$variant"; then
    echo "[covar-npu] skip complete $variant"
    return 0
  fi
  (
    export SEED="$seed"
    export SAVE_ROOT
    export MASTER_PORT
    source "$SCRIPT_DIR/common_voc_cirkdv2_npu.sh"
    run_variant "$variant" \
      --teacher-output-temp 1.0 \
      --covar-temp-mode newton \
      --covar-temp-base 1.0 \
      --covar-temp-min 0.5 \
      --covar-temp-max 8.0 \
      --covar-kd-temp-power 2.0 \
      --covar-grad-eta 0.6 \
      --covar-grad-max-iter 8 \
      --covar-newton-hessian-eps 1e-5 \
      --covar-newton-max-step 0.25 \
      --covar-grad-converge-thresh 0.01 \
      --covar-grad-detail-interval 100
  )
}

echo "[covar-npu] Phase E Tout=1.0 seed stability started at $(date -Is)"
echo "[covar-npu] phaseE_seeds=$PHASEE_SEEDS"
echo "[covar-npu] save_root=$SAVE_ROOT"
echo "[covar-npu] phaseD_root=$PHASED_ROOT"

for seed in $PHASEE_SEEDS; do
  echo "[covar-npu] >>> seed=$seed no-covar at $(date -Is)"
  run_no_covar "$seed"
  echo "[covar-npu] <<< seed=$seed no-covar done at $(date -Is)"

  echo "[covar-npu] >>> seed=$seed covar at $(date -Is)"
  run_covar "$seed"
  echo "[covar-npu] <<< seed=$seed covar done at $(date -Is)"
done

REPORT_DIR=${REPORT_DIR:-"$ROOT_DIR/reports"}
REPORT_PATH=${REPORT_PATH:-"$REPORT_DIR/$(date +%F)_phaseE_tout1_seed_stability.md"}
mkdir -p "$REPORT_DIR"

"${PYTHON:-/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python}" "$SCRIPT_DIR/summarize_phaseE_tout1_seed_stability.py" \
  --phase-e-root "$SAVE_ROOT" \
  --phase-d-root "$PHASED_ROOT" \
  --seeds "1234 $PHASEE_SEEDS" \
  --report "$REPORT_PATH" \
  --max-iterations "$MAX_ITERATIONS"

echo "[covar-npu] wrote report: $REPORT_PATH"
echo "[covar-npu] Phase E Tout=1.0 seed stability finished at $(date -Is)"
