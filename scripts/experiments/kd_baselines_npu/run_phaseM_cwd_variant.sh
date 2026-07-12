#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common_voc_kd_npu.sh"

: "${PHASEM_VARIANT:?Set PHASEM_VARIANT=fixed or covar}"

case "$MAX_ITERATIONS" in
  20) budget_label="smoke" ;;
  20000) budget_label="20k" ;;
  80000) budget_label="80k" ;;
  *) budget_label="${MAX_ITERATIONS}it" ;;
esac

case "$PHASEM_VARIANT" in
  fixed)
    variant="cwd_tout3_fixed_${budget_label}_seed${SEED}"
    method_args=(--teacher-output-temp 3.0)
    ;;
  covar)
    variant="cwd_covar_newton_tout3_${budget_label}_seed${SEED}"
    method_args=(
      --teacher-output-temp 3.0
      --use-covar
      --covar-temp-mode newton
      --covar-temp-base 1.0
      --covar-temp-min 0.5
      --covar-temp-max 8.0
      --covar-kd-temp-power 2.0
      --covar-grad-eta 0.6
      --covar-grad-max-iter 8
      --covar-newton-hessian-eps 1e-5
      --covar-newton-max-step 0.25
      --covar-reliability-mode full
    )
    ;;
  *)
    echo "[kd-baselines-npu] unsupported PHASEM_VARIANT=$PHASEM_VARIANT" >&2
    exit 2
    ;;
esac

variant_log="$LOG_ROOT/$variant/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"
if [[ -f "$variant_log" ]] \
  && grep -q "Iters: ${MAX_ITERATIONS}/${MAX_ITERATIONS}" "$variant_log" \
  && grep -q "Total training time:" "$variant_log"; then
  echo "[kd-baselines-npu] skip complete variant=$variant"
  exit 0
fi

extra_args=()
if [[ "${SKIP_VAL:-0}" == "1" ]]; then
  extra_args+=(--skip-val)
fi

echo "[kd-baselines-npu] Phase M variant=$PHASEM_VARIANT started at $(date -Is)"
run_variant "$variant" \
  --lambda-kd 1.0 \
  --lambda-d 0.1 \
  --lambda-adv 0.001 \
  --lambda-cwd-fea 50.0 \
  --lambda-cwd-logit 3.0 \
  "${method_args[@]}" \
  "${extra_args[@]}"
echo "[kd-baselines-npu] Phase M variant=$PHASEM_VARIANT finished at $(date -Is)"
