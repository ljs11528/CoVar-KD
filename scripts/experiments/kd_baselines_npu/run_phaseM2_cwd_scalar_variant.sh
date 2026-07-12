#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common_voc_kd_npu.sh"

: "${PHASEM2_KD_TEMP:?Set PHASEM2_KD_TEMP=0.5 or 0.6}"

case "$PHASEM2_KD_TEMP" in
  0.5)
    kd_temp="0.5"
    temp_label="0p5"
    ;;
  0.6)
    kd_temp="0.6"
    temp_label="0p6"
    ;;
  *)
    echo "[kd-baselines-npu] unsupported PHASEM2_KD_TEMP=$PHASEM2_KD_TEMP (expected 0.5 or 0.6)" >&2
    exit 2
    ;;
esac

case "$MAX_ITERATIONS" in
  20) budget_label="smoke" ;;
  20000) budget_label="20k" ;;
  *)
    echo "[kd-baselines-npu] unsupported Phase M2 budget MAX_ITERATIONS=$MAX_ITERATIONS (expected 20 or 20000)" >&2
    exit 2
    ;;
esac

variant="cwd_tout3_kdtemp${temp_label}_${budget_label}_seed${SEED}"
training_log="$LOG_ROOT/$variant/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"

if [[ -f "$training_log" ]] \
  && grep -Fq "Iters: ${MAX_ITERATIONS}/${MAX_ITERATIONS}" "$training_log" \
  && grep -Fq "Total training time:" "$training_log"; then
  echo "[kd-baselines-npu] skip complete Phase M2 variant=$variant log=$training_log"
  exit 0
fi

extra_args=()
if [[ "${SKIP_VAL:-0}" == "1" ]]; then
  extra_args+=(--skip-val)
fi

echo "[kd-baselines-npu] Phase M2 variant=$variant kd_temperature=$kd_temp teacher_output_temp=3.0 started at $(date -Is)"
run_variant "$variant" \
  --lambda-kd 1.0 \
  --lambda-d 0.1 \
  --lambda-adv 0.001 \
  --lambda-cwd-fea 50.0 \
  --lambda-cwd-logit 3.0 \
  --teacher-output-temp 3.0 \
  --kd-temperature "$kd_temp" \
  "${extra_args[@]}"
echo "[kd-baselines-npu] Phase M2 variant=$variant finished at $(date -Is)"
