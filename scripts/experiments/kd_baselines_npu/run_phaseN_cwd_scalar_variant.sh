#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common_voc_kd_npu.sh"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/phaseN_scalar_temperature_common.sh"

: "${PHASEN_KD_TEMP:?Set PHASEN_KD_TEMP=1.0 or 0.6}"

case "$PHASEN_KD_TEMP" in
  1.0)
    kd_temp="1.0"
    temp_label="1p0"
    ;;
  0.6)
    kd_temp="0.6"
    temp_label="0p6"
    ;;
  *)
    echo "[kd-baselines-npu] unsupported PHASEN_KD_TEMP=$PHASEN_KD_TEMP (expected 1.0 or 0.6)" >&2
    exit 2
    ;;
esac

if [[ "$SEED" != "1234" ]]; then
  echo "[kd-baselines-npu] unsupported Phase N seed=$SEED (expected 1234)" >&2
  exit 2
fi

case "$MAX_ITERATIONS" in
  20) budget_label="smoke" ;;
  80000) budget_label="80k" ;;
  *)
    echo "[kd-baselines-npu] unsupported Phase N budget MAX_ITERATIONS=$MAX_ITERATIONS (expected 20 or 80000)" >&2
    exit 2
    ;;
esac

variant="cwd_tout3_kdtemp${temp_label}_${budget_label}_seed${SEED}"
training_log="$LOG_ROOT/$variant/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"
expected_student_pretrained_base="$ROOT_DIR/data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth"

if [[ "$STUDENT_PRETRAINED_BASE" != "$expected_student_pretrained_base" ]]; then
  echo "[kd-baselines-npu] Phase N requires fresh ImageNet student init: $expected_student_pretrained_base" >&2
  exit 2
fi

expected_skip_val=False
expected_require_best=True
if [[ "${SKIP_VAL:-0}" == "1" ]]; then
  expected_skip_val=True
  expected_require_best=False
fi

# A completed run is immutable: repeated launch attempts must not overwrite it.
if phaseN_log_session_complete \
  "$training_log" "$MAX_ITERATIONS" "$kd_temp" "$SAVE_PER_ITERS" \
  "$VAL_PER_ITERS" "$expected_skip_val" "$expected_student_pretrained_base" \
  && phaseN_artifacts_complete "$SAVE_ROOT/$variant" "$expected_require_best"; then
  echo "[kd-baselines-npu] skip complete Phase N variant=$variant log=$training_log"
  exit 0
fi

extra_args=()
if [[ "${SKIP_VAL:-0}" == "1" ]]; then
  extra_args+=(--skip-val)
fi

echo "[kd-baselines-npu] Phase N variant=$variant kd_temperature=$kd_temp teacher_output_temp=3.0 started at $(date -Is)"
run_variant "$variant" \
  --lambda-kd 1.0 \
  --lambda-d 0.1 \
  --lambda-adv 0.001 \
  --lambda-cwd-fea 50.0 \
  --lambda-cwd-logit 3.0 \
  --teacher-output-temp 3.0 \
  --kd-temperature "$kd_temp" \
  "${extra_args[@]}"
echo "[kd-baselines-npu] Phase N variant=$variant finished at $(date -Is)"
