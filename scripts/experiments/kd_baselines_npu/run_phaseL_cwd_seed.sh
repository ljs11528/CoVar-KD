#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common_voc_kd_npu.sh"

VARIANT="cwd_80k_seed${SEED}"
VARIANT_LOG="$LOG_ROOT/$VARIANT/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"

if [[ -f "$VARIANT_LOG" ]] \
  && grep -q "Iters: ${MAX_ITERATIONS}/${MAX_ITERATIONS}" "$VARIANT_LOG" \
  && grep -q "Total training time:" "$VARIANT_LOG"; then
  echo "[kd-baselines-npu] skip complete variant=$VARIANT"
  exit 0
fi

echo "[kd-baselines-npu] Phase L CWD seed=$SEED started at $(date -Is)"
run_variant "$VARIANT" \
  --lambda-kd 1.0 \
  --lambda-d 0.1 \
  --lambda-adv 0.001 \
  --lambda-cwd-fea 50.0 \
  --lambda-cwd-logit 3.0
echo "[kd-baselines-npu] Phase L CWD seed=$SEED finished at $(date -Is)"
