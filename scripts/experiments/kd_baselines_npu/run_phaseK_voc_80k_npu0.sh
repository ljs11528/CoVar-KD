#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common_voc_kd_npu.sh"

variant_complete() {
  local variant="$1"
  local log_file="$LOG_ROOT/$variant/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"
  [[ -f "$log_file" ]] \
    && grep -q "Iters: ${MAX_ITERATIONS}/${MAX_ITERATIONS}" "$log_file" \
    && grep -q "Total training time:" "$log_file"
}

run_or_skip() {
  local variant="$1"
  shift

  if variant_complete "$variant"; then
    echo "[kd-baselines-npu] skip complete variant=$variant"
    return 0
  fi
  run_variant "$variant" "$@"
}

echo "[kd-baselines-npu] Phase K NPU0 queue started at $(date -Is)"

run_or_skip "cwd_80k" \
  --lambda-kd 1.0 \
  --lambda-d 0.1 \
  --lambda-adv 0.001 \
  --lambda-cwd-fea 50.0 \
  --lambda-cwd-logit 3.0

run_or_skip "ifvd_80k" \
  --lambda-kd 1.0 \
  --lambda-d 0.1 \
  --lambda-adv 0.001 \
  --lambda-ifv 20.0

echo "[kd-baselines-npu] Phase K NPU0 queue finished at $(date -Is)"
