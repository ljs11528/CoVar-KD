#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common_voc_kd_npu.sh"

run_variant "kdonly_20k" \
  --lambda-kd 1.0

run_variant "cwd_20k" \
  --lambda-kd 1.0 \
  --lambda-d 0.1 \
  --lambda-adv 0.001 \
  --lambda-cwd-fea 50.0 \
  --lambda-cwd-logit 3.0

run_variant "skd_20k" \
  --lambda-kd 1.0 \
  --lambda-d 0.1 \
  --lambda-adv 0.001 \
  --lambda-skd 10.0

run_variant "ifvd_20k" \
  --lambda-kd 1.0 \
  --lambda-d 0.1 \
  --lambda-adv 0.001 \
  --lambda-ifv 20.0

run_variant "fitnet_20k" \
  --lambda-kd 1.0 \
  --lambda-fitnet 10.0

run_variant "at_20k" \
  --lambda-kd 1.0 \
  --lambda-at 10000.0

run_variant "dsd_20k" \
  --lambda-psd 1000.0 \
  --lambda-csd 10.0
