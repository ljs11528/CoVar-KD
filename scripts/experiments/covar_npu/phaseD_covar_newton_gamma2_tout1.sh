#!/usr/bin/env bash
# Phase D2: CoVar Newton on the original teacher logits with default T^2 KD scaling.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common_voc_cirkdv2_npu.sh"

run_variant "phaseD_covar_newton_gamma2_tout1" \
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
