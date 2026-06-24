#!/usr/bin/env bash
# Phase B3: Adaptive anchor + lower t_min (0.3) + alpha=1.0.
# Hypothesis: lower t_min reduces unreachable rate in sharpen direction,
# allowing more pixels to reach strong sharpening temperatures.
# mc≈0.97 → anchor_q≈0.97 → ~97% pixels directed toward sharpen.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common_voc_cirkdv2.sh"
run_variant "B3_adaptive_a1_tmin0.3" \
  --covar-temp-mode calib_conf \
  --covar-calib-centered \
  --covar-calib-anchor-mode adaptive_mc \
  --covar-calib-anchor-power 1.0 \
  --covar-calib-anchor-alpha 1.0 \
  --covar-ref-temp 1.0 \
  --covar-temp-base 1.0 \
  --covar-temp-min 0.3 \
  --covar-temp-max 4.0 \
  --covar-kd-temp-power 0.0 \
  --covar-grad-eta 0.6 \
  --covar-grad-max-iter 8 \
  --covar-newton-hessian-eps 1e-5 \
  --covar-grad-converge-thresh 0.01 \
  --covar-grad-detail-interval 100
