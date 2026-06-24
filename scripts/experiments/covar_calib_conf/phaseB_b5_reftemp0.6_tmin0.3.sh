#!/usr/bin/env bash
# Phase B5: Adaptive anchor + ref_temp=0.6 to match newton's T_mean=0.6.
# Lowering ref_temp shifts the entire calibration baseline from T=1.0 to T=0.6.
# With anchor_q≈0.97, ~97% of pixels get T ≤ 0.6 (sharpen or identity at 0.6).
# Expected: T_mean drops from ~0.74 (B1/B3) to ~0.55-0.65 (matching newton).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common_voc_cirkdv2.sh"
run_variant "B5_adaptive_ref0.6_tmin0.3" \
  --covar-temp-mode calib_conf \
  --covar-calib-centered \
  --covar-calib-anchor-mode adaptive_mc \
  --covar-calib-anchor-power 1.0 \
  --covar-calib-anchor-alpha 1.0 \
  --covar-ref-temp 0.6 \
  --covar-temp-base 0.6 \
  --covar-temp-min 0.3 \
  --covar-temp-max 4.0 \
  --covar-kd-temp-power 0.0 \
  --covar-grad-eta 0.6 \
  --covar-grad-max-iter 8 \
  --covar-newton-hessian-eps 1e-5 \
  --covar-grad-converge-thresh 0.01 \
  --covar-grad-detail-interval 100
