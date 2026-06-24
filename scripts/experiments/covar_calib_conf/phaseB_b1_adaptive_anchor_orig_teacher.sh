#!/usr/bin/env bash
# Phase B1: Adaptive anchor with original (high-confidence) teacher.
# anchor-mode=adaptive_mc, power=1.0, t_min=0.5
# Expected: mean(mc)≈0.97 → anchor_q≈0.97 → ~97% pixels sharpened
# Should behave similarly to newton (P2: 0.639) but with adaptive tuning.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common_voc_cirkdv2.sh"
run_variant "B1_adaptive_anchor_mc_power1_tmin0.5" \
  --covar-temp-mode calib_conf \
  --covar-calib-centered \
  --covar-calib-anchor-mode adaptive_mc \
  --covar-calib-anchor-power 1.0 \
  --covar-ref-temp 1.0 \
  --covar-temp-base 1.0 \
  --covar-temp-min 0.5 \
  --covar-temp-max 4.0 \
  --covar-kd-temp-power 0.0 \
  --covar-grad-eta 0.6 \
  --covar-grad-max-iter 8 \
  --covar-newton-hessian-eps 1e-5 \
  --covar-grad-converge-thresh 0.01 \
  --covar-grad-detail-interval 100
