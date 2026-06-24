#!/usr/bin/env bash
# Phase B2: Adaptive anchor with low-confidence (T_out=3.0) teacher.
# anchor-mode=adaptive_mc, power=1.0, t_min=0.5, teacher-output-temp=3.0
# Expected: mean(mc)≈0.73 → anchor_q≈0.73 → ~73% sharpen, ~27% smooth
# Tests whether adaptive anchor generalizes across teacher quality.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common_voc_lowconf_teacher.sh"
run_variant "B2_adaptive_anchor_mc_power1_tmin0.5_Tout3.0" \
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
