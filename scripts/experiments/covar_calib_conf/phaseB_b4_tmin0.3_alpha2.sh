#!/usr/bin/env bash
# Phase B4: Adaptive anchor + t_min 0.3 + alpha=2.0 (steeper transition).
# alpha>1 reduces identity zone and creates sharper T differentiation.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common_voc_cirkdv2.sh"
run_variant "B4_adaptive_a1_tmin0.3_alpha2" \
  --covar-temp-mode calib_conf \
  --covar-calib-centered \
  --covar-calib-anchor-mode adaptive_mc \
  --covar-calib-anchor-power 1.0 \
  --covar-calib-anchor-alpha 2.0 \
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
