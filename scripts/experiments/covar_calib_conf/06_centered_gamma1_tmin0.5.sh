#!/usr/bin/env bash
# P0: Centered bidirectional calibration with gamma=1.0 (temperature compensation enabled).
# t_min=0.5 allows sharpening (T<1) for reliable pixels; t_max=4.0 smooths unreliable pixels.
# Expected to be the best variant based on 5k diagnostic (+0.040 vs old newton at step 4800).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common_voc_cirkdv2.sh"
run_variant "06_centered_gamma1_tmin0.5" \
  --covar-temp-mode calib_conf \
  --covar-calib-centered \
  --covar-ref-temp 1.0 \
  --covar-temp-base 1.0 \
  --covar-temp-min 0.5 \
  --covar-temp-max 4.0 \
  --covar-kd-temp-power 1.0 \
  --covar-grad-eta 0.6 \
  --covar-grad-max-iter 8 \
  --covar-newton-hessian-eps 1e-5 \
  --covar-grad-converge-thresh 0.01 \
  --covar-grad-detail-interval 100
