#!/usr/bin/env bash
# P1: Centered bidirectional calibration with gamma=0.0 (no temperature compensation).
# Tests whether the T^gamma compensation term in KD loss is necessary when using centered calibration.
# All other parameters identical to 06_centered_gamma1_tmin0.5.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common_voc_cirkdv2.sh"
run_variant "07_centered_gamma0_tmin0.5" \
  --covar-temp-mode calib_conf \
  --covar-calib-centered \
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
