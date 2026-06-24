#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common_voc_cirkdv2.sh"
run_variant "05_calib_gamma0_tmax8" \
  --covar-temp-mode calib_conf \
  --covar-ref-temp 1.0 \
  --covar-temp-base 1.0 \
  --covar-temp-min 1.0 \
  --covar-temp-max 8.0 \
  --covar-kd-temp-power 0.0 \
  --covar-grad-eta 0.6 \
  --covar-grad-max-iter 8 \
  --covar-newton-hessian-eps 1e-5 \
  --covar-grad-converge-thresh 0.01 \
  --covar-grad-detail-interval 100
