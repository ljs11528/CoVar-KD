#!/usr/bin/env bash
# Diagnostic: centered bidirectional calibration with T_min=0.5
# Tests whether allowing sharpening (T<1) for reliable pixels improves over smoothing-only calib_conf.
set -euo pipefail

# Must be set BEFORE sourcing common_voc_cirkdv2.sh which uses these in COMMON_ARGS
MAX_ITERATIONS=5000
VAL_PER_ITERS=800
SAVE_PER_ITERS=800

source "$(dirname "${BASH_SOURCE[0]}")/common_voc_cirkdv2.sh"

run_variant "diag_centered_tmin0.5" \
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
