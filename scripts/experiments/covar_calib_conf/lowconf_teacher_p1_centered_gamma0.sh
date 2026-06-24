#!/usr/bin/env bash
# P1: Centered bidirectional calibration gamma=0.0, LOW-CONFIDENCE TEACHER (T_out=3.0).
# Hypothesis: with reduced teacher confidence, centered calibration's smoothing (T>1)
# should provide an advantage over newton's sharpen-only approach.
# Teacher output temperature 3.0 reduces mc from ~0.97 to ~0.6-0.75.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common_voc_lowconf_teacher.sh"
run_variant "lc_p1_centered_gamma0" \
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
