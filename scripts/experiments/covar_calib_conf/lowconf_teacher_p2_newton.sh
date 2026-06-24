#!/usr/bin/env bash
# P2: Old newton method, LOW-CONFIDENCE TEACHER (T_out=3.0).
# Newton's sharpen-only approach may amplify noise when teacher confidence is reduced.
# Compare against P1 (centered) to test if bidirectional calibration helps in this scenario.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common_voc_lowconf_teacher.sh"
run_variant "lc_p2_newton" \
  --covar-temp-mode newton \
  --covar-temp-base 1.0 \
  --covar-temp-min 0.5 \
  --covar-temp-max 8.0 \
  --covar-grad-eta 0.6 \
  --covar-grad-max-iter 8 \
  --covar-newton-hessian-eps 1e-5 \
  --covar-newton-max-step 0.25 \
  --covar-grad-converge-thresh 0.01 \
  --covar-grad-detail-interval 100
