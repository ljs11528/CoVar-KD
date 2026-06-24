#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common_voc_cirkdv2.sh"
run_variant "02_old_newton" \
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
