#!/usr/bin/env bash
set -euo pipefail
export MAX_ITERATIONS=${MAX_ITERATIONS:-2}
export BATCH_SIZE=${BATCH_SIZE:-2}
export WORKERS=${WORKERS:-0}
export LOG_ITER=${LOG_ITER:-1}
export SAVE_PER_ITERS=${SAVE_PER_ITERS:-1000000}
export VAL_PER_ITERS=${VAL_PER_ITERS:-1000000}
export SAVE_ROOT=${SAVE_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/runs/covar_calib_conf_smoke"}
source "$(dirname "${BASH_SOURCE[0]}")/common_voc_cirkdv2.sh"
run_variant "smoke_calib_gamma0_tmax4" \
  --skip-val \
  --covar-temp-mode calib_conf \
  --covar-ref-temp 1.0 \
  --covar-temp-base 1.0 \
  --covar-temp-min 1.0 \
  --covar-temp-max 4.0 \
  --covar-kd-temp-power 0.0 \
  --covar-grad-eta 0.6 \
  --covar-grad-max-iter 8 \
  --covar-newton-hessian-eps 1e-5 \
  --covar-grad-converge-thresh 0.01 \
  --covar-grad-detail-interval 1
