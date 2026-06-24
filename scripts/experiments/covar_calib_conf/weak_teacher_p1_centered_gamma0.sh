#!/usr/bin/env bash
# P1: Centered bidirectional calibration with gamma=0.0, WEAK TEACHER (no VOC finetune).
# Hypothesis: with low-confidence teacher, centered calibration's smoothing (T>1)
# should be beneficial, potentially outperforming newton's sharpen-only approach.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common_voc_weak_teacher.sh"
run_variant "p1_centered_gamma0" \
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
