#!/usr/bin/env bash
# Phase D1: standard CIRKD control with the original teacher logits.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common_voc_cirkdv2_npu.sh"

run_variant "phaseD_cirkd_no_covar_tout1" \
  --teacher-output-temp 1.0 \
  --no-covar
