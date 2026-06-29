#!/usr/bin/env bash
# Phase C4: low-confidence teacher control without CoVar temperature maps.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common_voc_cirkdv2_npu.sh"

run_variant "phaseC_lc_no_covar_tout3" \
  --teacher-output-temp 3.0 \
  --no-covar
