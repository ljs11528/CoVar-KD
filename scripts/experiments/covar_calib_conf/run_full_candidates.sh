#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
cd "$ROOT_DIR"

export MAX_ITERATIONS=${MAX_ITERATIONS:-80000}
export VAL_PER_ITERS=${VAL_PER_ITERS:-800}
export SAVE_PER_ITERS=${SAVE_PER_ITERS:-800}
export SAVE_ROOT=${SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_calib_conf_full"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stage 1 selected 04 first. Keep 03 as the second calibrated full-run ablation.
bash "$SCRIPT_DIR/04_calib_gamma1_tmax4.sh"
bash "$SCRIPT_DIR/03_calib_gamma0_tmax4.sh"
