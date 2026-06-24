#!/usr/bin/env bash
set -euo pipefail
export MAX_ITERATIONS=${MAX_ITERATIONS:-20000}
export VAL_PER_ITERS=${VAL_PER_ITERS:-800}
export SAVE_PER_ITERS=${SAVE_PER_ITERS:-800}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/01_baseline_no_covar.sh"
bash "$SCRIPT_DIR/02_old_newton.sh"
bash "$SCRIPT_DIR/03_calib_gamma0_tmax4.sh"
bash "$SCRIPT_DIR/04_calib_gamma1_tmax4.sh"
bash "$SCRIPT_DIR/05_calib_gamma0_tmax8.sh"
