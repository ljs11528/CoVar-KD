#!/usr/bin/env bash
# Sequential runner for centered calibration 80k experiments.
# Runs P0 → P1 → P2 in order on a single GPU.
# Each variant runs full 80k iterations before the next begins.
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
cd "$ROOT_DIR"

export MAX_ITERATIONS=${MAX_ITERATIONS:-80000}
export VAL_PER_ITERS=${VAL_PER_ITERS:-800}
export SAVE_PER_ITERS=${SAVE_PER_ITERS:-800}
export SAVE_ROOT=${SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_calib_conf_centered_80k"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "[centered-80k] Starting sequential 80k training queue"
echo "[centered-80k] MAX_ITERATIONS=$MAX_ITERATIONS"
echo "[centered-80k] SAVE_ROOT=$SAVE_ROOT"
echo "[centered-80k] Start time: $(date -u)"
echo "============================================================"

# P0: centered calibration with gamma=1.0 (primary candidate)
echo ""
echo "=== [P0] 06_centered_gamma1_tmin0.5 ==="
bash "$SCRIPT_DIR/06_centered_gamma1_tmin0.5.sh"
echo "[centered-80k] P0 completed at $(date -u)"

# P1: centered calibration with gamma=0.0 (ablation)
echo ""
echo "=== [P1] 07_centered_gamma0_tmin0.5 ==="
bash "$SCRIPT_DIR/07_centered_gamma0_tmin0.5.sh"
echo "[centered-80k] P1 completed at $(date -u)"

# P2: old newton rerun (fair baseline)
echo ""
echo "=== [P2] 08_old_newton_rerun ==="
bash "$SCRIPT_DIR/08_old_newton_rerun.sh"
echo "[centered-80k] P2 completed at $(date -u)"

echo ""
echo "============================================================"
echo "[centered-80k] All 80k experiments complete!"
echo "[centered-80k] End time: $(date -u)"
echo "============================================================"
