#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"

# Reuse the repository's existing three-seed convention: 1234, 2025, 3407.
# Seed 1234 is complete; this runner intentionally launches only the two new seeds.
readonly NEW_SEEDS=(2025 3407)

for seed in "${NEW_SEEDS[@]}"; do
  echo "[paired-multiseed] seed=$seed fixed temperatures start"
  SEED="$seed" P1_TEMPERATURES="0.5 1.5" \
    bash scripts/experiments/covar_match/run_p1_teacher_only_temperature.sh

  echo "[paired-multiseed] seed=$seed P4a start"
  SEED="$seed" \
    bash scripts/experiments/covar_match/run_p4a_task_aligned_region.sh
done

echo "[paired-multiseed] all six new runs complete"
