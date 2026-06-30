#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${ROOT_DIR:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
MAX_ITERATIONS=${MAX_ITERATIONS:-80000}
SAVE_ROOT=${SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseC_80k"}
MASTER_PORT=${MASTER_PORT:-29630}
export ROOT_DIR MAX_ITERATIONS SAVE_ROOT MASTER_PORT

source "$SCRIPT_DIR/common_voc_cirkdv2_npu.sh"

VARIANTS=(
  phaseC_lc_newton_gamma2_repro
  phaseC_lc_no_covar_tout3
)

echo "[covar-npu] Phase C 80k pair started at $(date -Is)"
echo "[covar-npu] variants=${VARIANTS[*]}"

for variant in "${VARIANTS[@]}"; do
  echo "[covar-npu] >>> start $variant at $(date -Is)"
  bash "$SCRIPT_DIR/${variant}.sh"
  echo "[covar-npu] <<< done $variant at $(date -Is)"
done

REPORT_DIR=${REPORT_DIR:-"$ROOT_DIR/reports"}
REPORT_PATH=${REPORT_PATH:-"$REPORT_DIR/$(date +%F)_phaseC_npu_80k_pair.md"}
mkdir -p "$REPORT_DIR"

"$PYTHON" "$SCRIPT_DIR/summarize_phaseC_results.py" \
  --save-root "$SAVE_ROOT" \
  --report "$REPORT_PATH" \
  --max-iterations "$MAX_ITERATIONS" \
  --variants "${VARIANTS[@]}"

echo "[covar-npu] wrote report: $REPORT_PATH"

if [[ "${AUTO_GIT_REPORT:-1}" == "1" ]]; then
  set +e
  git add "$REPORT_PATH"
  if git diff --cached --quiet -- "$REPORT_PATH"; then
    echo "[covar-npu] no report changes to commit"
  else
    git commit -m "Add Phase C 80k NPU pair report"
    push_target=${REPORT_PUSH_URL:-${REPORT_PUSH_REMOTE:-origin}}
    push_timeout=${GIT_PUSH_TIMEOUT:-60}
    GIT_TERMINAL_PROMPT=0 \
      GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new" \
      timeout "$push_timeout" git push "$push_target" "$(git branch --show-current)"
  fi
  git_status=$?
  set -e
  if [[ "$git_status" -ne 0 ]]; then
    echo "[covar-npu] warning: report git sync failed with status $git_status"
  fi
fi

echo "[covar-npu] Phase C 80k pair finished at $(date -Is)"
