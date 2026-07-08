#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_voc_cirkdv2_npu.sh"

PHASEG_VARIANTS=${PHASEG_VARIANTS:-"off confidence variance full"}

if [[ "${SKIP_VAL:-0}" == "1" ]]; then
  COMMON_ARGS+=(--skip-val)
fi

variant_log_file() {
  local variant="$1"
  find "$SAVE_ROOT/$variant" -maxdepth 1 -name '*_log.txt' -print -quit 2>/dev/null || true
}

variant_complete() {
  local variant="$1"
  local log_file
  log_file="$(variant_log_file "$variant")"
  [[ -n "$log_file" ]] && grep -q "Iters: ${MAX_ITERATIONS}/${MAX_ITERATIONS}" "$log_file" && grep -q "Total training time" "$log_file"
}

COVAR_NEWTON_ARGS=(
  --teacher-output-temp 3.0
  --covar-temp-mode newton
  --covar-temp-base 1.0
  --covar-temp-min 0.5
  --covar-temp-max 8.0
  --covar-kd-temp-power 2.0
  --covar-grad-eta 0.6
  --covar-grad-max-iter 8
  --covar-newton-hessian-eps 1e-5
  --covar-newton-max-step 0.25
  --covar-grad-converge-thresh 0.01
  --covar-grad-detail-interval 100
)

run_no_covar() {
  local variant="phaseG_triage_no_covar_tout3_seed${SEED}"
  if variant_complete "$variant"; then
    echo "[covar-npu] skip complete $variant"
    return 0
  fi
  run_variant "$variant" \
    --teacher-output-temp 3.0 \
    --no-covar
}

run_confidence_only() {
  local variant="phaseG_triage_covar_confidence_only_tout3_seed${SEED}"
  if variant_complete "$variant"; then
    echo "[covar-npu] skip complete $variant"
    return 0
  fi
  run_variant "$variant" \
    "${COVAR_NEWTON_ARGS[@]}" \
    --covar-reliability-mode confidence \
    --covar-a 0
}

run_variance_only() {
  local variant="phaseG_triage_covar_variance_only_tout3_seed${SEED}"
  if variant_complete "$variant"; then
    echo "[covar-npu] skip complete $variant"
    return 0
  fi
  run_variant "$variant" \
    "${COVAR_NEWTON_ARGS[@]}" \
    --covar-reliability-mode variance
}

run_full() {
  local variant="phaseG_triage_covar_full_tout3_seed${SEED}"
  if variant_complete "$variant"; then
    echo "[covar-npu] skip complete $variant"
    return 0
  fi
  run_variant "$variant" \
    "${COVAR_NEWTON_ARGS[@]}" \
    --covar-reliability-mode full
}

echo "[covar-npu] Phase G component ablation triage started at $(date -Is)"
echo "[covar-npu] variants=$PHASEG_VARIANTS"
echo "[covar-npu] save_root=$SAVE_ROOT"

for variant in $PHASEG_VARIANTS; do
  case "$variant" in
    off|no_covar)
      echo "[covar-npu] >>> no-covar at $(date -Is)"
      run_no_covar
      echo "[covar-npu] <<< no-covar done at $(date -Is)"
      ;;
    confidence|confidence_only)
      echo "[covar-npu] >>> confidence-only at $(date -Is)"
      run_confidence_only
      echo "[covar-npu] <<< confidence-only done at $(date -Is)"
      ;;
    variance|variance_only)
      echo "[covar-npu] >>> variance-only at $(date -Is)"
      run_variance_only
      echo "[covar-npu] <<< variance-only done at $(date -Is)"
      ;;
    full|covar)
      echo "[covar-npu] >>> full CoVar at $(date -Is)"
      run_full
      echo "[covar-npu] <<< full CoVar done at $(date -Is)"
      ;;
    *)
      echo "[covar-npu] unknown PHASEG_VARIANTS entry: $variant" >&2
      exit 2
      ;;
  esac
done

REPORT_DIR=${REPORT_DIR:-"$ROOT_DIR/reports"}
REPORT_PATH=${REPORT_PATH:-"$REPORT_DIR/$(date +%F)_phaseG_component_ablation_triage.md"}
mkdir -p "$REPORT_DIR"

"$PYTHON" "$SCRIPT_DIR/summarize_phaseG_component_ablation.py" \
  --save-root "$SAVE_ROOT" \
  --report "$REPORT_PATH" \
  --max-iterations "$MAX_ITERATIONS" \
  --seed "$SEED"

echo "[covar-npu] wrote report: $REPORT_PATH"
echo "[covar-npu] Phase G component ablation triage finished at $(date -Is)"
