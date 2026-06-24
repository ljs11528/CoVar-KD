#!/usr/bin/env bash
# Common config for WEAK TEACHER experiments.
# Teacher: RANDOMLY INITIALIZED (no pretrained backbone, no VOC finetune).
# Max confidence ~1/21≈0.048 — extreme low-confidence scenario.
# Hypothesis: centered calibration smoothing (T>1) should outperform newton sharpening (T<1)
# which amplifies random noise.
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
PYTHON=${PYTHON:-"$ROOT_DIR/.venv/bin/python"}
GPU_ID=${GPU_ID:-0}
MAX_ITERATIONS=${MAX_ITERATIONS:-80000}
VAL_PER_ITERS=${VAL_PER_ITERS:-800}
SAVE_PER_ITERS=${SAVE_PER_ITERS:-800}

DATA_DIR=${DATA_DIR:-"$ROOT_DIR/dataset/VOCAug/"}
SAVE_ROOT=${SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_calib_conf_weak_teacher"}

mkdir -p "$SAVE_ROOT"

COMMON_ARGS=(
  --teacher-model deeplabv3
  --student-model deeplabv3_mobilenet_ssseg
  --teacher-backbone resnet101
  --student-backbone mobilenetv3_small
  --dataset voc
  --data "$DATA_DIR"
  --batch-size 16
  --crop-size 512 512
  --workers 8
  --lr 0.02
  --max-iterations "$MAX_ITERATIONS"
  --lambda-kd 1.0
  --lambda-fitnet 10.0
  --lambda-minibatch-pixel 1.0
  --lambda-minibatch-channel 1.0
  --lambda-memory-pixel 0.1
  --lambda-memory-region 0.1
  --lambda-memory-channel 0.1
  --lambda-channel-kd 100.0
  --log-iter 10
  --save-per-iters "$SAVE_PER_ITERS"
  --val-per-iters "$VAL_PER_ITERS"
  --topk-checkpoints 5
  --student-pretrained-base "$ROOT_DIR/data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth"
  --gpu-id "$GPU_ID"
)

run_variant() {
  local variant_name="$1"
  shift
  local save_dir="$SAVE_ROOT/$variant_name"
  mkdir -p "$save_dir"
  echo "[weak-teacher] variant=$variant_name"
  echo "[weak-teacher] python=$PYTHON"
  echo "[weak-teacher] save_dir=$save_dir"
  "$PYTHON" train_cirkdv2.py \
    "${COMMON_ARGS[@]}" \
    --save-dir "$SAVE_ROOT" \
    --save-dir-name "$variant_name" \
    "$@"
}
