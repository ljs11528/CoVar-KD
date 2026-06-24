#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
cd "$ROOT_DIR"

PYTHON=${PYTHON:-"$ROOT_DIR/.venv/bin/python"}
GPU_ID=${GPU_ID:-0}
DATA_DIR=${DATA_DIR:-"$ROOT_DIR/dataset/VOCAug/"}
SAVE_ROOT=${SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_calib_conf"}
TEACHER_PRETRAINED=${TEACHER_PRETRAINED:-"$ROOT_DIR/data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth"}
STUDENT_PRETRAINED_BASE=${STUDENT_PRETRAINED_BASE:-"$ROOT_DIR/data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth"}

MAX_ITERATIONS=${MAX_ITERATIONS:-80000}
BATCH_SIZE=${BATCH_SIZE:-16}
WORKERS=${WORKERS:-8}
LOG_ITER=${LOG_ITER:-10}
SAVE_PER_ITERS=${SAVE_PER_ITERS:-800}
VAL_PER_ITERS=${VAL_PER_ITERS:-800}
TOPK_CHECKPOINTS=${TOPK_CHECKPOINTS:-5}

COMMON_ARGS=(
  --teacher-model deeplabv3
  --student-model deeplabv3_mobilenet_ssseg
  --teacher-backbone resnet101
  --student-backbone mobilenetv3_small
  --dataset voc
  --data "$DATA_DIR"
  --batch-size "$BATCH_SIZE"
  --crop-size 512 512
  --workers "$WORKERS"
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
  --log-iter "$LOG_ITER"
  --save-per-iters "$SAVE_PER_ITERS"
  --val-per-iters "$VAL_PER_ITERS"
  --topk-checkpoints "$TOPK_CHECKPOINTS"
  --teacher-pretrained "$TEACHER_PRETRAINED"
  --student-pretrained-base "$STUDENT_PRETRAINED_BASE"
)

run_variant() {
  local variant_name="$1"
  shift

  mkdir -p "$SAVE_ROOT"
  echo "[covar-calib-conf] variant=$variant_name"
  echo "[covar-calib-conf] python=$PYTHON"
  echo "[covar-calib-conf] save_dir=$SAVE_ROOT/$variant_name"

  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" train_cirkdv2.py \
    "${COMMON_ARGS[@]}" \
    --save-dir "$SAVE_ROOT" \
    --save-dir-name "$variant_name" \
    "$@"
}
