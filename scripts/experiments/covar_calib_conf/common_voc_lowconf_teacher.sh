#!/usr/bin/env bash
# Common config for LOW-CONFIDENCE TEACHER experiments.
# Uses VOC-fine-tuned DeepLabV3-ResNet101 teacher with temperature-scaled outputs.
# --teacher-output-temp 3.0 reduces mean max confidence from ~0.97 to ~0.60-0.75.
# This creates a medium-confidence scenario where centered calibration's smoothing
# capability should be beneficial, while newton's sharpening may amplify noise.
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
PYTHON=${PYTHON:-"$ROOT_DIR/.venv/bin/python"}
GPU_ID=${GPU_ID:-0}
MAX_ITERATIONS=${MAX_ITERATIONS:-80000}
VAL_PER_ITERS=${VAL_PER_ITERS:-800}
SAVE_PER_ITERS=${SAVE_PER_ITERS:-800}

TEACHER_OUTPUT_TEMP=${TEACHER_OUTPUT_TEMP:-3.0}
TEACHER_PRETRAINED=${TEACHER_PRETRAINED:-"$ROOT_DIR/data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth"}

DATA_DIR=${DATA_DIR:-"$ROOT_DIR/dataset/VOCAug/"}
SAVE_ROOT=${SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_calib_conf_lowconf_teacher"}

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
  --teacher-pretrained "$TEACHER_PRETRAINED"
  --teacher-output-temp "$TEACHER_OUTPUT_TEMP"
  --student-pretrained-base "$ROOT_DIR/data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth"
  --gpu-id "$GPU_ID"
)

run_variant() {
  local variant_name="$1"
  shift
  local save_dir="$SAVE_ROOT/$variant_name"
  mkdir -p "$save_dir"
  echo "[lowconf] variant=$variant_name"
  echo "[lowconf] teacher_output_temp=$TEACHER_OUTPUT_TEMP"
  echo "[lowconf] python=$PYTHON"
  echo "[lowconf] save_dir=$save_dir"
  "$PYTHON" train_cirkdv2.py \
    "${COMMON_ARGS[@]}" \
    --save-dir "$SAVE_ROOT" \
    --save-dir-name "$variant_name" \
    "$@"
}
