#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
GPU_IDS="${GPU_IDS:-0,1}"
MAX_ITERATIONS=20000
BATCH_SIZE=16
WORKERS="${WORKERS:-4}"
SEED="${SEED:-1234}"
VARIANT="task_aligned_region_r8_20k_seed${SEED}"
SAVE_DIR="${SAVE_ROOT:-runs/covar_match/P4a_task_aligned_region/checkpoints}/$VARIANT"
LOG_DIR="${LOG_ROOT:-runs/covar_match/P4a_task_aligned_region/logs}/$VARIANT"
LOG_FILE="$LOG_DIR/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"
STATE_FILE="$SAVE_DIR/training_state_latest.pth"
TEACHER_PRETRAINED="${TEACHER_PRETRAINED:-data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth}"
STUDENT_PRETRAINED_BASE="${STUDENT_PRETRAINED_BASE:-data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth}"

IFS=',' read -r -a DEVICES <<< "$GPU_IDS"
if [[ "${#DEVICES[@]}" -ne 2 ]]; then
  echo "P4a formal protocol requires exactly two CUDA devices" >&2
  exit 2
fi

mkdir -p "$SAVE_DIR" "$LOG_DIR"
if [[ -f "$STATE_FILE" && -f "$LOG_FILE" ]] \
   && grep -Fq "Iters: 20000/20000" "$LOG_FILE" \
   && grep -Fq "P4a stats:" "$LOG_FILE" \
   && grep -Fq "Overall validation pixAcc:" "$LOG_FILE" \
   && grep -Fq "Total training time:" "$LOG_FILE"; then
  echo "[P4a] skip complete variant=$VARIANT"
  exit 0
fi
if [[ -f "$STATE_FILE" || -f "$LOG_FILE" ]]; then
  echo "[P4a] partial variant exists; refusing to overwrite: $VARIANT" >&2
  exit 3
fi

echo "[P4a] start variant=$VARIANT"
echo "[P4a] selector=hard_argmax temperatures=0.5,0.75,1.0,1.25,1.5,2.0 region=8 min_valid=16 exact_tie_fallback=1.5"
CUDA_VISIBLE_DEVICES="$GPU_IDS" OMP_NUM_THREADS=4 "$PYTHON" \
  -m torch.distributed.run --standalone --nproc-per-node=2 \
  train_kd.py \
  --device-type cuda --seed "$SEED" \
  --teacher-model deeplabv3 --teacher-backbone resnet101 \
  --student-model deeplabv3_mobilenet_ssseg \
  --student-backbone mobilenetv3_small \
  --dataset voc --data dataset/VOCAug/ \
  --batch-size "$BATCH_SIZE" --crop-size 512 512 --workers "$WORKERS" \
  --lr 0.02 --max-iterations "$MAX_ITERATIONS" \
  --lambda-kd 1.0 --kd-loss-mode task_aligned_region \
  --kd-temperature 1.5 --teacher-output-temp 1.0 \
  --log-iter 20 --save-per-iters 4000 \
  --val-per-iters "$MAX_ITERATIONS" \
  --teacher-pretrained "$TEACHER_PRETRAINED" \
  --student-pretrained-base "$STUDENT_PRETRAINED_BASE" \
  --save-dir "$SAVE_DIR" --log-dir "$LOG_DIR"
echo "[P4a] finish variant=$VARIANT"
