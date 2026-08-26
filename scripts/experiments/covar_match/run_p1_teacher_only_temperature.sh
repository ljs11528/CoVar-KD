#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
GPU_IDS="${GPU_IDS:-0,1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-1234}"
TEMPERATURES="${P1_TEMPERATURES:-0.5 0.75 1.0 1.5 2.0}"
SAVE_ROOT="${SAVE_ROOT:-runs/covar_match/P1_teacher_only_temperature/checkpoints}"
LOG_ROOT="${LOG_ROOT:-runs/covar_match/P1_teacher_only_temperature/logs}"
TEACHER_PRETRAINED="${TEACHER_PRETRAINED:-data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth}"
STUDENT_PRETRAINED_BASE="${STUDENT_PRETRAINED_BASE:-data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth}"

IFS=',' read -r -a DEVICES <<< "$GPU_IDS"
if [[ "${#DEVICES[@]}" -ne 2 ]]; then
  echo "P1 formal protocol requires exactly two CUDA devices" >&2
  exit 2
fi

temperature_label() {
  case "$1" in
    0.5) echo "T0p5" ;;
    0.75) echo "T0p75" ;;
    1|1.0) echo "T1p0" ;;
    1.5) echo "T1p5" ;;
    2|2.0) echo "T2p0" ;;
    *)
      echo "unsupported P1 temperature: $1" >&2
      return 2
      ;;
  esac
}

for temperature in $TEMPERATURES; do
  label="$(temperature_label "$temperature")"
  variant="${label}_20k_seed${SEED}"
  save_dir="$SAVE_ROOT/$variant"
  log_dir="$LOG_ROOT/$variant"
  log_file="$log_dir/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"
  state_file="$save_dir/training_state_latest.pth"

  mkdir -p "$save_dir" "$log_dir"
  if [[ -f "$state_file" && -f "$log_file" ]] \
     && grep -Fq "Iters: ${MAX_ITERATIONS}/${MAX_ITERATIONS}" "$log_file" \
     && grep -Fq "Overall validation pixAcc:" "$log_file" \
     && grep -Fq "Total training time:" "$log_file"; then
    echo "[P1] skip complete variant=$variant"
    continue
  fi
  if [[ -f "$state_file" || -f "$log_file" ]]; then
    echo "[P1] partial variant exists; refusing to overwrite: $variant" >&2
    exit 3
  fi

  keep_args=()
  if [[ "$temperature" == "1" || "$temperature" == "1.0" ]]; then
    keep_args=(--keep-checkpoint-iters 4000 12000 20000)
  fi

  echo "[P1] start variant=$variant temperature=$temperature"
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
    --lambda-kd 1.0 --kd-loss-mode teacher_only \
    --kd-temperature "$temperature" --teacher-output-temp 1.0 \
    --log-iter 20 --save-per-iters 4000 \
    --val-per-iters "$MAX_ITERATIONS" \
    --teacher-pretrained "$TEACHER_PRETRAINED" \
    --student-pretrained-base "$STUDENT_PRETRAINED_BASE" \
    --save-dir "$save_dir" --log-dir "$log_dir" \
    "${keep_args[@]}"
  echo "[P1] finish variant=$variant"
done
