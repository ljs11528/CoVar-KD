#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
GPU_IDS="${GPU_IDS:-0,1}"
WORKERS="${WORKERS:-4}"
readonly MAX_ITERATIONS=80000
readonly BATCH_SIZE=16
readonly SEEDS=(1234 2025 3407)
readonly TEMPERATURES=(0.75 1.25 2.0)
readonly MILESTONES=(20000 40000 60000 80000)
readonly RUN_ROOT="${RUN_ROOT:-runs/covar_match/P7_fixed_temperature_response}"
readonly TEACHER_PRETRAINED="${TEACHER_PRETRAINED:-data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth}"
readonly STUDENT_PRETRAINED_BASE="${STUDENT_PRETRAINED_BASE:-data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth}"
readonly LOG_NAME=deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt

IFS=',' read -r -a DEVICES <<< "$GPU_IDS"
if [[ "${#DEVICES[@]}" -ne 2 ]]; then
  echo "P7 formal protocol requires exactly two CUDA devices" >&2
  exit 2
fi
if [[ ! -f "$TEACHER_PRETRAINED" || ! -f "$STUDENT_PRETRAINED_BASE" ]]; then
  echo "P7 requires the locked P6 teacher and student initialization checkpoints" >&2
  exit 2
fi

temperature_tag() {
  case "$1" in
    0.75) printf 'T0p75' ;;
    1.25) printf 'T1p25' ;;
    2.0) printf 'T2p0' ;;
    *) echo "unknown P7 temperature: $1" >&2; return 2 ;;
  esac
}

variant_for() {
  local temperature="$1"
  local seed="$2"
  printf 'fixed_%s_80k_seed%s' "$(temperature_tag "$temperature")" "$seed"
}

is_complete() {
  local temperature="$1"
  local save_dir="$2"
  local log_file="$3"
  [[ -f "$save_dir/training_state_latest.pth" && -f "$log_file" ]] || return 1
  grep -Fq "Iters: 80000/80000" "$log_file" || return 1
  [[ "$(grep -Fc 'Overall validation pixAcc:' "$log_file")" -eq 4 ]] || return 1
  grep -Fq "Total training time:" "$log_file" || return 1
  printf -v expected_temperature 'Teacher-only target T: %.4f' "$temperature"
  grep -Fq "$expected_temperature" "$log_file" || return 1
  local milestone padded
  for milestone in "${MILESTONES[@]}"; do
    printf -v padded '%06d' "$milestone"
    [[ -f "$save_dir/training_state_iter${padded}.pth" ]] || return 1
  done
}

run_one() {
  local seed="$1"
  local temperature="$2"
  local variant save_dir log_dir log_file
  variant="$(variant_for "$temperature" "$seed")"
  save_dir="$RUN_ROOT/checkpoints/$variant"
  log_dir="$RUN_ROOT/logs/$variant"
  log_file="$log_dir/$LOG_NAME"

  if is_complete "$temperature" "$save_dir" "$log_file"; then
    echo "[P7] skip complete seed=$seed temperature=$temperature variant=$variant"
    return 0
  fi
  if [[ -e "$save_dir" || -e "$log_dir" ]]; then
    echo "[P7] partial output exists; refusing overwrite/resume: $variant" >&2
    return 3
  fi
  mkdir -p "$save_dir" "$log_dir"

  echo "[P7] start seed=$seed temperature=$temperature variant=$variant fresh_init=true max_iterations=$MAX_ITERATIONS"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" OMP_NUM_THREADS=4 "$PYTHON" \
    -m torch.distributed.run --standalone --nproc-per-node=2 \
    train_kd.py \
    --device-type cuda --seed "$seed" \
    --teacher-model deeplabv3 --teacher-backbone resnet101 \
    --student-model deeplabv3_mobilenet_ssseg \
    --student-backbone mobilenetv3_small \
    --dataset voc --data dataset/VOCAug/ \
    --batch-size "$BATCH_SIZE" --crop-size 512 512 --workers "$WORKERS" \
    --lr 0.02 --max-iterations "$MAX_ITERATIONS" \
    --lambda-kd 1.0 --kd-loss-mode teacher_only \
    --kd-temperature "$temperature" --teacher-output-temp 1.0 \
    --log-iter 20 --save-per-iters 20000 --val-per-iters 20000 \
    --keep-checkpoint-iters "${MILESTONES[@]}" \
    --teacher-pretrained "$TEACHER_PRETRAINED" \
    --student-pretrained-base "$STUDENT_PRETRAINED_BASE" \
    --save-dir "$save_dir" --log-dir "$log_dir"

  if ! is_complete "$temperature" "$save_dir" "$log_file"; then
    echo "[P7] completion contract failed: $variant" >&2
    return 4
  fi
  echo "[P7] finish seed=$seed temperature=$temperature variant=$variant"
}

echo "[P7] protocol=P6_locked_fresh_80k seeds=1234,2025,3407 temperatures=0.75,1.25,2.0 milestones=20k,40k,60k,80k"
for seed in "${SEEDS[@]}"; do
  for temperature in "${TEMPERATURES[@]}"; do
    run_one "$seed" "$temperature"
  done
done
echo "[P7] all nine fixed-temperature runs complete"
