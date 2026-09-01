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
readonly METHODS=(fixed_t0p5 fixed_t1p0 fixed_t1p5 p4a_r8)
readonly MILESTONES=(20000 40000 60000 80000)
readonly RUN_ROOT="${RUN_ROOT:-runs/covar_match/P6_full_schedule}"
readonly TEACHER_PRETRAINED="${TEACHER_PRETRAINED:-data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth}"
readonly STUDENT_PRETRAINED_BASE="${STUDENT_PRETRAINED_BASE:-data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth}"
readonly LOG_NAME=deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt

IFS=',' read -r -a DEVICES <<< "$GPU_IDS"
if [[ "${#DEVICES[@]}" -ne 2 ]]; then
  echo "P6 formal protocol requires exactly two CUDA devices" >&2
  exit 2
fi
if [[ ! -f "$TEACHER_PRETRAINED" || ! -f "$STUDENT_PRETRAINED_BASE" ]]; then
  echo "P6 requires the locked teacher and student initialization checkpoints" >&2
  exit 2
fi

variant_for() {
  local method="$1"
  local seed="$2"
  if [[ "$method" == fixed_t0p5 && "$seed" == 1234 ]]; then
    printf 'fixed_T0p5_80k_seed1234_retry1'
    return 0
  fi
  case "$method" in
    fixed_t0p5) printf 'fixed_T0p5_80k_seed%s' "$seed" ;;
    fixed_t1p0) printf 'fixed_T1p0_80k_seed%s' "$seed" ;;
    fixed_t1p5) printf 'fixed_T1p5_80k_seed%s' "$seed" ;;
    p4a_r8) printf 'p4a_r8_80k_seed%s' "$seed" ;;
    *) echo "unknown P6 method: $method" >&2; return 2 ;;
  esac
}

is_complete() {
  local method="$1"
  local save_dir="$2"
  local log_file="$3"
  [[ -f "$save_dir/training_state_latest.pth" && -f "$log_file" ]] || return 1
  grep -Fq "Iters: 80000/80000" "$log_file" || return 1
  [[ "$(grep -Fc 'Overall validation pixAcc:' "$log_file")" -eq 4 ]] || return 1
  grep -Fq "Total training time:" "$log_file" || return 1
  if [[ "$method" == p4a_r8 ]]; then
    grep -Fq "P4a stats:" "$log_file" || return 1
  fi
  local milestone padded
  for milestone in "${MILESTONES[@]}"; do
    printf -v padded '%06d' "$milestone"
    [[ -f "$save_dir/training_state_iter${padded}.pth" ]] || return 1
  done
}

run_one() {
  local seed="$1"
  local method="$2"
  local variant save_dir log_dir log_file temperature
  variant="$(variant_for "$method" "$seed")"
  save_dir="$RUN_ROOT/checkpoints/$variant"
  log_dir="$RUN_ROOT/logs/$variant"
  log_file="$log_dir/$LOG_NAME"

  if is_complete "$method" "$save_dir" "$log_file"; then
    echo "[P6] skip complete seed=$seed method=$method variant=$variant"
    return 0
  fi
  if [[ -e "$save_dir" || -e "$log_dir" ]]; then
    echo "[P6] partial output exists; refusing overwrite/resume: $variant" >&2
    return 3
  fi
  mkdir -p "$save_dir" "$log_dir"

  local -a kd_args
  case "$method" in
    fixed_t0p5) temperature=0.5 ;;
    fixed_t1p0) temperature=1.0 ;;
    fixed_t1p5) temperature=1.5 ;;
    p4a_r8) temperature=1.5 ;;
  esac
  if [[ "$method" == p4a_r8 ]]; then
    kd_args=(--kd-loss-mode task_aligned_region --kd-temperature 1.5)
  else
    kd_args=(--kd-loss-mode teacher_only --kd-temperature "$temperature")
  fi

  echo "[P6] start seed=$seed method=$method variant=$variant fresh_init=true max_iterations=$MAX_ITERATIONS"
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
    --lambda-kd 1.0 "${kd_args[@]}" --teacher-output-temp 1.0 \
    --log-iter 20 --save-per-iters 20000 --val-per-iters 20000 \
    --keep-checkpoint-iters "${MILESTONES[@]}" \
    --teacher-pretrained "$TEACHER_PRETRAINED" \
    --student-pretrained-base "$STUDENT_PRETRAINED_BASE" \
    --save-dir "$save_dir" --log-dir "$log_dir"

  if ! is_complete "$method" "$save_dir" "$log_file"; then
    echo "[P6] completion contract failed: $variant" >&2
    return 4
  fi
  echo "[P6] finish seed=$seed method=$method variant=$variant"
}

echo "[P6] protocol=fresh_80k seeds=1234,2025,3407 methods=T0.5,T1.0,T1.5,P4a_r8 milestones=20k,40k,60k,80k"
echo "[P6] excluded_infrastructure_attempt=fixed_T0p5_80k_seed1234 stopped_at=50600 replacement=fixed_T0p5_80k_seed1234_retry1"
for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    run_one "$seed" "$method"
  done
done
echo "[P6] all twelve full-schedule runs complete"
