#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${ROOT_DIR:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
cd "$ROOT_DIR"

ASCEND_ENV_SH=${ASCEND_ENV_SH:-"/usr/local/Ascend/cann-8.5.0/set_env.sh"}
if [[ -f "$ASCEND_ENV_SH" ]]; then
  # shellcheck source=/dev/null
  source "$ASCEND_ENV_SH"
fi

PYTHON=${PYTHON:-"/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python"}
if [[ ! -x "$PYTHON" && -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
fi

ASCEND_DEVICES=${ASCEND_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-"0,1"}}
export ASCEND_RT_VISIBLE_DEVICES="$ASCEND_DEVICES"
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-"$ASCEND_DEVICES"}
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -r -a _npu_devices <<< "$ASCEND_DEVICES"
  NPROC_PER_NODE=${#_npu_devices[@]}
fi

MASTER_PORT=${MASTER_PORT:-29660}
DATA_DIR=${DATA_DIR:-"$ROOT_DIR/dataset/VOCAug/"}
SAVE_ROOT=${SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseF_psp_mbv3small"}
TEACHER_PRETRAINED=${TEACHER_PRETRAINED:-"$ROOT_DIR/data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth"}
STUDENT_PRETRAINED_BASE=${STUDENT_PRETRAINED_BASE:-"$ROOT_DIR/data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth"}

MAX_ITERATIONS=${MAX_ITERATIONS:-80000}
BATCH_SIZE=${BATCH_SIZE:-16}
WORKERS=${WORKERS:-8}
LOG_ITER=${LOG_ITER:-20}
SAVE_PER_ITERS=${SAVE_PER_ITERS:-800}
VAL_PER_ITERS=${VAL_PER_ITERS:-800}
TOPK_CHECKPOINTS=${TOPK_CHECKPOINTS:-5}
SEED=${SEED:-1234}
PHASEF_VARIANTS=${PHASEF_VARIANTS:-"off on"}

COMMON_ARGS=(
  --device-type npu
  --seed "$SEED"
  --teacher-model deeplabv3
  --student-model psp_mobile
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

run_variant() {
  local variant_name="$1"
  shift

  mkdir -p "$SAVE_ROOT"
  echo "[covar-npu] $(date -Is) variant=$variant_name"
  echo "[covar-npu] root=$ROOT_DIR"
  echo "[covar-npu] python=$PYTHON"
  echo "[covar-npu] devices=$ASCEND_RT_VISIBLE_DEVICES nproc=$NPROC_PER_NODE master_port=$MASTER_PORT"
  echo "[covar-npu] save_dir=$SAVE_ROOT/$variant_name"
  echo "[covar-npu] max_iterations=$MAX_ITERATIONS batch_size=$BATCH_SIZE workers=$WORKERS seed=$SEED skip_val=${SKIP_VAL:-0}"

  "$PYTHON" -m torch.distributed.run \
    --nproc-per-node="$NPROC_PER_NODE" \
    --master-port="$MASTER_PORT" \
    train_cirkdv2.py \
    "${COMMON_ARGS[@]}" \
    --save-dir "$SAVE_ROOT" \
    --save-dir-name "$variant_name" \
    "$@"
}

run_no_covar() {
  local variant="phaseF_psp_mbv3small_no_covar_tout3_seed${SEED}"
  if variant_complete "$variant"; then
    echo "[covar-npu] skip complete $variant"
    return 0
  fi
  run_variant "$variant" \
    --teacher-output-temp 3.0 \
    --no-covar
}

run_covar() {
  local variant="phaseF_psp_mbv3small_covar_newton_gamma2_tout3_seed${SEED}"
  if variant_complete "$variant"; then
    echo "[covar-npu] skip complete $variant"
    return 0
  fi
  run_variant "$variant" \
    --teacher-output-temp 3.0 \
    --covar-temp-mode newton \
    --covar-temp-base 1.0 \
    --covar-temp-min 0.5 \
    --covar-temp-max 8.0 \
    --covar-kd-temp-power 2.0 \
    --covar-grad-eta 0.6 \
    --covar-grad-max-iter 8 \
    --covar-newton-hessian-eps 1e-5 \
    --covar-newton-max-step 0.25 \
    --covar-grad-converge-thresh 0.01 \
    --covar-grad-detail-interval 100
}

echo "[covar-npu] Phase F PSP-MobileNetV3-Small Tout=3.0 started at $(date -Is)"
echo "[covar-npu] variants=$PHASEF_VARIANTS"
echo "[covar-npu] save_root=$SAVE_ROOT"

for variant in $PHASEF_VARIANTS; do
  case "$variant" in
    off)
      echo "[covar-npu] >>> no-covar at $(date -Is)"
      run_no_covar
      echo "[covar-npu] <<< no-covar done at $(date -Is)"
      ;;
    on|covar)
      echo "[covar-npu] >>> covar at $(date -Is)"
      run_covar
      echo "[covar-npu] <<< covar done at $(date -Is)"
      ;;
    *)
      echo "[covar-npu] unknown PHASEF_VARIANTS entry: $variant" >&2
      exit 2
      ;;
  esac
done

echo "[covar-npu] Phase F PSP-MobileNetV3-Small Tout=3.0 finished at $(date -Is)"

