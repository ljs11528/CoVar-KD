#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
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

MASTER_PORT=${MASTER_PORT:-29740}
DATA_DIR=${DATA_DIR:-"$ROOT_DIR/dataset/VOCAug/"}
SAVE_ROOT=${SAVE_ROOT:-"$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseJ_voc_20k"}
LOG_ROOT=${LOG_ROOT:-"$ROOT_DIR/runs/logs/kd_baselines_npu/phaseJ_voc_20k"}
TEACHER_PRETRAINED=${TEACHER_PRETRAINED:-"$ROOT_DIR/data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth"}
STUDENT_PRETRAINED_BASE=${STUDENT_PRETRAINED_BASE:-"$ROOT_DIR/data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth"}

MAX_ITERATIONS=${MAX_ITERATIONS:-20000}
BATCH_SIZE=${BATCH_SIZE:-16}
WORKERS=${WORKERS:-8}
LOG_ITER=${LOG_ITER:-20}
SAVE_PER_ITERS=${SAVE_PER_ITERS:-800}
VAL_PER_ITERS=${VAL_PER_ITERS:-800}
SEED=${SEED:-1234}

COMMON_ARGS=(
  --device-type npu
  --seed "$SEED"
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
  --log-iter "$LOG_ITER"
  --save-per-iters "$SAVE_PER_ITERS"
  --val-per-iters "$VAL_PER_ITERS"
  --teacher-pretrained "$TEACHER_PRETRAINED"
  --student-pretrained-base "$STUDENT_PRETRAINED_BASE"
)

run_variant() {
  local variant_name="$1"
  shift

  mkdir -p "$SAVE_ROOT/$variant_name" "$LOG_ROOT/$variant_name"
  echo "[kd-baselines-npu] $(date -Is) variant=$variant_name"
  echo "[kd-baselines-npu] root=$ROOT_DIR"
  echo "[kd-baselines-npu] python=$PYTHON"
  echo "[kd-baselines-npu] devices=$ASCEND_RT_VISIBLE_DEVICES nproc=$NPROC_PER_NODE master_port=$MASTER_PORT"
  echo "[kd-baselines-npu] save_dir=$SAVE_ROOT/$variant_name"
  echo "[kd-baselines-npu] log_dir=$LOG_ROOT/$variant_name"
  echo "[kd-baselines-npu] max_iterations=$MAX_ITERATIONS batch_size=$BATCH_SIZE workers=$WORKERS seed=$SEED"

  if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
    "$PYTHON" train_kd.py \
      "${COMMON_ARGS[@]}" \
      --save-dir "$SAVE_ROOT/$variant_name" \
      --log-dir "$LOG_ROOT/$variant_name" \
      "$@"
  else
    "$PYTHON" -m torch.distributed.run \
      --nproc-per-node="$NPROC_PER_NODE" \
      --master-port="$MASTER_PORT" \
      train_kd.py \
      "${COMMON_ARGS[@]}" \
      --save-dir "$SAVE_ROOT/$variant_name" \
      --log-dir "$LOG_ROOT/$variant_name" \
      "$@"
  fi
}
