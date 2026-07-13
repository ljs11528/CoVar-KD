#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

VARIANT="${1:-}"
PHYSICAL_NPU="${2:-}"
MODE="${3:-fresh}"

case "$VARIANT" in
  neutral|unreliable_only) ;;
  *)
    echo "usage: $0 {neutral|unreliable_only} {0|1} [fresh|resume_audit]" >&2
    exit 2
    ;;
esac
case "$PHYSICAL_NPU" in
  0|1) ;;
  *)
    echo "physical NPU must be 0 or 1" >&2
    exit 2
    ;;
esac
case "$MODE" in
  fresh|resume_audit) ;;
  *)
    echo "mode must be fresh or resume_audit" >&2
    exit 2
    ;;
esac

PYTHON="/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python"
ASCEND_ENV_SH="/usr/local/Ascend/cann-8.5.0/set_env.sh"
CHECKER="$ROOT_DIR/scripts/experiments/kd_baselines_npu/check_phaseO_o12_smoke.py"
TRAIN_ENTRY="$ROOT_DIR/train_kd.py"
O12_MODULE="$ROOT_DIR/utils/rtc_o12_calibration.py"
O12_DIAGNOSE="$ROOT_DIR/scripts/diagnostics/diagnose_rtc_o12_budget.py"
O12_GATE_CHECKER="$ROOT_DIR/scripts/diagnostics/check_rtc_o12_gate.py"
O11_RTC_MODULE="$ROOT_DIR/utils/rtc_temperature.py"
O11_BUILD_CDF="$ROOT_DIR/scripts/diagnostics/build_rtc_cdf.py"
O11_DIAGNOSE="$ROOT_DIR/scripts/diagnostics/diagnose_rtc_routing.py"
O11_GATE_CHECKER="$ROOT_DIR/scripts/diagnostics/check_rtc_o11_gate.py"
DATA_DIR="$ROOT_DIR/dataset/VOCAug/"
TEACHER="$ROOT_DIR/data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth"
STUDENT_INIT="$ROOT_DIR/data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth"
CDF="$ROOT_DIR/runs/diagnostics/phaseO_o11/voc_train_rtc_confidence_cdf.pt"
PARAMETERS="$ROOT_DIR/runs/diagnostics/phaseO_o12/o12_budget_parameters.json"
GATE="$ROOT_DIR/runs/diagnostics/phaseO_o12/o12_joint_gate.json"
O11_GATE="$ROOT_DIR/runs/diagnostics/phaseO_o11/o11_confidence_gate.json"
TRAIN_LIST="$ROOT_DIR/dataset/list/voc/train_aug.txt"

BASE_NAME="o12b_${VARIANT}_smoke20_seed1234"
if [[ "$MODE" == "resume_audit" ]]; then
  RUN_NAME="${BASE_NAME}_resume_audit"
else
  RUN_NAME="$BASE_NAME"
fi
SAVE_DIR="$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseO_o12/$RUN_NAME"
LOG_DIR="$ROOT_DIR/runs/kd_baselines_npu/phaseO_o12/$RUN_NAME"
RUNTIME_DIR="$ROOT_DIR/runs/runtime/kd_baselines_npu/phaseO_o12/$RUN_NAME"
FRESH_STATE="$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseO_o12/$BASE_NAME/training_state_latest.pth"
LOCK_ROOT="$ROOT_DIR/runs/runtime/kd_baselines_npu/phaseO_o12/.locks"
LOCK_DIR="$LOCK_ROOT/$RUN_NAME.lock"

[[ -x "$PYTHON" ]] || { echo "python not executable: $PYTHON" >&2; exit 2; }
[[ -f "$ASCEND_ENV_SH" ]] || { echo "Ascend environment missing: $ASCEND_ENV_SH" >&2; exit 2; }
[[ -f "$CHECKER" ]] || { echo "smoke checker missing: $CHECKER" >&2; exit 2; }
command -v setsid >/dev/null || { echo "setsid is required" >&2; exit 2; }
command -v npu-smi >/dev/null || { echo "npu-smi is required" >&2; exit 2; }

require_sha() {
  local path="$1"
  local expected="$2"
  [[ -f "$path" ]] || { echo "required file missing: $path" >&2; exit 2; }
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 2
  }
}

require_sha "$TEACHER" "ac49b2c7720b21d565072e974e4404fcb009ba106288d95d1a4bb25f09c3fe58"
require_sha "$STUDENT_INIT" "47085aa164b2977003221458a2a5fdf5f46f434f5539b164dc71969b4dd4cd75"
require_sha "$CDF" "8ba8376a03c2f93835339d98670fa357b381b23a22cbf2a7fc48b0c2441d7e69"
require_sha "$PARAMETERS" "a9006972e165ed6c95e7a966c526927bb9f846fb6c7ec6528f5507dd21b8b5df"
require_sha "$GATE" "c1961cfec8f232c1fdf7fc8b90db5a6f4855901317ea2705536eea88cbc1be82"
require_sha "$O11_GATE" "47ff2f1f2ea68a4e50375bfa7efc7221c8197703372d5d8f22dfec9032088d3a"
require_sha "$TRAIN_LIST" "d1326bd532648d73bc4b1bd275434eba81930982dc7401a65a8cecb26c028e24"
require_sha "$TRAIN_ENTRY" "f50a130c17ca3b5f368f848b96c59fd58c408952e75bda447c051589505b316d"
require_sha "$O12_MODULE" "5e23f3e1bd526017aa2046faff621dd284093d91dead9088dd4d7d9ddb641d9e"
require_sha "$O12_DIAGNOSE" "cc391388f64505abae4cded5ac7b36122018a131b3c90f480290ff275a4cee50"
require_sha "$O12_GATE_CHECKER" "805a19625d496d3c3864d529e314a49d75584afad69fedf69e28cadc431ce085"
require_sha "$O11_RTC_MODULE" "01b7b6e6aa0d513561332510347b52ea9411330dfb0f2da54abdc36f2375fe59"
require_sha "$O11_BUILD_CDF" "c88bdfcb885cde01cbf437e2e7751c8eab510067aacf530b469df808ae6604dd"
require_sha "$O11_DIAGNOSE" "f838f25b70b8eadfc982873057e5fb68c54c81089a32e9121fd664e02916f9ef"
require_sha "$O11_GATE_CHECKER" "55553ec523ac2c2a979470b8542f31878462bbe5a919e0c52132edfbba4eb256"

RESUME_SOURCE_SHA256=""
if [[ "$MODE" == "resume_audit" ]]; then
  [[ -f "$FRESH_STATE" ]] || {
    echo "fresh checkpoint missing for resume audit: $FRESH_STATE" >&2
    exit 2
  }
  RESUME_SOURCE_SHA256="$(sha256sum "$FRESH_STATE" | awk '{print $1}')"
fi
mkdir -p "$LOCK_ROOT"
if ! mkdir "$LOCK_DIR"; then
  echo "O1.2-B run lock already exists: $LOCK_DIR" >&2
  exit 2
fi
for path in "$SAVE_DIR" "$LOG_DIR" "$RUNTIME_DIR"; do
  [[ ! -e "$path" ]] || {
    echo "refusing to overwrite existing O1.2-B output: $path" >&2
    exit 2
  }
done
mkdir -p "$SAVE_DIR" "$LOG_DIR" "$RUNTIME_DIR"

# shellcheck source=/dev/null
source "$ASCEND_ENV_SH"

export ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_NPU"
export ASCEND_VISIBLE_DEVICES="$PHYSICAL_NPU"
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4

RUNTIME_ENV_JSON="$("$PYTHON" -c '
import json
import platform
import sys
import torch
import torch_npu
available = bool(torch.npu.is_available())
count = int(torch.npu.device_count())
if not available or count < 1:
    raise SystemExit("Ascend NPU runtime is unavailable")
print(json.dumps({"python_executable": sys.executable, "python_version": platform.python_version(), "torch_version": torch.__version__, "torch_npu_version": torch_npu.__version__, "npu_available": available, "npu_device_count": count}, sort_keys=True))
')"

COMMAND=(
  "$PYTHON" "$ROOT_DIR/train_kd.py"
  --device-type npu
  --local-rank 0
  --seed 1234
  --teacher-model deeplabv3
  --teacher-backbone resnet101
  --student-model deeplabv3_mobilenet_ssseg
  --student-backbone mobilenetv3_small
  --dataset voc
  --data "$DATA_DIR"
  --crop-size 512 512
  --batch-size 16
  --workers 8
  --ignore-label -1
  --start_epoch 0
  --max-iterations 20
  --lr 0.02
  --momentum 0.9
  --weight-decay 0.0001
  --kd-loss-mode rtc_o12_teacher_target
  --rtc-o12-variant "$VARIANT"
  --rtc-o12-cdf-path "$CDF"
  --rtc-o12-parameters-path "$PARAMETERS"
  --rtc-o12-gate-path "$GATE"
  --teacher-output-temp 3.0
  --kd-temperature 1.0
  --lambda-kd 1.0
  --lambda-adv 0.001
  --lambda-d 0.1
  --lambda-cwd-fea 50.0
  --lambda-cwd-logit 3.0
  --lambda-skd 0.0
  --lambda-ifv 0.0
  --lambda-fitnet 0.0
  --lambda-at 0.0
  --lambda-psd 0.0
  --lambda-csd 0.0
  --teacher-pretrained-base None
  --teacher-pretrained "$TEACHER"
  --student-pretrained-base "$STUDENT_INIT"
  --student-pretrained None
  --log-iter 20
  --save-per-iters 800
  --val-per-iters 800
  --skip-val
  --save-dir "$SAVE_DIR"
  --log-dir "$LOG_DIR"
)
if [[ "$MODE" == "resume_audit" ]]; then
  COMMAND+=(--resume "$FRESH_STATE")
fi

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_SHA256="$(sha256sum "$SCRIPT_PATH" | awk '{print $1}')"
CHECKER_SHA256="$(sha256sum "$CHECKER" | awk '{print $1}')"
PYTHON_REALPATH="$(realpath "$PYTHON")"
PYTHON_SHA256="$(sha256sum "$PYTHON_REALPATH" | awk '{print $1}')"
ASCEND_ENV_SHA256="$(sha256sum "$ASCEND_ENV_SH" | awk '{print $1}')"
TEACHER_SHA256="$(sha256sum "$TEACHER" | awk '{print $1}')"
STUDENT_INIT_SHA256="$(sha256sum "$STUDENT_INIT" | awk '{print $1}')"
CDF_SHA256="$(sha256sum "$CDF" | awk '{print $1}')"
PARAMETERS_SHA256="$(sha256sum "$PARAMETERS" | awk '{print $1}')"
GATE_SHA256="$(sha256sum "$GATE" | awk '{print $1}')"
O11_GATE_SHA256="$(sha256sum "$O11_GATE" | awk '{print $1}')"
TRAIN_LIST_SHA256="$(sha256sum "$TRAIN_LIST" | awk '{print $1}')"
TRAIN_ENTRY_SHA256="$(sha256sum "$TRAIN_ENTRY" | awk '{print $1}')"
O12_MODULE_SHA256="$(sha256sum "$O12_MODULE" | awk '{print $1}')"
O12_DIAGNOSE_SHA256="$(sha256sum "$O12_DIAGNOSE" | awk '{print $1}')"
O12_GATE_CHECKER_SHA256="$(sha256sum "$O12_GATE_CHECKER" | awk '{print $1}')"
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
GIT_COMMIT="$(git rev-parse HEAD)"
GIT_DIRTY_COUNT="$(git status --porcelain --untracked-files=all | wc -l | tr -d ' ')"
PROVENANCE="$RUNTIME_DIR/provenance.txt"
CONSOLE_LOG="$RUNTIME_DIR/console.log"
NPU_LOG="$RUNTIME_DIR/npu_smi.log"
NPU_MAPPING="$RUNTIME_DIR/npu_mapping.txt"

{
  echo "phase=O1.2-B"
  echo "run_kind=20_step_smoke"
  echo "variant=$VARIANT"
  echo "mode=$MODE"
  echo "requested_physical_npu=$PHYSICAL_NPU"
  echo "world_size=1"
  echo "rank=0"
  echo "local_rank=0"
  echo "seed=1234"
  echo "start_utc=$START_UTC"
  echo "runtime_env_json=$RUNTIME_ENV_JSON"
  echo "python_path=$PYTHON"
  echo "python_realpath=$PYTHON_REALPATH"
  echo "python_sha256=$PYTHON_SHA256"
  echo "ascend_env_path=$ASCEND_ENV_SH"
  echo "ascend_env_sha256=$ASCEND_ENV_SHA256"
  echo "launcher_path=$SCRIPT_PATH"
  echo "launcher_sha256=$SCRIPT_SHA256"
  echo "checker_path=$CHECKER"
  echo "checker_sha256=$CHECKER_SHA256"
  echo "train_entry_sha256=$TRAIN_ENTRY_SHA256"
  echo "rtc_o12_calibration_sha256=$O12_MODULE_SHA256"
  echo "diagnose_rtc_o12_budget_sha256=$O12_DIAGNOSE_SHA256"
  echo "check_rtc_o12_gate_sha256=$O12_GATE_CHECKER_SHA256"
  echo "teacher_sha256=$TEACHER_SHA256"
  echo "student_init_sha256=$STUDENT_INIT_SHA256"
  echo "cdf_sha256=$CDF_SHA256"
  echo "parameters_sha256=$PARAMETERS_SHA256"
  echo "gate_sha256=$GATE_SHA256"
  echo "o11_gate_sha256=$O11_GATE_SHA256"
  echo "train_list_sha256=$TRAIN_LIST_SHA256"
  echo "resume_source_sha256=$RESUME_SOURCE_SHA256"
  echo "git_commit=$GIT_COMMIT"
  echo "git_dirty_count=$GIT_DIRTY_COUNT"
  echo "save_dir=$SAVE_DIR"
  echo "log_dir=$LOG_DIR"
  echo "runtime_dir=$RUNTIME_DIR"
  printf 'argv_shell='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
} > "$PROVENANCE"
git status --porcelain --untracked-files=all > "$RUNTIME_DIR/git_status.txt"
printf '%s\0' "${COMMAND[@]}" > "$RUNTIME_DIR/argv.nul"
npu-smi info -m > "$NPU_MAPPING" 2>&1 || true

echo "[O1.2-B] start variant=$VARIANT mode=$MODE physical_npu=$PHYSICAL_NPU"
echo "[O1.2-B] launcher_sha256=$SCRIPT_SHA256"
echo "[O1.2-B] runtime_dir=$RUNTIME_DIR"

CHILD_PID=""
RUNNER_PID=""
TRAIN_PID_FILE="$RUNTIME_DIR/training.pid"
END_RECORDED=0

record_end() {
  local status="$1"
  local reason="$2"
  if [[ "$END_RECORDED" -eq 0 ]]; then
    {
      echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
      echo "exit_status=$status"
      echo "exit_reason=$reason"
    } >> "$PROVENANCE"
    END_RECORDED=1
  fi
}

terminate_run() {
  local signal_name="$1"
  local status="$2"
  set +e
  if [[ -n "$CHILD_PID" ]]; then
    kill -TERM -- "-$CHILD_PID" 2>/dev/null
  fi
  if [[ -n "$RUNNER_PID" ]]; then
    for _ in {1..20}; do
      if ! kill -0 "$RUNNER_PID" 2>/dev/null; then
        break
      fi
      sleep 0.25
    done
    if kill -0 "$RUNNER_PID" 2>/dev/null; then
      if [[ -n "$CHILD_PID" ]]; then
        kill -KILL -- "-$CHILD_PID" 2>/dev/null
      fi
    fi
    wait "$RUNNER_PID" 2>/dev/null
  fi
  record_end "$status" "signal_$signal_name"
  exit "$status"
}

trap 'terminate_run INT 130' INT
trap 'terminate_run TERM 143' TERM

{
  echo "snapshot_kind=before_training"
  echo "snapshot_utc=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
  npu-smi info
} > "$NPU_LOG" 2>&1 || true

setsid --wait bash -c '
pid_file="$1"
shift
printf "%s\n" "$$" > "$pid_file"
exec "$@"
' bash "$TRAIN_PID_FILE" "${COMMAND[@]}" > "$CONSOLE_LOG" 2>&1 &
RUNNER_PID=$!
for _ in {1..200}; do
  if [[ -s "$TRAIN_PID_FILE" ]]; then
    break
  fi
  if ! kill -0 "$RUNNER_PID" 2>/dev/null; then
    break
  fi
  sleep 0.05
done
if [[ ! -s "$TRAIN_PID_FILE" ]]; then
  set +e
  kill -TERM "$RUNNER_PID" 2>/dev/null
  wait "$RUNNER_PID"
  TRAIN_STATUS=$?
  set -e
  if [[ "$TRAIN_STATUS" -eq 0 ]]; then
    TRAIN_STATUS=2
  fi
  record_end "$TRAIN_STATUS" "training_pid_capture_failed"
  echo "[O1.2-B] failed to capture training PID" >&2
  tail -n 80 "$CONSOLE_LOG" >&2 || true
  exit "${TRAIN_STATUS:-2}"
fi
CHILD_PID="$(<"$TRAIN_PID_FILE")"
if ! [[ "$CHILD_PID" =~ ^[0-9]+$ ]]; then
  kill -TERM "$RUNNER_PID" 2>/dev/null || true
  wait "$RUNNER_PID" 2>/dev/null || true
  record_end 2 "invalid_training_pid"
  echo "[O1.2-B] invalid training PID: $CHILD_PID" >&2
  exit 2
fi
echo "launcher_pid=$$" >> "$PROVENANCE"
echo "runner_pid=$RUNNER_PID" >> "$PROVENANCE"
echo "training_pid=$CHILD_PID" >> "$PROVENANCE"

while kill -0 "$RUNNER_PID" 2>/dev/null; do
  {
    echo "snapshot_kind=running"
    echo "snapshot_utc=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
    npu-smi info
  } >> "$NPU_LOG" 2>&1 || true
  sleep 0.5
done

set +e
wait "$RUNNER_PID"
TRAIN_STATUS=$?
set -e
RUNNER_PID=""
CHILD_PID=""
{
  echo "training_end_utc=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
  echo "training_exit_status=$TRAIN_STATUS"
} >> "$PROVENANCE"

if [[ "$TRAIN_STATUS" -ne 0 ]]; then
  record_end "$TRAIN_STATUS" "training_failed"
  echo "[O1.2-B] failed variant=$VARIANT mode=$MODE status=$TRAIN_STATUS" >&2
  tail -n 80 "$CONSOLE_LOG" >&2 || true
  exit "$TRAIN_STATUS"
fi

CHECKER_COMMON=(
  "$PYTHON" "$CHECKER"
  --mode "$MODE"
  --variant "$VARIANT"
  --save-dir "$SAVE_DIR"
  --log-dir "$LOG_DIR"
  --runtime-dir "$RUNTIME_DIR"
)
CHECKER_COMMAND=("${CHECKER_COMMON[@]}" --stage prefinal)
FINAL_CHECKER_COMMAND=("${CHECKER_COMMON[@]}" --stage final)
if [[ "$MODE" == "resume_audit" ]]; then
  CHECKER_COMMAND+=(--source-checkpoint "$FRESH_STATE")
  FINAL_CHECKER_COMMAND+=(--source-checkpoint "$FRESH_STATE")
fi
{
  printf 'checker_argv_shell='
  printf '%q ' "${CHECKER_COMMAND[@]}"
  printf '\n'
  printf 'final_checker_argv_shell='
  printf '%q ' "${FINAL_CHECKER_COMMAND[@]}"
  printf '\n'
} >> "$PROVENANCE"

set +e
"${CHECKER_COMMAND[@]}" > "$RUNTIME_DIR/checker.log" 2>&1
CHECK_STATUS=$?
set -e
{
  echo "prefinal_checker_end_utc=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
  echo "checker_exit_status=$CHECK_STATUS"
} >> "$PROVENANCE"
if [[ "$CHECK_STATUS" -ne 0 ]]; then
  record_end "$CHECK_STATUS" "checker_failed"
  echo "[O1.2-B] prefinal checker failed variant=$VARIANT mode=$MODE" >&2
  tail -n 120 "$RUNTIME_DIR/checker.log" >&2 || true
  exit "$CHECK_STATUS"
fi

record_end 0 "completed_and_checked"

set +e
"${FINAL_CHECKER_COMMAND[@]}" > "$RUNTIME_DIR/final_checker.log" 2>&1
FINAL_CHECK_STATUS=$?
set -e
if [[ "$FINAL_CHECK_STATUS" -ne 0 ]]; then
  echo "[O1.2-B] final checker failed variant=$VARIANT mode=$MODE" >&2
  tail -n 120 "$RUNTIME_DIR/final_checker.log" >&2 || true
  exit "$FINAL_CHECK_STATUS"
fi

echo "[O1.2-B] completed variant=$VARIANT mode=$MODE"
echo "[O1.2-B] console_log=$CONSOLE_LOG"
echo "[O1.2-B] acceptance=$RUNTIME_DIR/acceptance.json"
