#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 {neutral|unreliable_only} {0|1}" >&2
  exit 2
fi
VARIANT="$1"
PHYSICAL_NPU="$2"
case "$VARIANT:$PHYSICAL_NPU" in
  neutral:0|unreliable_only:1) ;;
  *)
    echo "C1 binding must be neutral:NPU0 or unreliable_only:NPU1" >&2
    exit 2
    ;;
esac

PYTHON="/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python"
ASCEND_ENV_SH="/usr/local/Ascend/cann-8.5.0/set_env.sh"
CHECKER="$ROOT_DIR/scripts/experiments/kd_baselines_npu/check_phaseO_o12_c1_run.py"
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
TRAIN_ARTIFACT="$ROOT_DIR/runs/diagnostics/phaseO_o12/o12_budget_train.json"
VAL_ARTIFACT="$ROOT_DIR/runs/diagnostics/phaseO_o12/o12_budget_val.json"
GATE="$ROOT_DIR/runs/diagnostics/phaseO_o12/o12_joint_gate.json"
O11_GATE="$ROOT_DIR/runs/diagnostics/phaseO_o11/o11_confidence_gate.json"
TRAIN_LIST="$ROOT_DIR/dataset/list/voc/train_aug.txt"
VAL_LIST="$ROOT_DIR/dataset/list/voc/val.txt"
PLAN="$ROOT_DIR/reports/2026-07-13_phaseO_rtc_o12_budgeted_routing_plan.md"
O12B_REPORT="$ROOT_DIR/reports/2026-07-13_phaseO_rtc_o12b_smoke_report.md"
BOOTSTRAP_INDICES="$ROOT_DIR/runs/diagnostics/phaseO_o12_c1/bootstrap_indices_pcg64_3407.npy"

BASE_NAME="o12c1_${VARIANT}_20k_seed1234"
SAVE_DIR="$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseO_o12_c1/$BASE_NAME"
LOG_DIR="$ROOT_DIR/runs/kd_baselines_npu/phaseO_o12_c1/$BASE_NAME"
RUNTIME_DIR="$ROOT_DIR/runs/runtime/kd_baselines_npu/phaseO_o12_c1/$BASE_NAME"
LOCK_ROOT="$ROOT_DIR/runs/runtime/kd_baselines_npu/phaseO_o12_c1/.locks"
LOCK_DIR="$LOCK_ROOT/$BASE_NAME.lock"

case "$VARIANT" in
  neutral)
    O12B_FRESH="$ROOT_DIR/runs/runtime/kd_baselines_npu/phaseO_o12/o12b_neutral_smoke20_seed1234/acceptance.json"
    O12B_FRESH_SHA256="90eb89fe7a7dff773845152d5d59a7efcc3d61299976137223be6cf7d2615e1c"
    O12B_RESUME="$ROOT_DIR/runs/runtime/kd_baselines_npu/phaseO_o12/o12b_neutral_smoke20_seed1234_resume_audit/acceptance.json"
    O12B_RESUME_SHA256="30fe00fa72b2f20293db2cfca3da7f0f9d960649fef8d153629b3d2b2f496677"
    ;;
  unreliable_only)
    O12B_FRESH="$ROOT_DIR/runs/runtime/kd_baselines_npu/phaseO_o12/o12b_unreliable_only_smoke20_seed1234/acceptance.json"
    O12B_FRESH_SHA256="d8a3363724c7c5d93c7629ba0ebf365a7f73d1c25c53138633e179e21fe623cd"
    O12B_RESUME="$ROOT_DIR/runs/runtime/kd_baselines_npu/phaseO_o12/o12b_unreliable_only_smoke20_seed1234_resume_audit/acceptance.json"
    O12B_RESUME_SHA256="af540bf53a0b1646227eb9024a6eb9f5d7bf1af250773c62d1074b77aaa82488"
    ;;
esac

EXPECTED_GIT_STATUS=$'?? scripts/experiments/kd_baselines_npu/check_phaseO_rtc_runs.py\n?? scripts/experiments/kd_baselines_npu/launch_phaseO_rtc.sh\n?? scripts/experiments/kd_baselines_npu/run_phaseO_rtc.sh\n?? scripts/experiments/kd_baselines_npu/run_phaseO_rtc_variant.sh'

fail() {
  echo "[O1.2-C1] $*" >&2
  exit 2
}

require_sha() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || fail "required file missing: $path"
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || fail \
    "SHA256 mismatch: $path expected=$expected actual=$actual"
}

verify_b_acceptance() {
  local path="$1" expected_sha="$2" expected_mode="$3"
  require_sha "$path" "$expected_sha"
  "$PYTHON" -c '
import json, sys
path, variant, mode = sys.argv[1:]
with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)
expected = {"schema_version": 2, "phase": "O1.2-B", "stage": "final",
            "mode": mode, "variant": variant, "pass": True, "errors": []}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f"B acceptance mismatch: {key}")
' "$path" "$VARIANT" "$expected_mode" || fail \
    "O1.2-B prerequisite did not pass: $path"
}

require_git_contract() {
  local actual tracked
  actual="$(git status --porcelain --untracked-files=all)"
  [[ "$actual" == "$EXPECTED_GIT_STATUS" ]] || {
    echo "[O1.2-C1] git status must contain exactly the four quarantined old scripts" >&2
    git status --short >&2
    exit 2
  }
  for tracked in \
    scripts/experiments/kd_baselines_npu/run_phaseO_o12_c1_variant.sh \
    scripts/experiments/kd_baselines_npu/check_phaseO_o12_c1_run.py \
    reports/2026-07-13_phaseO_rtc_o12_budgeted_routing_plan.md; do
    git ls-files --error-unmatch "$tracked" >/dev/null 2>&1 || \
      fail "required launch input is not tracked: $tracked"
  done
}

require_device_ready() {
  local mapping health processes
  mapping="$(npu-smi info -m 2>&1)" || fail "cannot query NPU mapping"
  awk -v chip="$PHYSICAL_NPU" '
    $1 == 0 && $2 == chip && $3 == chip && $4 == chip && $5 == "Ascend910" { found=1 }
    END { exit(found ? 0 : 1) }
  ' <<< "$mapping" || fail "physical/logical NPU mapping mismatch for chip $PHYSICAL_NPU"
  health="$(npu-smi info -t health -i 0 -c "$PHYSICAL_NPU" 2>&1)" || \
    fail "cannot query NPU health for chip $PHYSICAL_NPU"
  grep -Eq 'Health Status[[:space:]]*:[[:space:]]*OK' <<< "$health" || \
    fail "NPU chip $PHYSICAL_NPU is not healthy"
  processes="$(npu-smi info -t proc-mem -i 0 -c "$PHYSICAL_NPU" 2>&1)" || \
    fail "cannot query NPU processes for chip $PHYSICAL_NPU"
  grep -Fq "No process in device." <<< "$processes" || \
    fail "NPU chip $PHYSICAL_NPU is not idle"
}

[[ -x "$PYTHON" ]] || fail "python not executable: $PYTHON"
[[ -f "$ASCEND_ENV_SH" ]] || fail "Ascend environment missing: $ASCEND_ENV_SH"
[[ -f "$CHECKER" ]] || fail "C1 checker missing: $CHECKER"
command -v setsid >/dev/null || fail "setsid is required"
command -v npu-smi >/dev/null || fail "npu-smi is required"

require_sha "$TEACHER" "ac49b2c7720b21d565072e974e4404fcb009ba106288d95d1a4bb25f09c3fe58"
require_sha "$STUDENT_INIT" "47085aa164b2977003221458a2a5fdf5f46f434f5539b164dc71969b4dd4cd75"
require_sha "$CDF" "8ba8376a03c2f93835339d98670fa357b381b23a22cbf2a7fc48b0c2441d7e69"
require_sha "$PARAMETERS" "a9006972e165ed6c95e7a966c526927bb9f846fb6c7ec6528f5507dd21b8b5df"
require_sha "$TRAIN_ARTIFACT" "8deb4850a629e7a7b44a6ed52bf86b988857e98bf786ee39e6e948995f448b6e"
require_sha "$VAL_ARTIFACT" "d1dde29c569df7a6fabda32ec3daa50bb6a16758125768e5059c1762cb0b1a5f"
require_sha "$GATE" "c1961cfec8f232c1fdf7fc8b90db5a6f4855901317ea2705536eea88cbc1be82"
require_sha "$O11_GATE" "47ff2f1f2ea68a4e50375bfa7efc7221c8197703372d5d8f22dfec9032088d3a"
require_sha "$TRAIN_LIST" "d1326bd532648d73bc4b1bd275434eba81930982dc7401a65a8cecb26c028e24"
require_sha "$VAL_LIST" "cdc1326d12f69ce5153aa5da04a4d8783e146868d42d19f82f44e74b97ac907d"
require_sha "$PLAN" "c6ac659aea7019d8c2faed88ccdd596678e909129e3a468942cde921d6a6d8b9"
require_sha "$O12B_REPORT" "2bdd4b77fbcc568138f19711e8e6c1bf7e719f298b9e539e74822488e92bf1fb"
require_sha "$BOOTSTRAP_INDICES" "de2b18873dcd9f05f2d1d7acd9c0d94088680fb009441a501b8ba31ee8ce10b5"
require_sha "$TRAIN_ENTRY" "f50a130c17ca3b5f368f848b96c59fd58c408952e75bda447c051589505b316d"
require_sha "$O12_MODULE" "5e23f3e1bd526017aa2046faff621dd284093d91dead9088dd4d7d9ddb641d9e"
require_sha "$O12_DIAGNOSE" "cc391388f64505abae4cded5ac7b36122018a131b3c90f480290ff275a4cee50"
require_sha "$O12_GATE_CHECKER" "805a19625d496d3c3864d529e314a49d75584afad69fedf69e28cadc431ce085"
require_sha "$O11_RTC_MODULE" "01b7b6e6aa0d513561332510347b52ea9411330dfb0f2da54abdc36f2375fe59"
require_sha "$O11_BUILD_CDF" "c88bdfcb885cde01cbf437e2e7751c8eab510067aacf530b469df808ae6604dd"
require_sha "$O11_DIAGNOSE" "f838f25b70b8eadfc982873057e5fb68c54c81089a32e9121fd664e02916f9ef"
require_sha "$O11_GATE_CHECKER" "55553ec523ac2c2a979470b8542f31878462bbe5a919e0c52132edfbba4eb256"
verify_b_acceptance "$O12B_FRESH" "$O12B_FRESH_SHA256" fresh
verify_b_acceptance "$O12B_RESUME" "$O12B_RESUME_SHA256" resume_audit
require_git_contract
require_device_ready

for path in "$SAVE_DIR" "$LOG_DIR" "$RUNTIME_DIR" "$LOCK_DIR"; do
  [[ ! -e "$path" ]] || fail "refusing to overwrite existing C1 artifact: $path"
done
mkdir -p "$LOCK_ROOT"
mkdir "$LOCK_DIR" || fail "C1 run lock already exists: $LOCK_DIR"
for path in "$SAVE_DIR" "$LOG_DIR" "$RUNTIME_DIR"; do
  [[ ! -e "$path" ]] || fail "C1 output appeared during lock acquisition: $path"
done
mkdir -p "$SAVE_DIR" "$LOG_DIR" "$RUNTIME_DIR"

# shellcheck source=/dev/null
source "$ASCEND_ENV_SH"
export ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_NPU"
export ASCEND_VISIBLE_DEVICES="$PHYSICAL_NPU"
export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4

RUNTIME_ENV_JSON="$("$PYTHON" -c '
import json, platform, sys, torch, torch_npu
available = bool(torch.npu.is_available())
count = int(torch.npu.device_count())
if not available or count != 1:
    raise SystemExit("C1 requires exactly one visible Ascend NPU")
print(json.dumps({"python_executable": sys.executable,
                  "python_version": platform.python_version(),
                  "torch_version": torch.__version__,
                  "torch_npu_version": torch_npu.__version__,
                  "npu_available": available, "npu_device_count": count},
                 sort_keys=True))
')"

COMMAND=(
  "$PYTHON" "$TRAIN_ENTRY"
  --device-type npu --local-rank 0 --seed 1234
  --teacher-model deeplabv3 --teacher-backbone resnet101
  --student-model deeplabv3_mobilenet_ssseg --student-backbone mobilenetv3_small
  --dataset voc --data "$DATA_DIR" --crop-size 512 512
  --batch-size 16 --workers 8 --ignore-label -1 --start_epoch 0
  --max-iterations 20000 --lr 0.02 --momentum 0.9 --weight-decay 0.0001
  --kd-loss-mode rtc_o12_teacher_target --rtc-o12-variant "$VARIANT"
  --rtc-o12-cdf-path "$CDF" --rtc-o12-parameters-path "$PARAMETERS"
  --rtc-o12-gate-path "$GATE" --teacher-output-temp 3.0 --kd-temperature 1.0
  --lambda-kd 1.0 --lambda-adv 0.001 --lambda-d 0.1
  --lambda-cwd-fea 50.0 --lambda-cwd-logit 3.0
  --lambda-skd 0.0 --lambda-ifv 0.0 --lambda-fitnet 0.0
  --lambda-at 0.0 --lambda-psd 0.0 --lambda-csd 0.0
  --teacher-pretrained-base None --teacher-pretrained "$TEACHER"
  --student-pretrained-base "$STUDENT_INIT" --student-pretrained None
  --log-iter 20 --save-per-iters 800 --val-per-iters 800
  --save-dir "$SAVE_DIR" --log-dir "$LOG_DIR"
)

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_SHA256="$(sha256sum "$SCRIPT_PATH" | awk '{print $1}')"
CHECKER_SHA256="$(sha256sum "$CHECKER" | awk '{print $1}')"
PYTHON_REALPATH="$(realpath "$PYTHON")"
PYTHON_SHA256="$(sha256sum "$PYTHON_REALPATH" | awk '{print $1}')"
ASCEND_ENV_SHA256="$(sha256sum "$ASCEND_ENV_SH" | awk '{print $1}')"
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
GIT_COMMIT="$(git rev-parse HEAD)"
GIT_DIRTY_COUNT="$(git status --porcelain --untracked-files=all | wc -l | tr -d ' ')"
PROVENANCE="$RUNTIME_DIR/provenance.txt"
CONSOLE_LOG="$RUNTIME_DIR/console.log"
NPU_LOG="$RUNTIME_DIR/npu_smi.log"
NPU_MAPPING="$RUNTIME_DIR/npu_mapping.txt"
TRAIN_PID_FILE="$RUNTIME_DIR/training.pid"

{
  echo "phase=O1.2-C1"
  echo "run_kind=20k_signal_screen"
  echo "variant=$VARIANT"
  echo "mode=fresh"
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
  echo "train_entry_sha256=$(sha256sum "$TRAIN_ENTRY" | awk '{print $1}')"
  echo "rtc_o12_calibration_sha256=$(sha256sum "$O12_MODULE" | awk '{print $1}')"
  echo "diagnose_rtc_o12_budget_sha256=$(sha256sum "$O12_DIAGNOSE" | awk '{print $1}')"
  echo "check_rtc_o12_gate_sha256=$(sha256sum "$O12_GATE_CHECKER" | awk '{print $1}')"
  echo "teacher_sha256=$(sha256sum "$TEACHER" | awk '{print $1}')"
  echo "student_init_sha256=$(sha256sum "$STUDENT_INIT" | awk '{print $1}')"
  echo "cdf_sha256=$(sha256sum "$CDF" | awk '{print $1}')"
  echo "parameters_sha256=$(sha256sum "$PARAMETERS" | awk '{print $1}')"
  echo "train_artifact_sha256=$(sha256sum "$TRAIN_ARTIFACT" | awk '{print $1}')"
  echo "val_artifact_sha256=$(sha256sum "$VAL_ARTIFACT" | awk '{print $1}')"
  echo "gate_sha256=$(sha256sum "$GATE" | awk '{print $1}')"
  echo "o11_gate_sha256=$(sha256sum "$O11_GATE" | awk '{print $1}')"
  echo "train_list_sha256=$(sha256sum "$TRAIN_LIST" | awk '{print $1}')"
  echo "val_list_path=$VAL_LIST"
  echo "val_list_sha256=$(sha256sum "$VAL_LIST" | awk '{print $1}')"
  echo "bootstrap_indices_path=$BOOTSTRAP_INDICES"
  echo "bootstrap_indices_sha256=$(sha256sum "$BOOTSTRAP_INDICES" | awk '{print $1}')"
  echo "plan_path=$PLAN"
  echo "plan_sha256=$(sha256sum "$PLAN" | awk '{print $1}')"
  echo "o12b_report_path=$O12B_REPORT"
  echo "o12b_report_sha256=$(sha256sum "$O12B_REPORT" | awk '{print $1}')"
  echo "expected_order_sha256=10e600fd87537bba4329a7a90473bd71931e5fe2c775345af4dc483c3b9f8c5c"
  echo "o12b_fresh_acceptance_path=$O12B_FRESH"
  echo "o12b_fresh_acceptance_sha256=$O12B_FRESH_SHA256"
  echo "o12b_fresh_acceptance_pass=true"
  echo "o12b_resume_acceptance_path=$O12B_RESUME"
  echo "o12b_resume_acceptance_sha256=$O12B_RESUME_SHA256"
  echo "o12b_resume_acceptance_pass=true"
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

echo "[O1.2-C1] start variant=$VARIANT physical_npu=$PHYSICAL_NPU"
echo "[O1.2-C1] launcher_sha256=$SCRIPT_SHA256"
echo "[O1.2-C1] runtime_dir=$RUNTIME_DIR"

CHILD_PID=""
RUNNER_PID=""
END_RECORDED=0
record_end() {
  local status="$1" reason="$2"
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
  local signal_name="$1" status="$2"
  set +e
  [[ -z "$CHILD_PID" ]] || kill -TERM -- "-$CHILD_PID" 2>/dev/null
  if [[ -n "$RUNNER_PID" ]]; then
    for _ in {1..40}; do
      kill -0 "$RUNNER_PID" 2>/dev/null || break
      sleep 0.25
    done
    if kill -0 "$RUNNER_PID" 2>/dev/null && [[ -n "$CHILD_PID" ]]; then
      kill -KILL -- "-$CHILD_PID" 2>/dev/null
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
  [[ ! -s "$TRAIN_PID_FILE" ]] || break
  kill -0 "$RUNNER_PID" 2>/dev/null || break
  sleep 0.05
done
if [[ ! -s "$TRAIN_PID_FILE" ]]; then
  set +e
  kill -TERM "$RUNNER_PID" 2>/dev/null
  wait "$RUNNER_PID"
  TRAIN_STATUS=$?
  set -e
  [[ "$TRAIN_STATUS" -ne 0 ]] || TRAIN_STATUS=2
  record_end "$TRAIN_STATUS" training_pid_capture_failed
  tail -n 80 "$CONSOLE_LOG" >&2 || true
  exit "$TRAIN_STATUS"
fi
CHILD_PID="$(<"$TRAIN_PID_FILE")"
if ! [[ "$CHILD_PID" =~ ^[0-9]+$ ]]; then
  kill -TERM "$RUNNER_PID" 2>/dev/null || true
  wait "$RUNNER_PID" 2>/dev/null || true
  record_end 2 invalid_training_pid
  fail "invalid training PID: $CHILD_PID"
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
  sleep 5
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
  record_end "$TRAIN_STATUS" training_failed
  tail -n 120 "$CONSOLE_LOG" >&2 || true
  exit "$TRAIN_STATUS"
fi

CHECKER_COMMON=(
  "$PYTHON" "$CHECKER" --variant "$VARIANT"
  --save-dir "$SAVE_DIR" --log-dir "$LOG_DIR" --runtime-dir "$RUNTIME_DIR"
)
CHECKER_COMMAND=("${CHECKER_COMMON[@]}" --stage prefinal)
FINAL_CHECKER_COMMAND=("${CHECKER_COMMON[@]}" --stage final)
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
  record_end "$CHECK_STATUS" checker_failed
  tail -n 160 "$RUNTIME_DIR/checker.log" >&2 || true
  exit "$CHECK_STATUS"
fi
record_end 0 completed_and_checked
set +e
"${FINAL_CHECKER_COMMAND[@]}" > "$RUNTIME_DIR/final_checker.log" 2>&1
FINAL_CHECK_STATUS=$?
set -e
if [[ "$FINAL_CHECK_STATUS" -ne 0 ]]; then
  tail -n 160 "$RUNTIME_DIR/final_checker.log" >&2 || true
  exit "$FINAL_CHECK_STATUS"
fi
echo "[O1.2-C1] completed variant=$VARIANT"
echo "[O1.2-C1] console_log=$CONSOLE_LOG"
echo "[O1.2-C1] acceptance=$RUNTIME_DIR/acceptance.json"
