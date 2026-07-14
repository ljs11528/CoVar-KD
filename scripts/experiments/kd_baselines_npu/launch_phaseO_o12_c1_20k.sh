#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
[[ "$#" -eq 0 ]] || { echo "usage: $0" >&2; exit 2; }

SCRIPT_DIR="$ROOT_DIR/scripts/experiments/kd_baselines_npu"
RUNNER="$SCRIPT_DIR/run_phaseO_o12_c1_variant.sh"
PAIR="$SCRIPT_DIR/run_phaseO_o12_c1_pair.sh"
CHECKER="$SCRIPT_DIR/check_phaseO_o12_c1_run.py"
RUNTIME_ROOT="$ROOT_DIR/runs/runtime/kd_baselines_npu/phaseO_o12_c1"
PID_FILE="$RUNTIME_ROOT/controller.pid"
STATUS_FILE="$RUNTIME_ROOT/controller.status"
LOG_FILE="$RUNTIME_ROOT/controller.log"
LOCK_DIR="$RUNTIME_ROOT/.controller.lock"
PLAN="$ROOT_DIR/reports/2026-07-13_phaseO_rtc_o12_budgeted_routing_plan.md"
O12B_REPORT="$ROOT_DIR/reports/2026-07-13_phaseO_rtc_o12b_smoke_report.md"
BOOTSTRAP="$ROOT_DIR/runs/diagnostics/phaseO_o12_c1/bootstrap_indices_pcg64_3407.npy"

EXPECTED_GIT_STATUS=$'?? scripts/experiments/kd_baselines_npu/check_phaseO_rtc_runs.py\n?? scripts/experiments/kd_baselines_npu/launch_phaseO_rtc.sh\n?? scripts/experiments/kd_baselines_npu/run_phaseO_rtc.sh\n?? scripts/experiments/kd_baselines_npu/run_phaseO_rtc_variant.sh'

fail() { echo "[O1.2-C1] $*" >&2; exit 2; }
require_sha() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || fail "required file missing: $path"
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || fail \
    "SHA256 mismatch: $path expected=$expected actual=$actual"
}
require_b_acceptance() {
  local path="$1" sha="$2" variant="$3" mode="$4"
  require_sha "$path" "$sha"
  /home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python -c '
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    p = json.load(handle)
expected = {"schema_version": 2, "phase": "O1.2-B", "stage": "final",
            "variant": sys.argv[2], "mode": sys.argv[3], "pass": True, "errors": []}
if any(p.get(k) != v for k, v in expected.items()):
    raise SystemExit("invalid O1.2-B prerequisite")
' "$path" "$variant" "$mode" || fail "invalid B acceptance: $path"
}
require_device() {
  local chip="$1" mapping health processes
  mapping="$(npu-smi info -m 2>&1)" || fail "cannot query NPU mapping"
  awk -v chip="$chip" '
    $1 == 0 && $2 == chip && $3 == chip && $4 == chip && $5 == "Ascend910" { found=1 }
    END { exit(found ? 0 : 1) }
  ' <<< "$mapping" || fail "physical/logical mapping mismatch for NPU $chip"
  health="$(npu-smi info -t health -i 0 -c "$chip" 2>&1)" || fail "cannot query NPU $chip health"
  grep -Eq 'Health Status[[:space:]]*:[[:space:]]*OK' <<< "$health" || fail "NPU $chip is not healthy"
  processes="$(npu-smi info -t proc-mem -i 0 -c "$chip" 2>&1)" || fail "cannot query NPU $chip processes"
  grep -Fq "No process in device." <<< "$processes" || fail "NPU $chip is not idle"
}

for path in "$RUNNER" "$PAIR" "$CHECKER"; do
  [[ -x "$path" || "$path" == "$CHECKER" && -f "$path" ]] || fail "required C1 tool missing: $path"
done
command -v setsid >/dev/null || fail "setsid is required"
command -v npu-smi >/dev/null || fail "npu-smi is required"

require_sha "$ROOT_DIR/train_kd.py" "f50a130c17ca3b5f368f848b96c59fd58c408952e75bda447c051589505b316d"
require_sha "$ROOT_DIR/utils/rtc_o12_calibration.py" "5e23f3e1bd526017aa2046faff621dd284093d91dead9088dd4d7d9ddb641d9e"
require_sha "$ROOT_DIR/scripts/diagnostics/diagnose_rtc_o12_budget.py" "cc391388f64505abae4cded5ac7b36122018a131b3c90f480290ff275a4cee50"
require_sha "$ROOT_DIR/scripts/diagnostics/check_rtc_o12_gate.py" "805a19625d496d3c3864d529e314a49d75584afad69fedf69e28cadc431ce085"
require_sha "$ROOT_DIR/utils/rtc_temperature.py" "01b7b6e6aa0d513561332510347b52ea9411330dfb0f2da54abdc36f2375fe59"
require_sha "$ROOT_DIR/scripts/diagnostics/build_rtc_cdf.py" "c88bdfcb885cde01cbf437e2e7751c8eab510067aacf530b469df808ae6604dd"
require_sha "$ROOT_DIR/scripts/diagnostics/diagnose_rtc_routing.py" "f838f25b70b8eadfc982873057e5fb68c54c81089a32e9121fd664e02916f9ef"
require_sha "$ROOT_DIR/scripts/diagnostics/check_rtc_o11_gate.py" "55553ec523ac2c2a979470b8542f31878462bbe5a919e0c52132edfbba4eb256"
require_sha "$ROOT_DIR/data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth" "ac49b2c7720b21d565072e974e4404fcb009ba106288d95d1a4bb25f09c3fe58"
require_sha "$ROOT_DIR/data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth" "47085aa164b2977003221458a2a5fdf5f46f434f5539b164dc71969b4dd4cd75"
require_sha "$ROOT_DIR/runs/diagnostics/phaseO_o11/voc_train_rtc_confidence_cdf.pt" "8ba8376a03c2f93835339d98670fa357b381b23a22cbf2a7fc48b0c2441d7e69"
require_sha "$ROOT_DIR/runs/diagnostics/phaseO_o12/o12_budget_parameters.json" "a9006972e165ed6c95e7a966c526927bb9f846fb6c7ec6528f5507dd21b8b5df"
require_sha "$ROOT_DIR/runs/diagnostics/phaseO_o12/o12_budget_train.json" "8deb4850a629e7a7b44a6ed52bf86b988857e98bf786ee39e6e948995f448b6e"
require_sha "$ROOT_DIR/runs/diagnostics/phaseO_o12/o12_budget_val.json" "d1dde29c569df7a6fabda32ec3daa50bb6a16758125768e5059c1762cb0b1a5f"
require_sha "$ROOT_DIR/runs/diagnostics/phaseO_o12/o12_joint_gate.json" "c1961cfec8f232c1fdf7fc8b90db5a6f4855901317ea2705536eea88cbc1be82"
require_sha "$ROOT_DIR/runs/diagnostics/phaseO_o11/o11_confidence_gate.json" "47ff2f1f2ea68a4e50375bfa7efc7221c8197703372d5d8f22dfec9032088d3a"
require_sha "$ROOT_DIR/dataset/list/voc/train_aug.txt" "d1326bd532648d73bc4b1bd275434eba81930982dc7401a65a8cecb26c028e24"
require_sha "$ROOT_DIR/dataset/list/voc/val.txt" "cdc1326d12f69ce5153aa5da04a4d8783e146868d42d19f82f44e74b97ac907d"
require_sha "$PLAN" "c6ac659aea7019d8c2faed88ccdd596678e909129e3a468942cde921d6a6d8b9"
require_sha "$O12B_REPORT" "2bdd4b77fbcc568138f19711e8e6c1bf7e719f298b9e539e74822488e92bf1fb"
require_sha "$BOOTSTRAP" "de2b18873dcd9f05f2d1d7acd9c0d94088680fb009441a501b8ba31ee8ce10b5"

require_b_acceptance "$ROOT_DIR/runs/runtime/kd_baselines_npu/phaseO_o12/o12b_neutral_smoke20_seed1234/acceptance.json" \
  "90eb89fe7a7dff773845152d5d59a7efcc3d61299976137223be6cf7d2615e1c" neutral fresh
require_b_acceptance "$ROOT_DIR/runs/runtime/kd_baselines_npu/phaseO_o12/o12b_neutral_smoke20_seed1234_resume_audit/acceptance.json" \
  "30fe00fa72b2f20293db2cfca3da7f0f9d960649fef8d153629b3d2b2f496677" neutral resume_audit
require_b_acceptance "$ROOT_DIR/runs/runtime/kd_baselines_npu/phaseO_o12/o12b_unreliable_only_smoke20_seed1234/acceptance.json" \
  "d8a3363724c7c5d93c7629ba0ebf365a7f73d1c25c53138633e179e21fe623cd" unreliable_only fresh
require_b_acceptance "$ROOT_DIR/runs/runtime/kd_baselines_npu/phaseO_o12/o12b_unreliable_only_smoke20_seed1234_resume_audit/acceptance.json" \
  "af540bf53a0b1646227eb9024a6eb9f5d7bf1af250773c62d1074b77aaa82488" unreliable_only resume_audit

actual_status="$(git status --porcelain --untracked-files=all)"
[[ "$actual_status" == "$EXPECTED_GIT_STATUS" ]] || {
  echo "[O1.2-C1] git status must contain exactly the four quarantined old scripts" >&2
  git status --short >&2
  exit 2
}
for tracked in \
  scripts/experiments/kd_baselines_npu/run_phaseO_o12_c1_variant.sh \
  scripts/experiments/kd_baselines_npu/run_phaseO_o12_c1_pair.sh \
  scripts/experiments/kd_baselines_npu/launch_phaseO_o12_c1_20k.sh \
  scripts/experiments/kd_baselines_npu/monitor_phaseO_o12_c1_20k.sh \
  scripts/experiments/kd_baselines_npu/check_phaseO_o12_c1_run.py \
  scripts/diagnostics/evaluate_phaseO_o12_c1_final.py \
  scripts/diagnostics/check_phaseO_o12_c1_gate.py \
  tests/test_phaseO_o12_c1_checker.py \
  tests/test_phaseO_o12_c1_evaluator.py \
  reports/2026-07-13_phaseO_rtc_o12_budgeted_routing_plan.md; do
  git ls-files --error-unmatch "$tracked" >/dev/null 2>&1 || fail "required input is not tracked: $tracked"
done

if pgrep -af '[t]rain_kd.py' >/dev/null 2>&1; then
  echo "[O1.2-C1] refusing launch while train_kd.py is live:" >&2
  pgrep -af '[t]rain_kd.py' >&2 || true
  exit 2
fi
require_device 0
require_device 1

for path in "$PID_FILE" "$STATUS_FILE" "$LOG_FILE" "$LOCK_DIR"; do
  [[ ! -e "$path" ]] || fail "refusing repeated controller launch: $path"
done
for variant in neutral unreliable_only; do
  name="o12c1_${variant}_20k_seed1234"
  for path in \
    "$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseO_o12_c1/$name" \
    "$ROOT_DIR/runs/kd_baselines_npu/phaseO_o12_c1/$name" \
    "$RUNTIME_ROOT/$name" \
    "$RUNTIME_ROOT/.locks/$name.lock"; do
    [[ ! -e "$path" ]] || fail "refusing to overwrite existing C1 artifact: $path"
  done
done

mkdir -p "$RUNTIME_ROOT"
mkdir "$LOCK_DIR" || fail "controller lock acquisition failed: $LOCK_DIR"
setsid env O12_C1_CONTROLLER_AUTH=launch_phaseO_o12_c1_20k_v1 \
  bash "$PAIR" > "$LOG_FILE" 2>&1 < /dev/null &
controller_pid=$!
printf '%s\n' "$controller_pid" > "$PID_FILE"

for _ in {1..100}; do
  [[ -s "$STATUS_FILE" ]] && break
  kill -0 "$controller_pid" 2>/dev/null || break
  sleep 0.05
done
if [[ ! -s "$STATUS_FILE" ]]; then
  echo "[O1.2-C1] controller exited before writing status" >&2
  tail -n 120 "$LOG_FILE" >&2 || true
  exit 2
fi
status_state="$(awk -F= '$1 == "state" {print $2; exit}' "$STATUS_FILE")"
status_pid="$(awk -F= '$1 == "controller_pid" {print $2; exit}' "$STATUS_FILE")"
if [[ "$status_pid" != "$controller_pid" || ! "$status_state" =~ ^(starting|running)$ ]]; then
  echo "[O1.2-C1] controller did not enter a valid active state: pid=$status_pid state=$status_state" >&2
  tail -n 120 "$LOG_FILE" >&2 || true
  exit 2
fi
if ! kill -0 "$controller_pid" 2>/dev/null; then
  echo "[O1.2-C1] controller is not alive after active status publication" >&2
  tail -n 120 "$LOG_FILE" >&2 || true
  exit 2
fi

echo "[O1.2-C1] launched controller pid=$controller_pid"
echo "[O1.2-C1] status=$STATUS_FILE"
echo "[O1.2-C1] log=$LOG_FILE"
echo "[O1.2-C1] scope=neutral:NPU0 versus unreliable_only:NPU1; C2 is not authorized"
