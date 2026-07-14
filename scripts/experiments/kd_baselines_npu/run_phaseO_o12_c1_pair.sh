#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

[[ "$#" -eq 0 ]] || { echo "usage: $0" >&2; exit 2; }
[[ "${O12_C1_CONTROLLER_AUTH:-}" == "launch_phaseO_o12_c1_20k_v1" ]] || {
  echo "[O1.2-C1] pair controller must be started by launch_phaseO_o12_c1_20k.sh" >&2
  exit 2
}

RUNNER="$ROOT_DIR/scripts/experiments/kd_baselines_npu/run_phaseO_o12_c1_variant.sh"
RUNTIME_ROOT="$ROOT_DIR/runs/runtime/kd_baselines_npu/phaseO_o12_c1"
PID_FILE="$RUNTIME_ROOT/controller.pid"
STATUS_FILE="$RUNTIME_ROOT/controller.status"
LOCK_DIR="$RUNTIME_ROOT/.controller.lock"
NEUTRAL_RUNTIME="$RUNTIME_ROOT/o12c1_neutral_20k_seed1234"
UNRELIABLE_RUNTIME="$RUNTIME_ROOT/o12c1_unreliable_only_20k_seed1234"

[[ -x "$RUNNER" ]] || { echo "[O1.2-C1] runner missing: $RUNNER" >&2; exit 2; }
[[ -d "$LOCK_DIR" ]] || { echo "[O1.2-C1] controller lock missing: $LOCK_DIR" >&2; exit 2; }
mkdir -p "$RUNTIME_ROOT"
printf '%s\n' "$$" > "$PID_FILE"

START_UTC="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
NEUTRAL_PID=""
UNRELIABLE_PID=""
NEUTRAL_STATUS="pending"
UNRELIABLE_STATUS="pending"
SIGNAL_HANDLED=0

write_status() {
  local state="$1" reason="$2" controller_exit="$3"
  local tmp="$STATUS_FILE.tmp.$$"
  {
    echo "schema_version=1"
    echo "phase=O1.2-C1"
    echo "run_kind=20k_signal_screen_pair"
    echo "scope=neutral_vs_unreliable_only_training"
    echo "state=$state"
    echo "reason=$reason"
    echo "controller_pid=$$"
    echo "controller_pgid=$(ps -o pgid= -p $$ | tr -d ' ')"
    echo "controller_start_utc=$START_UTC"
    echo "status_utc=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
    echo "controller_exit_status=$controller_exit"
    echo "neutral_worker_pid=$NEUTRAL_PID"
    echo "neutral_worker_exit_status=$NEUTRAL_STATUS"
    echo "unreliable_only_worker_pid=$UNRELIABLE_PID"
    echo "unreliable_only_worker_exit_status=$UNRELIABLE_STATUS"
    echo "neutral_acceptance=$NEUTRAL_RUNTIME/acceptance.json"
    echo "unreliable_only_acceptance=$UNRELIABLE_RUNTIME/acceptance.json"
    echo "c2_started=false"
  } > "$tmp"
  mv "$tmp" "$STATUS_FILE"
}

terminate_group() {
  local pid="$1"
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 0
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in {1..40}; do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.25
  done
  kill -KILL -- "-$pid" 2>/dev/null || true
}

handle_signal() {
  local name="$1" code="$2"
  [[ "$SIGNAL_HANDLED" -eq 0 ]] || exit "$code"
  SIGNAL_HANDLED=1
  trap - INT TERM
  write_status terminating "signal_$name" "$code"
  terminate_group "$NEUTRAL_PID"
  terminate_group "$UNRELIABLE_PID"
  set +e
  [[ -z "$NEUTRAL_PID" ]] || wait "$NEUTRAL_PID" 2>/dev/null
  [[ -z "$UNRELIABLE_PID" ]] || wait "$UNRELIABLE_PID" 2>/dev/null
  set -e
  NEUTRAL_STATUS="terminated_$code"
  UNRELIABLE_STATUS="terminated_$code"
  write_status failed "signal_$name" "$code"
  exit "$code"
}
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM

write_status starting preflight_complete pending
echo "[O1.2-C1] pair controller pid=$$ start=$START_UTC"

setsid "$RUNNER" neutral 0 &
NEUTRAL_PID=$!
setsid "$RUNNER" unreliable_only 1 &
UNRELIABLE_PID=$!
write_status running workers_started pending
echo "[O1.2-C1] neutral worker pid=$NEUTRAL_PID physical_npu=0"
echo "[O1.2-C1] unreliable_only worker pid=$UNRELIABLE_PID physical_npu=1"

set +e
COMPLETED_PID=""
wait -n -p COMPLETED_PID "$NEUTRAL_PID" "$UNRELIABLE_PID"
FIRST_STATUS=$?
set -e

if [[ "$COMPLETED_PID" == "$NEUTRAL_PID" ]]; then
  NEUTRAL_STATUS="$FIRST_STATUS"
  OTHER_PID="$UNRELIABLE_PID"
  OTHER_VARIANT="unreliable_only"
else
  UNRELIABLE_STATUS="$FIRST_STATUS"
  OTHER_PID="$NEUTRAL_PID"
  OTHER_VARIANT="neutral"
fi

if [[ "$FIRST_STATUS" -ne 0 ]]; then
  write_status terminating "first_worker_failed_pid_${COMPLETED_PID}" "$FIRST_STATUS"
  echo "[O1.2-C1] worker pid=$COMPLETED_PID failed status=$FIRST_STATUS; terminating sibling" >&2
  terminate_group "$OTHER_PID"
  set +e
  wait "$OTHER_PID"
  OTHER_STATUS=$?
  set -e
  if [[ "$OTHER_VARIANT" == neutral ]]; then
    NEUTRAL_STATUS="$OTHER_STATUS"
  else
    UNRELIABLE_STATUS="$OTHER_STATUS"
  fi
  write_status failed worker_failed "$FIRST_STATUS"
  exit "$FIRST_STATUS"
fi

set +e
wait "$OTHER_PID"
OTHER_STATUS=$?
set -e
if [[ "$OTHER_VARIANT" == neutral ]]; then
  NEUTRAL_STATUS="$OTHER_STATUS"
else
  UNRELIABLE_STATUS="$OTHER_STATUS"
fi
if [[ "$OTHER_STATUS" -ne 0 ]]; then
  echo "[O1.2-C1] worker variant=$OTHER_VARIANT failed status=$OTHER_STATUS" >&2
  write_status failed second_worker_failed "$OTHER_STATUS"
  exit "$OTHER_STATUS"
fi

PYTHON="/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python"
if ! "$PYTHON" -c '
import json, sys
for path, variant in ((sys.argv[1], "neutral"), (sys.argv[2], "unreliable_only")):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {"phase": "O1.2-C1", "stage": "final", "mode": "fresh",
                "variant": variant, "pass": True, "errors": []}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise SystemExit(f"{variant} final acceptance mismatch: {key}")
' "$NEUTRAL_RUNTIME/acceptance.json" "$UNRELIABLE_RUNTIME/acceptance.json"; then
  write_status failed final_acceptance_invalid 2
  exit 2
fi

NEUTRAL_STATUS=0
UNRELIABLE_STATUS=0
write_status completed both_final_acceptances_passed 0
echo "[O1.2-C1] both 20k training runs completed and passed final acceptance"
echo "[O1.2-C1] C2 was not started"
