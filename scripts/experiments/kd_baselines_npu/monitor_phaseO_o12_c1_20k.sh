#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[[ "$#" -eq 0 ]] || { echo "usage: $0" >&2; exit 2; }

RUNTIME_ROOT="$ROOT_DIR/runs/runtime/kd_baselines_npu/phaseO_o12_c1"
LOG_ROOT="$ROOT_DIR/runs/kd_baselines_npu/phaseO_o12_c1"
PID_FILE="$RUNTIME_ROOT/controller.pid"
STATUS_FILE="$RUNTIME_ROOT/controller.status"
CONTROLLER_LOG="$RUNTIME_ROOT/controller.log"
LOGGER_NAME="deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"

show_pid() {
  local label="$1" pid="$2"
  if [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "$label: alive"
    ps -o pid=,ppid=,pgid=,stat=,etime=,args= -p "$pid" 2>/dev/null || true
  elif [[ -n "$pid" ]]; then
    echo "$label: not alive (recorded pid=$pid)"
  else
    echo "$label: no PID recorded"
  fi
}

status_value() {
  local key="$1"
  [[ -f "$STATUS_FILE" ]] || return 0
  awk -F= -v key="$key" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' "$STATUS_FILE"
}

echo "O1.2-C1 monitor $(date -Is)"
controller_pid=""
[[ ! -f "$PID_FILE" ]] || controller_pid="$(tr -d '[:space:]' < "$PID_FILE")"
show_pid controller "$controller_pid"

if [[ -f "$STATUS_FILE" ]]; then
  echo "controller status:"
  sed 's/^/  /' "$STATUS_FILE"
else
  echo "controller status: not created"
fi

neutral_worker="$(status_value neutral_worker_pid)"
unreliable_worker="$(status_value unreliable_only_worker_pid)"
show_pid neutral_worker "$neutral_worker"
show_pid unreliable_only_worker "$unreliable_worker"

for variant in neutral unreliable_only; do
  name="o12c1_${variant}_20k_seed1234"
  runtime="$RUNTIME_ROOT/$name"
  logger="$LOG_ROOT/$name/$LOGGER_NAME"
  training_pid=""
  [[ ! -f "$runtime/training.pid" ]] || training_pid="$(tr -d '[:space:]' < "$runtime/training.pid")"
  echo "$variant:"
  show_pid "  training" "$training_pid"
  if [[ -f "$logger" ]]; then
    latest_iter="$(grep -F 'Iters:' "$logger" 2>/dev/null | tail -n 1)"
    [[ -z "$latest_iter" ]] || echo "  latest iteration: $latest_iter"
    completed_validations="$(grep -Fc 'Sample: 1449, Validation' "$logger" 2>/dev/null || true)"
    echo "  completed validations: $completed_validations/25"
    latest_val="$(grep -F 'Sample: 1449, Validation' "$logger" 2>/dev/null | tail -n 1)"
    [[ -z "$latest_val" ]] || echo "  latest completed validation: $latest_val"
    active_val="$(grep -F 'Sample:' "$logger" 2>/dev/null | tail -n 1)"
    if [[ -n "$active_val" && "$active_val" != "$latest_val" ]]; then
      echo "  current validation progress: $active_val"
    fi
  else
    echo "  training logger: not created"
  fi
  if [[ -f "$runtime/acceptance.json" ]]; then
    /home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python -c '
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    p = json.load(handle)
v = p.get("validation", {})
print("  final acceptance: pass={} errors={} final_mIoU={} best_mIoU={} last10={}".format(
    p.get("pass"), len(p.get("errors", [])), v.get("final_mIoU"),
    v.get("best_mIoU"), v.get("last10_mean_mIoU")))
' "$runtime/acceptance.json" 2>/dev/null || echo "  final acceptance: unreadable"
  elif [[ -f "$runtime/acceptance.prefinal.json" ]]; then
    echo "  acceptance: prefinal exists; final pending"
  else
    echo "  acceptance: pending"
  fi
  if [[ -f "$runtime/console.log" ]]; then
    echo "  console tail:"
    tail -n 4 "$runtime/console.log" | sed 's/^/    /'
  fi
done

echo "NPU status:"
npu-smi info 2>&1 || true

if [[ -f "$CONTROLLER_LOG" ]]; then
  echo "controller log tail:"
  tail -n 30 "$CONTROLLER_LOG" | sed 's/^/  /'
else
  echo "controller log: not created"
fi
