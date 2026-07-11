#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
RUN_DIR="$ROOT_DIR/data/winycg/checkpoints/kd_baselines_npu/phaseK_voc_80k"

show_queue() {
  local queue_name="$1"
  local expected_variants="$2"
  local pid_file="$RUN_DIR/${queue_name}.pid"
  local queue_log="$RUN_DIR/${queue_name}.nohup.log"

  echo "[kd-baselines-npu] queue=$queue_name expected=$expected_variants"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "[kd-baselines-npu] state=running pid=$pid"
    else
      echo "[kd-baselines-npu] state=stopped pid=$pid"
    fi
  else
    echo "[kd-baselines-npu] state=not-launched"
  fi

  if [[ ! -f "$queue_log" ]]; then
    echo "[kd-baselines-npu] missing log=$queue_log"
    return 0
  fi

  local current_variant completed latest_iter latest_val
  current_variant="$(grep '\[kd-baselines-npu\].*variant=' "$queue_log" | tail -n 1 || true)"
  completed="$(grep -c 'Total training time:' "$queue_log" || true)"
  latest_iter="$(grep 'Iters:' "$queue_log" | tail -n 1 || true)"
  latest_val="$(grep 'Sample: 1449,' "$queue_log" | tail -n 1 || true)"

  echo "[kd-baselines-npu] completed_variants=$completed"
  [[ -n "$current_variant" ]] && echo "$current_variant"
  [[ -n "$latest_iter" ]] && echo "$latest_iter"
  [[ -n "$latest_val" ]] && echo "$latest_val"
  tail -n 3 "$queue_log"
}

show_queue "phaseK_npu0_cwd_ifvd" "cwd_80k,ifvd_80k"
show_queue "phaseK_npu1_skd_kdonly" "skd_80k,kdonly_80k"

npu-smi info
