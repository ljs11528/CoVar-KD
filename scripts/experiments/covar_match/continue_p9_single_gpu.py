#!/usr/bin/env python3
"""Continue the approved P9 grid on server 2 with fresh, isolated outputs."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.experiments.covar_match import run_p8_p9_single_gpu as queue

REFERENCE_REPORT = ROOT / "reports/covar_match/P8_P9_single_gpu"
REFERENCE_P9 = queue.P9_ROOT
SHARED_LOCK = queue.RUN_ROOT / "runtime/queue.lock"
RUN_ROOT = ROOT / "runs/covar_match/P9_single_gpu_server2"
P9_ROOT = ROOT / "runs/covar_match/P9_pair2_temperature_response_single_gpu_server2"
REPORT_ROOT = REFERENCE_REPORT / "server2"
RUNTIME_FIELDS = ("gpu", "python", "torch", "cuda", "cudnn", "numpy", "opencv")


def configure_paths():
    queue.RUN_ROOT = RUN_ROOT
    queue.P9_ROOT = P9_ROOT
    queue.REPORT_ROOT = REPORT_ROOT


def verify_reference():
    reference = json.loads((REFERENCE_REPORT / "protocol.json").read_text())
    for relative, expected in reference["sha256"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or queue.digest(path) != expected:
            raise RuntimeError(f"locked source or weight differs: {relative}")
    for seed in queue.SEEDS:
        queue.read_run(queue.P8_ROOT, queue.p8_variant(seed, "single_gpu"),
                       queue.SMALL_LOG_NAME, seed, "mobilenetv3_small",
                       0.0, "1.0", "single_gpu")
    return reference


def verify_runtime(reference, current):
    mismatches = {key: [reference[key], current[key]]
                  for key in RUNTIME_FIELDS if reference[key] != current[key]}
    if mismatches:
        raise RuntimeError(f"runtime differs from the approved single-GPU profile: {mismatches}")


def record_continuation(reference, smoke):
    queue.record_protocol()
    path = REPORT_ROOT / "protocol.json"
    current = json.loads(path.read_text())
    verify_runtime(reference, current)
    log = REFERENCE_P9 / "logs/fixed_T0p25_80k_seed1234" / queue.LARGE_LOG_NAME
    with log.open("rb") as handle:
        handle.seek(max(0, log.stat().st_size - 16384))
        text = handle.read().decode("utf-8", errors="replace")
    steps = re.findall(r"Iters:\s*(\d+)/80000", text)
    current.update(
        ce_runs=0, completed_ce_reference_seeds=list(queue.SEEDS),
        hostname=socket.gethostname(), smoke=smoke,
        reference_protocol=str(REFERENCE_REPORT / "protocol.json"),
        reference_protocol_sha256=queue.digest(REFERENCE_REPORT / "protocol.json"),
        p9_output_root=str(P9_ROOT),
        continuation_policy="fresh 80k for every P9 condition; preserve and exclude the interrupted attempt",
        interrupted_attempt={
            "root": str(REFERENCE_P9), "variant": "fixed_T0p25_80k_seed1234",
            "latest_logged_iteration": int(steps[-1]) if steps else None,
            "performance_summary_includes_attempt": False,
        },
    )
    current["sha256"][str(Path(__file__).relative_to(ROOT))] = queue.digest(__file__)
    queue.write_json(path, current)


def run_grid(temperatures, state, state_path):
    for seed in queue.SEEDS:
        for temperature in temperatures:
            queue.execute_run(seed, "kd", temperature, state, state_path)
            variant = queue.p9_variant(temperature, seed)
            if variant not in state["completed_runs"]:
                state["completed_runs"].append(variant)
                queue.write_json(state_path, state)


def run_approved_plan(state, state_path):
    run_grid(queue.P9_STAGE1_TEMPERATURES, state, state_path)
    queue.generate_reports("p9")
    payload = json.loads((REPORT_ROOT / "P9_temperature.json").read_text())
    gate = payload["stage1"]["phase2_gate"]
    queue.write_json(REPORT_ROOT / "stage2_decision.json", gate)
    for suffix in (".json", ".md", ".csv"):
        source = REPORT_ROOT / ("P9_temperature" + suffix)
        archive = REPORT_ROOT / ("P9_stage1" + suffix)
        if archive.exists():
            if archive.read_bytes() != source.read_bytes():
                raise RuntimeError(f"existing stage-1 archive differs: {archive}")
        else:
            shutil.copyfile(source, archive)
    state["stage2_required"] = gate["required"]
    queue.write_json(state_path, state)
    if gate["required"]:
        run_grid(queue.P9_STAGE2_TEMPERATURES, state, state_path)
        queue.generate_reports("p9")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="one isolated 20-step Large KD environment check")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4",
                      PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        os.environ.pop(key, None)
    for key, name in (("TMPDIR", "tmp"), ("MPLCONFIGDIR", "matplotlib"),
                      ("TORCH_HOME", "torch"), ("HF_HOME", "huggingface"),
                      ("XDG_CACHE_HOME", "xdg")):
        path = ROOT / ".runtime-cache" / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    # Share the original lock so both queue entry points cannot run together.
    with SHARED_LOCK.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configure_paths()
        runtime = RUN_ROOT / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        prefix = "smoke-" if args.smoke else ""
        state_path = runtime / (prefix + "state.json")
        if state_path.exists():
            raise RuntimeError(f"existing queue state is preserved: {state_path}")
        state = dict(status="PREPARING", pid=os.getpid(), hostname=socket.gethostname(),
                     smoke=args.smoke, completed_runs=[],
                     started_at=dt.datetime.now(dt.timezone.utc).isoformat())
        queue.write_json(state_path, state)
        (runtime / (prefix + "controller.pid")).write_text(str(os.getpid()) + "\n")
        try:
            reference = verify_reference()
            record_continuation(reference, args.smoke)
            if args.smoke:
                queue.execute_run(1234, "kd", "1.0", state, state_path, smoke=True)
            else:
                run_approved_plan(state, state_path)
            state.update(status="COMPLETE",
                         finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
            queue.write_json(state_path, state)
            print("QUEUE_COMPLETE", json.dumps(state), flush=True)
        except Exception as error:
            state.update(status="FAILED", error=str(error),
                         updated_at=dt.datetime.now(dt.timezone.utc).isoformat())
            queue.write_json(state_path, state)
            raise


if __name__ == "__main__":
    main()
