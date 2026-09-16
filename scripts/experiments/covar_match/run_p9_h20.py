#!/usr/bin/env python3
"""Run one fresh P9 cohort, with one GPU per paired seed and a stage barrier."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.experiments.covar_match import run_p8_p9_single_gpu as queue
from scripts.diagnostics import summarize_p8_p9_experiments as summary

RUN_ROOT = ROOT / "runs/covar_match/P9_h20"
P9_ROOT = ROOT / "runs/covar_match/P9_pair2_temperature_response_h20"
REPORT_ROOT = ROOT / "reports/covar_match/P9_h20"


def assignments(gpus):
    if not 1 <= len(gpus) <= len(queue.SEEDS) or len(set(gpus)) != len(gpus):
        raise ValueError("specify one to three distinct GPU indices")
    if any(gpu < 0 for gpu in gpus):
        raise ValueError("GPU indices must be nonnegative")
    return [(seed, gpus[index % len(gpus)]) for index, seed in enumerate(queue.SEEDS)]


def setup(gpu):
    os.chdir(ROOT)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
                      PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        os.environ.pop(key, None)
    for key, name in (("TMPDIR", "tmp"), ("MPLCONFIGDIR", "matplotlib"),
                      ("TORCH_HOME", "torch"), ("HF_HOME", "huggingface"),
                      ("XDG_CACHE_HOME", "xdg")):
        path = ROOT / ".runtime-cache" / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    queue.RUN_ROOT, queue.P9_ROOT, queue.REPORT_ROOT = RUN_ROOT, P9_ROOT, REPORT_ROOT


def record_protocol(gpus, smoke):
    reference = json.loads((ROOT / "reports/covar_match/P8_P9_single_gpu/protocol.json").read_text())
    for name, expected in reference["sha256"].items():
        if queue.digest(ROOT / name) != expected:
            raise RuntimeError(f"training source or weight differs from reference: {name}")
    inventory = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    rows = {int(row.split(",")[0]): row.split(",") for row in inventory.splitlines()}
    for gpu in gpus:
        row = rows[gpu]
        if int(row[-2]) > 1024 or int(row[-1]) > 10:
            raise RuntimeError(f"GPU {gpu} is already busy: {row}")
    queue.record_protocol()
    path = REPORT_ROOT / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol.update(cohort="P9_h20", ce_runs=0, smoke=smoke,
                    seed_gpu_mapping=dict(assignments(gpus)),
                    selected_gpus={gpu: rows[gpu] for gpu in gpus},
                    legacy_h100_results_pooled=False,
                    runtime_changes_from_h100={
                        key: {"reference": reference[key], "current": protocol[key]}
                        for key in ("gpu", "python", "torch", "cuda", "cudnn", "numpy", "opencv")
                        if reference[key] != protocol[key]},
                    initialization="fresh ImageNet backbone and same-seed head for each temperature",
                    cross_pair_scope="P7 differs in student capacity, execution and unverified software; capacity effect is not isolated")
    for path_source in [Path(__file__), ROOT / "scripts/diagnostics/summarize_p8_p9_experiments.py"]:
        protocol["sha256"][str(path_source.relative_to(ROOT))] = queue.digest(path_source)
    queue.write_json(path, protocol)


def worker(seed, gpu, stage, smoke):
    setup(gpu)
    prefix = "smoke-" if smoke else ""
    state_path = RUN_ROOT / "runtime" / f"{prefix}seed{seed}-stage{stage}.json"
    if state_path.exists():
        raise RuntimeError(f"existing worker state is preserved: {state_path}")
    state = dict(seed=seed, gpu=gpu, stage=stage, pid=os.getpid(), completed_runs=[])
    temperatures = ("1.0",) if smoke else (
        queue.P9_STAGE1_TEMPERATURES if stage == 1 else queue.P9_STAGE2_TEMPERATURES)
    try:
        for temperature in temperatures:
            queue.execute_run(seed, "kd", temperature, state, state_path, smoke=smoke)
        state["status"] = "COMPLETE"
    except Exception as error:
        state.update(status="FAILED", error=str(error))
        raise
    finally:
        queue.write_json(state_path, state)


def run_stage(gpus, stage, smoke):
    pairs = assignments(gpus)
    for offset in range(0, len(pairs), len(gpus)):
        children = []
        for seed, gpu in pairs[offset:offset + len(gpus)]:
            command = [sys.executable, "-B", "-u", __file__, "--worker-seed", str(seed),
                       "--gpu", str(gpu), "--stage", str(stage)]
            if smoke:
                command.append("--smoke")
            prefix = "smoke-" if smoke else ""
            log = (RUN_ROOT / "runtime" / f"{prefix}seed{seed}-stage{stage}.log").open("x")
            try:
                children.append((seed, subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                                       stdout=log, stderr=subprocess.STDOUT)))
            finally:
                log.close()
        # Finish the active wave before starting another seed or stage 2.
        failures = [(seed, code) for seed, process in children if (code := process.wait()) != 0]
        if failures:
            raise RuntimeError(f"workers failed; outputs preserved: {failures}")


def make_report():
    queue.generate_reports("p9")
    path = REPORT_ROOT / "P9_temperature.json"
    payload = json.loads(path.read_text())
    payload["runtime_protocol"] = json.loads((REPORT_ROOT / "protocol.json").read_text())
    temperatures = payload["final_grid_temperatures"]
    per_seed_near = {}
    gaps = {temperature: [] for temperature in temperatures}
    winner = payload["final_summary"]["mean_winner"]
    for seed, runs in payload["runs"].items():
        best = max(run["final_miou_percent"] for run in runs.values())
        per_seed_near[seed] = [t for t in temperatures if runs[t]["final_miou_percent"] >= best - 0.2]
        for temperature in temperatures:
            gaps[temperature].append(runs[winner]["final_miou_percent"] - runs[temperature]["final_miou_percent"])
    payload["per_seed_delta_optimal_grid_sets"] = per_seed_near
    payload["paired_mean_winner_minus_temperature_pp"] = gaps
    queue.write_json(path, payload)
    summary.write_p9_markdown(payload, path.with_suffix(".md"))
    with path.with_suffix(".md").open("a") as handle:
        handle.write("\n## H20 cohort\n\nAll runs use the runtime in [protocol.json](protocol.json). "
                     "Earlier H100 runs are not pooled. The nominally best temperature and delta sets "
                     "are descriptive on this finite grid with three seeds; they are not a proof of "
                     "a population optimum or of non-identifiability.\n\n")
        handle.write("Per-seed delta-near-optimal sets: `" + json.dumps(per_seed_near) + "`.\n\n")
        handle.write("Classification KD interactions were studied by [Frank and Davis (2026)]"
                     "(https://arxiv.org/abs/2603.02430). Here the measurements concern dense prediction, "
                     "temperature response and teacher CoVar coordinates.\n")
    return payload["stage1"]["phase2_gate"]


def main():
    global REPORT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[6, 7])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker-seed", type=int, choices=queue.SEEDS)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.worker_seed is not None:
        if args.gpu is None:
            parser.error("worker requires --gpu")
        worker(args.worker_seed, args.gpu, args.stage, args.smoke)
        return
    assignments(args.gpus)
    if args.smoke:
        REPORT_ROOT = REPORT_ROOT / "smoke"
    setup(args.gpus[0])
    runtime = RUN_ROOT / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = runtime / ("smoke-state.json" if args.smoke else "state.json")
        if state_path.exists():
            raise RuntimeError(f"existing cohort state is preserved: {state_path}")
        state = dict(status="PREPARING", pid=os.getpid(), gpus=args.gpus,
                     started_at=dt.datetime.now(dt.timezone.utc).isoformat())
        queue.write_json(state_path, state)
        try:
            record_protocol(args.gpus, args.smoke)
            state.update(status="RUNNING", stage=1)
            queue.write_json(state_path, state)
            run_stage(args.gpus, 1, args.smoke)
            if not args.smoke:
                gate = make_report()
                queue.write_json(REPORT_ROOT / "stage2_decision.json", gate)
                for suffix in (".json", ".md", ".csv"):
                    shutil.copyfile(REPORT_ROOT / ("P9_temperature" + suffix),
                                    REPORT_ROOT / ("P9_stage1" + suffix))
                state["stage2_required"] = gate["required"]
                if gate["required"]:
                    state["stage"] = 2
                    queue.write_json(state_path, state)
                    run_stage(args.gpus, 2, False)
                    make_report()
            state.update(status="COMPLETE", finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        except Exception as error:
            state.update(status="FAILED", error=str(error))
            raise
        finally:
            queue.write_json(state_path, state)


if __name__ == "__main__":
    main()
