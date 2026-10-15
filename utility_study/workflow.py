"""Bounded two-GPU diagnostic queue with an explicit scientific stopping gate."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import os
import statistics
import subprocess
import sys

from .io import read_json, sha256, verify_sources, write_json


def gate_from_values(values):
    datasets = {}
    for dataset in ("voc", "cityscapes"):
        stages = []
        for stage in (20000, 40000, 60000):
            seed_values = {str(seed): values[dataset, seed, stage] for seed in (1234, 2025, 3407)}
            stages.append({"checkpoint": stage, "seed_differences": seed_values,
                           "mean": statistics.mean(seed_values.values()),
                           "sample_sd": statistics.stdev(seed_values.values())})
        positive = sum(row["mean"] > 0 for row in stages)
        datasets[dataset] = {"stages": stages, "positive_stages": positive, "pass": positive >= 2}
    return {"pass": all(d["pass"] for d in datasets.values()), "datasets": datasets,
            "definition": "Shared c4 minus c1 pixel CE, matched_train, CE starts, 1 virtual update",
            "unit": "three training seeds; stages are repeated measurements",
            "role": "resource gate, not a statistical significance test",
            "full_training_authorized_by_this_gate_alone": False}


def primary_gate(out, registration_hash):
    values = {}
    for dataset in ("voc", "cityscapes"):
        for seed in (1234, 2025, 3407):
            for stage in (20000, 40000, 60000):
                base = Path(out) / "one_step" / dataset / "ce" / str(seed) / str(stage) / "matched_train"
                a, b = (read_json(base / (name + ".json")) for name in ("shared_c1", "shared_c4"))
                for row in (a, b):
                    assert row["status"] == "PASS" and row["registration_sha256"] == registration_hash
                    assert row["horizon"] == 1 and row["state_restored"]
                values[dataset, seed, stage] = b["after"]["1"]["pixel_ce"] - a["after"]["1"]["pixel_ce"]
    gate = gate_from_values(values)
    write_json(Path(out) / "one_step_gate.json", gate)
    return gate


def main(root, out, gpu_list):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "controller.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpus = [int(x) for x in gpu_list.split(",")]
    if len(gpus) != 2 or set(gpus) != {0, 2}:
        raise ValueError("Use exactly the registered two lanes, GPU 0 and GPU 2")
    state = {"status": "RUNNING", "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
             "max_gpus": 2, "gpus": gpus, "completed_phases": [], "full_training_runs_launched": 0,
             "checkpoint_files_written": 0}
    write_json(out / "workflow.json", state)

    def phase(name, cohorts="ce"):
        state["phase"] = name + ":" + cohorts
        write_json(out / "workflow.json", state)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = []
            for dataset, gpu in zip(("voc", "cityscapes"), gpus):
                command = [sys.executable, "-u", "-m", "utility_study.runner", "--registration", str(out / "registration.json"),
                           "--out", str(out), "--dataset", dataset, "--gpu", str(gpu), "--phase", name, "--cohorts", cohorts]
                logfile = out / (name + "_" + dataset + "_" + cohorts.replace(",", "_") + ".log")

                def execute(command=command, logfile=logfile, dataset=dataset, gpu=gpu):
                    with logfile.open("a") as log:
                        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                        write_json(out / "workers" / (dataset + ".json"), {"pid": child.pid, "command": command, "gpu": gpu, "phase": name})
                        code = child.wait()
                    if code:
                        raise RuntimeError(f"{dataset} {name} failed with code {code}: {logfile}")
                    return dataset

                jobs.append(pool.submit(execute))
            failures = []
            for job in as_completed(jobs):
                try:
                    print("LANE_PASS", job.result(), name, cohorts, flush=True)
                except Exception as error:
                    failures.append(repr(error))
            if failures:
                raise RuntimeError("; ".join(failures))
        state["completed_phases"].append(name + ":" + cohorts)
        write_json(out / "workflow.json", state)

    try:
        if not (out / "registration.json").exists():
            subprocess.run([sys.executable, "-u", "-m", "utility_study.register", "--root", str(root), "--out", str(out)], check=True)
        registered = read_json(out / "registration.json")
        verify_sources(registered["source_hashes"])
        state["registration_sha256"] = sha256(out / "registration.json")
        phase("smoke")
        phase("one_step", "ce")
        gate = primary_gate(out, state["registration_sha256"])
        state["primary_gate_pass"] = gate["pass"]
        if not gate["pass"]:
            state.update(status="SCIENTIFIC_STOP", phase="primary_gate",
                         reason="Historical local CE ordering did not replicate sufficiently under the matched training protocol.",
                         next_action="Review protocol dependence; selector training stopped by the registered gate.")
            write_json(out / "workflow.json", state)
            return
        phase("one_step", "shared_c1,shared_c4")
        phase("rollout", "ce,shared_c1,shared_c4")
        verify_sources(registered["source_hashes"])
        state.update(status="DIAGNOSTICS_COMPLETE", phase="complete",
                     next_action="Analyze matched-mode, trajectory, metric and horizon effects before selector GPU smoke.",
                     completed_at=datetime.now(timezone.utc).isoformat())
        write_json(out / "workflow.json", state)
    except Exception as error:
        state.update(status="FAILED", error=repr(error))
        write_json(out / "workflow.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpus", default="0,2")
    args = parser.parse_args()
    main(args.root, args.out, args.gpus)
