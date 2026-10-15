"""Independent seed jobs on registered idle GPUs, with a fixed scientific gate."""
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
import threading
import time

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


def phase_jobs(name, seeds, stages=(20000, 40000, 60000)):
    if name == "smoke":
        return [(dataset, 1234, 20000) for dataset in ("voc", "cityscapes")]
    # Native-resolution Cityscapes monitoring takes longer. Queue these first
    # so both lanes can contribute rather than leaving a VOC lane idle.
    return [(dataset, seed, stage) for dataset in ("cityscapes", "voc") for seed in seeds for stage in stages]


def job_inputs(registered, dataset, seed, stage, cohorts):
    runs = registered["datasets"][dataset]["runs"]
    return [next(r for r in runs if r["seed"] == seed and r["condition"] == cohort)["checkpoints"][str(stage)]
            for cohort in cohorts.split(",")]


def pop_ready_job(pending, registered, cohorts):
    # Caller holds the queue lock. Transfer files become visible by atomic rename
    # only after stream and readback SHA256 verification; runner rechecks hashes.
    for index, (dataset, seed, stage) in enumerate(pending):
        if all(Path(p).is_file() for p in job_inputs(registered, dataset, seed, stage, cohorts)):
            return pending.pop(index)
    return None


def validate_gpus(gpus, registered):
    if not gpus or len(set(gpus)) != len(gpus) or set(gpus) != set(registered["allowed_gpus"]):
        raise ValueError("GPU lanes must match the immutable registered allocation")
    if len(gpus) > registered["max_gpus"]:
        raise ValueError("GPU concurrency exceeds the registered allocation")


def main(root, out, gpu_list):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "controller.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpus = [int(x) for x in gpu_list.split(",")]
    state = {"status": "RUNNING", "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
             "max_gpus": len(gpus), "gpus": gpus, "completed_phases": [], "full_training_runs_launched": 0,
             "checkpoint_files_written": 0}
    write_json(out / "workflow.json", state)

    def phase(name, cohorts="ce"):
        state["phase"] = name + ":" + cohorts
        write_json(out / "workflow.json", state)
        pending = phase_jobs(name, registered["seeds"], registered["checkpoints"])
        queue_lock = threading.Lock()
        failed = threading.Event()
        waiting_since = time.monotonic()

        def lane(gpu):
            while not failed.is_set():
                with queue_lock:
                    if not pending:
                        return
                    ready = pop_ready_job(pending, registered, cohorts)
                if ready is None:
                    if time.monotonic() - waiting_since > 24 * 3600:
                        failed.set()
                        raise TimeoutError("Registered checkpoint copies did not arrive within 24 hours")
                    write_json(out / "lanes" / ("gpu" + str(gpu) + ".json"),
                               {"status": "WAITING_INPUT", "gpu": gpu, "phase": name,
                                "updated_at": datetime.now(timezone.utc).isoformat()})
                    failed.wait(5)
                    continue
                dataset, seed, stage = ready
                job_id = name + "_" + dataset + "_" + cohorts.replace(",", "_") + "_seed" + str(seed) + "_stage" + str(stage)
                command = [sys.executable, "-u", "-m", "utility_study.runner", "--registration", str(out / "registration.json"),
                           "--out", str(out), "--dataset", dataset, "--gpu", str(gpu), "--phase", name,
                           "--cohorts", cohorts, "--seeds", str(seed), "--stages", str(stage)]
                logfile = out / (job_id + ".log")
                worker_path = out / "workers" / (job_id + ".json")
                try:
                    with logfile.open("a") as log:
                        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                        worker = {"status": "RUNNING", "pid": child.pid, "command": command, "gpu": gpu,
                                  "phase": name, "dataset": dataset, "seed": seed, "checkpoint": stage}
                        write_json(worker_path, worker)
                        write_json(out / "lanes" / ("gpu" + str(gpu) + ".json"), worker)
                        code = child.wait()
                    worker.update(status="PASS" if code == 0 else "FAILED", returncode=code)
                    write_json(worker_path, worker)
                    if code:
                        raise RuntimeError(f"{dataset} {name} failed with code {code}: {logfile}")
                    print("LANE_PASS", dataset, seed, stage, name, cohorts, gpu, flush=True)
                except Exception:
                    failed.set()
                    raise

        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            jobs = [pool.submit(lane, gpu) for gpu in gpus]
            failures = []
            for job in as_completed(jobs):
                try:
                    job.result()
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
        validate_gpus(gpus, registered)
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
