#!/usr/bin/env python3
"""P10: frozen P9-compatible controls; wait for idle GPUs without preemption."""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.experiments.covar_match import run_p8_p9_single_gpu as base
from scripts.diagnostics import summarize_p8_p9_experiments as old

RUN_ROOT = ROOT / "runs/covar_match/P10_h20"
REPORT_ROOT = ROOT / "reports/covar_match/P10_h20"
REFERENCE = ROOT / "reports/covar_match/P9_h20"
SEEDS = old.SEEDS
TEMPERATURES = old.P9_STAGE1_TEMPERATURES
MILESTONES = old.MILESTONES


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def experiment_plan():
    groups = [(1, student, "ce", "1.0", 0.0) for student in ("small", "large")]
    groups += [(1, "small", "teacher_only", t, 1.0) for t in TEMPERATURES]
    groups += [(2, "large", "teacher_only", "4.0", 1.0)]
    groups += [(3, "large", "teacher_only", "2.0", 4.0),
               (3, "large", "masked", "2.0", 1.0),
               (3, "large", "masked", "2.0", 4.0)]
    runs = []
    for phase, student, kind, temperature, scale in groups:
        label = {"ce": "ce", "teacher_only": "teacher", "masked": "shared"}[kind]
        group = (f"{student}_ce" if kind == "ce" else
                 f"{student}_{label}_T{temperature.replace('.', 'p')}_L{scale:g}")
        for seed in SEEDS:
            runs.append(dict(phase=phase, group=group, student=student, seed=seed,
                             mode="teacher_only" if kind == "ce" else kind,
                             temperature=temperature, lambda_kd=scale,
                             variant=f"{group}_80k_seed{seed}"))
    return runs


def smoke_plan():
    return [s for s in experiment_plan() if s["seed"] == 1234 and
            (s["lambda_kd"] == 0 or s["phase"] > 1 or s["temperature"] == "0.25")]


def command_for(spec, smoke=False):
    if spec not in experiment_plan():
        raise ValueError("run is outside the registered P10 matrix")
    command = base.command_for(spec["seed"], "kd", "1.0", smoke=smoke)[0]
    root = RUN_ROOT / "smoke" if smoke else RUN_ROOT
    save_dir = root / "checkpoints" / spec["variant"]
    log_dir = root / "logs" / spec["variant"]
    student_base = base.BASES["ce" if spec["student"] == "small" else "kd"]
    options = {"--student-backbone": "mobilenetv3_" + spec["student"],
               "--student-pretrained-base": str(student_base),
               "--kd-temperature": spec["temperature"], "--kd-loss-mode": spec["mode"],
               "--lambda-kd": str(spec["lambda_kd"]),
               "--save-dir": str(save_dir), "--log-dir": str(log_dir)}
    for flag, value in options.items():
        command[command.index(flag) + 1] = value
    if spec["mode"] == "masked":
        # The total coefficient is lambda_kd. At T=2, lambda=4 is standard T² KD.
        command += ["--covar-kd-temp-power", "0.0"]
    return command, root, save_dir, log_dir


def gpu_inventory():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    rows = {}
    for line in output.splitlines():
        index, uuid, name, memory, utilization = [v.strip() for v in line.split(",")]
        rows[int(index)] = dict(uuid=uuid, name=name, memory_mib=int(memory),
                                utilization_percent=int(utilization))
    return rows


def idle(row):
    return row["memory_mib"] < 1024 and row["utilization_percent"] <= 10


def setup_environment(gpu=None):
    os.chdir(ROOT)
    os.environ.update(CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
                      OMP_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        os.environ.pop(key, None)
    for key, name in (("TMPDIR", "tmp"), ("MPLCONFIGDIR", "matplotlib"),
                      ("TORCH_HOME", "torch"), ("HF_HOME", "huggingface"),
                      ("XDG_CACHE_HOME", "xdg")):
        path = ROOT / ".runtime-cache" / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)


def register(gpus):
    import torch
    import numpy
    import cv2
    reference = json.loads((REFERENCE / "protocol.json").read_text())
    for name, expected in reference["sha256"].items():
        if base.digest(ROOT / name) != expected:
            raise RuntimeError(f"P9 source/weight changed: {name}")
    versions = dict(python=sys.version.split()[0], torch=torch.__version__,
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    numpy=numpy.__version__, opencv=cv2.__version__)
    if any(versions[k] != reference[k] for k in versions):
        raise RuntimeError(f"P9 environment differs: {versions}")
    inventory = gpu_inventory()
    if any(inventory[g]["name"] != reference["gpu"] for g in gpus):
        raise RuntimeError("P10 requires the P9 H20 GPU model")
    baseline = json.loads((REFERENCE / "P9_temperature.json").read_text())
    baseline_audit = json.loads((REFERENCE / "completion_audit.json").read_text())
    if baseline_audit["status"] != "PASS" or baseline_audit["completed_runs"] != 15:
        raise RuntimeError("P9 completed cohort is not audited")
    for seed in SEEDS:
        for temperature in TEMPERATURES:
            run = old.read_run(ROOT / "runs/covar_match/P9_pair2_temperature_response_h20",
                               old.p9_variant(temperature, seed), old.LARGE_LOG_NAME, seed,
                               "mobilenetv3_large", 1.0, temperature, "single_gpu")
            if run["final_miou_percent"] != baseline["runs"][str(seed)][temperature]["final_miou_percent"]:
                raise RuntimeError("P9 baseline log and report disagree")
            audited = baseline_audit["runs"][str(seed)][temperature]
            if base.digest(run["log_path"]) != audited["log_sha256"]:
                raise RuntimeError("P9 baseline log hash changed")
    extra_files = [Path(__file__), ROOT / "scripts/diagnostics/summarize_p10_h20.py",
                   ROOT / "tests/test_p10_h20.py", ROOT / "utils/teacher_only_kd.py",
                   ROOT / "utils/rtc_temperature.py", ROOT / "dataset/list/voc/train_aug.txt",
                   ROOT / "dataset/list/voc/val.txt"]
    hashes = {**reference["sha256"], **{str(p.relative_to(ROOT)): base.digest(p) for p in extra_files}}
    protocol = dict(cohort="P10_h20", registered_at=now(), runtime=versions,
                    reference_cohort="P9_h20", reference_sha256=base.digest(REFERENCE / "protocol.json"),
                    reference_results_sha256=base.digest(REFERENCE / "P9_temperature.json"),
                    baseline_audit_sha256=base.digest(REFERENCE / "completion_audit.json"),
                    training_source_unchanged_from_p9=True, source_sha256=hashes,
                    requested_gpus=gpus, initial_gpu_inventory=inventory,
                    execution_profile="single_gpu", global_batch_size=16, crop_size=[512, 512],
                    workers=4, iterations=80000, milestones=list(MILESTONES), seeds=list(SEEDS),
                    optimizer=reference["optimizer"], initialization=reference["initialization"],
                    new_runs=experiment_plan(), new_run_count=33, reused_p9_runs=15,
                    common_grid=list(TEMPERATURES), delta_pp=0.2,
                    boundary_rule="Evaluate T=4 once. Upper boundary is unresolved if T=4 belongs to the expanded grid's delta-near-optimal mean set. Never auto-expand to T=8.",
                    controls="Large Tt=2, Ts in {1,2}, total KL coefficient in {1,4}; mean-valid-pixel KL, CE unchanged. The Ts=1/coefficient=1 cell reuses P9. Shared Ts=2/coefficient=4 equals standard T^2 KD. No gradient-norm matching claim.",
                    preserved_p9_stage2_decision=json.loads((REFERENCE / "stage2_decision.json").read_text()),
                    scope="Frozen P9 protocol with new Small and CE controls; old P7/H100 outcomes are not pooled. Same-numbered seeds, no claim of identical random streams across capacities. T=2 loss control is a local sensitivity experiment, not a temperature-response grid for shared KD.")
    path = REPORT_ROOT / "protocol.json"
    if path.exists():
        saved = json.loads(path.read_text())
        for key in ("source_sha256", "runtime", "new_runs", "reference_results_sha256",
                    "reference_sha256", "baseline_audit_sha256", "requested_gpus"):
            if saved[key] != protocol[key]:
                raise RuntimeError(f"registered P10 protocol drift: {key}")
        return saved
    write_json(path, protocol)
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
                      if d.metadata.get("Name"))
    (REPORT_ROOT / "environment-freeze.txt").write_text("\n".join(packages) + "\n")
    return protocol


def read_run(spec, smoke=False):
    _, root, save_dir, log_dir = command_for(spec, smoke)
    name = old.SMALL_LOG_NAME if spec["student"] == "small" else old.LARGE_LOG_NAME
    log_path = log_dir / name
    text = log_path.read_text()
    maximum, trajectory = old.parse_log(text)
    checks = old.namespace_checks(text, seed=spec["seed"],
        student_backbone="mobilenetv3_" + spec["student"], lambda_kd=spec["lambda_kd"],
        temperature=spec["temperature"], execution_protocol="single_gpu")
    arguments = ast.parse("Namespace(" + re.findall(r"Namespace\(([^\n]+)\)", text)[0] + ")",
                          mode="eval").body
    args = {k.arg: ast.literal_eval(k.value) for k in arguments.keywords}
    expected_max = 20 if smoke else 80000
    milestones = (20,) if smoke else MILESTONES
    overrides = dict(kd_loss_mode=spec["mode"], max_iterations=expected_max,
                     skip_val=smoke, save_per_iters=20 if smoke else 20000,
                     keep_checkpoint_iters=list(milestones))
    checks.update({key: args.get(key) == value for key, value in overrides.items()})
    if spec["mode"] == "masked":
        checks["no_implicit_scaling"] = args.get("covar_kd_temp_power") == 0.0
    checks.update(final_iteration=f"Iters: {expected_max}/{expected_max}" in text,
                  maximum=maximum == expected_max, training_end=bool(old.TIME_RE.findall(text)),
                  finite_log=not bool(re.search(r"\b(?:nan|inf)\b", text, re.I)),
                  validations=tuple(sorted(trajectory)) == (() if smoke else MILESTONES))
    losses = [float(v) for v in re.findall(r"KD Loss:\s*([-+0-9.eE]+)", text)]
    checks["kd_exercised"] = bool(losses) and (all(v == 0 for v in losses)
                              if spec["lambda_kd"] == 0 else any(v > 0 for v in losses))
    for step in milestones:
        checks[f"checkpoint_{step}"] = (save_dir / f"training_state_iter{step:06d}.pth").is_file()
    checks["latest"] = (save_dir / "training_state_latest.pth").is_file()
    checks["finite_validation"] = all(math.isfinite(v) for row in trajectory.values() for v in row.values())
    if not all(checks.values()):
        raise RuntimeError(f"{spec['variant']}: failed checks {[k for k,v in checks.items() if not v]}")
    return dict(**spec, trajectory={str(k): v for k,v in trajectory.items()},
                final_miou_percent=None if smoke else trajectory[80000]["miou_percent"],
                log_path=str(log_path), log_sha256=base.digest(log_path), contract_pass=True,
                training_time=old.TIME_RE.findall(text)[-1].strip(), smoke=smoke)


def audit_checkpoints(spec, smoke=False):
    import torch
    def finite(value):
        if isinstance(value, torch.Tensor):
            return not value.is_floating_point() or bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)
    row = read_run(spec, smoke)
    save_dir = command_for(spec, smoke)[2]
    milestones = (20,) if smoke else MILESTONES
    records = []
    for name, iteration in [(f"training_state_iter{s:06d}.pth", s) for s in milestones] + [
                            ("training_state_latest.pth", milestones[-1])]:
        path = save_dir / name
        state = torch.load(path, map_location="cpu")
        args = state["args"]
        valid = (state["checkpoint_type"] == "train_kd_training_state" and
                 state["iteration"] == iteration and state["world_size"] == 1 and
                 args["seed"] == spec["seed"] and args["resume"] is None and
                 args["student_backbone"] == "mobilenetv3_" + spec["student"] and
                 args["kd_loss_mode"] == spec["mode"] and
                 args["kd_temperature"] == float(spec["temperature"]) and
                 args["lambda_kd"] == spec["lambda_kd"] and
                 args["max_iterations"] == milestones[-1] and args["batch_size"] == 16 and
                 bool(state["optimizer"]["state"]) and bool(state["rng_state"]) and
                 len(state["rng_state_by_rank"]) == 1 and finite(state["student"]) and finite(state["optimizer"]))
        if not valid:
            raise RuntimeError(f"invalid checkpoint: {path}")
        records.append(dict(path=str(path.relative_to(ROOT)), iteration=iteration,
                            sha256=base.digest(path), tensors_finite=True, optimizer_rng_present=True))
        del state
    row.update(checkpoints=records, audited_at=now(), status="PASS")
    return row


def worker(spec, gpu, smoke):
    setup_environment(gpu)
    if not idle(gpu_inventory()[gpu]):
        raise RuntimeError(f"GPU {gpu} became busy before worker launch; no training started")
    command, root, save_dir, log_dir = command_for(spec, smoke)
    if save_dir.exists() or log_dir.exists():
        raise RuntimeError("existing output preserved; partial runs require explicit audit")
    subprocess.run(command, cwd=ROOT, check=True, stdin=subprocess.DEVNULL)
    audit = audit_checkpoints(spec, smoke)
    write_json(root / "runtime/audits" / (spec["variant"] + ".json"), audit)


def completed(spec, smoke=False):
    root = RUN_ROOT / "smoke" if smoke else RUN_ROOT
    path = root / "runtime/audits" / (spec["variant"] + ".json")
    if not path.exists():
        return False
    audit = json.loads(path.read_text())
    return audit["status"] == "PASS" and audit["log_sha256"] == read_run(spec, smoke)["log_sha256"]


def run_phase(specs, gpus, state, smoke=False):
    pending = [s for s in specs if not completed(s, smoke)]
    active, free_since, failures = {}, {}, []
    runtime = RUN_ROOT / "runtime"
    while pending or active:
        for gpu, (spec, child) in list(active.items()):
            code = child.poll()
            if code is None:
                continue
            del active[gpu]
            free_since.pop(gpu, None)
            if code != 0 or not completed(spec, smoke):
                failures.append(dict(variant=spec["variant"], returncode=code))
            else:
                key = "smoke_completed" if smoke else "completed_runs"
                state.setdefault(key, []).append(spec["variant"])
        if failures and not active:
            raise RuntimeError(f"workers failed; outputs preserved: {failures}")
        inventory = gpu_inventory()
        for gpu in gpus:
            if gpu in active or not idle(inventory[gpu]):
                free_since.pop(gpu, None)
                continue
            free_since.setdefault(gpu, time.monotonic())
            if not pending or failures or time.monotonic() - free_since[gpu] < 30:
                continue
            spec = pending.pop(0)
            prefix = "smoke-" if smoke else ""
            command = [sys.executable, "-B", "-u", __file__, "--worker", spec["variant"], "--gpu", str(gpu)]
            if smoke:
                command.append("--smoke")
            with (runtime / f"{prefix}{spec['variant']}.log").open("a") as handle:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                         stdout=handle, stderr=subprocess.STDOUT)
            active[gpu] = spec, child
        state.update(status="SMOKE" if smoke and active else "RUNNING" if active else "WAITING_GPU",
                     updated_at=now(), pending_count=len(pending), gpu_inventory=inventory,
                     active_runs=[dict(**s, gpu=g, pid=p.pid, smoke=smoke) for g,(s,p) in active.items()])
        write_json(runtime / "state.json", state)
        if pending or active:
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[6, 7])
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--continue", dest="continue_queue", action="store_true")
    parser.add_argument("--worker")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if args.gpu is None:
            parser.error("worker requires --gpu")
        spec = next(s for s in experiment_plan() if s["variant"] == args.worker)
        worker(spec, args.gpu, args.smoke)
        return
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        parser.error("specify distinct nonnegative GPU indices")
    setup_environment()
    runtime = RUN_ROOT / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = register(args.gpus)
        if args.prepare:
            print(json.dumps(dict(status="REGISTERED", new_runs=protocol["new_run_count"])))
            return
        path = runtime / "state.json"
        if path.exists() and not args.continue_queue:
            raise RuntimeError("existing queue state preserved; use --continue after checking workers")
        if path.exists():
            prior = json.loads(path.read_text())
            for active in prior.get("active_runs", []):
                try:
                    os.kill(active["pid"], 0)
                except ProcessLookupError:
                    continue
                raise RuntimeError("an earlier worker PID is alive; inspect before continuing")
        state = dict(status="PREPARING", pid=os.getpid(), started_at=now(), gpus=args.gpus,
                     total_new_runs=33, completed_runs=[s["variant"] for s in experiment_plan() if completed(s)])
        write_json(path, state)
        try:
            run_phase(smoke_plan(), args.gpus, state, smoke=True)
            write_json(REPORT_ROOT / "smoke_checks.json", dict(status="PASS", completed_at=now(),
                       runs=[json.loads((RUN_ROOT / "smoke/runtime/audits" / (s["variant"] + ".json")).read_text())
                             for s in smoke_plan()]))
            for phase in (1, 2, 3):
                state["phase"] = phase
                run_phase([s for s in experiment_plan() if s["phase"] == phase], args.gpus, state)
                from scripts.diagnostics.summarize_p10_h20 import generate
                generate()
            register(args.gpus)  # Fail rather than publish if the frozen source/runtime drifted.
            state.update(status="COMPLETE", finished_at=now(), active_runs=[], pending_count=0)
        except Exception as error:
            state.update(status="FAILED", error=str(error), updated_at=now())
            raise
        finally:
            write_json(path, state)


if __name__ == "__main__":
    main()
