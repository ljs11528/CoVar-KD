"""Run bounded virtual-update diagnostics on immutable existing checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .io import read_json, sha256, source_hashes, verify_sources, write_json


def acquire_gpu(gpu):
    if gpu not in (0, 2):
        raise ValueError("Registered resource policy permits only GPU 0 or 2")
    locks = []
    for name in (f"/tmp/covar-kd-gpu-{gpu}.lock", f"/tmp/covar-p13-gpu-{gpu}.lock"):
        handle = open(name, "a+")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True)
    selected = next([int(x) for x in row.split(",")] for row in rows.splitlines() if int(row.split(",")[0]) == gpu)
    if selected[1] >= 1024 or selected[2] > 10:
        raise RuntimeError(f"GPU {gpu} is occupied: {selected}")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        os.environ.pop(key, None)
    return locks


def run(args):
    # CUDA visibility is selected by main() before importing PyTorch.
    import numpy as np
    import torch
    import cv2
    from PIL import Image
    from .core import (PooledMetrics, Snapshot, ce_loss, evaluation_mode, gradient_norm,
                       isolated_rng, kd_loss, restored_state, tree_equal)
    from .register import CITY_IDS

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    registration = read_json(args.registration)
    verify_sources(registration["source_hashes"])
    reg_hash = sha256(args.registration)
    config = registration["datasets"][args.dataset]
    out = Path(args.out).resolve()
    work = out / args.phase / args.dataset
    work.mkdir(parents=True, exist_ok=True)
    runtime = Path(config["runtime"])
    sys.path.insert(0, str(runtime))
    os.chdir(runtime)
    for name, digest in config["input_sha256"].items():
        if Path(name).suffix == ".py" and sha256(name) != digest:
            raise RuntimeError(f"Frozen runtime changed: {name}")
    for item in config["monitor"]:
        if sha256(item["img"]) != item["image_sha256"] or sha256(item["label"]) != item["label_sha256"]:
            raise RuntimeError("Monitoring image or label changed")
    if sha256(config["teacher"]) != config["input_sha256"][config["teacher"]]:
        raise RuntimeError("Teacher checkpoint changed")

    spec = importlib.util.spec_from_file_location("utility_frozen_trainer", runtime / "train_kd_3090.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.logger = logging.getLogger("utility-study")
    ce_reference = next(r for r in config["runs"] if r["condition"] == "ce" and r["seed"] == 1234)
    constructor = argparse.Namespace(**ce_reference["args"])
    constructor.batch_size = config["matched_batch"]
    constructor.workers, constructor.max_iterations, constructor.resume = 0, 1, None
    constructor.student_pretrained_base, constructor.student_pretrained = "None", "None"
    constructor.log_dir, constructor.save_dir = str(work / "constructor"), str(work / "unused-checkpoints")
    constructor.device, constructor.device_type = "cuda", "cuda"
    constructor.distributed, constructor.num_gpus, constructor.local_rank = False, 1, 0
    trainer = module.Trainer(constructor)
    trainer.t_model.eval()
    student, optimizer = trainer.s_model, trainer.optimizer
    parameters = tuple(student.parameters())
    if args.dataset == "voc":
        from dataset.voc import VOCDataTrainSet
        data = VOCDataTrainSet(config["data"], config["train_list"], crop_size=tuple(constructor.crop_size),
                               scale=True, mirror=True, ignore_label=-1)
    else:
        from dataset.cityscapes import CSTrainValSet
        data = CSTrainValSet(config["data"], config["train_list"], crop_size=tuple(constructor.crop_size),
                             scale=True, mirror=True, ignore_label=-1)
    assert len(data) == config["train_count"]
    monitor = config["monitor"][:8] if args.phase == "smoke" else config["monitor"]
    lookup = np.full(256, -1, dtype=np.int64)
    for k, raw in enumerate(CITY_IDS):
        lookup[raw] = k

    def monitor_scores():
        metrics = PooledMetrics(config["classes"])
        with evaluation_mode(student), torch.no_grad():
            for item in monitor:
                image = cv2.imread(item["img"], cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError("Monitoring image unreadable")
                image = image.astype(np.float32) - np.array([104.00698793, 116.66876762, 122.67891434])
                image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
                with Image.open(item["label"]) as handle:
                    label = np.asarray(handle).astype(np.int64)
                if args.dataset == "cityscapes":
                    label = lookup[label]
                else:
                    label[label == 255] = -1
                x = torch.from_numpy(image)[None].cuda()
                y = torch.from_numpy(label)[None].cuda()
                metrics.add(student(x)[0], y)
        result = metrics.result()
        if args.phase != "smoke" and not result["all_classes_present"]:
            raise RuntimeError("Monitoring lost a registered class")
        return result

    def update_batch(entry):
        with isolated_rng(entry["augmentation_seed"]):
            items = [data[index] for index in entry["indices"]]
        image = torch.stack([torch.from_numpy(x[0]) for x in items]).cuda()
        target = torch.stack([torch.from_numpy(x[1]).long() for x in items]).cuda()
        if not bool(((target == -1) | ((target >= 0) & (target < config["classes"]))).all()):
            raise RuntimeError("Invalid training labels")
        return image, target

    conditions = {"ce": (None, 0.), "shared_c1": ("shared", 1.), "shared_c4": ("shared", 4.),
                  "teacher_c1": ("teacher", 1.), "teacher_c4": ("teacher", 4.), "shared_norm": ("shared", None)}
    smoke = args.phase == "smoke"
    stages = [20000] if smoke else registration["checkpoints"]
    seeds = [1234] if smoke else registration["seeds"]
    cohorts = ["ce"] if smoke else args.cohorts.split(",")
    if set(cohorts) - set(registration["cohorts"]):
        raise ValueError("Unregistered starting cohort")
    horizon = 3 if smoke else 128 if args.phase == "rollout" else 1
    outputs, started = [], time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for cohort in cohorts:
        for seed in seeds:
            run_record = next(r for r in config["runs"] if r["condition"] == cohort and r["seed"] == seed)
            for stage in stages:
                checkpoint = run_record["checkpoints"][str(stage)]
                if sha256(checkpoint) != config["input_sha256"][checkpoint]:
                    raise RuntimeError("Source training checkpoint changed")
                checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                assert checkpoint_state["iteration"] == stage and checkpoint_state["args"]["seed"] == seed
                student.load_state_dict(checkpoint_state["student"], strict=True)
                for name in ("criterion_cwd", "criterion_fitnet"):
                    getattr(trainer, name).load_state_dict(checkpoint_state[name], strict=True)
                    getattr(trainer, name).eval()
                optimizer.load_state_dict(checkpoint_state["optimizer"])
                module.move_optimizer_state_to_device(optimizer, trainer.device)
                optimizer.zero_grad(set_to_none=True)
                student.eval()
                baseline = Snapshot.capture(student, optimizer)
                before = monitor_scores()
                baseline.verify(student, optimizer)
                base_lr = run_record["args"]["lr"]
                for mode in registration["modes"]:
                    stream = config["updates"][f"{seed}_{stage}_{mode}"]
                    for condition, (placement, coefficient) in conditions.items():
                        path = work / cohort / str(seed) / str(stage) / mode / (condition + ".json")
                        if path.exists():
                            result = read_json(path)
                            if result["registration_sha256"] != reg_hash or result["horizon"] != horizon or result["status"] != "PASS":
                                raise RuntimeError("Existing branch belongs to a different registration")
                            outputs.append(str(path))
                            continue
                        branch_start = time.monotonic()
                        coefficients, local_losses, checkpoints, first_gradient = [], [], {}, None
                        with restored_state(student, optimizer):
                            student.train(mode == "matched_train")
                            with isolated_rng(stream[0]["augmentation_seed"] + 90000000):
                                for step in range(horizon):
                                    x, y = update_batch(stream[step])
                                    for group in optimizer.param_groups:
                                        group["lr"] = base_lr * (1.0 - (stage + step + 1) / 80000.) ** .9
                                    optimizer.zero_grad(set_to_none=True)
                                    logits = student(x)[0]
                                    task = ce_loss(logits, y)
                                    alpha, norm_record = coefficient, None
                                    if placement is not None:
                                        with torch.no_grad():
                                            teacher = trainer.t_model(x)[0]
                                        knowledge = kd_loss(logits, teacher, y, placement)
                                        if condition == "shared_norm":
                                            teacher_loss = kd_loss(logits, teacher, y, "teacher")
                                            a = torch.autograd.grad(teacher_loss, parameters, retain_graph=True, allow_unused=True)
                                            b = torch.autograd.grad(knowledge, parameters, retain_graph=True, allow_unused=True)
                                            teacher_norm, shared_norm = gradient_norm(a), gradient_norm(b)
                                            if not math.isfinite(teacher_norm + shared_norm) or min(teacher_norm, shared_norm) <= 0:
                                                raise RuntimeError("Degenerate norm-control gradient")
                                            alpha = teacher_norm / shared_norm
                                            norm_record = {"teacher_norm": teacher_norm, "shared_norm": shared_norm}
                                            del a, b, teacher_loss
                                        objective = task + alpha * knowledge
                                    else:
                                        objective = task
                                    if not bool(torch.isfinite(objective)):
                                        raise RuntimeError("Nonfinite virtual objective")
                                    if step == 0:
                                        ce_grad = torch.autograd.grad(task, parameters, retain_graph=True, allow_unused=True)
                                        first_gradient = {"ce_norm": gradient_norm(ce_grad)}
                                        if placement is not None:
                                            kd_grad = torch.autograd.grad(knowledge, parameters, retain_graph=True, allow_unused=True)
                                            knorm = gradient_norm(kd_grad)
                                            dot = sum(float((a.detach().double() * b.detach().double()).sum())
                                                      for a, b in zip(ce_grad, kd_grad) if a is not None and b is not None)
                                            first_gradient.update(kd_norm=knorm, kd_ce_cosine=dot / max(1e-30, knorm * first_gradient["ce_norm"]),
                                                                  effective_kd_ce_norm_ratio=alpha * knorm / max(1e-30, first_gradient["ce_norm"]))
                                            del kd_grad
                                        del ce_grad
                                    objective.backward()
                                    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
                                        raise RuntimeError("Nonfinite virtual gradient")
                                    optimizer.step()
                                    coefficients.append(float(alpha))
                                    local_losses.append(float(task.detach()))
                                    del logits, task, objective, x, y
                                    if placement is not None:
                                        del teacher, knowledge
                                    if step + 1 in (1, horizon):
                                        checkpoints[str(step + 1)] = monitor_scores()
                                    if horizon > 3 and (step + 1) % 16 == 0:
                                        write_json(work / "state.json", {"status": "RUNNING", "phase": args.phase,
                                                   "cohort": cohort, "seed": seed, "checkpoint": stage, "mode": mode,
                                                   "condition": condition, "virtual_step": step + 1, "gpu": args.gpu,
                                                   "pid": os.getpid(), "completed_branches": len(outputs)})
                            # BN counters are deliberately part of the virtual
                            # state; training and frozen-BN branches must differ.
                            bn_updates = sum(int(value.item()) - int(baseline.model[name].item()) for name, value in student.state_dict().items()
                                             if name.endswith("num_batches_tracked"))
                            if mode == "matched_train" and bn_updates <= 0:
                                raise RuntimeError("Training-mode BN did not update")
                            if mode != "matched_train" and bn_updates != 0:
                                raise RuntimeError("Frozen-mode BN unexpectedly updated")
                        baseline.verify(student, optimizer)
                        result = {
                            "status": "PASS", "dataset": args.dataset, "cohort": cohort, "seed": seed,
                            "checkpoint": stage, "mode": mode, "condition": condition, "horizon": horizon,
                            "batch_size": len(stream[0]["indices"]), "monitor_images": len(monitor),
                            "registration_sha256": reg_hash, "checkpoint_sha256": config["input_sha256"][checkpoint],
                            "before": before, "after": checkpoints, "coefficients": coefficients,
                            "local_ce_losses": local_losses, "last_norm_control": norm_record,
                            "first_gradient": first_gradient,
                            "bn_counter_delta": bn_updates, "state_restored": True,
                            "seconds": time.monotonic() - branch_start,
                        }
                        write_json(path, result)
                        outputs.append(str(path))
                        write_json(work / "state.json", {"status": "RUNNING", "phase": args.phase, "gpu": args.gpu,
                                   "cohort": cohort, "seed": seed, "checkpoint": stage, "mode": mode,
                                   "condition": condition, "completed_branches": len(outputs), "pid": os.getpid(),
                                   "elapsed_seconds": time.monotonic() - started})
                        print("BRANCH_PASS", args.dataset, cohort, seed, stage, mode, condition,
                              f"CE={checkpoints[str(horizon)]['pixel_ce']:.8f}", flush=True)
                    baseline.verify(student, optimizer)
                if sha256(checkpoint) != config["input_sha256"][checkpoint]:
                    raise RuntimeError("Source checkpoint mutated")
                del checkpoint_state, baseline
    for item in config["monitor"]:
        if sha256(item["img"]) != item["image_sha256"] or sha256(item["label"]) != item["label_sha256"]:
            raise RuntimeError("Monitoring inputs mutated")
    verify_sources(registration["source_hashes"])
    summary = {"status": "PASS", "dataset": args.dataset, "phase": args.phase, "cohorts": cohorts,
               "horizon": horizon, "registration_sha256": reg_hash, "branches": len(outputs),
               "files": outputs, "seconds": time.monotonic() - started,
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
               "full_training_runs_launched": 0, "checkpoint_files_written": 0,
               "completed_at": datetime.now(timezone.utc).isoformat()}
    write_json(work / ("summary_" + "_".join(cohorts) + ".json"), summary)
    write_json(work / "state.json", summary)
    print("PHASE_PASS", args.dataset, args.phase, len(outputs), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", choices=("voc", "cityscapes"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--phase", choices=("smoke", "one_step", "rollout"), required=True)
    parser.add_argument("--cohorts", default="ce,shared_c1,shared_c4")
    args = parser.parse_args()
    locks = acquire_gpu(args.gpu)
    try:
        run(args)
    except Exception as error:
        write_json(Path(args.out) / args.phase / args.dataset / "failure.json",
                   {"status": "FAILED", "error": repr(error), "phase": args.phase,
                    "source_hashes": source_hashes(), "pid": os.getpid()})
        raise
    finally:
        for lock in locks:
            lock.close()


if __name__ == "__main__":
    main()
