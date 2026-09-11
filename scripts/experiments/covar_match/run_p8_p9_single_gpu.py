#!/usr/bin/env python3
"""Run the approved single-GPU CE baseline, then the conditional second-pair grid."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.summarize_p8_p9_experiments import (
    SEEDS, MILESTONES, P9_STAGE1_TEMPERATURES, P9_STAGE2_TEMPERATURES,
    SMALL_LOG_NAME, LARGE_LOG_NAME, p8_variant, p9_variant, read_run,
)

RUN_ROOT = ROOT / "runs/covar_match/P8_P9_single_gpu"
P8_ROOT = ROOT / "runs/covar_match/P8_ce_only_baseline_single_gpu"
P9_ROOT = ROOT / "runs/covar_match/P9_pair2_temperature_response_single_gpu"
REPORT_ROOT = ROOT / "reports/covar_match/P8_P9_single_gpu"
TEACHER = ROOT / "data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth"
BASES = {
    "ce": ROOT / "data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth",
    "kd": ROOT / "data/winycg/imagenet_pretrained/mobilenet_v3_large-bc2c3fd3.pth",
}


def command_for(seed, kind, temperature, *, smoke=False):
    if seed not in SEEDS or kind not in BASES:
        raise ValueError("run is outside the approved plan")
    if kind == "ce" and str(temperature) != "1.0":
        raise ValueError("CE uses the inert T=1 setting")
    if kind == "kd" and str(temperature) not in P9_STAGE1_TEMPERATURES + P9_STAGE2_TEMPERATURES:
        raise ValueError("temperature is outside the approved plan")
    variant = p8_variant(seed, "single_gpu") if kind == "ce" else p9_variant(temperature, seed)
    root = P8_ROOT if kind == "ce" else P9_ROOT
    if smoke:
        root = RUN_ROOT / "smoke" / kind
        variant = "smoke_" + variant
    save_dir, log_dir = root / "checkpoints" / variant, root / "logs" / variant
    maximum = 20 if smoke else 80000
    milestones = (20,) if smoke else MILESTONES
    student = "mobilenetv3_small" if kind == "ce" else "mobilenetv3_large"
    command = [
        sys.executable, "-u", "train_kd.py",
        "--device-type", "cuda", "--seed", str(seed),
        "--teacher-model", "deeplabv3", "--teacher-backbone", "resnet101",
        "--student-model", "deeplabv3_mobilenet_ssseg", "--student-backbone", student,
        "--dataset", "voc", "--data", "dataset/VOCAug/",
        "--batch-size", "16", "--crop-size", "512", "512", "--workers", "4",
        "--lr", "0.02", "--momentum", "0.9", "--weight-decay", "0.0001",
        "--max-iterations", str(maximum), "--lambda-kd", "0.0" if kind == "ce" else "1.0",
        "--kd-loss-mode", "teacher_only", "--kd-temperature", str(temperature),
        "--teacher-output-temp", "1.0", "--log-iter", "1" if smoke else "20",
        "--save-per-iters", str(20 if smoke else 20000),
        "--val-per-iters", "20000", "--keep-checkpoint-iters",
        *map(str, milestones), "--teacher-pretrained", str(TEACHER),
        "--student-pretrained-base", str(BASES[kind]),
        "--save-dir", str(save_dir), "--log-dir", str(log_dir),
    ]
    for name in ("adv", "d", "skd", "cwd-fea", "cwd-logit", "ifv",
                 "fitnet", "at", "psd", "csd"):
        command += ["--lambda-" + name, "0.0"]
    if smoke:
        command.append("--skip-val")
    return command, root, variant, save_dir, log_dir, student


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def record_protocol():
    import torch
    import numpy
    import cv2
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the approved profile requires exactly one visible CUDA GPU")
    files = [
        TEACHER, *BASES.values(), ROOT / "train_kd.py",
        ROOT / "dataset/voc.py", ROOT / "utils/distributed.py",
        ROOT / "models/deeplabv3_mobilenetv3.py",
        ROOT / "models/base_models/mobilenetv3.py",
        Path(__file__).resolve(),
    ]
    protocol = {
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "execution_profile": "single_gpu", "user_approved_deviation_from_p7": True,
        "strict_p7_pairing": False, "world_size": 1,
        "gpu": torch.cuda.get_device_name(0), "python": sys.version.split()[0],
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "numpy": numpy.__version__,
        "opencv": cv2.__version__, "python_executable": sys.executable,
        "seeds": list(SEEDS), "iterations": 80000, "milestones": list(MILESTONES),
        "optimizer": {"name": "SGD", "lr": 0.02, "momentum": 0.9, "weight_decay": 0.0001},
        "schedule": "existing train_kd.py poly schedule",
        "global_batch_size": 16, "crop_size": [512, 512], "workers": 4,
        "student_initialization": "ImageNet backbone plus fresh segmentation head for every run",
        "teacher": "DeepLabV3-ResNet101", "ce_student": "MobileNetV3-Small",
        "second_student": "MobileNetV3-Large",
        "augmentation": "original VOC random scale 0.5:0.1:2.0, random crop and mirror",
        "ce_runs": 3, "stage1_kd_runs": 15, "conditional_stage2_kd_runs": 6,
        "stage1_temperatures": list(P9_STAGE1_TEMPERATURES),
        "conditional_temperatures": list(P9_STAGE2_TEMPERATURES),
        "near_optimal_delta_pp": 0.2,
        "protocol_differences": [
            "Head SyncBatchNorm becomes BatchNorm; backbone local batch changes from 8 to 16.",
            "RandomSampler replaces DistributedSampler; rank/worker random streams differ.",
            "Single-rank validation uses each image once; the legacy two-rank sampler pads the odd-sized split.",
            "The original P7 logs do not establish that their software versions match this runtime.",
        ],
        "sha256": {str(path.relative_to(ROOT)): digest(path) for path in files},
    }
    write_json(REPORT_ROOT / "protocol.json", protocol)


def generate_reports(stage):
    command = [
        sys.executable, "scripts/diagnostics/summarize_p8_p9_experiments.py",
        "--execution-protocol", "single_gpu", "--stage", stage,
        "--p8-root", str(P8_ROOT), "--p9-root", str(P9_ROOT),
        "--p8-output-json", str(REPORT_ROOT / "P8_ce_only.json"),
        "--p8-output-markdown", str(REPORT_ROOT / "P8_ce_only.md"),
        "--p9-output-json", str(REPORT_ROOT / "P9_temperature.json"),
        "--p9-output-markdown", str(REPORT_ROOT / "P9_temperature.md"),
    ]
    subprocess.run(command, cwd=ROOT, check=True, stdin=subprocess.DEVNULL)


def execute_run(seed, kind, temperature, state, state_path, *, smoke=False):
    command, root, variant, save_dir, log_dir, student = command_for(
        seed, kind, temperature, smoke=smoke
    )
    log_name = SMALL_LOG_NAME if kind == "ce" else LARGE_LOG_NAME
    if not smoke:
        try:
            read_run(root, variant, log_name, seed, student,
                     0.0 if kind == "ce" else 1.0, temperature, "single_gpu")
        except RuntimeError:
            pass
        else:
            print("SKIP_COMPLETE", variant, flush=True)
            return
    if save_dir.exists() or log_dir.exists():
        raise RuntimeError(f"partial output exists; preserved without overwrite: {variant}")
    state.update(status="RUNNING", variant=variant, seed=seed,
                 kind=kind, temperature=temperature,
                 log_path=str(log_dir / log_name),
                 updated_at=dt.datetime.now(dt.timezone.utc).isoformat())
    write_json(state_path, state)
    print("START_RUN", json.dumps({"variant": variant, "kind": kind,
                                  "temperature": temperature, "smoke": smoke}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, stdin=subprocess.DEVNULL)
    if smoke:
        text = (log_dir / log_name).read_text()
        if "Iters: 20/20" not in text or "Total training time:" not in text:
            raise RuntimeError(f"incomplete smoke: {variant}")
        if re.search(r"\b(?:nan|inf)\b", text, re.IGNORECASE):
            raise RuntimeError(f"non-finite smoke log: {variant}")
        kd_losses = [float(x) for x in re.findall(r"KD Loss:\s*([-+0-9.eE]+)", text)]
        if not kd_losses or (kind == "ce" and any(x != 0 for x in kd_losses)):
            raise RuntimeError(f"CE/KD smoke loss contract failed: {variant}")
        if kind == "kd" and not any(x > 0 for x in kd_losses):
            raise RuntimeError("KD smoke did not exercise the distillation term")
    else:
        read_run(root, variant, log_name, seed, student,
                 0.0 if kind == "ce" else 1.0, temperature, "single_gpu")
    state.setdefault("completed_runs", []).append(variant)
    write_json(state_path, state)
    print("COMPLETE_RUN", variant, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="two separate 20-step engineering checks")
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
    runtime = RUN_ROOT / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    lock = (runtime / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = runtime / ("smoke-state.json" if args.smoke else "state.json")
    state = {"status": "PREPARING", "pid": os.getpid(), "smoke": args.smoke,
             "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
             "completed_runs": []}
    write_json(state_path, state)
    try:
        record_protocol()
        if args.smoke:
            execute_run(1234, "ce", "1.0", state, state_path, smoke=True)
            execute_run(1234, "kd", "1.0", state, state_path, smoke=True)
        else:
            for seed in SEEDS:
                execute_run(seed, "ce", "1.0", state, state_path)
            generate_reports("p8")
            for seed in SEEDS:
                for temperature in P9_STAGE1_TEMPERATURES:
                    execute_run(seed, "kd", temperature, state, state_path)
            generate_reports("all")
            payload = json.loads((REPORT_ROOT / "P9_temperature.json").read_text())
            gate = payload["stage1"]["phase2_gate"]
            write_json(REPORT_ROOT / "stage2_decision.json", gate)
            for suffix in (".json", ".md", ".csv"):
                archived = REPORT_ROOT / ("P9_stage1" + suffix)
                if not archived.exists():
                    shutil.copyfile(REPORT_ROOT / ("P9_temperature" + suffix), archived)
            if gate["required"]:
                for seed in SEEDS:
                    for temperature in P9_STAGE2_TEMPERATURES:
                        execute_run(seed, "kd", temperature, state, state_path)
                generate_reports("p9")
        state.update(status="COMPLETE", finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(state_path, state)
        print("QUEUE_COMPLETE", json.dumps(state), flush=True)
    except Exception as error:
        state.update(status="FAILED", error=str(error),
                     updated_at=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
