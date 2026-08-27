#!/usr/bin/env python3
"""P5: locate where P4a logit alignment stops transferring.

No optimizer or training step is created. Frozen checkpoints are probed with
autograd.grad, and virtual KD updates use copied parameter tensors through
torch.func.functional_call.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch.func import functional_call

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "diagnostics")]

from dataset.voc import VOCDataValSet
from region_teachability_oracle import STAGES, build_students, build_teacher
from summarize_p4a_task_aligned import (
    TEMPERATURES,
    aggregate_p4a,
    parse_log,
    phase_name,
    training_loss_summary,
)
from utils.task_aligned_temperature import (
    TASK_ALIGNED_FALLBACK_TEMPERATURE,
    build_task_aligned_region_selection,
    task_aligned_region_kd_loss,
)
from utils.teacher_only_kd import teacher_target_kd_loss

P1_LOG = ROOT / (
    "runs/covar_match/P1_teacher_only_temperature/logs/T1p5_20k_seed1234/"
    "deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"
)
P4A_LOG = ROOT / (
    "runs/covar_match/P4a_task_aligned_region/logs/"
    "task_aligned_region_r8_20k_seed1234/"
    "deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"
)
PHASES = ("early_1_4000", "middle_4001_12000", "late_12001_20000")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "dataset/VOCAug")
    parser.add_argument(
        "--list-path", type=Path,
        default=ROOT / "dataset/list/voc/val.txt",
    )
    parser.add_argument(
        "--teacher-pretrained", type=Path,
        default=ROOT / "data/winycg/cirkd/teachers/"
        "deeplabv3_resnet101_voc_best_model.pth",
    )
    parser.add_argument(
        "--checkpoint-root", type=Path,
        default=ROOT / "runs/covar_match/P1_teacher_only_temperature/"
        "checkpoints/T1p0_20k_seed1234",
    )
    parser.add_argument("--p1-log", type=Path, default=P1_LOG)
    parser.add_argument("--p4a-log", type=Path, default=P4A_LOG)
    parser.add_argument(
        "--p4a-report-json", type=Path,
        default=ROOT / "reports/covar_match/P4a_task_aligned_region.json",
    )
    parser.add_argument(
        "--raw-output", type=Path,
        default=ROOT / "runs/covar_match/P5_mechanism_audit/raw_results.json",
    )
    parser.add_argument(
        "--output-json", type=Path,
        default=ROOT / "reports/covar_match/P5_mechanism_audit.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batch-pairs", type=int, default=6)
    parser.add_argument("--sample-pool-size", type=int, default=30)
    parser.add_argument("--virtual-step-size", type=float, default=0.02)
    parser.add_argument("--ignore-label", type=int, default=-1)
    return parser.parse_args()


def training_ce_loss(logits, targets, ignore_label=-1):
    """Training CE with an algebraically exact deterministic reduction."""
    logits = F.interpolate(
        logits, targets.shape[-2:], mode="bilinear", align_corners=True
    )
    valid = targets != int(ignore_label)
    if not bool(valid.any()):
        raise ValueError("CE target contains no valid pixel")
    safe_targets = targets.masked_fill(~valid, 0)
    log_probability = F.log_softmax(logits, dim=1)
    nll = -torch.gather(
        log_probability, 1, safe_targets.unsqueeze(1)
    ).squeeze(1)
    return nll[valid].sum() / valid.sum().to(nll.dtype)


def gradient_norm(gradients):
    terms = [
        gradient.detach().double().square().sum()
        for gradient in gradients if gradient is not None
    ]
    return float(torch.stack(terms).sum().sqrt().item()) if terms else 0.0


def gradient_cosine(left, right):
    pairs = [
        (a.detach().double(), b.detach().double())
        for a, b in zip(left, right) if a is not None and b is not None
    ]
    if not pairs:
        raise RuntimeError("no common gradient tensors")
    dot = torch.stack([(a * b).sum() for a, b in pairs]).sum()
    left_norm = torch.stack([a.square().sum() for a, _ in pairs]).sum().sqrt()
    right_norm = torch.stack([b.square().sum() for _, b in pairs]).sum().sqrt()
    denominator = left_norm * right_norm
    if float(denominator.item()) <= 1e-30:
        raise RuntimeError("zero gradient norm")
    return float((dot / denominator).item())


def region_alignment_means(selection):
    """P4a's exact eligible-region-weighted native-logit objective."""
    eligible = selection.eligible_region_mask
    if not bool(eligible.any()):
        raise RuntimeError("minibatch has no eligible region")
    fixed_index = TEMPERATURES.index(TASK_ALIGNED_FALLBACK_TEMPERATURE)
    fixed = selection.region_scores[:, fixed_index]
    selected = torch.gather(
        selection.region_scores,
        1,
        selection.selected_region_index.unsqueeze(1),
    ).squeeze(1)
    return {
        "fixed": float(fixed[eligible].double().mean().item()),
        "p4a": float(selected[eligible].double().mean().item()),
        "eligible_regions": int(eligible.sum().item()),
    }


def mean_target_entropy(probabilities, valid):
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(1)
    return float(entropy[valid].double().mean().item())


def updated_parameters(model, gradients, step_size):
    return {
        name: (
            parameter.detach()
            if gradient is None
            else parameter.detach() - float(step_size) * gradient.detach()
        )
        for (name, parameter), gradient
        in zip(model.named_parameters(), gradients)
    }


def functional_logits(model, parameters, images):
    buffers = {name: value.detach() for name, value in model.named_buffers()}
    return functional_call(
        model, (parameters, buffers), (images,), strict=True
    )[0]


def materialize_batch(dataset, indices, ignore_label=-1):
    items = [dataset[index] for index in indices]
    images = [torch.from_numpy(item[0]) for item in items]
    targets = [torch.from_numpy(item[1]).long() for item in items]
    height = max(image.shape[-2] for image in images)
    width = max(image.shape[-1] for image in images)
    padded_images = [
        F.pad(image, (0, width - image.shape[-1], 0, height - image.shape[-2]))
        for image in images
    ]
    padded_targets = [
        F.pad(
            target,
            (0, width - target.shape[-1], 0, height - target.shape[-2]),
            value=int(ignore_label),
        )
        for target in targets
    ]
    return {
        "indices": list(indices),
        "names": [dataset.files[index]["name"] for index in indices],
        "images": torch.stack(padded_images),
        "targets": torch.stack(padded_targets),
    }


def fixed_probe_indices(dataset_size, seed, required, pool_size):
    if required <= 0 or required > dataset_size:
        raise ValueError("invalid image count")
    pool_count = min(dataset_size, max(required, pool_size))
    pool = sorted(random.Random(seed).sample(range(dataset_size), pool_count))
    return pool[:required]


def evaluate_pair(
    model, stage, iteration, pair_index, same_batch, next_batch,
    teacher_logits, device, step_size, ignore_label,
):
    images = same_batch["images"].to(device)
    targets = same_batch["targets"].to(device)
    next_images = next_batch["images"].to(device)
    next_targets = next_batch["targets"].to(device)
    teacher_logits = teacher_logits.to(device)

    model.eval()
    parameters = tuple(model.parameters())
    student_logits = model(images)[0]
    selection = build_task_aligned_region_selection(
        student_logits, teacher_logits, targets, ignore_label=ignore_label
    )
    ce_loss = training_ce_loss(student_logits, targets, ignore_label)
    fixed_kd = teacher_target_kd_loss(
        student_logits, teacher_logits, TASK_ALIGNED_FALLBACK_TEMPERATURE,
        targets != ignore_label,
    )
    p4a_kd = task_aligned_region_kd_loss(student_logits, selection)

    ce_gradients = torch.autograd.grad(
        ce_loss, parameters, retain_graph=True, allow_unused=True
    )
    fixed_gradients = torch.autograd.grad(
        fixed_kd, parameters, retain_graph=True, allow_unused=True
    )
    p4a_gradients = torch.autograd.grad(
        p4a_kd, parameters, allow_unused=True
    )
    alignment = region_alignment_means(selection)
    fixed_norm = gradient_norm(fixed_gradients)
    p4a_norm = gradient_norm(p4a_gradients)

    fixed_target = F.softmax(
        teacher_logits.detach() / TASK_ALIGNED_FALLBACK_TEMPERATURE, dim=1
    )
    fixed_parameters = updated_parameters(
        model, fixed_gradients, step_size
    )
    p4a_parameters = updated_parameters(model, p4a_gradients, step_size)
    with torch.no_grad():
        same_before = float(ce_loss.detach().item())
        next_before = float(training_ce_loss(
            model(next_images)[0], next_targets, ignore_label
        ).item())
        fixed_same_after = float(training_ce_loss(
            functional_logits(model, fixed_parameters, images),
            targets, ignore_label,
        ).item())
        fixed_next_after = float(training_ce_loss(
            functional_logits(model, fixed_parameters, next_images),
            next_targets, ignore_label,
        ).item())
        p4a_same_after = float(training_ce_loss(
            functional_logits(model, p4a_parameters, images),
            targets, ignore_label,
        ).item())
        p4a_next_after = float(training_ce_loss(
            functional_logits(model, p4a_parameters, next_images),
            next_targets, ignore_label,
        ).item())

    counts = torch.bincount(
        selection.selected_region_index[selection.eligible_region_mask],
        minlength=len(TEMPERATURES),
    ).cpu().tolist()
    return {
        "stage": stage,
        "checkpoint_iteration": iteration,
        "pair_index": pair_index,
        "same_image_ids": same_batch["names"],
        "next_image_ids": next_batch["names"],
        "eligible_regions": alignment["eligible_regions"],
        "temperature_counts": [int(value) for value in counts],
        "a_z_fixed": alignment["fixed"],
        "a_z_p4a": alignment["p4a"],
        "a_z_delta": alignment["p4a"] - alignment["fixed"],
        "a_theta_fixed": gradient_cosine(ce_gradients, fixed_gradients),
        "a_theta_p4a": gradient_cosine(ce_gradients, p4a_gradients),
        "ce_gradient_norm": gradient_norm(ce_gradients),
        "kd_gradient_norm_fixed": fixed_norm,
        "kd_gradient_norm_p4a": p4a_norm,
        "teacher_target_entropy_fixed": mean_target_entropy(
            fixed_target, selection.valid_mask
        ),
        "teacher_target_entropy_p4a": mean_target_entropy(
            selection.teacher_target, selection.valid_mask
        ),
        "delta_same_fixed": same_before - fixed_same_after,
        "delta_same_p4a": same_before - p4a_same_after,
        "delta_next_fixed": next_before - fixed_next_after,
        "delta_next_p4a": next_before - p4a_next_after,
        "virtual_update_norm_fixed": step_size * fixed_norm,
        "virtual_update_norm_p4a": step_size * p4a_norm,
    }


def mean_value(rows, key):
    return statistics.fmean(float(row[key]) for row in rows)


def summarize_rows(rows):
    groups = {
        stage: [row for row in rows if row["stage"] == stage]
        for stage, _ in STAGES
    }
    groups["overall"] = list(rows)
    summary = {}
    for name, group in groups.items():
        az_delta = [
            row["a_z_p4a"] - row["a_z_fixed"] for row in group
        ]
        theta_delta = [
            row["a_theta_p4a"] - row["a_theta_fixed"] for row in group
        ]
        same_uplift = [
            row["delta_same_p4a"] - row["delta_same_fixed"]
            for row in group
        ]
        next_uplift = [
            row["delta_next_p4a"] - row["delta_next_fixed"]
            for row in group
        ]
        summary[name] = {
            "minibatch_pairs": len(group),
            "p5_1": {
                "a_z_fixed_mean": mean_value(group, "a_z_fixed"),
                "a_z_p4a_mean": mean_value(group, "a_z_p4a"),
                "a_z_delta_mean": statistics.fmean(az_delta),
                "a_theta_fixed_mean": mean_value(group, "a_theta_fixed"),
                "a_theta_p4a_mean": mean_value(group, "a_theta_p4a"),
                "a_theta_delta_mean": statistics.fmean(theta_delta),
                "joint_break_fraction": statistics.fmean(
                    dz > 1e-12 and dt <= 0.0
                    for dz, dt in zip(az_delta, theta_delta)
                ),
                "ce_gradient_norm_mean": mean_value(
                    group, "ce_gradient_norm"
                ),
                "kd_gradient_norm_fixed_mean": mean_value(
                    group, "kd_gradient_norm_fixed"
                ),
                "kd_gradient_norm_p4a_mean": mean_value(
                    group, "kd_gradient_norm_p4a"
                ),
                "teacher_target_entropy_fixed_mean": mean_value(
                    group, "teacher_target_entropy_fixed"
                ),
                "teacher_target_entropy_p4a_mean": mean_value(
                    group, "teacher_target_entropy_p4a"
                ),
            },
            "p5_2": {
                "delta_same_fixed_mean": mean_value(
                    group, "delta_same_fixed"
                ),
                "delta_same_p4a_mean": mean_value(
                    group, "delta_same_p4a"
                ),
                "p4a_uplift_same_mean": statistics.fmean(same_uplift),
                "p4a_uplift_same_median": statistics.median(
                    same_uplift
                ),
                "delta_next_fixed_mean": mean_value(
                    group, "delta_next_fixed"
                ),
                "delta_next_p4a_mean": mean_value(
                    group, "delta_next_p4a"
                ),
                "p4a_uplift_next_mean": statistics.fmean(next_uplift),
                "p4a_uplift_next_median": statistics.median(
                    next_uplift
                ),
                "p4a_same_better_fraction": statistics.fmean(
                    value > 0.0 for value in same_uplift
                ),
                "p4a_next_better_fraction": statistics.fmean(
                    value > 0.0 for value in next_uplift
                ),
            },
        }
    return summary


def summarize_training_logs(p1_path, p4a_path, p4a_report_path):
    p1 = parse_log(p1_path, require_p4a=False)
    p4a = parse_log(p4a_path, require_p4a=True)
    p4a_report = json.loads(p4a_report_path.read_text())
    cumulative_delta = 0.0
    cumulative_regions = 0
    phases = {}
    for phase in PHASES:
        p1_rows = [
            row for row in p1["training"]
            if phase_name(row["iteration"]) == phase
        ]
        p4a_rows = [
            row for row in p4a["training"]
            if phase_name(row["iteration"]) == phase
        ]
        selector = aggregate_p4a([
            row for row in p4a["p4a"]
            if phase_name(row["iteration"]) == phase
        ])
        cumulative_delta += selector["raw_sums"]["delta_alignment"]
        cumulative_regions += selector["eligible_regions"]
        phases[phase] = {
            "p1_losses": training_loss_summary(p1_rows),
            "p4a_losses": training_loss_summary(p4a_rows),
            "temperature_fractions": selector["temperature_fractions"],
            "phase_mean_delta_alignment": selector[
                "mean_delta_alignment_vs_t1p5"
            ],
            "cumulative_delta_alignment_sum": cumulative_delta,
            "cumulative_mean_delta_alignment": (
                cumulative_delta / cumulative_regions
            ),
            "cumulative_eligible_regions": cumulative_regions,
        }
    return {
        "paths": {"p1": str(p1_path), "p4a": str(p4a_path)},
        "logged_points": {
            "p1": len(p1["training"]),
            "p4a": len(p4a["training"]),
            "p4a_selector": len(p4a["p4a"]),
        },
        "phases": phases,
        "validation": {
            "p1_points": len(p1["validations"]),
            "p4a_points": len(p4a["validations"]),
            "p1_final_miou_percent": p1["validations"][-1][
                "miou_percent"
            ],
            "p4a_final_miou_percent": p4a["validations"][-1][
                "miou_percent"
            ],
            "delta_final_miou_pp": (
                p4a["validations"][-1]["miou_percent"]
                - p1["validations"][-1]["miou_percent"]
            ),
        },
        "curve_availability": {
            "ce_kd": "1000 training log points per method",
            "miou": "one 20k endpoint per method; no intermediate curve",
            "temperature": "1000 P4a interval summaries",
            "target_entropy": "not logged; frozen P5 probe only",
            "kd_gradient_norm": "not logged; frozen P5 probe only",
        },
        "p4a_report_gate": bool(p4a_report["execution_gate_pass"]),
    }


def classify_breakpoint(summary, training):
    p51 = summary["overall"]["p5_1"]
    p52 = summary["overall"]["p5_2"]
    if p51["a_z_delta_mean"] > 0 and p51["a_theta_delta_mean"] <= 0:
        return "logit_to_parameter"
    if (
        p51["a_theta_delta_mean"] > 0
        and p52["p4a_uplift_same_mean"] <= 0
    ):
        if (
            p52["p4a_uplift_same_median"] > 0
            and p52["p4a_uplift_next_median"] <= 0
        ):
            return "finite_update_and_cross_batch_instability"
        return "parameter_first_order_to_same_batch_update"
    if (
        p52["p4a_uplift_same_mean"] > 0
        and p52["p4a_uplift_next_mean"] <= 0
    ):
        return "same_to_independent_batch"
    if (
        p52["p4a_uplift_next_mean"] > 0
        and training["validation"]["delta_final_miou_pp"] <= 0
    ):
        return "independent_batch_to_long_horizon_training"
    return "mixed"


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.batch_pairs <= 0:
        raise ValueError("batch size and pair count must be positive")
    if args.virtual_step_size <= 0:
        raise ValueError("virtual step size must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal P5 audit requires CUDA")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset = VOCDataValSet(
        str(args.data), str(args.list_path), crop_size=(512, 512),
        ignore_label=args.ignore_label,
    )
    required = args.batch_pairs * args.batch_size * 2
    indices = fixed_probe_indices(
        len(dataset), args.seed, required, args.sample_pool_size
    )
    batches = []
    for pair_index in range(args.batch_pairs):
        offset = pair_index * args.batch_size * 2
        batches.append({
            "pair_index": pair_index,
            "same": materialize_batch(
                dataset, indices[offset:offset + args.batch_size]
            ),
            "next": materialize_batch(
                dataset,
                indices[
                    offset + args.batch_size:
                    offset + 2 * args.batch_size
                ],
            ),
        })

    teacher = build_teacher(args, device)
    students, checkpoint_contracts = build_students(args, device)
    for batch in batches:
        with torch.inference_mode():
            batch["teacher_logits"] = teacher(
                batch["same"]["images"].to(device)
            )[0].cpu()
    del teacher
    torch.cuda.empty_cache()

    rows = []
    for stage, iteration in STAGES:
        model = students[stage]
        for batch in batches:
            row = evaluate_pair(
                model, stage, iteration, batch["pair_index"],
                batch["same"], batch["next"], batch["teacher_logits"],
                device, args.virtual_step_size, args.ignore_label,
            )
            rows.append(row)
            print(
                f"[P5] {stage} pair={batch['pair_index'] + 1}/"
                f"{args.batch_pairs} dz={row['a_z_p4a'] - row['a_z_fixed']:+.3e} "
                f"dtheta={row['a_theta_p4a'] - row['a_theta_fixed']:+.3e} "
                f"same={row['delta_same_p4a'] - row['delta_same_fixed']:+.3e} "
                f"next={row['delta_next_p4a'] - row['delta_next_fixed']:+.3e}",
                flush=True,
            )
        del students[stage]
        torch.cuda.empty_cache()

    names = [
        name for batch in batches for side in ("same", "next")
        for name in batch[side]["names"]
    ]
    raw_checks = {
        "row_count": len(rows) == len(STAGES) * args.batch_pairs,
        "unique_images": len(names) == len(set(names)) == required,
        "finite": all(
            math.isfinite(value)
            for row in rows for value in row.values()
            if isinstance(value, float)
        ),
        "hard_argmax_nonnegative": all(
            row["a_z_p4a"] - row["a_z_fixed"] >= -1e-7
            for row in rows
        ),
        "checkpoint_contracts": all(
            all(item["contract"].values())
            for item in checkpoint_contracts.values()
        ),
        "no_optimizer_or_checkpoint_write": True,
    }
    raw_payload = {
        "stage": "P5",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "batch_pairs": args.batch_pairs,
            "virtual_step_size": args.virtual_step_size,
            "model_mode": "eval; frozen BN buffers",
            "ce": "full-resolution bilinear-upsampled mean CE",
            "kd": "native-grid mean-valid KL",
            "a_z": "eligible-region mean P4a native-grid score",
        },
        "scope": {
            "dataset": "Pascal VOC val",
            "unique_images": required,
            "image_ids": names,
            "image_ids_sha256": hashlib.sha256(
                "\n".join(names).encode()
            ).hexdigest(),
        },
        "environment": {
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(device),
        },
        "checkpoints": checkpoint_contracts,
        "rows": rows,
        "checks": raw_checks,
        "execution_gate_pass": all(raw_checks.values()),
    }
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(
        json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n"
    )

    probe_summary = summarize_rows(rows)
    training = summarize_training_logs(
        args.p1_log, args.p4a_log, args.p4a_report_json
    )
    report_checks = {
        "raw_gate": raw_payload["execution_gate_pass"],
        "training_log_points": training["logged_points"] == {
            "p1": 1000, "p4a": 1000, "p4a_selector": 1000,
        },
        "single_validation_endpoint": (
            training["validation"]["p1_points"] == 1
            and training["validation"]["p4a_points"] == 1
        ),
        "p4a_report_gate": training["p4a_report_gate"],
        "p4a_delta_locked": math.isclose(
            training["validation"]["delta_final_miou_pp"],
            -0.343633, rel_tol=0, abs_tol=1e-6,
        ),
    }
    payload = {
        "stage": "P5",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": raw_payload["config"],
        "scope": {
            "batch_pairs": args.batch_pairs,
            "batch_size": args.batch_size,
            "unique_images": required,
            "image_ids_sha256": raw_payload["scope"][
                "image_ids_sha256"
            ],
            "raw_results": str(args.raw_output),
        },
        "probe_summary": probe_summary,
        "training_process": training,
        "breakpoint": classify_breakpoint(probe_summary, training),
        "checks": report_checks,
        "execution_gate_pass": all(report_checks.values()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({
        "execution_gate_pass": payload["execution_gate_pass"],
        "breakpoint": payload["breakpoint"],
        "overall": payload["probe_summary"]["overall"],
        "output": str(args.output_json),
    }, ensure_ascii=False, indent=2))
    if not payload["execution_gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
