#!/usr/bin/env python3
"""P0 theory-to-implementation audit on the frozen VOC teacher."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.voc import VOCDataValSet
from models.model_zoo import get_segmentation_model
from utils.covar_metrics import (
    covar_coefficient,
    covar_components_from_sorted_logits,
    covar_derivatives_from_sorted_logits,
    sort_logits_for_covar,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit CoVar theory and empirical temperature trajectories.")
    parser.add_argument("--data", default=str(ROOT / "dataset" / "VOCAug"))
    parser.add_argument("--list-path", default=str(ROOT / "dataset" / "list" / "voc" / "val.txt"))
    parser.add_argument("--teacher-pretrained", default=str(
        ROOT / "data" / "winycg" / "cirkd" / "teachers" / "deeplabv3_resnet101_voc_best_model.pth"
    ))
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / "covar_match"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--max-pixels-per-image", type=int, default=4096)
    parser.add_argument("--temperatures", default="0.5,0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--num-classes", type=int, default=21)
    parser.add_argument("--ignore-label", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def parse_temperatures(raw):
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("temperatures must be positive")
    if values != sorted(set(values)):
        raise ValueError("temperatures must be unique and increasing")
    return values


def resolve_device(raw):
    if raw.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(raw)


def build_teacher(args, device):
    teacher = get_segmentation_model(
        model="deeplabv3",
        backbone="resnet101",
        local_rank=0,
        pretrained_base="None",
        pretrained=args.teacher_pretrained,
        aux=True,
        norm_layer=nn.BatchNorm2d,
        num_class=args.num_classes,
    ).to(device)
    teacher.eval()
    return teacher


def error_summary(actual, expected):
    absolute = (actual - expected).abs()
    relative = absolute / expected.abs().clamp_min(1e-8)
    return {
        "max_abs": float(absolute.max().item()),
        "mean_abs": float(absolute.mean().item()),
        "p95_relative": float(torch.quantile(relative, 0.95).item()),
        "max_relative": float(relative.max().item()),
    }


def derivative_audit(seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    logits = torch.sort(
        torch.randn(4096, 21, dtype=torch.float64, generator=generator),
        dim=-1,
        descending=True,
    ).values
    temperature = (
        0.5 + 1.5 * torch.rand(4096, dtype=torch.float64, generator=generator)
    ).requires_grad_(True)
    components = covar_components_from_sorted_logits(logits, temperature)
    autograd_first = torch.autograd.grad(components["r"].sum(), temperature, create_graph=True)[0]
    autograd_second = torch.autograd.grad(autograd_first.sum(), temperature)[0]
    closed = covar_derivatives_from_sorted_logits(logits, temperature.detach())

    step = 1e-4
    center = covar_components_from_sorted_logits(logits, temperature.detach())["r"]
    plus = covar_components_from_sorted_logits(logits, temperature.detach() + step)["r"]
    minus = covar_components_from_sorted_logits(logits, temperature.detach() - step)["r"]
    finite_first = (plus - minus) / (2.0 * step)
    finite_second = (plus - 2.0 * center + minus) / (step ** 2)
    checks = {
        "autograd_first": torch.allclose(closed["dr_dT"], autograd_first.detach(), rtol=1e-5, atol=1e-7),
        "autograd_second": torch.allclose(closed["d2r_dT2"], autograd_second.detach(), rtol=1e-5, atol=1e-7),
        "finite_first": torch.allclose(closed["dr_dT"], finite_first, rtol=1e-5, atol=1e-7),
        "finite_second": torch.allclose(closed["d2r_dT2"], finite_second, rtol=1e-4, atol=1e-6),
    }
    return {
        "num_vectors": int(logits.shape[0]),
        "dtype": "float64",
        "closed_vs_autograd_first": error_summary(closed["dr_dT"], autograd_first.detach()),
        "closed_vs_autograd_second": error_summary(closed["d2r_dT2"], autograd_second.detach()),
        "closed_vs_finite_first": error_summary(closed["dr_dT"], finite_first),
        "closed_vs_finite_second": error_summary(closed["d2r_dT2"], finite_second),
        "allclose": {key: bool(value) for key, value in checks.items()},
    }


def empty_accumulator():
    keys = [
        "confidence", "r_c", "r_v", "r", "entropy",
        "residual_variance", "normalized_residual_variance", "dr_dT",
    ]
    return {
        "count": 0,
        "finite_count": 0,
        "dr_negative_count": 0,
        "d2_negative_count": 0,
        "sums": {key: 0.0 for key in keys},
        "max_decomposition_abs": 0.0,
        "max_stable_legacy_rv_abs": 0.0,
        "max_variance_identity_abs": 0.0,
    }


def update_accumulator(accumulator, closed, valid):
    count = int(valid.sum().item())
    accumulator["count"] += count
    finite = torch.isfinite(closed["r"]) & torch.isfinite(closed["dr_dT"]) & torch.isfinite(closed["d2r_dT2"])
    accumulator["finite_count"] += int(finite[valid].sum().item())
    accumulator["dr_negative_count"] += int((closed["dr_dT"][valid] < 0).sum().item())
    accumulator["d2_negative_count"] += int((closed["d2r_dT2"][valid] < 0).sum().item())
    for key in accumulator["sums"]:
        accumulator["sums"][key] += float(closed[key][valid].double().sum().item())

    decomposition = (closed["r"] - closed["r_c"] - closed["r_v"]).abs()[valid]
    legacy_rv = (
        covar_coefficient(closed["probability"].shape[-1])
        * closed["residual_variance"] / closed["residual_mass"]
    )
    stable_legacy = (closed["r_v"] - legacy_rv).abs()[valid]
    variance_identity = (
        closed["residual_variance"]
        - closed["residual_mass"] ** 2 * closed["normalized_residual_variance"]
    ).abs()[valid]
    accumulator["max_decomposition_abs"] = max(accumulator["max_decomposition_abs"], float(decomposition.max().item()))
    accumulator["max_stable_legacy_rv_abs"] = max(accumulator["max_stable_legacy_rv_abs"], float(stable_legacy.max().item()))
    accumulator["max_variance_identity_abs"] = max(accumulator["max_variance_identity_abs"], float(variance_identity.max().item()))


def finalize_row(temperature, accumulator):
    count = max(accumulator["count"], 1)
    row = {
        "temperature": temperature,
        "valid_pixels": accumulator["count"],
        "finite_fraction": accumulator["finite_count"] / count,
    }
    for key, value in accumulator["sums"].items():
        row[f"{key}_mean"] = value / count
    row["dr_negative_fraction"] = accumulator["dr_negative_count"] / count
    row["d2r_negative_fraction"] = accumulator["d2_negative_count"] / count
    for key in ("max_decomposition_abs", "max_stable_legacy_rv_abs", "max_variance_identity_abs"):
        row[key] = accumulator[key]
    return row


def render_report(summary, rows):
    lines = [
        "# P0 CoVar 理论—实现一致性门禁报告",
        "",
        f"- 数据：Pascal VOC val，处理 {summary['scope']['processed_images']} / {summary['scope']['dataset_size']} 张图像",
        "- 教师：DeepLabV3-ResNet101，teacher output temperature 固定为 1.0",
        f"- 类别数：{summary['config']['num_classes']}，a={summary['config']['coefficient_a']:.1f}",
        f"- 设备：{summary['config']['device']}",
        f"- P0 门禁：{'通过' if summary['gate_pass'] else '未通过'}",
        "",
        "## 1. 代码审计",
        "",
        "- 余类方差采用总体方差，即除以 K-1；不是方差求和或无偏样本方差。",
        "- 权威分解为 r_c=-log(C)，r_v=a(1-C)V，r=r_c+r_v。",
        "- 训练入口、PCOS 统计和独立 Newton 工具统一复用 utils/covar_metrics.py。",
        "- 本阶段直接使用原始教师 logits，不施加 outer teacher temperature。",
        "",
        "## 2. 数值公式门禁",
        "",
        "| 检查 | max abs | p95 relative | allclose |",
        "|---|---:|---:|---|",
    ]
    audit = summary["derivative_audit"]
    mapping = [
        ("闭式一阶导 vs autograd", "closed_vs_autograd_first", "autograd_first"),
        ("闭式二阶导 vs autograd", "closed_vs_autograd_second", "autograd_second"),
        ("闭式一阶导 vs 有限差分", "closed_vs_finite_first", "finite_first"),
        ("闭式二阶导 vs 有限差分", "closed_vs_finite_second", "finite_second"),
    ]
    for label, stats_key, pass_key in mapping:
        stats = audit[stats_key]
        lines.append(
            f"| {label} | {stats['max_abs']:.3e} | {stats['p95_relative']:.3e} | "
            f"{'pass' if audit['allclose'][pass_key] else 'fail'} |"
        )
    lines += [
        "",
        "## 3. 真实教师输出的温度轨迹",
        "",
        "| T | C | r_c | r_v | r | H | V | r'<0 | r''<0 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['temperature']:.2f} | {row['confidence_mean']:.6f} | {row['r_c_mean']:.6f} | "
            f"{row['r_v_mean']:.6f} | {row['r_mean']:.6f} | {row['entropy_mean']:.6f} | "
            f"{row['normalized_residual_variance_mean']:.6e} | "
            f"{row['dr_negative_fraction']:.4%} | {row['d2r_negative_fraction']:.4%} |"
        )
    empirical = summary["empirical"]
    lines += [
        "",
        "## 4. 结论与边界",
        "",
        f"- 扫描温度下平均 r 的最小点为 T={empirical['temperature_min_mean_r']:.2f}。",
        f"- 平均 r 随温度严格递增：{empirical['mean_r_strictly_increasing']}。",
        f"- 采样像素中至少一次离散下降的比例为 {empirical['sampled_pixel_any_decrease_fraction']:.4%}；"
        f"轨迹方向翻转比例为 {empirical['sampled_pixel_turn_fraction']:.4%}。",
        "- T 对输出复杂度具有可测影响，但平均轨迹不能替代逐像素非单调性分析。",
        "- 本阶段没有训练学生，不提供 mIoU、蒸馏收益或最优温度结论。",
        "",
    ]
    return "\n".join(lines).rstrip()


def main():
    args = parse_args()
    temperatures = parse_temperatures(args.temperatures)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dataset = VOCDataValSet(args.data, args.list_path, crop_size=(512, 512), ignore_label=args.ignore_label)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False)
    teacher = build_teacher(args, device)
    accumulators = {temperature: empty_accumulator() for temperature in temperatures}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    sampled_any_decrease = 0
    sampled_turn = 0
    sampled_total = 0
    processed_images = 0

    with torch.no_grad():
        for index, (image, target, _) in enumerate(loader):
            if args.max_images > 0 and index >= args.max_images:
                break
            image = image.to(device)
            target = target.long().to(device)
            valid = target != args.ignore_label
            if not valid.any():
                continue
            output = teacher(image)
            logits = output[0] if isinstance(output, (list, tuple)) else output
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=True)
            sorted_logits = sort_logits_for_covar(logits, class_dim=1)

            valid_indices = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(1).cpu()
            if args.max_pixels_per_image > 0 and valid_indices.numel() > args.max_pixels_per_image:
                order = torch.randperm(valid_indices.numel(), generator=generator)[:args.max_pixels_per_image]
                sampled_indices = valid_indices[order]
            else:
                sampled_indices = valid_indices
            sampled_r = []
            for temperature in temperatures:
                temp_map = torch.full(sorted_logits.shape[:-1], temperature, device=device, dtype=sorted_logits.dtype)
                closed = covar_derivatives_from_sorted_logits(sorted_logits, temp_map)
                update_accumulator(accumulators[temperature], closed, valid)
                sampled_r.append(closed["r"].reshape(-1)[sampled_indices.to(device)].detach().cpu())

            trajectory = torch.stack(sampled_r, dim=0)
            difference = trajectory[1:] - trajectory[:-1]
            sampled_any_decrease += int((difference < 0).any(dim=0).sum().item())
            if difference.shape[0] > 1:
                sign = torch.sign(difference)
                sampled_turn += int(((sign[1:] * sign[:-1]) < 0).any(dim=0).sum().item())
            sampled_total += int(trajectory.shape[1])
            processed_images += 1

    if processed_images == 0:
        raise RuntimeError("no valid images were processed")

    rows = [finalize_row(temperature, accumulators[temperature]) for temperature in temperatures]
    mean_r = [row["r_mean"] for row in rows]
    audit = derivative_audit(args.seed)
    gate_pass = (
        all(audit["allclose"].values())
        and all(row["finite_fraction"] == 1.0 for row in rows)
        and max(row["max_decomposition_abs"] for row in rows) < 1e-6
        and max(row["max_stable_legacy_rv_abs"] for row in rows) < 1e-5
        and max(row["max_variance_identity_abs"] for row in rows) < 1e-7
    )
    summary = {
        "scope": {
            "dataset": "Pascal VOC val",
            "dataset_size": len(dataset),
            "processed_images": processed_images,
            "sampled_trajectory_pixels": sampled_total,
        },
        "config": {
            "device": str(device),
            "temperatures": temperatures,
            "num_classes": args.num_classes,
            "coefficient_a": covar_coefficient(args.num_classes),
            "teacher_output_temperature": 1.0,
            "seed": args.seed,
        },
        "derivative_audit": audit,
        "empirical": {
            "temperature_min_mean_r": temperatures[int(np.argmin(mean_r))],
            "mean_r_strictly_increasing": all(
                mean_r[index + 1] > mean_r[index] for index in range(len(mean_r) - 1)
            ),
            "sampled_pixel_any_decrease_fraction": sampled_any_decrease / max(sampled_total, 1),
            "sampled_pixel_turn_fraction": sampled_turn / max(sampled_total, 1),
        },
        "gate_pass": gate_pass,
    }

    csv_path = output_dir / "P0_temperature_complexity_trajectory.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "P0_metric_theory_audit.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_path = output_dir / "P0_metric_theory_audit.md"
    report_path.write_text(render_report(summary, rows) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
