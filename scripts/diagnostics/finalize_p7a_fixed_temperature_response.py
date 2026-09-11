#!/usr/bin/env python3
"""Finalize P7 without training: CoVar coordinates, figures, and seed stability."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    sort_logits_for_covar,
)


SEEDS = (1234, 2025, 3407)
TEMPERATURES = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
MILESTONE = "80000"
METRICS = ("r_c", "r_v", "r")
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
QUANTILE_NAMES = ("q10", "q25", "q50", "q75", "q90")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p7-json",
        type=Path,
        default=ROOT
        / "reports"
        / "covar_match"
        / "P7_fixed_temperature_response_lower_boundary.json",
    )
    parser.add_argument(
        "--p0-csv",
        type=Path,
        default=ROOT
        / "reports"
        / "covar_match"
        / "P0_temperature_complexity_trajectory.csv",
    )
    parser.add_argument("--data", default=str(ROOT / "dataset" / "VOCAug"))
    parser.add_argument(
        "--list-path",
        default=str(ROOT / "dataset" / "list" / "voc" / "val.txt"),
    )
    parser.add_argument(
        "--teacher-pretrained",
        default=str(
            ROOT
            / "data"
            / "winycg"
            / "cirkd"
            / "teachers"
            / "deeplabv3_resnet101_voc_best_model.pth"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--max-pixels-per-image", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-classes", type=int, default=21)
    parser.add_argument("--ignore-label", type=int, default=-1)
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument(
        "--covar-json",
        type=Path,
        default=ROOT / "reports" / "covar_match" / "P7A_covar_statistics.json",
    )
    parser.add_argument(
        "--covar-csv",
        type=Path,
        default=ROOT / "reports" / "covar_match" / "P7A_covar_statistics.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "reports" / "covar_match" / "P7A_finalization.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=ROOT / "reports" / "covar_match" / "P7A_finalization.md",
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=ROOT / "figures" / "covar_match"
    )
    parser.add_argument(
        "--reuse-covar",
        action="store_true",
        help="Reuse --covar-json after validating its locked protocol.",
    )
    return parser.parse_args()


def temperature_key(value):
    return str(float(value))


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


def quantile_summary(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("quantiles require a non-empty one-dimensional sample")
    if not np.isfinite(array).all():
        raise ValueError("quantile sample contains non-finite values")
    estimates = np.quantile(array, QUANTILES, method="linear")
    normalized = [
        0.0 if float(value) == 0.0 else float(value)
        for value in estimates
    ]
    return {name: value for name, value in zip(QUANTILE_NAMES, normalized)}


def load_p0_reference(path):
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = temperature_key(row["temperature"])
            rows[key] = {
                "valid_pixels": int(row["valid_pixels"]),
                "r_c_mean": float(row["r_c_mean"]),
                "r_v_mean": float(row["r_v_mean"]),
                "r_mean": float(row["r_mean"]),
            }
    expected = {temperature_key(value) for value in TEMPERATURES[1:]}
    missing = sorted(expected - set(rows))
    if missing:
        raise RuntimeError(f"P0 reference is missing temperatures: {missing}")
    return rows


def _empty_covar_accumulator():
    return {
        temperature_key(temperature): {
            "sums": {metric: 0.0 for metric in METRICS},
            "samples": {metric: [] for metric in METRICS},
        }
        for temperature in TEMPERATURES
    }


def compute_covar_statistics(args):
    if args.max_pixels_per_image <= 0:
        raise ValueError("max-pixels-per-image must be positive for bounded quantiles")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    dataset = VOCDataValSet(
        args.data,
        args.list_path,
        crop_size=(512, 512),
        ignore_label=args.ignore_label,
    )
    loader = data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )
    teacher = build_teacher(args, device)
    accumulator = _empty_covar_accumulator()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    valid_pixels = 0
    sampled_pixels = 0
    processed_images = 0

    with torch.no_grad():
        for index, (image, target, _) in enumerate(loader):
            if args.max_images > 0 and index >= args.max_images:
                break
            image = image.to(device)
            target = target.long().to(device)
            valid = target != args.ignore_label
            image_valid_pixels = int(valid.sum().item())
            if image_valid_pixels == 0:
                continue
            output = teacher(image)
            logits = output[0] if isinstance(output, (list, tuple)) else output
            logits = F.interpolate(
                logits,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            sorted_logits = sort_logits_for_covar(logits, class_dim=1)

            valid_indices = torch.nonzero(
                valid.reshape(-1), as_tuple=False
            ).squeeze(1).cpu()
            if valid_indices.numel() > args.max_pixels_per_image:
                order = torch.randperm(valid_indices.numel(), generator=generator)[
                    : args.max_pixels_per_image
                ]
                sampled_indices = valid_indices[order]
            else:
                sampled_indices = valid_indices
            sampled_indices_device = sampled_indices.to(device)

            for temperature in TEMPERATURES:
                key = temperature_key(temperature)
                components = covar_components_from_sorted_logits(
                    sorted_logits, temperature
                )
                for metric in METRICS:
                    values = components[metric][valid]
                    if not torch.isfinite(values).all():
                        raise RuntimeError(
                            f"non-finite {metric} at T={temperature}, image={index}"
                        )
                    accumulator[key]["sums"][metric] += float(
                        values.double().sum().item()
                    )
                    sampled = components[metric].reshape(-1).index_select(
                        0, sampled_indices_device
                    )
                    accumulator[key]["samples"][metric].append(
                        sampled.detach().float().cpu()
                    )

            valid_pixels += image_valid_pixels
            sampled_pixels += int(sampled_indices.numel())
            processed_images += 1
            if processed_images % 100 == 0 or processed_images == len(dataset):
                print(
                    f"P7A CoVar: {processed_images}/{len(dataset)} images, "
                    f"valid={valid_pixels}, sampled={sampled_pixels}",
                    flush=True,
                )

    if processed_images == 0 or valid_pixels == 0:
        raise RuntimeError("no valid images were processed")

    rows = {}
    for temperature in TEMPERATURES:
        key = temperature_key(temperature)
        row = {
            "temperature": temperature,
            "valid_pixels": valid_pixels,
            "quantile_sample_pixels": sampled_pixels,
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = (
                accumulator[key]["sums"][metric] / valid_pixels
            )
            samples = torch.cat(accumulator[key]["samples"][metric]).numpy()
            for name, value in quantile_summary(samples).items():
                row[f"{metric}_{name}"] = value
            accumulator[key]["samples"][metric].clear()
        rows[key] = row

    reference = load_p0_reference(args.p0_csv)
    differences = {}
    for key, p0_row in reference.items():
        differences[key] = {
            metric: abs(rows[key][f"{metric}_mean"] - p0_row[f"{metric}_mean"])
            for metric in METRICS
        }
    maximum_reference_difference = max(
        value for row in differences.values() for value in row.values()
    )
    full_scope = args.max_images <= 0
    reference_valid_pixels = next(iter(reference.values()))["valid_pixels"]
    if full_scope and valid_pixels != reference_valid_pixels:
        raise RuntimeError(
            f"valid-pixel mismatch: P7A={valid_pixels}, P0={reference_valid_pixels}"
        )
    if full_scope and maximum_reference_difference > 5e-6:
        raise RuntimeError(
            "P7A CoVar means do not reproduce P0: "
            f"max abs difference={maximum_reference_difference:.3e}"
        )

    mean_path = [rows[temperature_key(value)]["r_mean"] for value in TEMPERATURES]
    return {
        "stage": "P7A_covar_statistics",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {
            "dataset": "Pascal VOC val",
            "dataset_size": len(dataset),
            "processed_images": processed_images,
            "teacher": "DeepLabV3-ResNet101",
            "teacher_checkpoint": str(Path(args.teacher_pretrained)),
            "teacher_output_temperature": 1.0,
            "temperatures": list(TEMPERATURES),
            "num_classes": args.num_classes,
            "coefficient_a": covar_coefficient(args.num_classes),
            "ignore_label": args.ignore_label,
            "mean_reduction": "global pixel-weighted mean over target != ignore_label",
            "valid_pixels": valid_pixels,
            "quantile_sampling": (
                "P0-locked deterministic per-image sample without replacement"
            ),
            "max_pixels_per_image": args.max_pixels_per_image,
            "quantile_sample_pixels": sampled_pixels,
            "seed": args.seed,
            "device": str(device),
            "student_training_performed": False,
        },
        "rows": rows,
        "gates": {
            "p0_reference_valid_pixels": reference_valid_pixels,
            "p0_mean_abs_differences": differences,
            "p0_mean_max_abs_difference": maximum_reference_difference,
            "p0_reproduction_pass": (
                not full_scope or maximum_reference_difference <= 5e-6
            ),
            "mean_r_strictly_increasing": all(
                right > left for left, right in zip(mean_path, mean_path[1:])
            ),
        },
    }


def write_covar_outputs(payload, json_path, csv_path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = [payload["rows"][temperature_key(value)] for value in TEMPERATURES]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def load_covar_payload(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "P7A_covar_statistics":
        raise RuntimeError("unexpected CoVar cache stage")
    protocol = payload["protocol"]
    if tuple(float(value) for value in protocol["temperatures"]) != TEMPERATURES:
        raise RuntimeError("CoVar cache temperature grid does not match P7A")
    if protocol["processed_images"] != protocol["dataset_size"]:
        raise RuntimeError("CoVar cache is not a full-dataset run")
    if not payload["gates"]["p0_reproduction_pass"]:
        raise RuntimeError("CoVar cache failed the P0 reproduction gate")
    return payload


def extract_final_miou(p7_payload):
    protocol = p7_payload["protocol"]
    if tuple(protocol["seeds"]) != SEEDS:
        raise RuntimeError("P7 seed set does not match P7A")
    if tuple(float(value) for value in protocol["temperatures"]) != TEMPERATURES:
        raise RuntimeError("P7 temperature grid does not match P7A")
    rows = {}
    for seed_row in p7_payload["seed_rows"]:
        seed = int(seed_row["seed"])
        rows[seed] = {
            temperature_key(temperature): float(
                seed_row["runs"][temperature_key(temperature)]["trajectory"][
                    MILESTONE
                ]["miou_percent"]
            )
            for temperature in TEMPERATURES
        }
    if tuple(sorted(rows)) != tuple(sorted(SEEDS)):
        raise RuntimeError("P7 JSON is missing a locked seed")
    return rows


def summarize_temperature_response(final_miou, delta):
    if delta < 0:
        raise ValueError("delta must be non-negative")
    summaries = {}
    for temperature in TEMPERATURES:
        key = temperature_key(temperature)
        values = [final_miou[seed][key] for seed in SEEDS]
        summaries[key] = {
            "values": values,
            "mean": mean(values),
            "sample_std": stdev(values),
        }
    winner = max(
        TEMPERATURES,
        key=lambda value: (
            summaries[temperature_key(value)]["mean"],
            -TEMPERATURES.index(value),
        ),
    )
    winner_key = temperature_key(winner)
    threshold = summaries[winner_key]["mean"] - delta
    near_optimal = [
        temperature_key(value)
        for value in TEMPERATURES
        if summaries[temperature_key(value)]["mean"] >= threshold
    ]
    return {
        "temperatures": summaries,
        "mean_winner": winner_key,
        "delta_miou_pp": delta,
        "delta_optimal_grid_set": near_optimal,
    }


def leave_one_seed_out_selection(final_miou):
    rows = []
    for held_out in SEEDS:
        selection_seeds = [seed for seed in SEEDS if seed != held_out]
        selection_means = {
            temperature_key(temperature): mean(
                final_miou[seed][temperature_key(temperature)]
                for seed in selection_seeds
            )
            for temperature in TEMPERATURES
        }
        selected = max(
            TEMPERATURES,
            key=lambda value: (
                selection_means[temperature_key(value)],
                -TEMPERATURES.index(value),
            ),
        )
        held_out_best = max(
            TEMPERATURES,
            key=lambda value: (
                final_miou[held_out][temperature_key(value)],
                -TEMPERATURES.index(value),
            ),
        )
        selected_key = temperature_key(selected)
        best_key = temperature_key(held_out_best)
        selected_miou = final_miou[held_out][selected_key]
        best_miou = final_miou[held_out][best_key]
        rows.append(
            {
                "held_out_seed": held_out,
                "selection_seeds": selection_seeds,
                "selected_temperature": selected_key,
                "held_out_best_temperature": best_key,
                "selected_miou_percent": selected_miou,
                "held_out_best_miou_percent": best_miou,
                "selection_regret_pp": best_miou - selected_miou,
            }
        )
    return {
        "rows": rows,
        "mean_selection_regret_pp": mean(
            row["selection_regret_pp"] for row in rows
        ),
        "interpretation": "post_hoc_descriptive_not_a_strict_generalization_estimate",
    }


def _configure_temperature_axis(axis):
    axis.set_xscale("log", base=2)
    axis.set_xticks(TEMPERATURES)
    axis.set_xticklabels([temperature_key(value) for value in TEMPERATURES])
    axis.set_xlabel("Teacher-only temperature T (log scale)")
    axis.grid(True, alpha=0.25, linewidth=0.8)


def render_figures(final_miou, response, covar, figure_dir):
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "seed_curves": figure_dir / "P7A_seed_curves.png",
        "mean_sd": figure_dir / "P7A_mean_sd.png",
        "covar_utility": figure_dir / "P7A_covar_utility.png",
    }
    colors = ("#0072B2", "#D55E00", "#009E73")

    figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    for seed, color in zip(SEEDS, colors):
        axis.plot(
            TEMPERATURES,
            [final_miou[seed][temperature_key(value)] for value in TEMPERATURES],
            marker="o",
            linewidth=1.8,
            markersize=5,
            color=color,
            label=f"seed {seed}",
        )
    _configure_temperature_axis(axis)
    axis.set_ylabel("80k validation mIoU (%)")
    axis.set_title("P7 fixed-temperature response by training seed")
    axis.legend(frameon=False, ncol=3, fontsize=9)
    figure.savefig(paths["seed_curves"], dpi=220)
    plt.close(figure)

    means = [
        response["temperatures"][temperature_key(value)]["mean"]
        for value in TEMPERATURES
    ]
    standard_deviations = [
        response["temperatures"][temperature_key(value)]["sample_std"]
        for value in TEMPERATURES
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    axis.errorbar(
        TEMPERATURES,
        means,
        yerr=standard_deviations,
        marker="o",
        color="#333333",
        ecolor="#777777",
        capsize=4,
        linewidth=1.7,
        label="mean ± sample SD",
    )
    near_values = [float(value) for value in response["delta_optimal_grid_set"]]
    axis.scatter(
        near_values,
        [means[TEMPERATURES.index(value)] for value in near_values],
        s=95,
        facecolors="none",
        edgecolors="#009E73",
        linewidths=2,
        label="δ=0.2 near-optimal",
        zorder=4,
    )
    winner = float(response["mean_winner"])
    axis.scatter(
        [winner],
        [means[TEMPERATURES.index(winner)]],
        marker="*",
        s=180,
        color="#D55E00",
        label=f"mean winner T={response['mean_winner']}",
        zorder=5,
    )
    _configure_temperature_axis(axis)
    axis.set_ylabel("80k validation mIoU (%)")
    axis.set_title("P7 mean fixed-temperature response")
    axis.legend(frameon=False, fontsize=9)
    figure.savefig(paths["mean_sd"], dpi=220)
    plt.close(figure)

    r_means = [
        covar["rows"][temperature_key(value)]["r_mean"]
        for value in TEMPERATURES
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    axis.plot(r_means, means, color="#666666", linewidth=1.3, alpha=0.8)
    axis.scatter(r_means, means, s=55, color="#0072B2", zorder=3)
    offsets = (
        (5, 7),
        (5, 7),
        (5, -13),
        (5, -13),
        (5, -13),
        (5, 7),
        (5, -13),
    )
    for temperature, x_value, y_value, offset in zip(
        TEMPERATURES, r_means, means, offsets
    ):
        axis.annotate(
            f"T={temperature_key(temperature)}",
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
        )
    axis.set_xlabel("Mean teacher CoVar complexity r")
    axis.set_ylabel("Mean 80k validation mIoU (%)")
    axis.set_title("CoVar complexity is not a monotonic utility coordinate")
    axis.grid(True, alpha=0.25, linewidth=0.8)
    axis.margins(x=0.05, y=0.12)
    figure.savefig(paths["covar_utility"], dpi=220)
    plt.close(figure)
    serialized = {}
    for name, path in paths.items():
        try:
            serialized[name] = str(path.relative_to(ROOT))
        except ValueError:
            serialized[name] = str(path.resolve())
    return serialized


def report_figure_link(value):
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return f"../../{path}"


def build_report(payload):
    response = payload["temperature_response"]
    covar = payload["covar"]
    stability = payload["selection_stability"]
    figure_paths = payload["figures"]
    rows = covar["rows"]
    near_text = ", ".join(
        f"T={value}" for value in response["delta_optimal_grid_set"]
    )
    t025 = rows["0.25"]
    lines = [
        "# P7A Finalization：固定温度响应的离线收尾",
        "",
        "## 结论先行",
        "",
        "- 本阶段未训练 student，也未新增温度点、seed 或自适应方法。",
        f"- T=0.25 的完整 CoVar 坐标为：r_c={t025['r_c_mean']:.6f}，"
        f"r_v={t025['r_v_mean']:.6f}，r={t025['r_mean']:.6f}。",
        f"- 七点平均 CoVar complexity 随 T 严格递增："
        f"{covar['gates']['mean_r_strictly_increasing']}；mean mIoU 对 r 不单调："
        f"{payload['conclusions']['mean_utility_nonmonotonic_in_r']}。",
        f"- 样本均值赢家仍为 T={response['mean_winner']}；"
        f"δ=0.2 pp 离散近优集合为 {near_text}。",
        f"- Leave-one-seed-out 的平均 selection regret 为 "
        f"{stability['mean_selection_regret_pp']:.6f} pp。",
        "- 结论：同一固定 teacher–student–protocol 内，CoVar complexity 的温度轨迹稳定，"
        "但基于少量训练轨迹识别出的效用最优温度不能稳定迁移到新的 seed。",
        "",
        "## 1. 锁定协议与统计范围",
        "",
        "- 教师、checkpoint、Pascal VOC val、预处理、有效像素定义和 CoVar 实现均与 P0 一致。",
        "- 均值使用全部有效像素的全局 pixel-weighted reduction；"
        f"有效像素数为 {covar['protocol']['valid_pixels']:,}。",
        "- 分位数使用 P0 已锁定的确定性逐图无放回像素样本：每图最多 "
        f"{covar['protocol']['max_pixels_per_image']:,} 个，"
        f"共 {covar['protocol']['quantile_sample_pixels']:,} 个像素；"
        "因此是高精度描述性样本分位数，不冒充全量精确分位数。",
        f"- 与 P0 六点均值复算的最大绝对差为 "
        f"{covar['gates']['p0_mean_max_abs_difference']:.3e}，一致性门禁通过。",
        "",
        "## 2. 七点 CoVar—效用轨迹",
        "",
        "| T | mean mIoU | sample SD | r_c mean | r_v mean | r mean |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for temperature in TEMPERATURES:
        key = temperature_key(temperature)
        response_row = response["temperatures"][key]
        covar_row = rows[key]
        lines.append(
            f"| {key} | {response_row['mean']:.6f} | "
            f"{response_row['sample_std']:.6f} | {covar_row['r_c_mean']:.6f} | "
            f"{covar_row['r_v_mean']:.6f} | {covar_row['r_mean']:.6f} |"
        )

    for metric, label in (("r_c", "r_c"), ("r_v", "r_v"), ("r", "r")):
        lines += [
            "",
            f"### {label} 像素分位数",
            "",
            "| T | q10 | q25 | q50 | q75 | q90 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for temperature in TEMPERATURES:
            key = temperature_key(temperature)
            row = rows[key]
            lines.append(
                f"| {key} | "
                + " | ".join(
                    f"{row[f'{metric}_{name}']:.6f}" for name in QUANTILE_NAMES
                )
                + " |"
            )

    lines += [
        "",
        "## 3. 最终图",
        "",
        "![七条 seed 温度响应曲线]("
        f"{report_figure_link(figure_paths['seed_curves'])})",
        "",
        "![均值与样本标准差]("
        f"{report_figure_link(figure_paths['mean_sd'])})",
        "",
        "![CoVar complexity 与效用]("
        f"{report_figure_link(figure_paths['covar_utility'])})",
        "",
        "图 C 直接显示：平均 r 随温度稳定上升，但 student utility 没有随 complexity 单调变化。",
        "",
        "## 4. Leave-one-seed-out selection stability",
        "",
        "| held-out seed | selection seeds | selected T | held-out best T | selection regret (pp) |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in stability["rows"]:
        selection_seeds = ", ".join(str(seed) for seed in row["selection_seeds"])
        lines.append(
            f"| {row['held_out_seed']} | {selection_seeds} | "
            f"{row['selected_temperature']} | {row['held_out_best_temperature']} | "
            f"{row['selection_regret_pp']:.6f} |"
        )
    lines += [
        "",
        f"平均 selection regret：{stability['mean_selection_regret_pp']:.6f} pp。",
        "",
        "该分析是 post-hoc 描述性稳定性检查，不是严格泛化误差估计。它支持的有限结论是："
        "即使模型对和协议固定，用两个 seed 选择的温度也可能无法迁移到第三条训练轨迹。",
        "",
        "## 5. 证据边界与下一门禁",
        "",
        "- 当前 VOC val 已参与温度响应刻画，不把本分析写成无偏最终测试结论。",
        "- 七点仍是离散粗网格；本报告不声称存在连续空间的唯一最优温度。",
        "- CoVar 在这里是稳定的 complexity coordinate，不是 utility objective。",
        "- P7A 没有运行 CE-only。若继续训练，下一项最小门禁是与 P7 完全同协议的 "
        "CE-only 三 seed 80k 基线；第二模型对在其后。",
        "- 未运行 T=0.125、T=0.375、T=3.0、Hard Teacher Target、P4a、"
        "CoVar-Match、shared temperature 或 T² compensation。",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    if args.reuse_covar:
        covar_payload = load_covar_payload(args.covar_json)
    else:
        covar_payload = compute_covar_statistics(args)
        write_covar_outputs(covar_payload, args.covar_json, args.covar_csv)

    p7_payload = json.loads(args.p7_json.read_text(encoding="utf-8"))
    final_miou = extract_final_miou(p7_payload)
    response = summarize_temperature_response(final_miou, args.delta)
    expected_selection = p7_payload["selection_summary"]
    if response["mean_winner"] != expected_selection["best_mean_temperature"]:
        raise RuntimeError("P7A mean winner does not reproduce P7")
    if (
        response["delta_optimal_grid_set"]
        != expected_selection["delta_optimal_grid_set"]
    ):
        raise RuntimeError("P7A delta-optimal set does not reproduce P7")
    stability = leave_one_seed_out_selection(final_miou)
    figures = render_figures(
        final_miou, response, covar_payload, args.figure_dir
    )

    r_path = [
        covar_payload["rows"][temperature_key(value)]["r_mean"]
        for value in TEMPERATURES
    ]
    utility_path = [
        response["temperatures"][temperature_key(value)]["mean"]
        for value in TEMPERATURES
    ]
    utility_nonmonotonic = not (
        all(right >= left for left, right in zip(utility_path, utility_path[1:]))
        or all(right <= left for left, right in zip(utility_path, utility_path[1:]))
    )
    payload = {
        "stage": "P7A_finalization",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "training_performed": False,
        "temperature_response": response,
        "covar": covar_payload,
        "selection_stability": stability,
        "figures": figures,
        "conclusions": {
            "mean_r_strictly_increasing": all(
                right > left for left, right in zip(r_path, r_path[1:])
            ),
            "mean_utility_nonmonotonic_in_r": utility_nonmonotonic,
            "global_case": "case_C_no_unique_reproducible_optimum",
            "next_training_gate": (
                "CE-only, same three seeds and locked 80k protocol"
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(build_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "t0p25_covar": covar_payload["rows"]["0.25"],
                "selection_stability": stability,
                "conclusions": payload["conclusions"],
                "figures": figures,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
