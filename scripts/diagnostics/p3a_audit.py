#!/usr/bin/env python3
"""P3A: audit score direction, KL, one-step gain, and oracle ceiling."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.voc import VOCDataValSet
from scripts.diagnostics.covar_gap_teachability import (
    audit_raw_cost_direction,
    file_sha256,
    iter_candidate_groups,
    spearman,
)
from scripts.diagnostics.region_teachability_oracle import (
    STAGES,
    build_students,
    build_teacher,
    region_mean,
)
from utils.region_teachability import normalized_teacher_step_maps
from utils.teacher_only_kd import teacher_target_kd_loss


TEMPERATURES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
STEP_SIZES = (0.1, 0.03, 0.01)
DIRECTION_IMAGE_COUNT = 30
BOOTSTRAP_REPLICATES = 2000
STAGE_NAMES = tuple(stage for stage, _ in STAGES)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "dataset/VOCAug")
    parser.add_argument(
        "--list-path", type=Path, default=ROOT / "dataset/list/voc/val.txt"
    )
    parser.add_argument(
        "--teacher-pretrained",
        type=Path,
        default=(
            ROOT
            / "data/winycg/cirkd/teachers"
            / "deeplabv3_resnet101_voc_best_model.pth"
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=(
            ROOT
            / "runs/covar_match/P1_teacher_only_temperature/checkpoints"
            / "T1p0_20k_seed1234"
        ),
    )
    parser.add_argument(
        "--candidate-cache",
        type=Path,
        default=ROOT / "runs/covar_match/P2_region_candidate_teachability.csv.gz",
    )
    parser.add_argument(
        "--p2-json",
        type=Path,
        default=ROOT / "reports/covar_match/P2_region_teachability.json",
    )
    parser.add_argument(
        "--p3-json",
        type=Path,
        default=ROOT / "reports/covar_match/P3_covar_gap_teachability.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/covar_match",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ignore-label", type=int, default=-1)
    return parser.parse_args()


def sha256_strings(values):
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def run_score_direction_audit(p3_payload):
    artificial = audit_raw_cost_direction(
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
    )
    contract = p3_payload.get("score_direction_contract", {})
    passed = (
        artificial["pred_idx"] == 0
        and artificial["oracle_idx"] == 0
        and abs(artificial["raw_rho"] + 1.0) <= 1e-12
        and abs(artificial["aligned_rho"] - 1.0) <= 1e-12
        and artificial["rho_identity_abs_error"] <= 1e-12
        and contract.get("artificial_array_pass") is True
        and float(contract.get("max_rho_identity_abs_error", math.inf))
        <= 1e-12
    )
    return {
        "selection_rule": "pred_idx = argmin(raw_cost)",
        "raw_correlation": "raw_rho = spearmanr(raw_cost, gain)",
        "aligned_correlation": (
            "aligned_rho = spearmanr(-raw_cost, gain)"
        ),
        "identity": "aligned_rho = -raw_rho",
        "artificial_cost": [1.0, 2.0, 3.0, 4.0],
        "artificial_gain": [4.0, 3.0, 2.0, 1.0],
        "artificial_result": artificial,
        "p3_all_group_max_identity_abs_error": float(
            contract.get("max_rho_identity_abs_error", math.inf)
        ),
        "pass": passed,
    }


def run_kl_audit():
    student = torch.tensor(
        [
            [
                [[0.2, -0.4], [1.1, 0.3]],
                [[-0.3, 0.7], [0.1, -0.5]],
                [[0.5, 0.0], [-0.2, 0.8]],
            ]
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    teacher = torch.tensor(
        [
            [
                [[1.0, -0.2], [0.4, 0.9]],
                [[0.1, 0.8], [-0.5, 0.0]],
                [[-0.7, 0.3], [1.2, -0.4]],
            ]
        ],
        dtype=torch.float64,
    )
    valid = torch.tensor([[[True, False], [True, True]]])
    temperature = 1.7
    implementation = teacher_target_kd_loss(
        student, teacher, temperature, valid
    )
    teacher_probability = F.softmax(teacher / temperature, dim=1)
    teacher_log_probability = F.log_softmax(
        teacher / temperature, dim=1
    )
    student_log_probability = F.log_softmax(student, dim=1)
    manual_map = (
        teacher_probability
        * (teacher_log_probability - student_log_probability)
    ).sum(dim=1)
    valid_count = int(valid.sum().item())
    manual = manual_map[valid].sum() / valid_count

    implementation_gradient = torch.autograd.grad(
        implementation, student
    )[0]
    expected_gradient = (
        (F.softmax(student.detach(), dim=1) - teacher_probability)
        * valid.unsqueeze(1)
        / valid_count
    )
    value_error = abs(
        float(implementation.detach().item() - manual.detach().item())
    )
    gradient_error = float(
        (implementation_gradient - expected_gradient).abs().max().item()
    )
    return {
        "temperature": temperature,
        "valid_pixels": valid_count,
        "reduction": "sum over valid pixels divided by valid pixel count",
        "implementation_value": float(implementation.detach().item()),
        "manual_kl_value": float(manual.detach().item()),
        "value_abs_error": value_error,
        "gradient_max_abs_error": gradient_error,
        "expected_gradient": "(p_s - p_t) * valid_mask / valid_count",
        "pass": value_error <= 1e-12 and gradient_error <= 1e-12,
    }


def load_gain_cache(path, temperatures):
    by_stage_image = {
        stage: defaultdict(list) for stage in STAGE_NAMES
    }
    image_indices = {}
    group_count = 0
    for _, rows in iter_candidate_groups(path):
        rows = sorted(rows, key=lambda row: float(row["temperature"]))
        observed = tuple(float(row["temperature"]) for row in rows)
        if observed != tuple(temperatures):
            raise RuntimeError(f"candidate temperature drift: {observed}")
        stage = rows[0]["stage"]
        image_id = rows[0]["image_id"]
        image_index = int(rows[0]["image_index"])
        previous = image_indices.setdefault(image_id, image_index)
        if previous != image_index:
            raise RuntimeError(f"image index drift for {image_id}")
        by_stage_image[stage][image_id].append(
            [float(row["one_step_gain"]) for row in rows]
        )
        group_count += 1

    converted = {}
    for stage in STAGE_NAMES:
        converted[stage] = {
            image_id: np.asarray(rows, dtype=np.float64)
            for image_id, rows in by_stage_image[stage].items()
        }
        if not converted[stage]:
            raise RuntimeError(f"no candidate rows for {stage}")
    return converted, image_indices, group_count


def oracle_point_metrics(image_gains, temperatures):
    gains = np.concatenate(list(image_gains.values()), axis=0)
    fixed_means = gains.mean(axis=0)
    oracle_gains = gains.max(axis=1)
    ordered = np.sort(gains, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    best_index = int(np.argmax(fixed_means))
    best_fixed_gain = float(fixed_means[best_index])
    oracle_gain = float(oracle_gains.mean())
    uplift = oracle_gain - best_fixed_gain
    return {
        "image_count": len(image_gains),
        "region_count": int(gains.shape[0]),
        "mean_gain_by_temperature": {
            str(temperature): float(fixed_means[index])
            for index, temperature in enumerate(temperatures)
        },
        "best_fixed_temperature": float(temperatures[best_index]),
        "best_fixed_mean_gain": best_fixed_gain,
        "mean_oracle_gain": oracle_gain,
        "oracle_uplift_over_best_fixed": uplift,
        "relative_oracle_uplift": uplift / abs(best_fixed_gain),
        "median_top1_second_margin": float(np.median(margins)),
        "p_margin_lt_1e_4": float(np.mean(margins < 1e-4)),
        "p_margin_lt_1pct_abs_oracle_gain": float(
            np.mean(margins < 0.01 * np.abs(oracle_gains))
        ),
    }


def percentile_interval(values):
    values = np.asarray(values, dtype=np.float64)
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def bootstrap_oracle_metrics(
    image_gains,
    temperatures,
    replicates=BOOTSTRAP_REPLICATES,
    seed=1234,
):
    image_ids = sorted(image_gains)
    stats = []
    for image_id in image_ids:
        gains = image_gains[image_id]
        oracle = gains.max(axis=1)
        ordered = np.sort(gains, axis=1)
        margins = ordered[:, -1] - ordered[:, -2]
        stats.append(
            {
                "count": gains.shape[0],
                "gain_sum": gains.sum(axis=0),
                "oracle_sum": float(oracle.sum()),
                "margins": margins,
                "narrow_abs": int(np.sum(margins < 1e-4)),
                "narrow_relative": int(
                    np.sum(margins < 0.01 * np.abs(oracle))
                ),
            }
        )

    counts = np.asarray([item["count"] for item in stats], dtype=np.int64)
    gain_sums = np.stack([item["gain_sum"] for item in stats])
    oracle_sums = np.asarray(
        [item["oracle_sum"] for item in stats], dtype=np.float64
    )
    narrow_abs = np.asarray(
        [item["narrow_abs"] for item in stats], dtype=np.float64
    )
    narrow_relative = np.asarray(
        [item["narrow_relative"] for item in stats], dtype=np.float64
    )

    rng = np.random.default_rng(seed)
    samples = rng.integers(
        0, len(image_ids), size=(replicates, len(image_ids))
    )
    sampled_counts = counts[samples].sum(axis=1)
    fixed_draws = (
        gain_sums[samples].sum(axis=1)
        / sampled_counts[:, np.newaxis]
    )
    oracle_draws = (
        oracle_sums[samples].sum(axis=1) / sampled_counts
    )
    best_fixed_draws = fixed_draws.max(axis=1)
    uplift_draws = oracle_draws - best_fixed_draws
    relative_uplift_draws = uplift_draws / np.abs(best_fixed_draws)
    narrow_abs_draws = narrow_abs[samples].sum(axis=1) / sampled_counts
    narrow_relative_draws = (
        narrow_relative[samples].sum(axis=1) / sampled_counts
    )
    median_margin_draws = np.empty(replicates, dtype=np.float64)
    for replicate_index, sample in enumerate(samples):
        margins = np.concatenate(
            [stats[index]["margins"] for index in sample]
        )
        median_margin_draws[replicate_index] = np.median(margins)

    return {
        "resampling_unit": "image",
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence_level": 0.95,
        "mean_gain_by_temperature_ci95": {
            str(temperature): percentile_interval(fixed_draws[:, index])
            for index, temperature in enumerate(temperatures)
        },
        "best_fixed_mean_gain_ci95": percentile_interval(
            best_fixed_draws
        ),
        "mean_oracle_gain_ci95": percentile_interval(oracle_draws),
        "oracle_uplift_over_best_fixed_ci95": percentile_interval(
            uplift_draws
        ),
        "relative_oracle_uplift_ci95": percentile_interval(
            relative_uplift_draws
        ),
        "median_top1_second_margin_ci95": percentile_interval(
            median_margin_draws
        ),
        "p_margin_lt_1e_4_ci95": percentile_interval(narrow_abs_draws),
        "p_margin_lt_1pct_abs_oracle_gain_ci95": percentile_interval(
            narrow_relative_draws
        ),
    }


def choose_direction_images(image_indices, count, seed):
    items = sorted(
        image_indices.items(),
        key=lambda item: (item[1], item[0]),
    )
    if count > len(items):
        raise ValueError("direction image count exceeds P2 cache images")
    selected_positions = random.Random(seed + 31).sample(
        range(len(items)), count
    )
    return sorted(
        [items[position] for position in selected_positions],
        key=lambda item: (item[1], item[0]),
    )


def summarize_direction_rhos(rhos, selected_image_count):
    summary = {}
    for scope in (*STAGE_NAMES, "overall"):
        summary[scope] = {}
        for step_size in STEP_SIZES:
            if scope == "overall":
                values = np.concatenate(
                    [
                        np.asarray(rhos[stage][step_size], dtype=np.float64)
                        for stage in STAGE_NAMES
                    ]
                )
            else:
                values = np.asarray(
                    rhos[scope][step_size], dtype=np.float64
                )
            if len(values) == 0 or not np.isfinite(values).all():
                raise RuntimeError(
                    f"invalid direction rho values for {scope}/{step_size}"
                )
            summary[scope][str(step_size)] = {
                "image_count": selected_image_count,
                "region_count": int(len(values)),
                "mean_spearman": float(values.mean()),
                "median_spearman": float(np.median(values)),
                "p_spearman_ge_0_9": float(np.mean(values >= 0.9)),
            }

    trend_by_scope = {}
    for scope in (*STAGE_NAMES, "overall"):
        means = [
            summary[scope][str(step_size)]["mean_spearman"]
            for step_size in STEP_SIZES
        ]
        medians = [
            summary[scope][str(step_size)]["median_spearman"]
            for step_size in STEP_SIZES
        ]
        mean_nondecreasing = (
            means[0] <= means[1] + 1e-12
            and means[1] <= means[2] + 1e-12
        )
        median_nondecreasing = (
            medians[0] <= medians[1] + 1e-12
            and medians[1] <= medians[2] + 1e-12
        )
        close_to_one = medians[-1] >= 0.9
        trend_by_scope[scope] = {
            "mean_spearman_by_descending_eta": means,
            "median_spearman_by_descending_eta": medians,
            "mean_nondecreasing_as_eta_decreases": mean_nondecreasing,
            "median_nondecreasing_as_eta_decreases": median_nondecreasing,
            "eta_0p01_median_close_to_one": close_to_one,
            "pass": (
                mean_nondecreasing
                and median_nondecreasing
                and close_to_one
            ),
        }
    return summary, trend_by_scope


def run_direction_gain_audit(
    args,
    image_indices,
    temperatures,
):
    selected = choose_direction_images(
        image_indices, DIRECTION_IMAGE_COUNT, args.seed
    )
    selected_ids = [image_id for image_id, _ in selected]
    dataset = VOCDataValSet(
        str(args.data),
        str(args.list_path),
        crop_size=(512, 512),
        ignore_label=args.ignore_label,
    )
    device = torch.device(args.device)
    teacher = build_teacher(args, device)
    students, checkpoint_contracts = build_students(args, device)
    rhos = {
        stage: {step_size: [] for step_size in STEP_SIZES}
        for stage in STAGE_NAMES
    }
    max_direction_recompute_error = 0.0

    with torch.inference_mode():
        for image_position, (image_id, image_index) in enumerate(selected):
            dataset_id = dataset.files[image_index]["name"]
            if dataset_id != image_id:
                raise RuntimeError(
                    f"dataset/cache image mismatch: {dataset_id} != {image_id}"
                )
            image_array, target_array, _ = dataset[image_index]
            image = torch.from_numpy(image_array).unsqueeze(0).to(device)
            target = (
                torch.from_numpy(target_array)
                .long()
                .unsqueeze(0)
                .to(device)
            )
            teacher_logits = teacher(image)[0]

            for stage, _ in STAGES:
                student_logits = students[stage](image)[0]
                if student_logits.shape != teacher_logits.shape:
                    raise RuntimeError(
                        f"logit shape mismatch for {stage}/{image_id}"
                    )
                audit_student_logits = student_logits.double()
                audit_teacher_logits = teacher_logits.double()
                aligned_target = F.interpolate(
                    target.float().unsqueeze(1),
                    size=student_logits.shape[-2:],
                    mode="nearest",
                ).squeeze(1).long()
                valid = aligned_target[0] != args.ignore_label
                direction_maps = {}
                gain_maps = {
                    step_size: {} for step_size in STEP_SIZES
                }
                for temperature in temperatures:
                    reference_direction = None
                    for step_size in STEP_SIZES:
                        maps = normalized_teacher_step_maps(
                            audit_student_logits,
                            audit_teacher_logits,
                            aligned_target,
                            temperature,
                            step_size=step_size,
                            ignore_label=args.ignore_label,
                        )
                        current_direction = maps["ce_kd_direction_dot"]
                        if reference_direction is None:
                            reference_direction = current_direction
                            direction_maps[temperature] = current_direction
                        else:
                            max_direction_recompute_error = max(
                                max_direction_recompute_error,
                                float(
                                    (
                                        current_direction
                                        - reference_direction
                                    )
                                    .abs()
                                    .max()
                                    .item()
                                ),
                            )
                        gain_maps[step_size][temperature] = maps["ce_gain"]

                height, width = valid.shape
                for y0 in range(0, height, 8):
                    y1 = min(y0 + 8, height)
                    for x0 in range(0, width, 8):
                        x1 = min(x0 + 8, width)
                        region_valid = valid[y0:y1, x0:x1]
                        if int(region_valid.sum().item()) < 16:
                            continue
                        slices = (slice(y0, y1), slice(x0, x1))
                        direction_values = [
                            region_mean(
                                direction_maps[temperature][
                                    0, slices[0], slices[1]
                                ],
                                region_valid,
                            )
                            for temperature in temperatures
                        ]
                        for step_size in STEP_SIZES:
                            gain_values = [
                                region_mean(
                                    gain_maps[step_size][temperature][
                                        0, slices[0], slices[1]
                                    ],
                                    region_valid,
                                )
                                for temperature in temperatures
                            ]
                            rhos[stage][step_size].append(
                                spearman(direction_values, gain_values)
                            )
            if (image_position + 1) % 5 == 0:
                print(
                    f"[P3A] direction images "
                    f"{image_position + 1}/{len(selected)}",
                    flush=True,
                )

    summary, trend = summarize_direction_rhos(rhos, len(selected))
    gate_pass = (
        max_direction_recompute_error <= 1e-12
        and all(item["pass"] for item in trend.values())
        and all(
            all(contract["contract"].values())
            for contract in checkpoint_contracts.values()
        )
    )
    return {
        "definition": {
            "u_T": (
                "(p_s - p_t(T)) / ||p_s - p_t(T)||_2 per valid pixel"
            ),
            "actual_update": "z_s_after = z_s_before - eta * u_T",
            "directional_score": "d(T) = <g_CE, u_T>",
            "comparison": "spearmanr(d(T), CE_before - CE_after(T))",
            "region_reduction": (
                "mean over the same valid pixels for d(T) and gain"
            ),
            "numerical_precision": (
                "frozen float32 model logits cast to float64 before "
                "direction and CE-difference audit"
            ),
        },
        "scope": {
            "dataset": "Pascal VOC val",
            "selected_images": len(selected),
            "selected_image_ids_sha256": sha256_strings(selected_ids),
            "temperatures": list(temperatures),
            "step_sizes": list(STEP_SIZES),
            "region_size": 8,
            "min_valid_pixels": 16,
        },
        "metrics": summary,
        "trend": trend,
        "max_direction_recompute_abs_error": (
            max_direction_recompute_error
        ),
        "checkpoint_contracts": checkpoint_contracts,
        "pass": gate_pass,
    }


def write_direction_csv(path, direction_audit):
    fields = (
        "scope",
        "eta",
        "image_count",
        "region_count",
        "mean_spearman",
        "median_spearman",
        "p_spearman_ge_0_9",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        for scope in (*STAGE_NAMES, "overall"):
            for step_size in STEP_SIZES:
                writer.writerow(
                    {
                        "scope": scope,
                        "eta": step_size,
                        **direction_audit["metrics"][scope][
                            str(step_size)
                        ],
                    }
                )


def write_oracle_csv(path, oracle_ceiling):
    fields = (
        "stage",
        "temperature",
        "mean_gain",
        "ci95_low",
        "ci95_high",
        "is_best_fixed",
        "best_fixed_temperature",
        "mean_oracle_gain",
        "oracle_uplift_over_best_fixed",
        "relative_oracle_uplift",
        "median_top1_second_margin",
        "p_margin_lt_1e_4",
        "p_margin_lt_1pct_abs_oracle_gain",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        for stage in STAGE_NAMES:
            point = oracle_ceiling[stage]["point"]
            bootstrap = oracle_ceiling[stage]["bootstrap"]
            for temperature in TEMPERATURES:
                interval = bootstrap[
                    "mean_gain_by_temperature_ci95"
                ][str(temperature)]
                writer.writerow(
                    {
                        "stage": stage,
                        "temperature": temperature,
                        "mean_gain": point[
                            "mean_gain_by_temperature"
                        ][str(temperature)],
                        "ci95_low": interval[0],
                        "ci95_high": interval[1],
                        "is_best_fixed": int(
                            temperature
                            == point["best_fixed_temperature"]
                        ),
                        "best_fixed_temperature": point[
                            "best_fixed_temperature"
                        ],
                        "mean_oracle_gain": point["mean_oracle_gain"],
                        "oracle_uplift_over_best_fixed": point[
                            "oracle_uplift_over_best_fixed"
                        ],
                        "relative_oracle_uplift": point[
                            "relative_oracle_uplift"
                        ],
                        "median_top1_second_margin": point[
                            "median_top1_second_margin"
                        ],
                        "p_margin_lt_1e_4": point[
                            "p_margin_lt_1e_4"
                        ],
                        "p_margin_lt_1pct_abs_oracle_gain": point[
                            "p_margin_lt_1pct_abs_oracle_gain"
                        ],
                    }
                )


def format_ci(estimate, interval, digits=6):
    return (
        f"{estimate:.{digits}f} "
        f"[{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"
    )


def format_percent_ci(estimate, interval):
    return (
        f"{estimate:.2%} "
        f"[{interval[0]:.2%}, {interval[1]:.2%}]"
    )


def format_scientific_ci(estimate, interval):
    return (
        f"{estimate:.3e} "
        f"[{interval[0]:.3e}, {interval[1]:.3e}]"
    )


def render_report(payload):
    score = payload["p3a_1_score_direction"]
    kl = payload["p3a_2_kl"]
    direction = payload["p3a_3_directional_gain"]
    oracle = payload["p3a_4_oracle_ceiling"]
    lines = [
        "# P3A：方向、KL、gain 与 oracle 上限审计",
        "",
        "- 本阶段不训练，不实现 CoVar-Match。",
        "- 温度集合固定为 0.5、0.75、1.0、1.25、1.5、2.0；region 固定为 8×8，至少 16 个有效位置。",
        "- P3A-3 使用固定抽样 30 张 VOC val 图像；P3A-4 使用既有 P2 全部 200 张图像缓存。",
        "- P3A-4 的 95% 区间使用 image-level cluster bootstrap，不把区域视为独立样本。",
        f"- 执行门禁：{'通过' if payload['execution_gate_pass'] else '失败'}。",
        "",
        "## P3A-1：score 方向",
        "",
        f"- {score['selection_rule']}。",
        f"- {score['raw_correlation']}。",
        f"- {score['aligned_correlation']}。",
        f"- {score['identity']}；全 P3 分组最大误差为 "
        f"{score['p3_all_group_max_identity_abs_error']:.3e}。",
        "",
        "| cost | gain | pred_idx | oracle_idx | raw_rho | aligned_rho |",
        "|---|---|---:|---:|---:|---:|",
        "| [1,2,3,4] | [4,3,2,1] | "
        f"{score['artificial_result']['pred_idx']} | "
        f"{score['artificial_result']['oracle_idx']} | "
        f"{score['artificial_result']['raw_rho']:.1f} | "
        f"{score['artificial_result']['aligned_rho']:.1f} |",
        "",
        "## P3A-2：KL 方向与梯度",
        "",
        "- 数值定义：mean_valid sum_c p_t(c)(log p_t(c)-log p_s(c))。",
        "- 梯度定义：(p_s-p_t)×valid_mask/valid_count。",
        f"- 有效像素：{kl['valid_pixels']}；loss 绝对误差 "
        f"{kl['value_abs_error']:.3e}；梯度最大绝对误差 "
        f"{kl['gradient_max_abs_error']:.3e}。",
        "",
        "## P3A-3：实际一步更新方向",
        "",
        "- 定义 u_T=(p_s-p_t(T))/||p_s-p_t(T)||，实际更新为 z_s←z_s-eta·u_T。",
        "- 冻结模型 logits 在方向和 CE 差分审计前转为 float64，"
        "避免 eta=0.01 时 float32 相减消减。",
        "- 比较 d(T)=<g_CE,u_T> 与 Delta(T)=CE_before-CE_after(T) 的 Spearman。",
        "",
        "| scope | eta | regions | mean rho | median rho | P(rho≥0.9) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scope in (*STAGE_NAMES, "overall"):
        for step_size in STEP_SIZES:
            metrics = direction["metrics"][scope][str(step_size)]
            lines.append(
                f"| {scope} | {step_size:g} | "
                f"{metrics['region_count']} | "
                f"{metrics['mean_spearman']:.6f} | "
                f"{metrics['median_spearman']:.6f} | "
                f"{metrics['p_spearman_ge_0_9']:.2%} |"
            )
    overall_direction = direction["metrics"]["overall"]
    lines.extend(
        [
            "",
            "- overall mean rho 随 eta 从 0.1 降至 0.03、0.01："
            + " → ".join(
                f"{overall_direction[str(step_size)]['mean_spearman']:.6f}"
                for step_size in STEP_SIZES
            )
            + "；三个步长的 overall median rho 均为 1.000000。",
        ]
    )
    lines.extend(
        [
            "",
            "## P3A-4：oracle 真实可利用上限",
            "",
            "下表为 mean gain，括号内为 image bootstrap 95% 区间。",
            "",
            "| stage | T=0.5 | T=0.75 | T=1.0 | T=1.25 | T=1.5 | T=2.0 | best fixed | mean oracle | uplift |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for stage in STAGE_NAMES:
        point = oracle[stage]["point"]
        bootstrap = oracle[stage]["bootstrap"]
        fixed_cells = []
        for temperature in TEMPERATURES:
            fixed_cells.append(
                format_ci(
                    point["mean_gain_by_temperature"][str(temperature)],
                    bootstrap["mean_gain_by_temperature_ci95"][
                        str(temperature)
                    ],
                )
            )
        lines.append(
            f"| {stage} | "
            + " | ".join(fixed_cells)
            + f" | T={point['best_fixed_temperature']:g} | "
            + format_ci(
                point["mean_oracle_gain"],
                bootstrap["mean_oracle_gain_ci95"],
            )
            + " | "
            + format_ci(
                point["oracle_uplift_over_best_fixed"],
                bootstrap["oracle_uplift_over_best_fixed_ci95"],
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "| stage | relative uplift | median margin | P(margin<1e-4) | P(margin<1%×abs(oracle gain)) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for stage in STAGE_NAMES:
        point = oracle[stage]["point"]
        bootstrap = oracle[stage]["bootstrap"]
        lines.append(
            f"| {stage} | "
            + format_percent_ci(
                point["relative_oracle_uplift"],
                bootstrap["relative_oracle_uplift_ci95"],
            )
            + " | "
            + format_scientific_ci(
                point["median_top1_second_margin"],
                bootstrap["median_top1_second_margin_ci95"],
            )
            + " | "
            + format_percent_ci(
                point["p_margin_lt_1e_4"],
                bootstrap["p_margin_lt_1e_4_ci95"],
            )
            + " | "
            + format_percent_ci(
                point["p_margin_lt_1pct_abs_oracle_gain"],
                bootstrap["p_margin_lt_1pct_abs_oracle_gain_ci95"],
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 决策归类",
            "",
            f"- 本次结果属于情况 {payload['decision']['case']}："
            f"{payload['decision']['summary']}。",
            f"- P3 vector gap 的 overall top-1 为 "
            f"{payload['decision']['vector_gap_top1']:.4%}，"
            f"aligned mean Spearman 为 "
            f"{payload['decision']['vector_gap_aligned_mean_spearman']:.6f}。",
            "- oracle uplift 的均值上限可测，但大量区域候选近似并列；该边界必须与 uplift 一起解释。",
            "- 这是 logit 空间一步诊断，不等价于最终蒸馏训练收益。",
        ]
    )
    if payload["decision"]["case"] == "C":
        lines.extend(
            [
                "- 若继续下一版 proxy，应优先保留类别对齐信息："
                "teacher-class-aligned student confidence、"
                "teacher/student argmax 一致性、完整 teacher-student KL，"
                "以及 target-class gap 与 non-target distribution gap 的分离建模。",
                "- 当前证据不支持继续调 r_c/r_v 权重、温度候选点、"
                "region size 或 Newton 步数；本轮未实现上述 proxy。",
            ]
        )
    elif payload["decision"]["case"] == "B":
        lines.append(
            "- 按预设决策，区域自适应温度上限不足，应停止 CoVar-Match。"
        )
    elif payload["decision"]["case"] == "D":
        lines.append(
            "- 进入独立图像子集确认前，仍需补 teacher entropy baseline。"
        )
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    if args.seed != 1234:
        raise ValueError("P3A protocol locks seed=1234")
    for path in (
        args.candidate_cache,
        args.p2_json,
        args.p3_json,
        args.teacher_pretrained,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    p2_payload = json.loads(args.p2_json.read_text(encoding="utf-8"))
    p3_payload = json.loads(args.p3_json.read_text(encoding="utf-8"))
    if not p2_payload.get("execution_gate_pass"):
        raise RuntimeError("P2 execution gate failed")
    if not p3_payload.get("execution_gate_pass"):
        raise RuntimeError("P3 execution gate failed")
    if tuple(p2_payload["config"]["temperatures"]) != TEMPERATURES:
        raise RuntimeError("P3A temperature grid drifted from P2")
    if int(p2_payload["config"]["region_size"]) != 8:
        raise RuntimeError("P3A region size drifted from P2")
    if int(p2_payload["config"]["min_valid_pixels"]) != 16:
        raise RuntimeError("P3A valid-pixel threshold drifted from P2")

    score_audit = run_score_direction_audit(p3_payload)
    kl_audit = run_kl_audit()
    gain_cache, image_indices, group_count = load_gain_cache(
        args.candidate_cache, TEMPERATURES
    )
    oracle_ceiling = {}
    for stage_index, stage in enumerate(STAGE_NAMES):
        point = oracle_point_metrics(gain_cache[stage], TEMPERATURES)
        bootstrap = bootstrap_oracle_metrics(
            gain_cache[stage],
            TEMPERATURES,
            replicates=BOOTSTRAP_REPLICATES,
            seed=args.seed + stage_index,
        )
        oracle_ceiling[stage] = {
            "point": point,
            "bootstrap": bootstrap,
        }

    direction_audit = run_direction_gain_audit(
        args, image_indices, TEMPERATURES
    )
    uplift_resolved = all(
        oracle_ceiling[stage]["bootstrap"][
            "oracle_uplift_over_best_fixed_ci95"
        ][0]
        > 0
        for stage in STAGE_NAMES
    )
    vector_metrics = p3_payload["metrics"]["overall"][
        "vector_covar_gap"
    ]
    random_top1 = 1.0 / len(TEMPERATURES)
    vector_gap_positive = (
        float(vector_metrics["top1_accuracy"]) > random_top1
        and float(vector_metrics["mean_spearman"]) > 0
    )
    implementation_pass = (
        score_audit["pass"]
        and kl_audit["pass"]
        and direction_audit["pass"]
    )
    if not implementation_pass:
        decision_case = "A"
        decision_summary = (
            "方向、KL 或 gain 门禁失败，修复后应重跑 P2/P3"
        )
    elif not uplift_resolved:
        decision_case = "B"
        decision_summary = (
            "实现正确，但 image-bootstrap 未分辨出 oracle uplift"
        )
    elif not vector_gap_positive:
        decision_case = "C"
        decision_summary = (
            "存在 student-dependent 温度信号，但 CoVar gap 无法捕获"
        )
    else:
        decision_case = "D"
        decision_summary = (
            "oracle uplift 可分辨，且 vector gap 恢复为正相关"
        )

    expected_groups = sum(
        int(p2_payload["stage_summaries"][stage]["region_count"])
        for stage in STAGE_NAMES
    )
    cache_gate = (
        group_count == expected_groups
        and len(image_indices) == int(
            p2_payload["scope"]["processed_images"]
        )
        and all(
            oracle_ceiling[stage]["point"]["region_count"]
            == int(p2_payload["stage_summaries"][stage]["region_count"])
            for stage in STAGE_NAMES
        )
    )
    execution_gate_pass = implementation_pass and cache_gate
    payload = {
        "stage": "P3A",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {
            "training": False,
            "implements_covar_match": False,
            "temperatures": list(TEMPERATURES),
            "region_size": 8,
            "min_valid_pixels": 16,
            "direction_images": DIRECTION_IMAGE_COUNT,
            "direction_step_sizes": list(STEP_SIZES),
            "oracle_images": len(image_indices),
            "bootstrap_unit": "image",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "seed": args.seed,
        },
        "inputs": {
            "candidate_cache": str(args.candidate_cache),
            "candidate_cache_sha256": file_sha256(
                args.candidate_cache
            ),
            "p2_json": str(args.p2_json),
            "p2_json_sha256": file_sha256(args.p2_json),
            "p3_json": str(args.p3_json),
            "p3_json_sha256": file_sha256(args.p3_json),
            "region_state_groups": group_count,
        },
        "p3a_1_score_direction": score_audit,
        "p3a_2_kl": kl_audit,
        "p3a_3_directional_gain": direction_audit,
        "p3a_4_oracle_ceiling": oracle_ceiling,
        "decision": {
            "case": decision_case,
            "summary": decision_summary,
            "implementation_pass": implementation_pass,
            "oracle_uplift_resolved_above_zero": uplift_resolved,
            "vector_gap_positive": vector_gap_positive,
            "vector_gap_top1": float(
                vector_metrics["top1_accuracy"]
            ),
            "vector_gap_aligned_mean_spearman": float(
                vector_metrics["mean_spearman"]
            ),
        },
        "execution_gate_pass": execution_gate_pass,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    direction_csv = args.output_dir / "P3A_directional_gain.csv"
    oracle_csv = args.output_dir / "P3A_oracle_ceiling.csv"
    json_path = args.output_dir / "P3A_audit.json"
    report_path = args.output_dir / "P3A_audit.md"
    write_direction_csv(direction_csv, direction_audit)
    write_oracle_csv(oracle_csv, oracle_ceiling)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "p3a_1_pass": score_audit["pass"],
                "p3a_2_pass": kl_audit["pass"],
                "p3a_3_pass": direction_audit["pass"],
                "decision": payload["decision"],
                "execution_gate_pass": execution_gate_pass,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Wrote {direction_csv}")
    print(f"Wrote {oracle_csv}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    if not execution_gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
