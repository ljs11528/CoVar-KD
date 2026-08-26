#!/usr/bin/env python3
"""P2: offline regional teachability and student-state dependence."""

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.voc import VOCDataValSet
from models.model_zoo import get_segmentation_model
from utils.covar_metrics import (
    covar_components_from_sorted_logits,
    sort_logits_for_covar,
)
from utils.region_teachability import (
    aggregate_gradient_cosine,
    normalized_teacher_step_maps,
)


STAGES = (
    ("early", 4000),
    ("middle", 12000),
    ("late", 20000),
)
CANDIDATE_FIELDS = (
    "stage",
    "checkpoint_iteration",
    "image_id",
    "image_index",
    "region_row",
    "region_col",
    "y0",
    "y1",
    "x0",
    "x1",
    "valid_pixels",
    "valid_fraction",
    "temperature",
    "one_step_gain",
    "gradient_cosine",
    "teacher_student_kl",
    "teacher_r_c",
    "teacher_r_v",
    "teacher_r",
    "student_r_c",
    "student_r_v",
    "student_r",
)
ORACLE_FIELDS = (
    "stage",
    "checkpoint_iteration",
    "image_id",
    "image_index",
    "region_row",
    "region_col",
    "valid_pixels",
    "oracle_temperature",
    "oracle_gain",
    "oracle_margin",
    "oracle_cosine_temperature",
    "oracle_cosine",
    "positive_oracle_gain",
)


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
        "--cache-path",
        type=Path,
        default=(
            ROOT
            / "runs/covar_match/P2_region_candidate_teachability.csv.gz"
        ),
    )
    parser.add_argument(
        "--oracle-cache-path",
        type=Path,
        default=(
            ROOT / "runs/covar_match/P2_region_oracle_summary.csv.gz"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "reports/covar_match"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--region-size", type=int, default=8)
    parser.add_argument("--min-valid-pixels", type=int, default=16)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument(
        "--temperatures", default="0.5,0.75,1.0,1.25,1.5,2.0"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ignore-label", type=int, default=-1)
    return parser.parse_args()


def parse_temperatures(raw):
    values = [float(item) for item in raw.split(",") if item.strip()]
    if values != sorted(set(values)) or any(value <= 0 for value in values):
        raise ValueError("temperatures must be unique, positive, and increasing")
    return values


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_state_dict(state):
    if isinstance(state, dict) and "student" in state:
        state = state["student"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def build_teacher(args, device):
    model = get_segmentation_model(
        model="deeplabv3",
        backbone="resnet101",
        local_rank=0,
        pretrained_base="None",
        pretrained=str(args.teacher_pretrained),
        aux=True,
        norm_layer=nn.BatchNorm2d,
        num_class=21,
    ).to(device)
    model.eval()
    return model


def student_checkpoint_path(root, iteration):
    return root / (
        "kd_deeplabv3_mobilenet_ssseg_mobilenetv3_small_voc_"
        f"iter{iteration:06d}.pth"
    )


def build_students(args, device):
    students = {}
    contracts = {}
    for stage, iteration in STAGES:
        checkpoint_path = student_checkpoint_path(
            args.checkpoint_root, iteration
        )
        state_path = (
            args.checkpoint_root
            / f"training_state_iter{iteration:06d}.pth"
        )
        if not checkpoint_path.is_file() or not state_path.is_file():
            raise FileNotFoundError(
                f"missing P1 checkpoint pair for {stage}: "
                f"{checkpoint_path}, {state_path}"
            )
        training_state = torch.load(
            state_path, map_location="cpu", weights_only=False
        )
        saved_args = training_state.get("args", {})
        contract = {
            "iteration": int(training_state.get("iteration", -1))
            == iteration,
            "seed": int(saved_args.get("seed", -1)) == args.seed,
            "kd_mode": saved_args.get("kd_loss_mode") == "teacher_only",
            "temperature": float(saved_args.get("kd_temperature", -1.0))
            == 1.0,
            "student_temperature": True,
            "no_temperature_squared_compensation": True,
            "outer_teacher_temperature": float(
                saved_args.get("teacher_output_temp", -1.0)
            )
            == 1.0,
        }
        if not all(contract.values()):
            raise RuntimeError(
                f"P1 checkpoint contract failed for {stage}: {contract}"
            )

        model = get_segmentation_model(
            model="deeplabv3_mobilenet_ssseg",
            backbone="mobilenetv3_small",
            local_rank=0,
            pretrained_base="None",
            pretrained="None",
            aux=False,
            norm_layer=nn.BatchNorm2d,
            num_class=21,
        ).to(device)
        state = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        model.load_state_dict(clean_state_dict(state), strict=True)
        model.eval()
        students[stage] = model
        contracts[stage] = {
            "iteration": iteration,
            "student_checkpoint": str(checkpoint_path),
            "student_checkpoint_sha256": file_sha256(checkpoint_path),
            "training_state": str(state_path),
            "training_state_sha256": file_sha256(state_path),
            "contract": contract,
        }
    return students, contracts


def region_mean(value, valid):
    return float(value[valid].double().mean().item())


def candidate_region_row(
    base,
    temperature,
    diagnostics,
    teacher_components,
    student_components,
    valid,
    slices,
):
    y_slice, x_slice = slices
    region_valid = valid[y_slice, x_slice]
    diagnostic_slice = {
        key: value[0, y_slice, x_slice]
        for key, value in diagnostics.items()
        if key != "valid"
    }
    cosine = aggregate_gradient_cosine(
        diagnostic_slice["gradient_dot"],
        diagnostic_slice["supervised_gradient_sq"],
        diagnostic_slice["kd_gradient_sq"],
    )
    return {
        **base,
        "temperature": temperature,
        "one_step_gain": region_mean(
            diagnostic_slice["ce_gain"], region_valid
        ),
        "gradient_cosine": float(cosine.double().item()),
        "teacher_student_kl": region_mean(
            diagnostic_slice["teacher_student_kl"], region_valid
        ),
        "teacher_r_c": region_mean(
            teacher_components["r_c"][0, y_slice, x_slice], region_valid
        ),
        "teacher_r_v": region_mean(
            teacher_components["r_v"][0, y_slice, x_slice], region_valid
        ),
        "teacher_r": region_mean(
            teacher_components["r"][0, y_slice, x_slice], region_valid
        ),
        "student_r_c": region_mean(
            student_components["r_c"][0, y_slice, x_slice], region_valid
        ),
        "student_r_v": region_mean(
            student_components["r_v"][0, y_slice, x_slice], region_valid
        ),
        "student_r": region_mean(
            student_components["r"][0, y_slice, x_slice], region_valid
        ),
    }


def stage_summary(rows, temperatures):
    gains = np.asarray([row["oracle_gain"] for row in rows], dtype=np.float64)
    margins = np.asarray(
        [row["oracle_margin"] for row in rows], dtype=np.float64
    )
    distribution = Counter(row["oracle_temperature"] for row in rows)
    return {
        "region_count": len(rows),
        "mean_oracle_gain": float(gains.mean()),
        "median_oracle_gain": float(np.median(gains)),
        "mean_oracle_margin": float(margins.mean()),
        "positive_oracle_gain_fraction": float((gains > 0).mean()),
        "gain_cosine_oracle_agreement": float(
            np.mean(
                [
                    row["oracle_temperature"]
                    == row["oracle_cosine_temperature"]
                    for row in rows
                ]
            )
        ),
        "oracle_temperature_distribution": {
            str(temperature): {
                "count": int(distribution[temperature]),
                "fraction": distribution[temperature] / max(len(rows), 1),
            }
            for temperature in temperatures
        },
    }


def state_transition(rows_by_stage, left, right, temperatures):
    left_map = {
        (row["image_id"], row["region_row"], row["region_col"]): row
        for row in rows_by_stage[left]
    }
    right_map = {
        (row["image_id"], row["region_row"], row["region_col"]): row
        for row in rows_by_stage[right]
    }
    keys = sorted(set(left_map) & set(right_map))
    index = {temperature: position for position, temperature in enumerate(temperatures)}
    left_temperature = np.asarray(
        [left_map[key]["oracle_temperature"] for key in keys]
    )
    right_temperature = np.asarray(
        [right_map[key]["oracle_temperature"] for key in keys]
    )
    distance = np.asarray(
        [
            abs(index[left_map[key]["oracle_temperature"]]
                - index[right_map[key]["oracle_temperature"]])
            for key in keys
        ],
        dtype=np.float64,
    )
    return {
        "left": left,
        "right": right,
        "common_regions": len(keys),
        "exact_agreement": float(np.mean(left_temperature == right_temperature)),
        "adjacent_agreement": float(np.mean(distance <= 1)),
        "mean_candidate_index_shift": float(distance.mean()),
        "mean_absolute_temperature_shift": float(
            np.mean(np.abs(left_temperature - right_temperature))
        ),
    }


def write_distribution_csv(path, summaries, temperatures):
    fields = (
        "stage",
        "temperature",
        "oracle_count",
        "oracle_fraction",
        "region_count",
        "mean_oracle_gain",
        "median_oracle_gain",
        "mean_oracle_margin",
        "positive_oracle_gain_fraction",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for stage, _ in STAGES:
            summary = summaries[stage]
            for temperature in temperatures:
                distribution = summary[
                    "oracle_temperature_distribution"
                ][str(temperature)]
                writer.writerow(
                    {
                        "stage": stage,
                        "temperature": temperature,
                        "oracle_count": distribution["count"],
                        "oracle_fraction": distribution["fraction"],
                        "region_count": summary["region_count"],
                        "mean_oracle_gain": summary["mean_oracle_gain"],
                        "median_oracle_gain": summary["median_oracle_gain"],
                        "mean_oracle_margin": summary["mean_oracle_margin"],
                        "positive_oracle_gain_fraction": summary[
                            "positive_oracle_gain_fraction"
                        ],
                    }
                )


def render_report(payload):
    lines = [
        "# P2 H2：区域 teachability 与学生状态依赖报告",
        "",
        "- 本阶段不训练；复用 P1 的 T=1 学生 4k/12k/20k 检查点。",
        f"- 数据：固定 seed 抽样的 VOC val {payload['scope']['processed_images']} 张图像。",
        f"- 区域：原生 logits 网格上的 {payload['config']['region_size']}×{payload['config']['region_size']} 块；"
        f"至少 {payload['config']['min_valid_pixels']} 个有效位置。",
        "- oracle：教师 KD 梯度逐像素归一化后，以 eta=0.1 做一次 student-logit 更新，选择监督 CE 降幅最大的 T。",
        f"- 执行门禁：{'通过' if payload['execution_gate_pass'] else '失败'}",
        "",
        "## 各学生状态的区域 oracle",
        "",
        "| 状态 | iteration | 区域数 | mean gain | median gain | mean margin | gain>0 | gain/cos oracle 一致 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, iteration in STAGES:
        summary = payload["stage_summaries"][stage]
        lines.append(
            f"| {stage} | {iteration} | {summary['region_count']} | "
            f"{summary['mean_oracle_gain']:.6e} | "
            f"{summary['median_oracle_gain']:.6e} | "
            f"{summary['mean_oracle_margin']:.6e} | "
            f"{summary['positive_oracle_gain_fraction']:.4%} | "
            f"{summary['gain_cosine_oracle_agreement']:.4%} |"
        )
    lines.extend(
        [
            "",
            "### oracle 温度分布",
            "",
            "| 状态 | " + " | ".join(
                f"T={temperature:g}" for temperature in payload["config"]["temperatures"]
            ) + " |",
            "|---|" + "|".join("---:" for _ in payload["config"]["temperatures"]) + "|",
        ]
    )
    for stage, _ in STAGES:
        distribution = payload["stage_summaries"][stage][
            "oracle_temperature_distribution"
        ]
        lines.append(
            f"| {stage} | "
            + " | ".join(
                f"{distribution[str(temperature)]['fraction']:.2%}"
                for temperature in payload["config"]["temperatures"]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 学生状态依赖",
            "",
            "| 状态对 | common regions | exact | adjacent | mean index shift | mean abs T shift |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for transition in payload["state_transitions"]:
        lines.append(
            f"| {transition['left']}→{transition['right']} | "
            f"{transition['common_regions']} | "
            f"{transition['exact_agreement']:.4%} | "
            f"{transition['adjacent_agreement']:.4%} | "
            f"{transition['mean_candidate_index_shift']:.6f} | "
            f"{transition['mean_absolute_temperature_shift']:.6f} |"
        )
    early_late = payload["state_transitions"][-1]
    changed_fraction = 1.0 - early_late["exact_agreement"]
    if changed_fraction > 0:
        state_text = (
            "观察到区域 oracle 随学生状态变化，支持在该短程轨迹内存在状态依赖。"
        )
    else:
        state_text = (
            "未观察到区域 oracle 随学生状态变化，当前结果不支持状态依赖。"
        )
    lines.extend(
        [
            "",
            "## 结论与边界",
            "",
            f"- early→late 的区域最优温度改变比例为 "
            f"{changed_fraction:.4%}；{state_text}",
            "- oracle 使用真实标签，只用于诊断和 P3 的预测目标，不能在测试时直接使用。",
            "- 这是 logit 空间的一步局部代理，不等同于重新训练后的最终 mIoU 因果效应。",
            "- 三个状态来自同一 T=1、20k 训练轨迹；结论不外推到 80k 或其它数据集。",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    temperatures = parse_temperatures(args.temperatures)
    if temperatures != [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        raise ValueError("P2 formal temperature grid changed")
    if args.max_images != 200:
        raise ValueError("P2 formal protocol requires 200 images")
    if args.region_size != 8 or args.step_size != 0.1:
        raise ValueError("P2 formal region size and step size changed")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = VOCDataValSet(
        str(args.data),
        str(args.list_path),
        crop_size=(512, 512),
        ignore_label=args.ignore_label,
    )
    selected_indices = sorted(
        random.Random(args.seed).sample(
            range(len(dataset)), min(args.max_images, len(dataset))
        )
    )
    selected_ids = [dataset.files[index]["name"] for index in selected_indices]
    selected_sha256 = hashlib.sha256(
        "\n".join(selected_ids).encode("utf-8")
    ).hexdigest()

    teacher = build_teacher(args, device)
    students, checkpoint_contracts = build_students(args, device)
    args.cache_path.parent.mkdir(parents=True, exist_ok=True)
    args.oracle_cache_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_count = 0
    finite = True
    rows_by_stage = {stage: [] for stage, _ in STAGES}

    with gzip.open(
        args.cache_path, "wt", newline="", encoding="utf-8"
    ) as candidate_handle, gzip.open(
        args.oracle_cache_path, "wt", newline="", encoding="utf-8"
    ) as oracle_handle:
        candidate_writer = csv.DictWriter(
            candidate_handle, fieldnames=CANDIDATE_FIELDS, lineterminator="\n"
        )
        oracle_writer = csv.DictWriter(
            oracle_handle, fieldnames=ORACLE_FIELDS, lineterminator="\n"
        )
        candidate_writer.writeheader()
        oracle_writer.writeheader()

        with torch.inference_mode():
            for selection_position, image_index in enumerate(selected_indices):
                image_array, target_array, _ = dataset[image_index]
                image_id = dataset.files[image_index]["name"]
                image = torch.from_numpy(image_array).unsqueeze(0).to(device)
                target = torch.from_numpy(target_array).long().unsqueeze(0).to(device)
                teacher_output = teacher(image)
                teacher_logits = teacher_output[0]
                sorted_teacher = sort_logits_for_covar(
                    teacher_logits, class_dim=1
                )
                teacher_components = {
                    temperature: covar_components_from_sorted_logits(
                        sorted_teacher, temperature
                    )
                    for temperature in temperatures
                }

                for stage, iteration in STAGES:
                    student_output = students[stage](image)
                    student_logits = student_output[0]
                    if student_logits.shape != teacher_logits.shape:
                        raise RuntimeError(
                            f"logit shape mismatch for {image_id}: "
                            f"{tuple(student_logits.shape)} vs "
                            f"{tuple(teacher_logits.shape)}"
                        )
                    aligned_target = F.interpolate(
                        target.float().unsqueeze(1),
                        size=student_logits.shape[-2:],
                        mode="nearest",
                    ).squeeze(1).long()
                    valid = aligned_target[0] != args.ignore_label
                    sorted_student = sort_logits_for_covar(
                        student_logits, class_dim=1
                    )
                    student_components = (
                        covar_components_from_sorted_logits(
                            sorted_student, 1.0
                        )
                    )
                    diagnostics = {
                        temperature: normalized_teacher_step_maps(
                            student_logits,
                            teacher_logits,
                            aligned_target,
                            temperature,
                            step_size=args.step_size,
                            ignore_label=args.ignore_label,
                        )
                        for temperature in temperatures
                    }

                    height, width = valid.shape
                    region_row = 0
                    for y0 in range(0, height, args.region_size):
                        y1 = min(y0 + args.region_size, height)
                        region_col = 0
                        for x0 in range(0, width, args.region_size):
                            x1 = min(x0 + args.region_size, width)
                            region_valid = valid[y0:y1, x0:x1]
                            valid_pixels = int(region_valid.sum().item())
                            if valid_pixels < args.min_valid_pixels:
                                region_col += 1
                                continue
                            base = {
                                "stage": stage,
                                "checkpoint_iteration": iteration,
                                "image_id": image_id,
                                "image_index": image_index,
                                "region_row": region_row,
                                "region_col": region_col,
                                "y0": y0,
                                "y1": y1,
                                "x0": x0,
                                "x1": x1,
                                "valid_pixels": valid_pixels,
                                "valid_fraction": (
                                    valid_pixels
                                    / float((y1 - y0) * (x1 - x0))
                                ),
                            }
                            candidate_rows = []
                            for temperature in temperatures:
                                row = candidate_region_row(
                                    base,
                                    temperature,
                                    diagnostics[temperature],
                                    teacher_components[temperature],
                                    student_components,
                                    valid,
                                    (
                                        slice(y0, y1),
                                        slice(x0, x1),
                                    ),
                                )
                                numeric = [
                                    row[field]
                                    for field in CANDIDATE_FIELDS
                                    if field
                                    not in (
                                        "stage",
                                        "image_id",
                                    )
                                ]
                                finite = finite and all(
                                    np.isfinite(float(value))
                                    for value in numeric
                                )
                                candidate_writer.writerow(row)
                                candidate_rows.append(row)
                                candidate_count += 1

                            gain_order = sorted(
                                candidate_rows,
                                key=lambda row: row["one_step_gain"],
                                reverse=True,
                            )
                            cosine_order = sorted(
                                candidate_rows,
                                key=lambda row: row["gradient_cosine"],
                                reverse=True,
                            )
                            oracle_row = {
                                "stage": stage,
                                "checkpoint_iteration": iteration,
                                "image_id": image_id,
                                "image_index": image_index,
                                "region_row": region_row,
                                "region_col": region_col,
                                "valid_pixels": valid_pixels,
                                "oracle_temperature": gain_order[0][
                                    "temperature"
                                ],
                                "oracle_gain": gain_order[0][
                                    "one_step_gain"
                                ],
                                "oracle_margin": (
                                    gain_order[0]["one_step_gain"]
                                    - gain_order[1]["one_step_gain"]
                                ),
                                "oracle_cosine_temperature": cosine_order[0][
                                    "temperature"
                                ],
                                "oracle_cosine": cosine_order[0][
                                    "gradient_cosine"
                                ],
                                "positive_oracle_gain": int(
                                    gain_order[0]["one_step_gain"] > 0
                                ),
                            }
                            oracle_writer.writerow(oracle_row)
                            rows_by_stage[stage].append(oracle_row)
                            region_col += 1
                        region_row += 1
                if (selection_position + 1) % 20 == 0:
                    print(
                        f"[P2] processed "
                        f"{selection_position + 1}/{len(selected_indices)} images"
                    )

    stage_summaries = {
        stage: stage_summary(rows_by_stage[stage], temperatures)
        for stage, _ in STAGES
    }
    transitions = [
        state_transition(rows_by_stage, "early", "middle", temperatures),
        state_transition(rows_by_stage, "middle", "late", temperatures),
        state_transition(rows_by_stage, "early", "late", temperatures),
    ]
    region_counts = [
        stage_summaries[stage]["region_count"] for stage, _ in STAGES
    ]
    gate_pass = (
        len(selected_indices) == args.max_images
        and min(region_counts) > 0
        and len(set(region_counts)) == 1
        and candidate_count == sum(region_counts) * len(temperatures)
        and finite
        and all(
            all(contract["contract"].values())
            for contract in checkpoint_contracts.values()
        )
        and all(
            transition["common_regions"] == region_counts[0]
            for transition in transitions
        )
    )
    payload = {
        "stage": "P2",
        "hypothesis": "regional teachability depends on teacher target temperature and the current student state",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {
            "temperatures": temperatures,
            "region_size": args.region_size,
            "min_valid_pixels": args.min_valid_pixels,
            "step_size": args.step_size,
            "normalization": "per-pixel KD gradient L2 normalization",
            "oracle": "maximum regional mean supervised CE decrease after one normalized logit step",
            "seed": args.seed,
        },
        "scope": {
            "dataset": "Pascal VOC val",
            "dataset_size": len(dataset),
            "processed_images": len(selected_indices),
            "selected_image_ids_sha256": selected_sha256,
            "candidate_rows": candidate_count,
            "candidate_cache": str(args.cache_path),
            "oracle_cache": str(args.oracle_cache_path),
        },
        "checkpoints": checkpoint_contracts,
        "stage_summaries": stage_summaries,
        "state_transitions": transitions,
        "execution_gate_pass": gate_pass,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    distribution_path = (
        args.output_dir / "P2_region_oracle_distribution.csv"
    )
    json_path = args.output_dir / "P2_region_teachability.json"
    report_path = args.output_dir / "P2_region_teachability.md"
    write_distribution_csv(
        distribution_path, stage_summaries, temperatures
    )
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(payload), encoding="utf-8")
    cache_metadata = {
        "candidate_fields": CANDIDATE_FIELDS,
        "oracle_fields": ORACLE_FIELDS,
        "payload": payload,
    }
    args.cache_path.with_suffix(".metadata.json").write_text(
        json.dumps(cache_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(
        {
            "stage_summaries": stage_summaries,
            "state_transitions": transitions,
            "execution_gate_pass": gate_pass,
        },
        indent=2,
        ensure_ascii=False,
    ))
    print(f"Wrote {distribution_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {args.cache_path}")
    print(f"Wrote {args.oracle_cache_path}")
    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
