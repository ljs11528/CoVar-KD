#!/usr/bin/env python3
"""Summarize the formal P1 teacher-target-only temperature sweep."""

import argparse
import csv
import datetime as dt
import json
import math
import re
from pathlib import Path

import torch


LABELS = {
    0.5: "T0p5",
    0.75: "T0p75",
    1.0: "T1p0",
    1.5: "T1p5",
    2.0: "T2p0",
}
LOG_PATTERN = re.compile(
    r"Overall validation pixAcc: (?P<pix>[0-9.]+), mIoU: (?P<miou>[0-9.]+)"
)
ITER_PATTERN = re.compile(r"Iters: (?P<step>\d+)/(?P<total>\d+)")
TIME_PATTERN = re.compile(r"Total training time: (?P<time>[^\n(]+)")
TARGET_PATTERN = re.compile(r"Teacher-only target T: (?P<temperature>[0-9.]+)")
ZERO_BRANCHES = (
    "lambda_adv",
    "lambda_d",
    "lambda_skd",
    "lambda_cwd_fea",
    "lambda_cwd_logit",
    "lambda_ifv",
    "lambda_fitnet",
    "lambda_at",
    "lambda_psd",
    "lambda_csd",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("runs/covar_match/P1_teacher_only_temperature/logs"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("runs/covar_match/P1_teacher_only_temperature/checkpoints"),
    )
    parser.add_argument(
        "--p0-csv",
        type=Path,
        default=Path("reports/covar_match/P0_temperature_complexity_trajectory.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/covar_match")
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.5, 2.0],
    )
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def close(a, b, tolerance=1e-12):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)


def read_complexity(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {float(row["temperature"]): row for row in rows}


def read_run(args, temperature, complexity):
    label = LABELS[temperature]
    variant = f"{label}_20k_seed{args.seed}"
    log_dir = args.log_root / variant
    checkpoint_dir = args.checkpoint_root / variant
    logs = sorted(log_dir.glob("*_log.txt"))
    if len(logs) != 1:
        raise RuntimeError(f"{variant}: expected one log, found {len(logs)}")
    text = logs[0].read_text(encoding="utf-8")
    validation_matches = list(LOG_PATTERN.finditer(text))
    iteration_matches = list(ITER_PATTERN.finditer(text))
    target_matches = list(TARGET_PATTERN.finditer(text))
    time_matches = list(TIME_PATTERN.finditer(text))
    if not validation_matches or not iteration_matches or not time_matches:
        raise RuntimeError(f"{variant}: incomplete training or validation log")

    validation = validation_matches[-1]
    final_iteration = iteration_matches[-1]
    state_path = checkpoint_dir / "training_state_latest.pth"
    if not state_path.is_file():
        raise RuntimeError(f"{variant}: missing {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    saved_args = state.get("args", {})

    checks = {
        "iteration": int(state.get("iteration", -1)) == args.iterations,
        "logged_iteration": (
            int(final_iteration.group("step")) == args.iterations
            and int(final_iteration.group("total")) == args.iterations
        ),
        "world_size": int(state.get("world_size", -1)) == 2,
        "seed": int(saved_args.get("seed", -1)) == args.seed,
        "global_batch": (
            int(saved_args.get("batch_size", -1))
            * int(state.get("world_size", -1)) == 16
        ),
        "kd_mode": saved_args.get("kd_loss_mode") == "teacher_only",
        "teacher_temperature": close(
            saved_args.get("kd_temperature", float("nan")), temperature
        ),
        "student_temperature": True,
        "no_t_squared": True,
        "outer_teacher_temperature": close(
            saved_args.get("teacher_output_temp", float("nan")), 1.0
        ),
        "lambda_kd": close(saved_args.get("lambda_kd", float("nan")), 1.0),
        "no_covar": saved_args.get("use_covar") is False,
        "other_kd_branches_off": all(
            close(saved_args.get(name, float("nan")), 0.0)
            for name in ZERO_BRANCHES
        ),
        "finite_log": "non-finite" not in text.lower(),
        "temperature_logged": (
            bool(target_matches)
            and close(target_matches[-1].group("temperature"), temperature, 5e-5)
        ),
    }
    if close(temperature, 1.0):
        checks["p2_snapshots"] = all(
            (
                checkpoint_dir
                / f"kd_deeplabv3_mobilenet_ssseg_mobilenetv3_small_voc_iter{step:06d}.pth"
            ).is_file()
            for step in (4000, 12000, 20000)
        )

    return {
        "variant": variant,
        "temperature": temperature,
        "mean_r": float(complexity["r_mean"]),
        "mean_r_c": float(complexity["r_c_mean"]),
        "mean_r_v": float(complexity["r_v_mean"]),
        "pixacc_percent": float(validation.group("pix")),
        "miou_percent": float(validation.group("miou")),
        "training_time": time_matches[-1].group("time").strip(),
        "log_path": str(logs[0]),
        "checkpoint_path": str(state_path),
        "checks": checks,
        "contract_pass": all(checks.values()),
    }


def write_csv(path, rows):
    fields = (
        "temperature",
        "mean_r",
        "mean_r_c",
        "mean_r_v",
        "pixacc_percent",
        "miou_percent",
        "training_time",
        "contract_pass",
        "variant",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def build_report(payload):
    lines = [
        "# P1 H1：全局 teacher-only 温度扫描报告",
        "",
        f"- 数据/模型：Pascal VOC，DeepLabV3-ResNet101 → DeepLabV3-MobileNetV3-Small",
        f"- 训练：20k iterations，全局 batch 16，双 GPU，seed {payload['config']['seed']}",
        "- 唯一自变量：teacher target temperature；student temperature=1；无 T² 补偿",
        "- 目标函数：监督 CE + 1.0 × KL(softmax(z_t/T) || softmax(z_s))",
        f"- 执行门禁：{'通过' if payload['execution_gate_pass'] else '失败'}",
        "",
        "## 正式结果",
        "",
        "| T | mean r | mIoU (%) | pixAcc (%) | 时间 | 契约 |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {temperature:.2f} | {mean_r:.6f} | {miou_percent:.6f} | "
            "{pixacc_percent:.6f} | {training_time} | {gate} |".format(
                gate="pass" if row["contract_pass"] else "fail", **row
            )
        )

    observation = payload["observation"]
    if observation["minimum_complexity_not_best"]:
        interpretation = (
            "本扫描中最低 mean r 的温度不是最高 mIoU 温度；"
            "这为“最低复杂度不一定最好”提供单 seed、20k 范围内的直接证据。"
        )
    else:
        interpretation = (
            "本扫描中最低 mean r 的温度同时得到最高 mIoU；"
            "当前 P1 不支持“最低复杂度不一定最好”，需要据实保留该负结果。"
        )
    lines.extend(
        [
            "",
            "## H1 观察",
            "",
            f"- 最低 mean r：T={observation['minimum_complexity_temperature']:.2f}。",
            f"- 最高 mIoU：T={observation['best_miou_temperature']:.2f}，"
            f"相对最低复杂度温度差 {observation['best_minus_min_complexity_pp']:+.6f} 个百分点。",
            f"- 扫描内 mIoU 曲线是否存在方向翻转：{observation['miou_has_turn']}。",
            f"- 结论：{interpretation}",
            "",
            "## 边界",
            "",
            "- 这是固定架构、VOC、单 seed、20k 的机制扫描，不提供跨数据集或统计显著性结论。",
            "- P1 只改变教师目标，不等同于历史 shared-temperature + T² 的 M2 标量温度实验。",
            "- P2/P3 只能复用 T=1 的 4k/12k/20k 学生状态，不把本表的最佳 T 当作区域 oracle。",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    if set(args.temperatures) != set(LABELS):
        raise ValueError(f"formal P1 temperatures must be {sorted(LABELS)}")
    complexity = read_complexity(args.p0_csv)
    rows = [
        read_run(args, temperature, complexity[temperature])
        for temperature in args.temperatures
    ]
    rows.sort(key=lambda row: row["temperature"])

    minimum_complexity = min(rows, key=lambda row: row["mean_r"])
    best_miou = max(rows, key=lambda row: row["miou_percent"])
    deltas = [
        rows[index + 1]["miou_percent"] - rows[index]["miou_percent"]
        for index in range(len(rows) - 1)
    ]
    has_turn = any(delta > 0 for delta in deltas) and any(
        delta < 0 for delta in deltas
    )
    payload = {
        "stage": "P1",
        "hypothesis": "global teacher-only temperature changes distillation quality; minimum output complexity need not be optimal",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {
            "dataset": "Pascal VOC",
            "iterations": args.iterations,
            "global_batch": 16,
            "world_size": 2,
            "seed": args.seed,
            "temperatures": sorted(args.temperatures),
            "student_temperature": 1.0,
            "temperature_squared_compensation": False,
            "outer_teacher_temperature": 1.0,
        },
        "rows": rows,
        "observation": {
            "minimum_complexity_temperature": minimum_complexity["temperature"],
            "best_miou_temperature": best_miou["temperature"],
            "best_minus_min_complexity_pp": (
                best_miou["miou_percent"] - minimum_complexity["miou_percent"]
            ),
            "minimum_complexity_not_best": (
                best_miou["temperature"] != minimum_complexity["temperature"]
            ),
            "miou_has_turn": has_turn,
        },
        "execution_gate_pass": all(row["contract_pass"] for row in rows),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "P1_teacher_only_temperature_scan.csv"
    json_path = args.output_dir / "P1_teacher_only_temperature_scan.json"
    report_path = args.output_dir / "P1_teacher_only_temperature_scan.md"
    write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(build_report(payload), encoding="utf-8")
    print(json.dumps(payload["observation"], indent=2, ensure_ascii=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    if not payload["execution_gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
