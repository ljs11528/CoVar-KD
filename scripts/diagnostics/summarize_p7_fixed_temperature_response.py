#!/usr/bin/env python3
"""Add the T=0.25 lower-boundary gate to the locked P7 response grid."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path
from statistics import mean, stdev


SEEDS = (1234, 2025, 3407)
TEMPERATURES = ("0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "2.0")
COVAR_TEMPERATURES = TEMPERATURES[1:]
MILESTONES = (20000, 40000, 60000, 80000)
LOG_NAME = "deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"
ITERATION_RE = re.compile(r"Iters:\s*(?P<iteration>\d+)/(?P<maximum>\d+)")
VALIDATION_RE = re.compile(
    r"Overall validation pixAcc: (?P<pixacc>[-+0-9.eE]+), "
    r"mIoU: (?P<miou>[-+0-9.eE]+)"
)
TIME_RE = re.compile(r"Total training time: (?P<time>[^\n(]+)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p6-root", type=Path, default=Path("runs/covar_match/P6_full_schedule")
    )
    parser.add_argument(
        "--p7-root",
        type=Path,
        default=Path("runs/covar_match/P7_fixed_temperature_response"),
    )
    parser.add_argument(
        "--covar-csv",
        type=Path,
        default=Path("reports/covar_match/P0_temperature_complexity_trajectory.csv"),
    )
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "reports/covar_match/P7_fixed_temperature_response_lower_boundary.json"
        ),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path(
            "reports/covar_match/P7_fixed_temperature_response_lower_boundary.md"
        ),
    )
    return parser.parse_args()


def variant_for(temperature, seed):
    variants = {
        "0.25": f"fixed_T0p25_80k_seed{seed}",
        "0.5": (
            "fixed_T0p5_80k_seed1234_retry1"
            if seed == 1234
            else f"fixed_T0p5_80k_seed{seed}"
        ),
        "0.75": f"fixed_T0p75_80k_seed{seed}",
        "1.0": f"fixed_T1p0_80k_seed{seed}",
        "1.25": f"fixed_T1p25_80k_seed{seed}",
        "1.5": f"fixed_T1p5_80k_seed{seed}",
        "2.0": f"fixed_T2p0_80k_seed{seed}",
    }
    try:
        return variants[temperature]
    except KeyError as error:
        raise ValueError(f"unknown temperature: {temperature}") from error


def root_for(temperature, p6_root, p7_root):
    return p6_root if temperature in {"0.5", "1.0", "1.5"} else p7_root


def parse_log(text):
    current_iteration = None
    maximum = None
    trajectory = {}
    for line in text.splitlines():
        iteration_match = ITERATION_RE.search(line)
        if iteration_match:
            current_iteration = int(iteration_match.group("iteration"))
            maximum = int(iteration_match.group("maximum"))
        validation_match = VALIDATION_RE.search(line)
        if validation_match:
            if current_iteration is None:
                raise RuntimeError("validation appeared before any logged iteration")
            if current_iteration in trajectory:
                raise RuntimeError(
                    f"duplicate validation at iteration {current_iteration}"
                )
            trajectory[current_iteration] = {
                "pixacc_percent": float(validation_match.group("pixacc")),
                "miou_percent": float(validation_match.group("miou")),
            }
    return maximum, trajectory


def read_run(root, temperature, seed):
    variant = variant_for(temperature, seed)
    log_path = root / "logs" / variant / LOG_NAME
    state_dir = root / "checkpoints" / variant
    latest_state = state_dir / "training_state_latest.pth"
    if not log_path.is_file() or not latest_state.is_file():
        raise RuntimeError(f"{variant}: missing log or latest training state")
    text = log_path.read_text(encoding="utf-8")
    maximum, trajectory = parse_log(text)
    times = TIME_RE.findall(text)
    expected_temperature = f"Teacher-only target T: {float(temperature):.4f}"
    checks = {
        "maximum_80k": maximum == 80000,
        "iteration_80k": "Iters: 80000/80000" in text,
        "milestones": tuple(sorted(trajectory)) == MILESTONES,
        "training_time": bool(times),
        "finite": "non-finite" not in text.lower(),
        "teacher_only_temperature": expected_temperature in text,
    }
    for milestone in MILESTONES:
        checks[f"state_{milestone}"] = (
            state_dir / f"training_state_iter{milestone:06d}.pth"
        ).is_file()
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"{variant}: failed checks {failed}")
    values = [trajectory[milestone]["miou_percent"] for milestone in MILESTONES]
    return {
        "variant": variant,
        "trajectory": {str(key): value for key, value in trajectory.items()},
        "best_observed_miou_percent": max(values),
        "final_miou_percent": values[-1],
        "final_pixacc_percent": trajectory[80000]["pixacc_percent"],
        "training_time": times[-1].strip(),
        "contract_pass": True,
        "log_path": str(log_path),
        "checkpoint_path": str(latest_state),
    }


def sample_statistics(values):
    values = [float(value) for value in values]
    if len(values) < 2:
        raise ValueError("sample standard deviation requires at least two values")
    return {
        "values": values,
        "mean": mean(values),
        "sample_std": stdev(values),
    }


def load_covar_path(path):
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = str(float(row["temperature"]))
            rows[key] = {
                "valid_pixels": int(row["valid_pixels"]),
                "r_c_mean": float(row["r_c_mean"]),
                "r_v_mean": float(row["r_v_mean"]),
                "r_mean": float(row["r_mean"]),
            }
    missing = [
        temperature for temperature in COVAR_TEMPERATURES if temperature not in rows
    ]
    if missing:
        raise RuntimeError(f"CoVar path missing temperatures: {missing}")
    return {temperature: rows.get(temperature) for temperature in TEMPERATURES}


def rank_temperatures(runs, milestone):
    return sorted(
        TEMPERATURES,
        key=lambda temperature: (
            -runs[temperature]["trajectory"][str(milestone)]["miou_percent"],
            TEMPERATURES.index(temperature),
        ),
    )


def count_rank_transitions(rankings):
    return sum(left != right for left, right in zip(rankings, rankings[1:]))


def classify(near_optimal, per_seed_best):
    if len(set(per_seed_best.values())) > 1:
        return "case_C_no_unique_reproducible_optimum"
    indices = [TEMPERATURES.index(temperature) for temperature in near_optimal]
    if len(indices) > 1 and max(indices) - min(indices) + 1 == len(indices):
        return "case_B_broad_near_optimal_grid_set"
    return "case_A_candidate_unique_grid_optimum_requires_confirmation"


def boundary_status(near_optimal):
    touches_boundary = TEMPERATURES[0] in near_optimal or TEMPERATURES[-1] in near_optimal
    return (
        "search_boundary_not_closed"
        if touches_boundary
        else "search_boundary_closed_on_current_grid"
    )


def lower_boundary_case(near_optimal, best_mean_temperature):
    if "0.25" not in near_optimal:
        return "case_1_lower_boundary_closed"
    if best_mean_temperature != "0.25":
        return "case_2_low_temperature_near_optimal_not_best"
    return "case_3_t0p25_sample_mean_best"


def summarize(p6_root, p7_root, covar_csv, delta):
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    covar_path = load_covar_path(covar_csv)
    seed_rows = []
    for seed in SEEDS:
        runs = {
            temperature: read_run(
                root_for(temperature, p6_root, p7_root), temperature, seed
            )
            for temperature in TEMPERATURES
        }
        rankings = [
            rank_temperatures(runs, milestone) for milestone in MILESTONES
        ]
        final_values = {
            temperature: runs[temperature]["final_miou_percent"]
            for temperature in TEMPERATURES
        }
        best_temperature = max(
            TEMPERATURES,
            key=lambda temperature: (
                final_values[temperature],
                -TEMPERATURES.index(temperature),
            ),
        )
        seed_rows.append(
            {
                "seed": seed,
                "runs": runs,
                "final_best_temperature": best_temperature,
                "final_best_miou_percent": final_values[best_temperature],
                "rankings": {
                    str(milestone): ranking
                    for milestone, ranking in zip(MILESTONES, rankings)
                },
                "rank_transition_count": count_rank_transitions(rankings),
            }
        )

    temperature_summaries = {}
    for temperature in TEMPERATURES:
        milestone_stats = {}
        for milestone in MILESTONES:
            milestone_stats[str(milestone)] = sample_statistics(
                [
                    row["runs"][temperature]["trajectory"][str(milestone)][
                        "miou_percent"
                    ]
                    for row in seed_rows
                ]
            )
        temperature_summaries[temperature] = {
            "milestones": milestone_stats,
            "final": milestone_stats["80000"],
            "best_observed": sample_statistics(
                [
                    row["runs"][temperature]["best_observed_miou_percent"]
                    for row in seed_rows
                ]
            ),
            "covar": covar_path[temperature],
        }

    best_mean_temperature = max(
        TEMPERATURES,
        key=lambda temperature: (
            temperature_summaries[temperature]["final"]["mean"],
            -TEMPERATURES.index(temperature),
        ),
    )
    best_mean = temperature_summaries[best_mean_temperature]["final"]["mean"]
    near_optimal = [
        temperature
        for temperature in TEMPERATURES
        if temperature_summaries[temperature]["final"]["mean"] >= best_mean - delta
    ]
    per_seed_best = {
        str(row["seed"]): row["final_best_temperature"] for row in seed_rows
    }
    candidate_pairs = {}
    for temperature in TEMPERATURES:
        if temperature == best_mean_temperature:
            continue
        candidate_pairs[temperature] = sample_statistics(
            [
                row["runs"][best_mean_temperature]["final_miou_percent"]
                - row["runs"][temperature]["final_miou_percent"]
                for row in seed_rows
            ]
        )
    lower_case = lower_boundary_case(near_optimal, best_mean_temperature)
    lower_boundary_closed = lower_case == "case_1_lower_boundary_closed"

    return {
        "stage": "P7_fixed_temperature_response_lower_boundary",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {
            "p6_locked": True,
            "fresh_from_student_initialization": True,
            "max_iterations": 80000,
            "validation_milestones": list(MILESTONES),
            "seeds": list(SEEDS),
            "temperatures": list(TEMPERATURES),
            "teacher_only_temperature": True,
            "student_temperature": 1.0,
            "t_squared_compensation": False,
            "delta_miou_pp": delta,
            "p_value_reported": False,
        },
        "seed_rows": seed_rows,
        "temperature_summaries": temperature_summaries,
        "selection_summary": {
            "best_mean_temperature": best_mean_temperature,
            "best_mean_final_miou_percent": best_mean,
            "delta_optimal_grid_set": near_optimal,
            "per_seed_best_temperature": per_seed_best,
            "candidate_paired_differences_pp": candidate_pairs,
            "boundary_status": boundary_status(near_optimal),
            "lower_boundary_case": lower_case,
            "lower_boundary_closed": lower_boundary_closed,
            "hard_target_next_gate": not lower_boundary_closed,
            "provisional_case": classify(near_optimal, per_seed_best),
        },
        "evidence_boundary": {
            "voc_val_used_for_response_characterization": True,
            "unbiased_final_test_claim": False,
            "continuous_curve_fit_performed": False,
            "bootstrap_performed": False,
            "covar_means_reused_from_p0_full_voc_val": True,
            "covar_t0p25_computed": False,
            "covar_quantiles_included": False,
        },
    }


def build_report(payload):
    protocol = payload["protocol"]
    selection = payload["selection_summary"]
    summaries = payload["temperature_summaries"]
    rows = {row["seed"]: row for row in payload["seed_rows"]}
    near_text = ", ".join(f"T={value}" for value in selection["delta_optimal_grid_set"])
    bests = ", ".join(
        f"{seed}: T={temperature}"
        for seed, temperature in selection["per_seed_best_temperature"].items()
    )
    lower_case = selection["lower_boundary_case"]
    if lower_case == "case_1_lower_boundary_closed":
        boundary_text = (
            "情况 1：T=0.25 不在 δ-近优集合内，下边界闭合；"
            "不继续在 T=0.25 与 T=0.5 之间细分。"
        )
        claim_text = (
            "Within the evaluated range T in [0.25, 2.0], no uniquely "
            "reproducible global optimum is identified. The sample-mean maximizer "
            f"is T={selection['best_mean_temperature']}, while the near-optimal "
            "set and per-seed winners remain non-unique."
        )
    elif lower_case == "case_2_low_temperature_near_optimal_not_best":
        boundary_text = (
            "情况 2：T=0.25 进入 δ-近优集合但不是最高均值，低温区尚未闭合；"
            "下一门禁为 Hard Teacher Target，本阶段未运行。"
        )
        claim_text = (
            "T=0.25 enters the near-optimal set without becoming the sample-mean "
            "maximizer; the finite low-temperature boundary remains open."
        )
    else:
        boundary_text = (
            "情况 3：T=0.25 成为新的平均最佳点，仍不将其称为唯一最优温度；"
            "下一门禁为 Hard Teacher Target，本阶段未运行。"
        )
        claim_text = (
            "T=0.25 becomes the sample-mean maximizer, but it is not yet evidence "
            "for a uniquely identifiable finite optimum."
        )

    lines = [
        "# P7：固定全局温度响应曲线，下边界闭合",
        "",
        "## 结论先行",
        "",
        f"- 三 seed 的 80k 平均 mIoU 在离散网格上的最高点为 "
        f"T={selection['best_mean_temperature']}，"
        f"mean={selection['best_mean_final_miou_percent']:.6f}。",
        f"- 预先固定 δ={protocol['delta_miou_pp']:.3f} pp 时，"
        f"离散近优集合为：{near_text}。",
        f"- 各 seed 的 80k 最佳温度为：{bests}。",
        f"- 初步 A/B/C 判定：{selection['provisional_case']}；"
        f"边界门禁：{selection['boundary_status']}。",
        f"- 下边界决策：{selection['lower_boundary_case']}。",
        f"- {boundary_text}",
        f"- 推荐结论：{claim_text}",
        "",
        "## 锁定协议",
        "",
        "- 本阶段只新增 T=0.25，运行 seed=1234、2025、3407，"
        "共 3 条全新 80k run。",
        "- T=0.5、1.0、1.5 复用 P6，T=0.75、1.25、2.0 复用 P7 阶段 1；"
        "P4a 不进入固定温度响应曲线。",
        "- teacher-only temperature，student T=1，无 T² 补偿，CE+KD，"
        "其余 teacher、student 初始化、optimizer、poly schedule、batch size、"
        "augmentation 和验证里程碑均与 P6 一致。",
        "- 每条 run 在 20k、40k、60k、80k 验证并保留训练状态；"
        "primary endpoint 为 80k final mIoU。",
        "",
        "## 原始 mIoU 轨迹",
        "",
        "| seed | T | 20k | 40k | 60k | 80k | best observed |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        for temperature in TEMPERATURES:
            run = rows[seed]["runs"][temperature]
            values = [
                run["trajectory"][str(milestone)]["miou_percent"]
                for milestone in MILESTONES
            ]
            lines.append(
                f"| {seed} | {temperature} | "
                + " | ".join(f"{value:.6f}" for value in values)
                + f" | {run['best_observed_miou_percent']:.6f} |"
            )

    lines += [
        "",
        "## 80k 期望响应与 CoVar 坐标",
        "",
        "| T | final mean | sample SD | r_c mean | r_v mean | r mean |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for temperature in TEMPERATURES:
        summary = summaries[temperature]
        covar = summary["covar"]
        covar_cells = (
            "— | — | —"
            if covar is None
            else (
                f"{covar['r_c_mean']:.6f} | {covar['r_v_mean']:.6f} | "
                f"{covar['r_mean']:.6f}"
            )
        )
        lines.append(
            f"| {temperature} | {summary['final']['mean']:.6f} | "
            f"{summary['final']['sample_std']:.6f} | {covar_cells} |"
        )

    lines += [
        "",
        f"以下 paired difference 定义为 T={selection['best_mean_temperature']} "
        "减去对应温度的 80k mIoU。",
        "",
        "| comparison T | paired mean (pp) | sample SD (pp) | seed values (pp) |",
        "|---:|---:|---:|---|",
    ]
    for temperature, stats in selection["candidate_paired_differences_pp"].items():
        values = ", ".join(f"{value:+.6f}" for value in stats["values"])
        lines.append(
            f"| {temperature} | {stats['mean']:+.6f} | "
            f"{stats['sample_std']:.6f} | {values} |"
        )

    lines += [
        "",
        "## 训练轨迹中的温度排名",
        "",
        "| seed | 20k | 40k | 60k | 80k | transitions |",
        "|---:|---|---|---|---|---:|",
    ]
    for seed in SEEDS:
        row = rows[seed]
        ranking_cells = [
            " > ".join(f"T={temperature}" for temperature in row["rankings"][str(step)])
            for step in MILESTONES
        ]
        lines.append(
            f"| {seed} | " + " | ".join(ranking_cells)
            + f" | {row['rank_transition_count']} |"
        )

    lines += [
        "",
        "## 证据边界与下一门禁",
        "",
        "- 这是七点离散粗网格，不把最高样本均值直接称为唯一最优温度。",
        "- n=3 仅报告均值、样本标准差和 paired differences，不报告 p-value。",
        "- δ-optimal 结果是离散网格集合，不是连续温度区间；"
        "低自由度曲线、bootstrap 和 profile 区间属于下一阶段。",
        "- 当前 VOC val 同时用于响应曲线刻画，因此本报告不声称无偏最终测试性能；"
        "正式选温度仍需独立 selection split 和未参与选择的最终报告集。",
        "- T=0.5 至 T=2.0 的 CoVar 均值直接复用 P0 锁定结果；"
        "本次最小边界门禁未额外计算 T=0.25 的 CoVar，也未补分位数，"
        "因此不据此宣称稳定的 CoVar 近优区间。",
        "- 未运行 T=0.375、T=0.125、T=3.0、新 seed、Hard Teacher Target、"
        "P4a、CoVar-Match、shared temperature 或 T² compensation。",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    payload = summarize(args.p6_root, args.p7_root, args.covar_csv, args.delta)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(build_report(payload), encoding="utf-8")
    print(json.dumps(payload["selection_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
