#!/usr/bin/env python3
"""Summarize the P6 fresh 80k, four-method, three-seed trajectories."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from statistics import mean, stdev


SEEDS = (1234, 2025, 3407)
METHODS = ("fixed_t0p5", "fixed_t1p0", "fixed_t1p5", "p4a_r8")
MILESTONES = (20000, 40000, 60000, 80000)
METHOD_LABELS = {
    "fixed_t0p5": "T=0.5",
    "fixed_t1p0": "T=1.0",
    "fixed_t1p5": "T=1.5",
    "p4a_r8": "P4a 8x8",
}
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
        "--root", type=Path, default=Path("runs/covar_match/P6_full_schedule")
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/covar_match/P6_full_schedule.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("reports/covar_match/P6_full_schedule.md"),
    )
    return parser.parse_args()


def variant_for(method, seed):
    if method == "fixed_t0p5" and seed == 1234:
        return "fixed_T0p5_80k_seed1234_retry1"
    variants = {
        "fixed_t0p5": f"fixed_T0p5_80k_seed{seed}",
        "fixed_t1p0": f"fixed_T1p0_80k_seed{seed}",
        "fixed_t1p5": f"fixed_T1p5_80k_seed{seed}",
        "p4a_r8": f"p4a_r8_80k_seed{seed}",
    }
    try:
        return variants[method]
    except KeyError as error:
        raise ValueError(f"unknown P6 method: {method}") from error


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


def read_run(root, method, seed):
    variant = variant_for(method, seed)
    log_path = root / "logs" / variant / LOG_NAME
    state_dir = root / "checkpoints" / variant
    latest_state = state_dir / "training_state_latest.pth"
    if not log_path.is_file() or not latest_state.is_file():
        raise RuntimeError(f"{variant}: missing log or latest training state")
    text = log_path.read_text(encoding="utf-8")
    maximum, trajectory = parse_log(text)
    times = TIME_RE.findall(text)
    checks = {
        "maximum_80k": maximum == 80000,
        "iteration_80k": "Iters: 80000/80000" in text,
        "milestones": tuple(sorted(trajectory)) == MILESTONES,
        "training_time": bool(times),
        "finite": "non-finite" not in text.lower(),
        "p4a_stats": method != "p4a_r8" or "P4a stats:" in text,
        "t0p5_logged": method != "fixed_t0p5"
        or "Teacher-only target T: 0.5000" in text,
        "t1p0_logged": method != "fixed_t1p0"
        or "Teacher-only target T: 1.0000" in text,
        "t1p5_logged": method != "fixed_t1p5"
        or "Teacher-only target T: 1.5000" in text,
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
        "best_miou_percent": max(values),
        "final_miou_percent": values[-1],
        "final_pixacc_percent": trajectory[80000]["pixacc_percent"],
        "validation_checkpoints": len(values),
        "training_time": times[-1].strip(),
        "contract_pass": True,
        "log_path": str(log_path),
        "checkpoint_path": str(latest_state),
    }


def paired_statistics(values):
    values = [float(value) for value in values]
    if len(values) < 2:
        raise ValueError("sample standard deviation requires at least two pairs")
    return {
        "values_pp": values,
        "mean_pp": mean(values),
        "sample_std_pp": stdev(values),
    }


def miou_at(methods, method, milestone):
    return methods[method]["trajectory"][str(milestone)]["miou_percent"]


def rank_methods(methods, milestone):
    return sorted(
        METHODS,
        key=lambda method: (-miou_at(methods, method, milestone), METHODS.index(method)),
    )


def rank_transition_count(rankings):
    return sum(left != right for left, right in zip(rankings, rankings[1:]))


def sign_change_count(values):
    nonzero = [value for value in values if value != 0.0]
    return sum(left * right < 0.0 for left, right in zip(nonzero, nonzero[1:]))


def terminal_decision(rows):
    adaptive = [row["differences_pp"]["adaptive_final"] for row in rows]
    p4a_above_all_fixed = []
    for row in rows:
        methods = row["methods"]
        best_fixed = max(
            methods[method]["final_miou_percent"] for method in METHODS[:-1]
        )
        p4a_above_all_fixed.append(
            methods["p4a_r8"]["final_miou_percent"] > best_fixed
        )
    if all(p4a_above_all_fixed):
        return "case_B_p4a_above_all_fixed_all_seeds"
    if any(value > 0.0 for value in adaptive) and any(
        value < 0.0 for value in adaptive
    ):
        return "case_A_mixed_adaptive_final"
    if all(value <= 0.0 for value in adaptive):
        return "no_reliable_adaptive_improvement"
    return "not_case_B_incomplete_consistency"


def summarize(root):
    rows = []
    for seed in SEEDS:
        methods = {method: read_run(root, method, seed) for method in METHODS}
        adaptive_trajectory = {
            str(milestone): miou_at(methods, "p4a_r8", milestone)
            - miou_at(methods, "fixed_t1p5", milestone)
            for milestone in MILESTONES
        }
        complexity_trajectory = {
            str(milestone): miou_at(methods, "fixed_t1p5", milestone)
            - miou_at(methods, "fixed_t0p5", milestone)
            for milestone in MILESTONES
        }
        rankings = [rank_methods(methods, milestone) for milestone in MILESTONES]
        rows.append(
            {
                "seed": seed,
                "methods": methods,
                "differences_pp": {
                    "adaptive_trajectory": adaptive_trajectory,
                    "complexity_trajectory": complexity_trajectory,
                    "adaptive_best": methods["p4a_r8"]["best_miou_percent"]
                    - methods["fixed_t1p5"]["best_miou_percent"],
                    "adaptive_final": adaptive_trajectory["80000"],
                    "complexity_best": methods["fixed_t1p5"]["best_miou_percent"]
                    - methods["fixed_t0p5"]["best_miou_percent"],
                    "complexity_final": complexity_trajectory["80000"],
                },
                "rankings": {
                    str(milestone): ranking
                    for milestone, ranking in zip(MILESTONES, rankings)
                },
                "rank_transition_count": rank_transition_count(rankings),
                "adaptive_sign_change_count": sign_change_count(
                    list(adaptive_trajectory.values())
                ),
            }
        )

    paired_trajectory = {}
    for contrast in ("adaptive_trajectory", "complexity_trajectory"):
        paired_trajectory[contrast] = {
            str(milestone): paired_statistics(
                [row["differences_pp"][contrast][str(milestone)] for row in rows]
            )
            for milestone in MILESTONES
        }
    paired_terminal = {
        key: paired_statistics([row["differences_pp"][key] for row in rows])
        for key in (
            "adaptive_best",
            "adaptive_final",
            "complexity_best",
            "complexity_final",
        )
    }
    return {
        "stage": "P6_full_schedule",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {
            "fresh_from_student_initialization": True,
            "max_iterations": 80000,
            "validation_milestones": list(MILESTONES),
            "seeds": list(SEEDS),
            "methods": list(METHODS),
            "p_value_reported": False,
        },
        "excluded_attempt": {
            "variant": "fixed_T0p5_80k_seed1234",
            "last_iteration": 50600,
            "reason": "rank_1_SIGSEGV_exitcode_-11",
            "formal_replacement": "fixed_T0p5_80k_seed1234_retry1",
            "partial_metrics_used": False,
        },
        "rows": rows,
        "paired_trajectory": paired_trajectory,
        "paired_terminal": paired_terminal,
        "decision": {
            "terminal": terminal_decision(rows),
            "trajectory": (
                "case_C_trajectory_dependent_rankings"
                if all(row["rank_transition_count"] == len(MILESTONES) - 1 for row in rows)
                else "no_universal_rank_flip_across_all_seeds"
            ),
            "rank_transition_counts": {
                str(row["seed"]): row["rank_transition_count"] for row in rows
            },
            "adaptive_sign_change_counts": {
                str(row["seed"]): row["adaptive_sign_change_count"] for row in rows
            },
        },
    }


def build_report(payload):
    terminal = payload["decision"]["terminal"]
    decision_text = {
        "case_B_p4a_above_all_fixed_all_seeds": (
            "情况 B：P4a 在三个 seed 的 80k 终点均超过全部固定温度。"
            "20k 负/混合结果不能再作为方法失败证据。"
        ),
        "case_A_mixed_adaptive_final": (
            "情况 A：P4a−T=1.5 的 80k paired difference 有正有负，"
            "未形成一致正收益；其均值与离散度见下表。"
        ),
        "no_reliable_adaptive_improvement": (
            "P4a 在三个 seed 的 80k 终点均未超过 T=1.5；"
            "不存在可靠的区域贪心自适应正收益。"
        ),
        "not_case_B_incomplete_consistency": (
            "P4a 并未在三个 seed 上一致超过全部固定温度，"
            "因此不满足情况 B。"
        ),
    }[terminal]
    adaptive_final = payload["paired_terminal"]["adaptive_final"]
    rows = {row["seed"]: row for row in payload["rows"]}
    lines = [
        "# P6：标准完整 schedule 复核",
        "",
        "## 结论先行",
        "",
        f"- {decision_text}",
        (
            "- 情况 C 同时成立：三个 seed 的完整四方法排名在每个相邻验证点均改变"
            "（每个 seed 3 次变化），因此温度效用与方法排序依赖训练轨迹，"
            "不能视为静态属性。"
            if payload["decision"]["trajectory"]
            == "case_C_trajectory_dependent_rankings"
            else "- 未观察到三个 seed 共同的全程排名翻转模式。"
        ),
        "- P4a−T=1.5 的 80k paired mean 为 "
        f"{adaptive_final['mean_pp']:+.6f} pp，样本 SD 为 "
        f"{adaptive_final['sample_std_pp']:.6f} pp（n=3，不报告 p-value）。",
        "- 方法排名是否随训练反复变化，按每个 seed 的四点完整排名和"
        "相邻排名变化次数原样报告，不由单个终点替代。",
        "",
        "## 协议",
        "",
        "- 所有 run 都从同一 student ImageNet 初始化重新启动；没有从 20k checkpoint 续训。",
        "- max_iterations=80k；验证并保留 20k、40k、60k、80k checkpoint。",
        "- 方法仅包括 T=0.5、T=1.0、T=1.5、P4a 8×8；seed=1234、2025、3407。",
        "- 其余设置沿用 P1/P4a：VOC、相同 teacher/student、双 GPU、"
        "global batch=16、CE+KD、lr=0.02 和相同 poly schedule。",
        "- seed=1234、T=0.5 的首次基础设施尝试在 50,600 iteration 因 rank 1 "
        "SIGSEGV 中断；该尝试的局部结果完全排除，正式结果来自相同初始化的"
        "全新 retry1 80k run，未使用恢复训练。",
        "",
        "## 原始 mIoU 轨迹",
        "",
        "| seed | method | 20k | 40k | 60k | 80k | best | final |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        for method in METHODS:
            run = rows[seed]["methods"][method]
            values = [
                run["trajectory"][str(step)]["miou_percent"] for step in MILESTONES
            ]
            lines.append(
                f"| {seed} | {METHOD_LABELS[method]} | "
                + " | ".join(f"{value:.6f}" for value in values)
                + f" | {run['best_miou_percent']:.6f} | {run['final_miou_percent']:.6f} |"
            )

    lines += [
        "",
        "## P4a 相对 T=1.5 的 paired 轨迹",
        "",
        "| seed | d(20k) | d(40k) | d(60k) | d(80k) | sign changes |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        row = rows[seed]
        values = row["differences_pp"]["adaptive_trajectory"]
        lines.append(
            f"| {seed} | "
            + " | ".join(f"{values[str(step)]:+.6f}" for step in MILESTONES)
            + f" | {row['adaptive_sign_change_count']} |"
        )
    lines += [
        "",
        "| milestone | paired mean (pp) | sample SD (pp) |",
        "|---:|---:|---:|",
    ]
    for milestone in MILESTONES:
        stats = payload["paired_trajectory"]["adaptive_trajectory"][str(milestone)]
        lines.append(
            f"| {milestone // 1000}k | {stats['mean_pp']:+.6f} | "
            f"{stats['sample_std_pp']:.6f} |"
        )

    lines += [
        "",
        "## Complexity 对照与方法排名",
        "",
        "d_complexity=mIoU(T=1.5)−mIoU(T=0.5)。",
        "",
        "| milestone | d_complexity paired mean (pp) | sample SD (pp) |",
        "|---:|---:|---:|",
    ]
    for milestone in MILESTONES:
        stats = payload["paired_trajectory"]["complexity_trajectory"][str(milestone)]
        lines.append(
            f"| {milestone // 1000}k | {stats['mean_pp']:+.6f} | "
            f"{stats['sample_std_pp']:.6f} |"
        )
    lines += [
        "",
        "| seed | 20k ranking | 40k ranking | 60k ranking | 80k ranking | transitions |",
        "|---:|---|---|---|---|---:|",
    ]
    for seed in SEEDS:
        row = rows[seed]
        ranking_cells = []
        for milestone in MILESTONES:
            ranking_cells.append(
                " > ".join(
                    METHOD_LABELS[method]
                    for method in row["rankings"][str(milestone)]
                )
            )
        lines.append(
            f"| {seed} | "
            + " | ".join(ranking_cells)
            + f" | {row['rank_transition_count']} |"
        )
    lines += [
        "",
        "## 证据边界",
        "",
        "- 本轮只回答完整 schedule 下的 seed 稳定性与训练轨迹，不新增方法或机制指标。",
        "- best 仅取四个预先固定验证点；final 为 80k，不能推断未观测迭代的峰值。",
        "- n=3 只报告 paired mean 和样本标准差，不强调统计显著性。",
        "- 未运行其它温度、region size、margin gate、CIRKD、第二数据集或不同 student。",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    payload = summarize(args.root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(build_report(payload), encoding="utf-8")
    print(json.dumps(payload["paired_terminal"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
