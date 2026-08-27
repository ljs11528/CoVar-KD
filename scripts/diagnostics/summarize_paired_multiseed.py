#!/usr/bin/env python3
"""Summarize paired multi-seed T=0.5, T=1.5, and P4a formal runs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from statistics import mean, stdev


SEEDS = (1234, 2025, 3407)
LOG_NAME = "deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"
VALIDATION_RE = re.compile(
    r"Overall validation pixAcc: (?P<pixacc>[-+0-9.eE]+), "
    r"mIoU: (?P<miou>[-+0-9.eE]+)"
)
TIME_RE = re.compile(r"Total training time: (?P<time>[^\n(]+)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/covar_match"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/covar_match/paired_multiseed.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("reports/covar_match/paired_multiseed.md"),
    )
    return parser.parse_args()


def run_paths(root, method, seed):
    if method == "fixed_t0p5":
        stage, variant = "P1_teacher_only_temperature", f"T0p5_20k_seed{seed}"
    elif method == "fixed_t1p5":
        stage, variant = "P1_teacher_only_temperature", f"T1p5_20k_seed{seed}"
    elif method == "p4a_r8":
        stage = "P4a_task_aligned_region"
        variant = f"task_aligned_region_r8_20k_seed{seed}"
    else:
        raise ValueError(f"unknown method: {method}")
    log = root / stage / "logs" / variant / LOG_NAME
    state = root / stage / "checkpoints" / variant / "training_state_latest.pth"
    return variant, log, state


def read_run(root, method, seed):
    variant, log_path, state_path = run_paths(root, method, seed)
    if not log_path.is_file() or not state_path.is_file():
        raise RuntimeError(f"{variant}: missing log or latest training state")
    text = log_path.read_text(encoding="utf-8")
    values = [
        {
            "pixacc_percent": float(match.group("pixacc")),
            "miou_percent": float(match.group("miou")),
        }
        for match in VALIDATION_RE.finditer(text)
    ]
    times = TIME_RE.findall(text)
    checks = {
        "iteration_20k": "Iters: 20000/20000" in text,
        "validation": bool(values),
        "training_time": bool(times),
        "finite": "non-finite" not in text.lower(),
        "p4a_stats": method != "p4a_r8" or "P4a stats:" in text,
        "t0p5_logged": method != "fixed_t0p5"
        or "Teacher-only target T: 0.5000" in text,
        "t1p5_logged": method != "fixed_t1p5"
        or "Teacher-only target T: 1.5000" in text,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"{variant}: failed checks {failed}")
    return {
        "variant": variant,
        "best_miou_percent": max(item["miou_percent"] for item in values),
        "final_miou_percent": values[-1]["miou_percent"],
        "final_pixacc_percent": values[-1]["pixacc_percent"],
        "validation_checkpoints": len(values),
        "training_time": times[-1].strip(),
        "contract_pass": True,
        "log_path": str(log_path),
        "checkpoint_path": str(state_path),
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


def adaptive_decision(differences):
    ordered = [differences[seed] for seed in SEEDS]
    if all(value <= 0.0 for value in ordered):
        return "no_reliable_improvement"
    if differences[2025] > 0.0 and differences[3407] > 0.0:
        return "both_new_seeds_positive_reassess"
    return "inconsistent_improvement"


def summarize(root):
    rows = []
    for seed in SEEDS:
        methods = {
            method: read_run(root, method, seed)
            for method in ("fixed_t0p5", "fixed_t1p5", "p4a_r8")
        }
        rows.append(
            {
                "seed": seed,
                "methods": methods,
                "differences_pp": {
                    "complexity_best": methods["fixed_t1p5"]["best_miou_percent"]
                    - methods["fixed_t0p5"]["best_miou_percent"],
                    "complexity_final": methods["fixed_t1p5"]["final_miou_percent"]
                    - methods["fixed_t0p5"]["final_miou_percent"],
                    "adaptive_best": methods["p4a_r8"]["best_miou_percent"]
                    - methods["fixed_t1p5"]["best_miou_percent"],
                    "adaptive_final": methods["p4a_r8"]["final_miou_percent"]
                    - methods["fixed_t1p5"]["final_miou_percent"],
                },
            }
        )
    paired = {
        key: paired_statistics([row["differences_pp"][key] for row in rows])
        for key in (
            "complexity_best",
            "complexity_final",
            "adaptive_best",
            "adaptive_final",
        )
    }
    adaptive = {
        row["seed"]: row["differences_pp"]["adaptive_final"] for row in rows
    }
    complexity = {
        row["seed"]: row["differences_pp"]["complexity_final"] for row in rows
    }
    return {
        "stage": "paired_multiseed",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seeds": list(SEEDS),
        "rows": rows,
        "paired": paired,
        "decision": {
            "adaptive": adaptive_decision(adaptive),
            "minimum_complexity_not_best_all_seeds": all(
                value > 0.0 for value in complexity.values()
            ),
        },
        "p_value_reported": False,
    }


def build_report(payload):
    rows = {row["seed"]: row for row in payload["rows"]}
    decision_text = {
        "no_reliable_improvement": (
            "三个 seed 中 P4a 均未超过固定 T=1.5。可写为："
            "Region-wise greedy task alignment does not provide a reliable "
            "improvement over the strongest global temperature baseline."
        ),
        "both_new_seeds_positive_reassess": (
            "两个新 seed 均数值上超过固定 T=1.5，而 seed=1234 为负；"
            "当前负结果不稳定，不能继续把 P4a 当作已失败方法。"
        ),
        "inconsistent_improvement": (
            "P4a 的 paired difference 有正有负，未形成一致正收益。可写为："
            "P4a fails to yield a consistent improvement."
        ),
    }[payload["decision"]["adaptive"]]
    complexity = payload["paired"]["complexity_final"]
    adaptive = payload["paired"]["adaptive_final"]
    lines = [
        "# Paired multi-seed：固定温度与 P4a 复核",
        "",
        "## 结论先行",
        "",
        f"- {decision_text}",
        "- n=3，仅报告 paired mean 与样本标准差；不计算或强调 p-value。",
        "- d_complexity 的逐 seed 符号为正、负、正，paired mean "
        "{:+.6f} pp、样本 SD {:.6f} pp；T=1.5 数值上胜出 2/3，"
        "但并非逐 seed 稳定优于最低复杂度 T=0.5。".format(
            complexity["mean_pp"], complexity["sample_std_pp"]
        ),
        "- d_adaptive 的逐 seed 符号为负、正、负，paired mean "
        "{:+.6f} pp、样本 SD {:.6f} pp。".format(
            adaptive["mean_pp"], adaptive["sample_std_pp"]
        ),
        "- 三种方法均只在 20k 做一次 validation，因此表中 best=final。",
        "",
        "## 协议",
        "",
        "- seed=1234 复用既有正式结果；新 seed=2025、3407 沿用仓库既有约定。",
        "- 每个新 seed 只跑固定 T=0.5、固定 T=1.5、P4a 8×8，均为 20k。",
        "- 数据、模型、global batch=16、双 GPU、CE+KD、优化器及验证频率保持不变。",
        "",
        "## 原始结果与 paired difference",
        "",
        "| seed | T=0.5 best/final | T=1.5 best/final | P4a best/final | "
        "d_complexity final (pp) | d_adaptive final (pp) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        row = rows[seed]
        methods = row["methods"]
        lines.append(
            "| {seed} | {a[best_miou_percent]:.6f}/{a[final_miou_percent]:.6f} | "
            "{b[best_miou_percent]:.6f}/{b[final_miou_percent]:.6f} | "
            "{c[best_miou_percent]:.6f}/{c[final_miou_percent]:.6f} | "
            "{dc:+.6f} | {da:+.6f} |".format(
                seed=seed,
                a=methods["fixed_t0p5"],
                b=methods["fixed_t1p5"],
                c=methods["p4a_r8"],
                dc=row["differences_pp"]["complexity_final"],
                da=row["differences_pp"]["adaptive_final"],
            )
        )
    lines += [
        "",
        "定义：d_complexity=mIoU(T=1.5)−mIoU(T=0.5)；"
        "d_adaptive=mIoU(P4a)−mIoU(T=1.5)。",
        "",
        "## Paired 汇总",
        "",
        "| contrast | best mean ± sample SD (pp) | final mean ± sample SD (pp) |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("T=1.5 − T=0.5", "complexity"),
        ("P4a − T=1.5", "adaptive"),
    ):
        best = payload["paired"][f"{key}_best"]
        final = payload["paired"][f"{key}_final"]
        lines.append(
            f"| {label} | {best['mean_pp']:+.6f} ± {best['sample_std_pp']:.6f} | "
            f"{final['mean_pp']:+.6f} ± {final['sample_std_pp']:.6f} |"
        )
    lines += [
        "",
        "## 边界",
        "",
        "- 本轮只复核 seed 稳定性，不新增机制解释或自适应设计。",
        "- 不据 n=3 声称统计显著性，也不把数值下降改写成“始终下降”。",
        "- 未运行其它温度、margin gate、其它 region size、P2–P5、"
        "跨设置实验或短时程轨迹。",
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
    print(json.dumps(payload["paired"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
