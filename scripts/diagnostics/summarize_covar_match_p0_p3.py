#!/usr/bin/env python3
"""Build the non-redundant P0-P3 CoVar Match execution report."""

import argparse
import datetime as dt
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-dir", type=Path, default=Path("reports/covar_match")
    )
    parser.add_argument("--base-commit", default="f102a71")
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def render(payload):
    p0 = payload["stages"]["P0"]
    p1 = payload["stages"]["P1"]
    p2 = payload["stages"]["P2"]
    p3 = payload["stages"]["P3"]
    chain = payload["evidence_chain"]
    p1_rows = p1["rows"]
    early_late = next(
        item
        for item in p2["state_transitions"]
        if item["left"] == "early" and item["right"] == "late"
    )
    lines = [
        "# CoVar Match P0–P3 实验执行总报告",
        "",
        f"- 基线代码提交：{payload['base_commit']}",
        "- 数据/架构：Pascal VOC；DeepLabV3-ResNet101 teacher；DeepLabV3-MobileNetV3-Small student。",
        "- 正式训练仅发生在 P1；P0、P2、P3 均为冻结模型或既有检查点上的离线诊断。",
        f"- 四阶段执行门禁：{'全部通过' if payload['all_execution_gates_pass'] else '存在失败'}。",
        "",
        "## 1. 主线结论",
        "",
        "| 主张 | 证据 | 状态 |",
        "|---|---|---|",
        f"| r 理论与实现一致 | 一/二阶闭式导数对 autograd 与有限差分均通过；VOC val 全量数值有限 | {'通过' if chain['r_theory_implementation_consistent'] else '未通过'} |",
        f"| T 能改变教师输出复杂度 | mean r 从 {chain['mean_r_min']:.6f} 变到 {chain['mean_r_max']:.6f} | {'支持' if chain['temperature_changes_complexity'] else '不支持'} |",
        f"| 最低复杂度不一定最好 | 最低 r 温度 T={chain['minimum_complexity_temperature']:.2f}；最佳 mIoU 温度 T={chain['best_miou_temperature']:.2f} | {'支持' if chain['minimum_complexity_not_best'] else '不支持'} |",
        f"| teachability 依赖学生状态 | early→late 区域 oracle 改变 {chain['early_late_oracle_change_fraction']:.4%} | {'支持' if chain['teachability_is_state_dependent'] else '不支持'} |",
        f"| CoVar gap 预测 teachability | overall top-1={chain['vector_gap_top1']:.4%}，mean Spearman={chain['vector_gap_mean_spearman']:.6f} | {'支持' if chain['covar_gap_predictive'] else '不支持'} |",
        f"| 状态匹配带来最终蒸馏收益 | P0–P3 没有训练自适应匹配策略 | {'尚未验证' if not chain['adaptive_distillation_gain_demonstrated'] else '已验证'} |",
        "",
        "## 2. P0：理论—实现一致性门禁",
        "",
        f"- VOC val：{p0['scope']['processed_images']} 张；有效像素 {p0['trajectory'][0]['valid_pixels']}。",
        f"- 闭式一阶导 vs autograd max abs：{p0['derivative_audit']['closed_vs_autograd_first']['max_abs']:.3e}。",
        f"- 闭式二阶导 vs autograd max abs：{p0['derivative_audit']['closed_vs_autograd_second']['max_abs']:.3e}。",
        f"- 采样像素至少一次 r 下降：{p0['empirical']['sampled_pixel_any_decrease_fraction']:.4%}；轨迹转折：{p0['empirical']['sampled_pixel_turn_fraction']:.4%}。",
        "",
        "## 3. P1：全局 teacher-only 温度扫描",
        "",
        "- 20k iterations，global batch 16，双 GPU，seed 1234；学生温度固定 1，无 T²，所有其它 KD 分支关闭。",
        "",
        "| T | mean r | mIoU (%) | pixAcc (%) |",
        "|---:|---:|---:|---:|",
    ]
    for row in p1_rows:
        lines.append(
            f"| {row['temperature']:.2f} | {row['mean_r']:.6f} | "
            f"{row['miou_percent']:.6f} | {row['pixacc_percent']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- 最佳温度相对最低复杂度温度的 mIoU 差："
            f"{p1['observation']['best_minus_min_complexity_pp']:+.6f} 个百分点。",
            "- 该结果只支持固定全局 teacher-target 温度会改变短程蒸馏结果；不等同于区域自适应策略收益。",
            "",
            "## 4. P2：区域 oracle 与学生状态",
            "",
            f"- 固定抽样 VOC val {p2['scope']['processed_images']} 张；"
            f"候选缓存 {p2['scope']['candidate_rows']} 行。",
            "- oracle 为 8×8 原生 logit 区域中，归一化 teacher-only KD 梯度一步更新后的监督 CE 最大降幅。",
            "",
            "| 状态 | iteration | 区域数 | mean oracle gain | oracle gain>0 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for stage, iteration in (("early", 4000), ("middle", 12000), ("late", 20000)):
        summary = p2["stage_summaries"][stage]
        lines.append(
            f"| {stage} | {iteration} | {summary['region_count']} | "
            f"{summary['mean_oracle_gain']:.6e} | "
            f"{summary['positive_oracle_gain_fraction']:.4%} |"
        )
    lines.extend(
        [
            "",
            f"- early→late exact agreement：{early_late['exact_agreement']:.4%}；"
            f"adjacent agreement：{early_late['adjacent_agreement']:.4%}。",
            "- 这是标签可见的一步局部 oracle，只是机制诊断。",
            "",
            "## 5. P3：gap 对 teachability 的预测",
            "",
            "| 分数 | overall top-1 | adjacent | mean regret | mean aligned rho |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for score in (
        "teacher_min_r",
        "scalar_r_gap",
        "vector_covar_gap",
        "teacher_student_kl",
    ):
        metrics = p3["metrics"]["overall"][score]
        lines.append(
            f"| {score} | {metrics['top1_accuracy']:.4%} | "
            f"{metrics['adjacent_accuracy']:.4%} | "
            f"{metrics['mean_regret']:.6e} | "
            f"{metrics['mean_spearman']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- overall 最强 top-1：{p3['observation']['best_top1_score']}；"
            f"最低 mean regret：{p3['observation']['best_regret_score']}。",
            "- r_c/r_v 缩放来自同一无标签分析缓存，没有独立校准集；P3 结果属于探索性机制证据。",
            "",
            "## 6. 测试验证",
            "",
            f"- P0–P3 聚焦测试：{payload['verification']['focused_passed']} passed。",
            f"- 完整测试套件：{payload['verification']['full_passed']} passed，"
            f"{payload['verification']['full_failed']} failed。",
            f"- 唯一失败原因：缺少历史冻结 artifact {payload['verification']['missing_artifact']}；项目内无副本。",
            "- 远端系统 pytest 的自动插件 anyio 与 pytest 版本不兼容，测试使用 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1；未修改依赖。",
            "",
            "## 7. 可复现入口与产物",
            "",
            "- P0：scripts/diagnostics/covar_metric_theory_audit.py",
            "- P1：scripts/experiments/covar_match/run_p1_teacher_only_temperature.sh",
            "- P1 汇总：scripts/diagnostics/summarize_p1_teacher_only_temperature.py",
            "- P2：scripts/diagnostics/region_teachability_oracle.py",
            "- P3：scripts/diagnostics/covar_gap_teachability.py",
            "- 分阶段详细报告：reports/covar_match/P0_metric_theory_audit.md、P1_teacher_only_temperature_scan.md、P2_region_teachability.md、P3_covar_gap_teachability.md。",
            "",
            "## 8. 结论边界",
            "",
            "- 本轮完成了从 r 一致性、温度效应、全局蒸馏结果、状态依赖 oracle 到 gap 预测的 P0–P3 链条。",
            "- 若要闭合“教师复杂度与学生能力匹配 → 最终蒸馏收益”，下一步仍需把不使用标签的匹配规则放回训练，并与最强全局 T 对照；本轮没有执行该训练。",
            "- 所有支持/不支持判断均由实际结果条件生成，负结果和混合结果不改写为正结论。",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    report_dir = args.report_dir
    p0 = read_json(report_dir / "P0_metric_theory_audit.json")
    with (report_dir / "P0_temperature_complexity_trajectory.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        import csv

        trajectory = [
            {
                key: (
                    float(value)
                    if key != "valid_pixels"
                    else int(value)
                )
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]
    p0["trajectory"] = trajectory
    p1 = read_json(report_dir / "P1_teacher_only_temperature_scan.json")
    p2 = read_json(report_dir / "P2_region_teachability.json")
    p3 = read_json(report_dir / "P3_covar_gap_teachability.json")
    gates = {
        "P0": bool(p0["gate_pass"]),
        "P1": bool(p1["execution_gate_pass"]),
        "P2": bool(p2["execution_gate_pass"]),
        "P3": bool(p3["execution_gate_pass"]),
    }
    early_late = next(
        item
        for item in p2["state_transitions"]
        if item["left"] == "early" and item["right"] == "late"
    )
    vector = p3["metrics"]["overall"]["vector_covar_gap"]
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_commit": args.base_commit,
        "execution_gates": gates,
        "all_execution_gates_pass": all(gates.values()),
        "verification": {
            "focused_passed": 19,
            "full_passed": 137,
            "full_failed": 1,
            "missing_artifact": (
                "runs/diagnostics/phaseO_o11/voc_train_rtc_confidence_cdf.pt"
            ),
        },
        "evidence_chain": {
            "r_theory_implementation_consistent": gates["P0"],
            "temperature_changes_complexity": (
                max(row["r_mean"] for row in trajectory)
                > min(row["r_mean"] for row in trajectory)
            ),
            "mean_r_min": min(row["r_mean"] for row in trajectory),
            "mean_r_max": max(row["r_mean"] for row in trajectory),
            "minimum_complexity_temperature": p1["observation"][
                "minimum_complexity_temperature"
            ],
            "best_miou_temperature": p1["observation"][
                "best_miou_temperature"
            ],
            "minimum_complexity_not_best": p1["observation"][
                "minimum_complexity_not_best"
            ],
            "early_late_oracle_change_fraction": (
                1.0 - early_late["exact_agreement"]
            ),
            "teachability_is_state_dependent": (
                early_late["exact_agreement"] < 1.0
            ),
            "vector_gap_top1": vector["top1_accuracy"],
            "vector_gap_mean_spearman": vector["mean_spearman"],
            "covar_gap_predictive": p3["observation"][
                "vector_gap_predictive"
            ],
            "adaptive_distillation_gain_demonstrated": False,
        },
        "stages": {
            "P0": p0,
            "P1": p1,
            "P2": p2,
            "P3": p3,
        },
    }
    json_path = report_dir / "CoVar_Match_P0_P3_execution_report.json"
    markdown_path = report_dir / "CoVar_Match_P0_P3_execution_report.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render(payload), encoding="utf-8")
    print(json.dumps(
        {
            "execution_gates": gates,
            "evidence_chain": payload["evidence_chain"],
        },
        indent=2,
        ensure_ascii=False,
    ))
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    if not payload["all_execution_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
