#!/usr/bin/env python3
"""Report only audited P10 runs; keep common-grid and boundary analyses separate."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.covar_match import run_p10_h20 as queue
from scripts.diagnostics.summarize_p8_p9_experiments import sample_statistics


def analyze(rows):
    groups = {}
    for row in rows:
        group = groups.setdefault(row["group"], {})
        if row["seed"] in group:
            raise ValueError("duplicate seed within a group")
        group[row["seed"]] = row["final_miou_percent"]
    statistics = {g: sample_statistics([v[s] for s in queue.SEEDS])
                  for g, v in groups.items() if set(v) == set(queue.SEEDS)}
    def difference(left, right):
        if left not in statistics or right not in statistics:
            return None
        return sample_statistics([groups[left][s] - groups[right][s] for s in queue.SEEDS])
    def group(student, t):
        return f"{student}_teacher_T{t.replace('.', 'p')}_L1"
    response, gains, selections = {}, {}, {}
    for student in ("small", "large"):
        ts = queue.TEMPERATURES
        if not all(group(student, t) in statistics for t in ts):
            continue
        means = {t: statistics[group(student, t)]["mean"] for t in ts}
        winner = max(ts, key=means.get)
        response[student] = dict(mean_winner=winner, means=means,
            delta_near_optimal=[t for t in ts if means[t] >= means[winner] - 0.2],
            per_seed_winners={s: max(ts, key=lambda t: groups[group(student, t)][s]) for s in queue.SEEDS})
        gains[student] = {t: difference(group(student, t), student + "_ce") for t in ts}
        selected, regrets = {}, []
        for seed in queue.SEEDS:
            other = [s for s in queue.SEEDS if s != seed]
            chosen = max(ts, key=lambda t: sum(groups[group(student, t)][s] for s in other))
            regret = max(groups[group(student, t)][seed] for t in ts) - groups[group(student, chosen)][seed]
            selected[seed] = dict(temperature=chosen, heldout_seed_regret_pp=regret)
            regrets.append(regret)
        selections[student] = dict(folds=selected, regret_pp=sample_statistics(regrets),
            interpretation="Exploratory leave-one-seed-out on the same validation set; not independent test-set validation.")
    boundary = None
    if "large" in response and group("large", "4.0") in statistics:
        means = {**response["large"]["means"], "4.0": statistics[group("large", "4.0")]["mean"]}
        best = max(means.values())
        near = [t for t,v in means.items() if v >= best - 0.2]
        boundary = dict(expanded_means=means, delta_near_optimal=near,
                        upper_boundary_unresolved="4.0" in near,
                        t4_minus_t2_pp=difference(group("large", "4.0"), group("large", "2.0")),
                        t4_minus_ce_pp=difference(group("large", "4.0"), "large_ce"),
                        automatic_further_runs=False,
                        scope="Finite registered grid only; no continuous/global optimum claim.")
    control_names = dict(teacher_1="large_teacher_T2p0_L1", teacher_4="large_teacher_T2p0_L4",
                         shared_1="large_shared_T2p0_L1", shared_4="large_shared_T2p0_L4")
    controls = None
    if all(g in statistics for g in control_names.values()):
        c = control_names
        controls = {"cells": {k: statistics[g] for k,g in c.items()},
            "teacher_scale_4_minus_1_pp": difference(c["teacher_4"], c["teacher_1"]),
            "shared_scale_4_minus_1_pp": difference(c["shared_4"], c["shared_1"]),
            "shared_minus_teacher_at_scale1_pp": difference(c["shared_1"], c["teacher_1"]),
            "shared_minus_teacher_at_scale4_pp": difference(c["shared_4"], c["teacher_4"]),
            "interaction_pp": sample_statistics([
                (groups[c["shared_4"]][s] - groups[c["shared_1"]][s]) -
                (groups[c["teacher_4"]][s] - groups[c["teacher_1"]][s]) for s in queue.SEEDS])}
    capacity = None
    if all(s in response for s in ("small", "large")):
        capacity = {t: dict(large_minus_small_pp=difference(group("large", t), group("small", t)),
            temperature_effect_interaction_vs_t1_pp=sample_statistics([
                (groups[group("large", t)][s] - groups[group("large", "1.0")][s]) -
                (groups[group("small", t)][s] - groups[group("small", "1.0")][s]) for s in queue.SEEDS]))
            for t in queue.TEMPERATURES}
    return dict(group_statistics=statistics, common_grid_response=response, kd_minus_ce_pp=gains,
                capacity_comparison=capacity, boundary_extension=boundary, loss_controls=controls,
                leave_one_seed_out=selections)


def markdown(payload):
    def fmt(stats):
        return "pending" if stats is None else f"{stats['mean']:.6f} ± {stats['sample_std']:.6f}"
    a = payload["analysis"]
    lines = ["# P10：同协议学生容量、CE、上边界和损失尺度对照", "",
        f"状态：**{payload['status']}**；新增实验完成 **{payload['completed_new_runs']}/33**。"
        "另复用已完成且审计通过的 P9 Large 五点温度 × 三 seed。", "",
        "各组均从 ImageNet backbone 和同编号 seed 的新分割头开始，单 GPU、80k、"
        "batch=16；teacher、数据、优化器、验证里程碑、训练源码与 P9 一致。"
        "完整冻结信息见 [protocol.json](protocol.json)。旧 P7/H100 结果不并入。", "",
        "## 80k 结果", "", "mIoU 为百分数，± 为三 seed 样本标准差，不是置信区间。", "",
        "| 组别 | seed 1234 | seed 2025 | seed 3407 | 均值 ± SD |",
        "|---|---:|---:|---:|---:|"]
    for name, stats in a["group_statistics"].items():
        lines.append(f"| {name} | " + " | ".join(f"{v:.6f}" for v in stats["values"]) + f" | {fmt(stats)} |")
    lines += ["", "## 同协议 KD − CE 配对差值", "", "单位为 mIoU 个百分点（pp）。", "",
              "| T | Small：均值 ± SD | Large：均值 ± SD |", "|---:|---:|---:|"]
    for t in queue.TEMPERATURES:
        lines.append(f"| {t} | {fmt(a['kd_minus_ce_pp'].get('small', {}).get(t))} | "
                     f"{fmt(a['kd_minus_ce_pp'].get('large', {}).get(t))} |")
    lines += ["", "## 共同五点网格与选择稳定性", ""]
    for student, response in a["common_grid_response"].items():
        loso = a["leave_one_seed_out"][student]
        lines.append(f"- {student}：均值赢家 T={response['mean_winner']}；δ=0.2 pp 近优集合 "
                     f"{response['delta_near_optimal']}；逐 seed 赢家 {response['per_seed_winners']}；"
                     f"留一 seed 选择损失 {fmt(loso['regret_pp'])} pp。")
    lines += ["", "留一 seed 分析仍使用同一验证集，仅为探索性诊断。赢家变化不证明总体最优点不存在；"
              "近优集合不是统计等效性检验。模型容量比较限定在共同五点网格；T=4 单独分析。", "",
              "## Large 上边界扩展", ""]
    boundary = a["boundary_extension"]
    if boundary:
        lines += [f"T=4 − T=2：{fmt(boundary['t4_minus_t2_pp'])} pp。",
                  f"扩展网格近优集合：{boundary['delta_near_optimal']}。",
                  "T=4 仍在近优集合内，上边界未闭合。" if boundary["upper_boundary_unresolved"] else
                  "T=4 不在近优集合内，本次扩展未标记上边界问题；这不是连续空间最优性的证明。"]
    else:
        lines.append("等待全部三 seed。")
    lines += ["", "此项为独立追加的边界实验，保留 P9 原来的不补 0.75/1.25 决定，不自动扩展到 T=8。", "",
              "## 共同温度与损失尺度", "",
              "固定 Large、teacher T=2；student T∈{1,2}，KL 总系数∈{1,4}。"
              "所有格子使用相同有效像素均值归一化。student T=2、系数=4 等价于该温度下的标准 T² KD；"
              "代码的额外 temperature power 固定为 0，避免重复补偿。", ""]
    if a["loss_controls"]:
        for key, value in a["loss_controls"].items():
            if key != "cells":
                lines.append(f"- {key}: {fmt(value)} pp。")
    else:
        lines.append("等待三个新增格子的三 seed。")
    lines += ["", "该 2×2 实验区分改变学生温度和改变 KL 系数的局部效应；"
              "不等于梯度范数匹配，也不能代替共同温度配方的完整响应曲线。", "",
              "## 完整性与文件", "",
              f"新增完成实验的 {payload['audited_checkpoint_count']} 份训练状态已逐一在 CPU 加载，"
              "核验 iteration、seed、损失配置、有限张量、optimizer 与 RNG 状态。"
              "每条完整实验应有 4 个验证点和 5 份被审计状态。", "",
              "[逐次验证 CSV](results.csv) · [机器可读结果](results.json) · "
              "[完整性审计](completion_audit.json) · [下载哈希](download-manifest.json)", ""]
    if payload["completed_new_runs"] == 33:
        lines += ["![温度响应](temperature_response.png)", "", "![KD 与 CE 差值](kd_minus_ce.png)", ""]
    return "\n".join(lines)


def figures(analysis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = list(queue.TEMPERATURES)
    x = [float(t) for t in ts]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, student in zip(axes, ("small", "large")):
        stats = [analysis["group_statistics"][f"{student}_teacher_T{t.replace('.', 'p')}_L1"] for t in ts]
        for i, seed in enumerate(queue.SEEDS):
            ax.plot(x, [s["values"][i] for s in stats], ".--", alpha=.45, label=f"seed {seed}")
        ax.errorbar(x, [s["mean"] for s in stats], [s["sample_std"] for s in stats],
                    fmt="o-", color="black", capsize=3, label="mean ± sample SD")
        ce = analysis["group_statistics"][student + "_ce"]["mean"]
        ax.axhline(ce, color="tab:red", linestyle=":", label="CE-only mean")
        ax.set(title=f"R101 → MobileNetV3-{student.title()}", xlabel="Teacher temperature", ylabel="80k mIoU (%)")
        ax.set_xticks(x); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle("P10 matched protocol: common five-point grid, n=3 seeds")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(queue.REPORT_ROOT / f"temperature_response.{suffix}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for student, offset in (("small", -.015), ("large", .015)):
        stats = [analysis["kd_minus_ce_pp"][student][t] for t in ts]
        ax.errorbar([v+offset for v in x], [s["mean"] for s in stats],
                    [s["sample_std"] for s in stats], fmt="o-", capsize=4, label=student.title())
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(xlabel="Teacher temperature", ylabel="KD − CE (mIoU pp)",
           title="Paired differences: mean ± sample SD, n=3")
    ax.set_xticks(x); ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(queue.REPORT_ROOT / f"kd_minus_ce.{suffix}", dpi=180)
    plt.close(fig)


def generate():
    baseline = json.loads((queue.REFERENCE / "P9_temperature.json").read_text())
    rows, new_audits = [], []
    for seed in queue.SEEDS:
        for t in queue.TEMPERATURES:
            rows.append(dict(baseline["runs"][str(seed)][t], seed=seed, student="large",
                temperature=t, mode="teacher_only", lambda_kd=1.0, reused=True,
                group=f"large_teacher_T{t.replace('.', 'p')}_L1"))
    for spec in queue.experiment_plan():
        if not queue.completed(spec):
            continue
        path = queue.RUN_ROOT / "runtime/audits" / (spec["variant"] + ".json")
        record = json.loads(path.read_text())
        new_audits.append(record)
        rows.append(dict(record, reused=False))
    analysis = analyze(rows)
    payload = dict(generated_at=queue.now(), status="COMPLETE" if len(new_audits) == 33 else "PARTIAL",
        completed_new_runs=len(new_audits), reused_runs=15, rows=rows, analysis=analysis,
        audited_checkpoint_count=sum(len(r["checkpoints"]) for r in new_audits))
    queue.write_json(queue.REPORT_ROOT / "results.json", payload)
    queue.write_json(queue.REPORT_ROOT / "completion_audit.json", dict(status=payload["status"],
        completed_runs=len(new_audits), checkpoints_checked=payload["audited_checkpoint_count"],
        runs=new_audits, p9_audit_reference="../P9_h20/completion_audit.json"))
    with (queue.REPORT_ROOT / "results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "seed", "iteration", "miou_percent", "pixacc_percent", "reused_p9"])
        for row in rows:
            for step, values in row["trajectory"].items():
                writer.writerow([row["group"], row["seed"], step, values["miou_percent"],
                                 values["pixacc_percent"], row["reused"]])
    if payload["status"] == "COMPLETE":
        figures(analysis)
    (queue.REPORT_ROOT / "final_report.md").write_text(markdown(payload))
    files = {p.name: dict(sha256=queue.base.digest(p), bytes=p.stat().st_size)
             for p in sorted(queue.REPORT_ROOT.iterdir())
             if p.is_file() and p.name != "download-manifest.json" and not p.name.endswith(".tmp")}
    queue.write_json(queue.REPORT_ROOT / "download-manifest.json", dict(created_at=queue.now(),
        source="lyf_H200_141G:" + str(queue.REPORT_ROOT), files=files))
    return payload


if __name__ == "__main__":
    generate()
