#!/usr/bin/env python3
"""P3: test whether CoVar gap predicts regional teachability."""

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np


SCORES = (
    "teacher_min_r",
    "scalar_r_gap",
    "vector_covar_gap",
    "teacher_student_kl",
)
SCORE_LABELS = {
    "teacher_min_r": "min teacher r",
    "scalar_r_gap": "scalar abs(r_t-r_s)",
    "vector_covar_gap": "2D CoVar gap",
    "teacher_student_kl": "teacher-student KL",
}
SCOPES = ("early", "middle", "late", "overall")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-cache",
        type=Path,
        default=Path(
            "runs/covar_match/P2_region_candidate_teachability.csv.gz"
        ),
    )
    parser.add_argument(
        "--p2-json",
        type=Path,
        default=Path("reports/covar_match/P2_region_teachability.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/covar_match")
    )
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        average = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average
        start = end
    return ranks


def spearman(values_a, values_b):
    rank_a = average_ranks(values_a)
    rank_b = average_ranks(values_b)
    centered_a = rank_a - rank_a.mean()
    centered_b = rank_b - rank_b.mean()
    denominator = math.sqrt(
        float(np.dot(centered_a, centered_a))
        * float(np.dot(centered_b, centered_b))
    )
    if denominator == 0:
        return 0.0
    return float(np.dot(centered_a, centered_b) / denominator)


def row_scores(row, component_scales):
    teacher_r = float(row["teacher_r"])
    student_r = float(row["student_r"])
    delta_rc = float(row["teacher_r_c"]) - float(row["student_r_c"])
    delta_rv = float(row["teacher_r_v"]) - float(row["student_r_v"])
    return {
        "teacher_min_r": teacher_r,
        "scalar_r_gap": abs(teacher_r - student_r),
        "vector_covar_gap": math.sqrt(
            (delta_rc / component_scales["r_c"]) ** 2
            + (delta_rv / component_scales["r_v"]) ** 2
        ),
        "teacher_student_kl": float(row["teacher_student_kl"]),
    }


def evaluate_candidate_group(rows, component_scales, temperatures):
    rows = sorted(rows, key=lambda row: float(row["temperature"]))
    observed_temperatures = [float(row["temperature"]) for row in rows]
    if observed_temperatures != list(temperatures):
        raise RuntimeError(
            f"candidate grid mismatch: {observed_temperatures}"
        )
    gains = np.asarray(
        [float(row["one_step_gain"]) for row in rows], dtype=np.float64
    )
    oracle_index = int(np.argmax(gains))
    result = {
        "stage": rows[0]["stage"],
        "oracle_temperature": temperatures[oracle_index],
        "oracle_gain": float(gains[oracle_index]),
        "scores": {},
    }
    for score_name in SCORES:
        values = np.asarray(
            [
                row_scores(row, component_scales)[score_name]
                for row in rows
            ],
            dtype=np.float64,
        )
        prediction_index = int(np.argmin(values))
        index_distance = abs(prediction_index - oracle_index)
        result["scores"][score_name] = {
            "prediction_temperature": temperatures[prediction_index],
            "exact": float(index_distance == 0),
            "adjacent": float(index_distance <= 1),
            "regret": max(
                float(gains[oracle_index] - gains[prediction_index]), 0.0
            ),
            "spearman": spearman(-values, gains),
        }
    return result


def group_key(row):
    return (
        row["stage"],
        row["image_id"],
        int(row["region_row"]),
        int(row["region_col"]),
    )


def iter_candidate_groups(path):
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        current_key = None
        current_rows = []
        for row in reader:
            key = group_key(row)
            if current_key is not None and key != current_key:
                yield current_key, current_rows
                current_rows = []
            current_key = key
            current_rows.append(row)
        if current_rows:
            yield current_key, current_rows


def component_scales(path):
    count = 0
    mean_rc = 0.0
    mean_rv = 0.0
    m2_rc = 0.0
    m2_rv = 0.0
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            delta_rc = (
                float(row["teacher_r_c"]) - float(row["student_r_c"])
            )
            delta_rv = (
                float(row["teacher_r_v"]) - float(row["student_r_v"])
            )
            count += 1
            rc_difference = delta_rc - mean_rc
            mean_rc += rc_difference / count
            m2_rc += rc_difference * (delta_rc - mean_rc)
            rv_difference = delta_rv - mean_rv
            mean_rv += rv_difference / count
            m2_rv += rv_difference * (delta_rv - mean_rv)
    if count == 0:
        raise RuntimeError("P2 candidate cache is empty")
    scales = {
        "r_c": math.sqrt(m2_rc / count),
        "r_v": math.sqrt(m2_rv / count),
    }
    if min(scales.values()) <= 0 or not all(
        math.isfinite(value) for value in scales.values()
    ):
        raise RuntimeError(f"invalid component scales: {scales}")
    return scales, count


def empty_accumulator():
    return {
        scope: {
            score: {
                "exact": [],
                "adjacent": [],
                "regret": [],
                "spearman": [],
            }
            for score in SCORES
        }
        for scope in SCOPES
    }


def add_result(accumulator, result):
    stage = result["stage"]
    for score, metrics in result["scores"].items():
        for scope in (stage, "overall"):
            for metric, value in metrics.items():
                if metric == "prediction_temperature":
                    continue
                accumulator[scope][score][metric].append(float(value))


def finalize_metrics(accumulator):
    rows = []
    nested = {}
    for scope in SCOPES:
        nested[scope] = {}
        for score in SCORES:
            values = accumulator[scope][score]
            if not values["exact"]:
                raise RuntimeError(f"no P3 rows for {scope}/{score}")
            metrics = {
                "region_count": len(values["exact"]),
                "top1_accuracy": float(np.mean(values["exact"])),
                "adjacent_accuracy": float(np.mean(values["adjacent"])),
                "mean_regret": float(np.mean(values["regret"])),
                "median_regret": float(np.median(values["regret"])),
                "mean_spearman": float(np.mean(values["spearman"])),
                "median_spearman": float(np.median(values["spearman"])),
            }
            nested[scope][score] = metrics
            rows.append({"scope": scope, "score": score, **metrics})
    return nested, rows


def write_csv(path, rows):
    fields = (
        "scope",
        "score",
        "region_count",
        "top1_accuracy",
        "adjacent_accuracy",
        "mean_regret",
        "median_regret",
        "mean_spearman",
        "median_spearman",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_report(payload):
    lines = [
        "# P3：CoVar gap 是否预测区域 teachability",
        "",
        "- 本阶段不训练；完全复用 P2 的区域×状态×候选温度缓存。",
        "- 预测规则统一为分数越小越优；oracle 为 P2 的一步监督 CE gain 最大温度。",
        "- 二维 CoVar gap 使用全分析缓存的总体标准差缩放 r_c/r_v；不使用标签拟合权重。",
        f"- 执行门禁：{'通过' if payload['execution_gate_pass'] else '失败'}",
        "",
        "## 预测结果",
        "",
        "| 状态 | 分数 | top-1 | adjacent | mean regret | median regret | mean Spearman | median Spearman |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in SCOPES:
        for score in SCORES:
            metrics = payload["metrics"][scope][score]
            lines.append(
                f"| {scope} | {SCORE_LABELS[score]} | "
                f"{metrics['top1_accuracy']:.4%} | "
                f"{metrics['adjacent_accuracy']:.4%} | "
                f"{metrics['mean_regret']:.6e} | "
                f"{metrics['median_regret']:.6e} | "
                f"{metrics['mean_spearman']:.6f} | "
                f"{metrics['median_spearman']:.6f} |"
            )
    observation = payload["observation"]
    if observation["vector_gap_predictive"]:
        predictive_text = (
            "二维 CoVar gap 的 overall top-1 高于随机六选一，且 mean Spearman 为正。"
        )
    else:
        predictive_text = (
            "二维 CoVar gap 未同时满足 overall top-1 高于随机六选一和 mean Spearman 为正。"
        )
    if observation["vector_outperforms_scalar_gap"]:
        comparison_text = (
            "二维分解同时提高了相对标量 r-gap 的 top-1，并降低 mean regret。"
        )
    else:
        comparison_text = (
            "二维分解没有同时优于标量 r-gap 的 top-1 与 mean regret；"
            "因此不能声称二维 gap 提供稳定增益。"
        )
    lines.extend(
        [
            "",
            "## 结论与边界",
            "",
            f"- overall 最强 top-1 分数：{SCORE_LABELS[observation['best_top1_score']]}。",
            f"- overall 最低 mean regret 分数：{SCORE_LABELS[observation['best_regret_score']]}。",
            "- min teacher r 实际偏向最低温度；其高 top-1 与 oracle 在 T=0.5 的多数质量一致，不能单独视为细粒度排序能力。",
            f"- {predictive_text}",
            f"- {comparison_text}",
            f"- 二维 CoVar gap 相对标量 gap 的 top-1 差为 "
            f"{observation['vector_minus_scalar_top1']:+.4%}，mean regret 差为 "
            f"{observation['vector_minus_scalar_mean_regret']:+.6e}。",
            "- 分量尺度在同一分析缓存上估计，结果属于机制诊断；没有独立校准集或跨数据集验证。",
            "- 这些分数只预测局部一步 teachability，不直接等价于完整蒸馏训练后的 mIoU 收益。",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    if not args.candidate_cache.is_file():
        raise FileNotFoundError(args.candidate_cache)
    p2_payload = json.loads(args.p2_json.read_text(encoding="utf-8"))
    if not p2_payload.get("execution_gate_pass"):
        raise RuntimeError("P2 gate did not pass")
    temperatures = tuple(p2_payload["config"]["temperatures"])
    if temperatures != (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        raise RuntimeError("P2 formal temperature grid drifted")

    scales, candidate_row_count = component_scales(
        args.candidate_cache
    )
    accumulator = empty_accumulator()
    group_count = 0
    stages_seen = set()
    finite = True
    for _, rows in iter_candidate_groups(args.candidate_cache):
        result = evaluate_candidate_group(rows, scales, temperatures)
        stages_seen.add(result["stage"])
        add_result(accumulator, result)
        group_count += 1
        finite = finite and all(
            math.isfinite(metric)
            for score_metrics in result["scores"].values()
            for name, metric in score_metrics.items()
            if name != "prediction_temperature"
        )

    metrics, csv_rows = finalize_metrics(accumulator)
    overall = metrics["overall"]
    vector = overall["vector_covar_gap"]
    scalar = overall["scalar_r_gap"]
    random_top1 = 1.0 / len(temperatures)
    best_top1 = max(
        SCORES, key=lambda score: overall[score]["top1_accuracy"]
    )
    best_regret = min(
        SCORES, key=lambda score: overall[score]["mean_regret"]
    )
    observation = {
        "random_top1": random_top1,
        "best_top1_score": best_top1,
        "best_regret_score": best_regret,
        "vector_gap_predictive": (
            vector["top1_accuracy"] > random_top1
            and vector["mean_spearman"] > 0
        ),
        "vector_outperforms_scalar_gap": (
            vector["top1_accuracy"] > scalar["top1_accuracy"]
            and vector["mean_regret"] < scalar["mean_regret"]
        ),
        "vector_minus_scalar_top1": (
            vector["top1_accuracy"] - scalar["top1_accuracy"]
        ),
        "vector_minus_scalar_mean_regret": (
            vector["mean_regret"] - scalar["mean_regret"]
        ),
    }
    expected_groups = sum(
        summary["region_count"]
        for summary in p2_payload["stage_summaries"].values()
    )
    gate_pass = (
        stages_seen == {"early", "middle", "late"}
        and group_count == expected_groups
        and candidate_row_count == group_count * len(temperatures)
        and finite
        and all(
            metrics[scope][score]["region_count"]
            == (
                p2_payload["stage_summaries"][scope]["region_count"]
                if scope != "overall"
                else expected_groups
            )
            for scope in SCOPES
            for score in SCORES
        )
    )
    payload = {
        "stage": "P3",
        "hypothesis": "distance between teacher and current student CoVar coordinates predicts regional teachability",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": {
            "candidate_cache": str(args.candidate_cache),
            "candidate_cache_sha256": file_sha256(
                args.candidate_cache
            ),
            "p2_json": str(args.p2_json),
            "p2_json_sha256": file_sha256(args.p2_json),
            "candidate_rows": candidate_row_count,
            "region_state_groups": group_count,
        },
        "score_semantics": {
            "teacher_min_r": "teacher r(T)",
            "scalar_r_gap": "absolute teacher r(T) minus student r(1)",
            "vector_covar_gap": "Euclidean distance in standardized r_c/r_v coordinates",
            "teacher_student_kl": "KL(teacher(T) || student(1))",
            "selection": "minimum score",
            "component_population_standard_deviation": scales,
            "scale_source": "same unlabeled P2 candidate cache; exploratory normalization",
        },
        "metrics": metrics,
        "observation": observation,
        "execution_gate_pass": gate_pass,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "P3_covar_gap_teachability.csv"
    json_path = args.output_dir / "P3_covar_gap_teachability.json"
    report_path = args.output_dir / "P3_covar_gap_teachability.md"
    write_csv(csv_path, csv_rows)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(
        {
            "component_scales": scales,
            "observation": observation,
            "execution_gate_pass": gate_pass,
        },
        indent=2,
        ensure_ascii=False,
    ))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
