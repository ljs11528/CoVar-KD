#!/usr/bin/env python3
"""Summarize the paired Phase N 80k scalar-temperature comparison."""

import argparse
import datetime as dt
import math
import re
import statistics
from pathlib import Path


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ITER_RE = re.compile(r"Iters:\s*(\d+)\s*/\s*(\d+)")
SAMPLE_VAL_RE = re.compile(rf"Sample:\s*1449\s*,.*?\bmIoU:\s*({NUMBER})")
OVERALL_VAL_RE = re.compile(rf"Overall validation .*?\bmIoU:\s*({NUMBER})")
TIME_RE = re.compile(r"Total training time:\s*(.+?)\s*$")
LOG_NAME = "deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"
EXPECTED_ITERATIONS = 80000


def _empty_session():
    return {
        "last_iter": 0,
        "declared_max_iter": 0,
        "validations": [],
        "runtime": "missing",
        "saw_iteration": False,
        "saw_namespace": False,
    }


def parse_training_log(path, expected_iterations=EXPECTED_ITERATIONS):
    """Parse only the latest session from a log that may contain restarts."""
    path = Path(path)
    sessions = [_empty_session()]

    if path.is_file():
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                current = sessions[-1]
                if "Namespace(" in line:
                    if current["saw_namespace"] or current["saw_iteration"] or current["validations"] or current["runtime"] != "missing":
                        current = _empty_session()
                        sessions.append(current)
                    current["saw_namespace"] = True
                    continue
                iter_match = ITER_RE.search(line)
                if iter_match:
                    iteration = int(iter_match.group(1))
                    maximum = int(iter_match.group(2))
                    if current["saw_iteration"] and iteration < current["last_iter"]:
                        current = _empty_session()
                        sessions.append(current)
                    current["last_iter"] = iteration
                    current["declared_max_iter"] = maximum
                    current["saw_iteration"] = True

                val_match = SAMPLE_VAL_RE.search(line) or OVERALL_VAL_RE.search(line)
                if val_match:
                    value = float(val_match.group(1))
                    if value > 1.0:
                        value /= 100.0
                    current["validations"].append((current["last_iter"], value))

                time_match = TIME_RE.search(line)
                if time_match:
                    current["runtime"] = time_match.group(1).strip()

    session = sessions[-1]
    validations = session["validations"]
    best = math.nan
    best_iter = None
    final = math.nan
    last10_mean = math.nan
    if validations:
        best_iter, best = max(validations, key=lambda item: item[1])
        final = validations[-1][1]
        last10_mean = statistics.mean(value for _, value in validations[-10:])

    complete = (
        session["saw_namespace"]
        and session["last_iter"] == expected_iterations
        and session["declared_max_iter"] == expected_iterations
        and session["runtime"] != "missing"
        and len(validations) == expected_iterations // 800
        and validations[-1][0] == expected_iterations
        and bool(validations)
    )
    return {
        "path": path,
        "last_iter": session["last_iter"],
        "max_iter": session["declared_max_iter"],
        "validations": len(validations),
        "best": best,
        "best_iter": best_iter,
        "final": final,
        "last10_mean": last10_mean,
        "runtime": session["runtime"],
        "complete": complete,
    }


def format_metric(value):
    return "missing" if math.isnan(value) else f"{value:.6f}"


def format_iter(value):
    return "missing" if value is None else str(value)


def write_report(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def summarize(root, report):
    root = Path(root)
    log_root = root / "runs/logs/kd_baselines_npu/phaseN_scalar_temperature_80k/main_80k"
    variants = [
        (
            "CWD, Tout=3, scalar KD T=1.0",
            log_root / "cwd_tout3_kdtemp1p0_80k_seed1234" / LOG_NAME,
        ),
        (
            "CWD, Tout=3, scalar KD T=0.6",
            log_root / "cwd_tout3_kdtemp0p6_80k_seed1234" / LOG_NAME,
        ),
    ]
    results = [(label, parse_training_log(path)) for label, path in variants]

    lines = [
        "# Phase N Scalar-Temperature 80k Comparison",
        "",
        f"- Generated: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        "- Dataset/model: Pascal VOC, DeepLabV3-ResNet101 teacher, DeepLabV3-MobileNetV3-Small student.",
        "- Seed/budget: `1234`, `80000` iterations.",
        "- Controlled change: both runs use the official CWD recipe and `Tout=3`; only scalar logit KD temperature differs.",
        "",
        "## Results",
        "",
        "| Variant | Best mIoU | Best iter | Final mIoU | Last-10 val mean | Validations | Runtime | Complete |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for label, result in results:
        lines.append(
            f"| {label} | {format_metric(result['best'])} | "
            f"{format_iter(result['best_iter'])} | {format_metric(result['final'])} | "
            f"{format_metric(result['last10_mean'])} | {result['validations']} | "
            f"{result['runtime']} | {str(result['complete']).lower()} |"
        )

    t10 = results[0][1]
    t06 = results[1][1]
    lines.extend(["", "## Paired Deltas (T=0.6 - T=1.0)", ""])
    if t10["complete"] and t06["complete"]:
        lines.extend(
            [
                f"- Best mIoU: `{t06['best'] - t10['best']:+.6f}`.",
                f"- Final mIoU: `{t06['final'] - t10['final']:+.6f}`.",
                f"- Last-10 validation mean: `{t06['last10_mean'] - t10['last10_mean']:+.6f}`.",
            ]
        )
    else:
        complete_count = sum(result["complete"] for _, result in results)
        lines.append(
            f"- Incomplete comparison: `{complete_count}/2` runs are complete; no paired conclusion yet."
        )

    lines.extend(["", "## Log Paths", ""])
    for label, result in results:
        lines.append(f"- {label}: `{result['path']}`")

    write_report(report, lines)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument(
        "--report",
        type=Path,
        help=(
            "Output Markdown path (default: "
            "reports/2026-07-13_phaseN_scalar_temperature_80k.md under --root)."
        ),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    report = args.report or (
        root / "reports/2026-07-13_phaseN_scalar_temperature_80k.md"
    )
    if not report.is_absolute():
        report = root / report
    results = summarize(root, report)
    complete_count = sum(result["complete"] for _, result in results)
    print(f"[kd-baselines-npu] wrote Phase N report: {report}")
    print(f"[kd-baselines-npu] complete runs: {complete_count}/2")


if __name__ == "__main__":
    main()
