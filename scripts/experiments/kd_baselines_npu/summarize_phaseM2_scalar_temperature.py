#!/usr/bin/env python3
"""Summarize Phase M2 scalar-temperature controls and Phase M references."""

import argparse
import datetime as dt
import math
import re
import statistics
from pathlib import Path


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ITER_RE = re.compile(r"Iters:\s*(\d+)\s*/\s*(\d+)")
SAMPLE_VAL_RE = re.compile(
    rf"Sample:\s*1449\s*,.*?\bmIoU:\s*({NUMBER})"
)
OVERALL_VAL_RE = re.compile(rf"Overall validation .*?\bmIoU:\s*({NUMBER})")
TIME_RE = re.compile(r"Total training time:\s*(.+?)\s*$")
LOG_NAME = "deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"


def _empty_session():
    return {
        "last_iter": 0,
        "declared_max_iter": 0,
        "validations": [],
        "runtime": "missing",
        "saw_iteration": False,
    }


def parse_training_log(path, expected_iterations=20000):
    """Parse the latest training session in a possibly appended log file."""
    path = Path(path)
    sessions = [_empty_session()]

    if path.is_file():
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                current = sessions[-1]
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
        session["last_iter"] >= expected_iterations
        and session["declared_max_iter"] == expected_iterations
        and session["runtime"] != "missing"
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
    phase_m2_root = root / "runs/logs/kd_baselines_npu/phaseM2_scalar_temperature/triage_20k"
    phase_m_root = root / "runs/logs/kd_baselines_npu/phaseM_cwd_covar/triage_20k"
    variants = [
        (
            "CWD, Tout=3, scalar KD T=0.5",
            phase_m2_root / "cwd_tout3_kdtemp0p5_20k_seed1234" / LOG_NAME,
        ),
        (
            "CWD, Tout=3, scalar KD T=0.6",
            phase_m2_root / "cwd_tout3_kdtemp0p6_20k_seed1234" / LOG_NAME,
        ),
        (
            "CWD, Tout=3, scalar KD T=1.0",
            phase_m_root / "cwd_tout3_fixed_20k_seed1234" / LOG_NAME,
        ),
        (
            "CWD, Tout=3, Newton CoVar",
            phase_m_root / "cwd_covar_newton_tout3_20k_seed1234" / LOG_NAME,
        ),
    ]
    results = [(label, parse_training_log(path)) for label, path in variants]

    lines = [
        "# Phase M2 Scalar-Temperature Controls",
        "",
        f"- Generated: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        "- Dataset/model: Pascal VOC, DeepLabV3-ResNet101 teacher, DeepLabV3-MobileNetV3-Small student.",
        "- Seed/budget: `1234`, `20000` iterations.",
        "- Shared recipe: CWD with teacher-output temperature `Tout=3`; only the logit KD temperature mechanism differs.",
        "- Metrics are rendered to 6 decimal places. Historical Phase M logs contain only 3-decimal mIoU values, so trailing zeros do not imply added precision.",
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

    result_by_label = {label: result for label, result in results}
    t05 = result_by_label["CWD, Tout=3, scalar KD T=0.5"]
    t06 = result_by_label["CWD, Tout=3, scalar KD T=0.6"]
    t10 = result_by_label["CWD, Tout=3, scalar KD T=1.0"]
    covar = result_by_label["CWD, Tout=3, Newton CoVar"]
    lines.extend(["", "## Controlled Deltas", ""])
    if all(item["complete"] for item in (t05, t06, t10, covar)):
        scalar_controls = [(0.5, t05), (0.6, t06), (1.0, t10)]
        best_temperature, best_scalar = max(
            scalar_controls, key=lambda item: item[1]["final"]
        )
        lines.extend(
            [
                f"- CoVar - scalar T=0.5 final mIoU: `{covar['final'] - t05['final']:+.6f}`.",
                f"- CoVar - scalar T=0.6 final mIoU: `{covar['final'] - t06['final']:+.6f}`.",
                f"- CoVar - scalar T=1.0 final mIoU: `{covar['final'] - t10['final']:+.6f}`.",
                f"- Strongest scalar control by final mIoU: `T={best_temperature:.1f}` ({best_scalar['final']:.6f}); CoVar delta: `{covar['final'] - best_scalar['final']:+.6f}`.",
            ]
        )
    else:
        complete_count = sum(result["complete"] for _, result in results)
        lines.append(
            f"- Incomplete comparison: `{complete_count}/4` runs are complete; no scalar-control conclusion yet."
        )

    lines.extend(["", "## Log Paths", ""])
    for label, result in results:
        lines.append(f"- {label}: `{result['path']}`")

    write_report(report, lines)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Summarize Phase M2 T=0.5/T=0.6 and Phase M T=1/CoVar logs."
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Output Markdown path (default: reports/2026-07-12_phaseM2_scalar_temperature.md under --root).",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    report = args.report or (
        root / "reports/2026-07-12_phaseM2_scalar_temperature.md"
    )
    if not report.is_absolute():
        report = root / report
    results = summarize(root, report)
    complete_count = sum(result["complete"] for _, result in results)
    print(f"[kd-baselines-npu] wrote Phase M2 report: {report}")
    print(f"[kd-baselines-npu] complete runs: {complete_count}/4")


if __name__ == "__main__":
    main()
