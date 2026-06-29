#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import re
from pathlib import Path


LOG_NAME = "deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Phase C NPU experiment logs.")
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    return parser.parse_args()


def normalize_miou(value):
    value = float(value)
    return value / 100.0 if value > 1.0 else value


def parse_log(path):
    result = {
        "exists": path.exists(),
        "last_iter": None,
        "best_miou": None,
        "final_miou": None,
        "num_validations": 0,
        "last_temp": None,
        "last_newton": None,
        "total_time": None,
    }
    if not path.exists():
        return result

    iter_re = re.compile(r"Iters:\s+(\d+)/(\d+)")
    val_re = re.compile(r"Overall validation pixAcc:\s+([0-9.]+), mIoU:\s+([0-9.]+)")
    temp_re = re.compile(r"T_mean:\s+([0-9.]+).*T_min:\s+([0-9.]+).*T_max:\s+([0-9.]+)")
    newton_re = re.compile(r"(NewtonT:[^\n]+)")
    total_re = re.compile(r"Total training time:\s+(.+)$")

    miou_values = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            iter_match = iter_re.search(line)
            if iter_match:
                result["last_iter"] = int(iter_match.group(1))

            val_match = val_re.search(line)
            if val_match:
                miou_values.append(normalize_miou(val_match.group(2)))

            temp_match = temp_re.search(line)
            if temp_match:
                result["last_temp"] = tuple(float(temp_match.group(i)) for i in range(1, 4))

            newton_match = newton_re.search(line)
            if newton_match:
                result["last_newton"] = newton_match.group(1).strip()

            total_match = total_re.search(line)
            if total_match:
                result["total_time"] = total_match.group(1).strip()

    result["num_validations"] = len(miou_values)
    if miou_values:
        result["best_miou"] = max(miou_values)
        result["final_miou"] = miou_values[-1]
    return result


def format_float(value):
    if value is None:
        return "pending"
    return f"{value:.4f}"


def main():
    args = parse_args()
    save_root = Path(args.save_root)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in args.variants:
        log_path = save_root / variant / LOG_NAME
        parsed = parse_log(log_path)
        rows.append((variant, log_path, parsed))

    lines = [
        "# Phase C NPU Triage Report",
        "",
        f"- Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- Save root: `{save_root}`",
        f"- Max iterations per variant: `{args.max_iterations}`",
        f"- Variants: `{', '.join(args.variants)}`",
        "",
        "## Summary",
        "",
        "| Variant | Last iter | Validations | Best mIoU | Final mIoU | Last T mean/min/max |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for variant, _, parsed in rows:
        if parsed["last_temp"] is None:
            temp = "n/a"
        else:
            temp = "{:.4f}/{:.4f}/{:.4f}".format(*parsed["last_temp"])
        last_iter = "pending" if parsed["last_iter"] is None else str(parsed["last_iter"])
        lines.append(
            f"| `{variant}` | {last_iter} | {parsed['num_validations']} | "
            f"{format_float(parsed['best_miou'])} | {format_float(parsed['final_miou'])} | {temp} |"
        )

    lines.extend([
        "",
        "## Details",
        "",
    ])

    for variant, log_path, parsed in rows:
        lines.extend([
            f"### {variant}",
            "",
            f"- Log: `{log_path}`",
            f"- Training complete: `{parsed['total_time'] is not None}`",
            f"- Total time: `{parsed['total_time'] or 'pending'}`",
            f"- Last Newton diagnostic: `{parsed['last_newton'] or 'n/a'}`",
            "",
        ])

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
