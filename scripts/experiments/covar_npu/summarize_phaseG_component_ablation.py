#!/usr/bin/env python3
import argparse
import datetime as dt
import re
from pathlib import Path


LOG_NAME = "deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Phase G component ablation triage.")
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--seed", default="1234")
    return parser.parse_args()


def normalize_miou(value):
    value = float(value)
    return value / 100.0 if value > 1.0 else value


def parse_log(path):
    result = {
        "exists": path.exists(),
        "last_iter": None,
        "validations": 0,
        "best_miou": None,
        "final_miou": None,
        "final_pixacc": None,
        "total_time": None,
        "last_temp": None,
        "last_mode": None,
        "last_newton": None,
        "complete": False,
    }
    if not path.exists():
        return result

    iter_re = re.compile(r"Iters:\s+(\d+)/(\d+)")
    val_re = re.compile(r"Overall validation pixAcc:\s+([0-9.]+), mIoU:\s+([0-9.]+)")
    total_re = re.compile(r"Total training time:\s+(.+)$")
    temp_re = re.compile(r"T_mean:\s+([0-9.]+).*T_min:\s+([0-9.]+).*T_max:\s+([0-9.]+)")
    mode_re = re.compile(r"NewtonT\[([^\]]+)\]")
    newton_re = re.compile(r"(NewtonT\[[^\]]+\]:[^\n]+)")

    vals = []
    pixaccs = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = iter_re.search(line)
            if match:
                result["last_iter"] = int(match.group(1))
            match = val_re.search(line)
            if match:
                pixaccs.append(normalize_miou(match.group(1)))
                vals.append(normalize_miou(match.group(2)))
            match = total_re.search(line)
            if match:
                result["total_time"] = match.group(1).strip()
            match = temp_re.search(line)
            if match:
                result["last_temp"] = tuple(float(match.group(i)) for i in range(1, 4))
            match = mode_re.search(line)
            if match:
                result["last_mode"] = match.group(1)
            match = newton_re.search(line)
            if match:
                result["last_newton"] = match.group(1).strip()

    result["validations"] = len(vals)
    if vals:
        result["best_miou"] = max(vals)
        result["final_miou"] = vals[-1]
        result["final_pixacc"] = pixaccs[-1]
    return result


def fmt(value):
    return "pending" if value is None else f"{value:.4f}"


def temp_fmt(value):
    if value is None:
        return "n/a"
    return "{:.4f}/{:.4f}/{:.4f}".format(*value)


def main():
    args = parse_args()
    save_root = Path(args.save_root)
    variants = [
        ("off", f"phaseG_triage_no_covar_tout3_seed{args.seed}"),
        ("confidence", f"phaseG_triage_covar_confidence_only_tout3_seed{args.seed}"),
        ("variance", f"phaseG_triage_covar_variance_only_tout3_seed{args.seed}"),
        ("full", f"phaseG_triage_covar_full_tout3_seed{args.seed}"),
    ]

    rows = []
    for label, variant in variants:
        log_path = save_root / variant / LOG_NAME
        parsed = parse_log(log_path)
        parsed["complete"] = parsed["last_iter"] == args.max_iterations and parsed["total_time"] is not None
        rows.append({
            "label": label,
            "variant": variant,
            "log_path": log_path,
            **parsed,
        })

    off_final = next((row["final_miou"] for row in rows if row["label"] == "off"), None)
    full_final = next((row["final_miou"] for row in rows if row["label"] == "full"), None)

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase G Component Ablation Triage Report",
        "",
        f"- Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- Save root: `{save_root}`",
        f"- Max iterations: `{args.max_iterations}`",
        f"- Seed: `{args.seed}`",
        "",
        "## Summary",
        "",
        "| Label | Variant | Last iter | Validations | Best mIoU | Final mIoU | Delta vs off | Delta vs full | Final pixAcc | T mean/min/max | Complete |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    for row in rows:
        final = row["final_miou"]
        delta_off = None if final is None or off_final is None else final - off_final
        delta_full = None if final is None or full_final is None else final - full_final
        last_iter = "pending" if row["last_iter"] is None else str(row["last_iter"])
        lines.append(
            f"| {row['label']} | `{row['variant']}` | {last_iter} | {row['validations']} | "
            f"{fmt(row['best_miou'])} | {fmt(row['final_miou'])} | {fmt(delta_off)} | {fmt(delta_full)} | "
            f"{fmt(row['final_pixacc'])} | {temp_fmt(row['last_temp'])} | {row['complete']} |"
        )

    lines.extend(["", "## Details", ""])
    for row in rows:
        lines.extend([
            f"### {row['label']}",
            "",
            f"- Variant: `{row['variant']}`",
            f"- Log: `{row['log_path']}`",
            f"- Last reliability mode: `{row['last_mode'] or 'n/a'}`",
            f"- Total time: `{row['total_time'] or 'pending'}`",
            f"- Last Newton diagnostic: `{row['last_newton'] or 'n/a'}`",
            "",
        ])

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
