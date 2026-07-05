#!/usr/bin/env python3
import argparse
import datetime as dt
import math
import re
from pathlib import Path


LOG_NAME = "deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Phase E Tout=3.0 seed stability.")
    parser.add_argument("--phase-e-root", required=True)
    parser.add_argument("--phase-c-root", required=True)
    parser.add_argument("--seeds", required=True, help="space separated seed list, including existing 1234")
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
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
        "total_time": None,
        "complete": False,
    }
    if not path.exists():
        return result

    iter_re = re.compile(r"Iters:\s+(\d+)/(\d+)")
    val_re = re.compile(r"Overall validation pixAcc:\s+([0-9.]+), mIoU:\s+([0-9.]+)")
    total_re = re.compile(r"Total training time:\s+(.+)$")
    vals = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = iter_re.search(line)
            if match:
                result["last_iter"] = int(match.group(1))
            match = val_re.search(line)
            if match:
                vals.append(normalize_miou(match.group(2)))
            match = total_re.search(line)
            if match:
                result["total_time"] = match.group(1).strip()

    result["validations"] = len(vals)
    if vals:
        result["best_miou"] = max(vals)
        result["final_miou"] = vals[-1]
    return result


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def fmt(value):
    return "pending" if value is None else f"{value:.4f}"


def main():
    args = parse_args()
    phase_e_root = Path(args.phase_e_root)
    phase_c_root = Path(args.phase_c_root)
    seeds = args.seeds.split()

    rows = []
    for seed in seeds:
        variants = [
            ("off", "phaseC_lc_no_covar_tout3" if seed == "1234" else f"phaseE_seed{seed}_cirkd_no_covar_tout3"),
            ("on", "phaseC_lc_newton_gamma2_repro" if seed == "1234" else f"phaseE_seed{seed}_covar_newton_gamma2_tout3"),
        ]
        for covar, variant in variants:
            root = phase_c_root if seed == "1234" else phase_e_root
            log_path = root / variant / LOG_NAME
            parsed = parse_log(log_path)
            parsed["complete"] = parsed["last_iter"] == args.max_iterations and parsed["total_time"] is not None
            rows.append({
                "seed": seed,
                "covar": covar,
                "variant": variant,
                "log_path": log_path,
                **parsed,
            })

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Phase E Tout=3.0 Seed Stability Report",
        "",
        f"- Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- Max iterations: `{args.max_iterations}`",
        f"- Phase E root: `{phase_e_root}`",
        f"- Existing seed-1234 root: `{phase_c_root}`",
        f"- Seeds: `{', '.join(seeds)}`",
        "",
        "## Per-Seed Results",
        "",
        "| Seed | CoVar | Variant | Last iter | Validations | Best mIoU | Final mIoU | Complete | Total time |",
        "|---:|---|---|---:|---:|---:|---:|---|---|",
    ]

    for row in rows:
        last_iter = "pending" if row["last_iter"] is None else str(row["last_iter"])
        lines.append(
            f"| {row['seed']} | {row['covar']} | `{row['variant']}` | {last_iter} | {row['validations']} | "
            f"{fmt(row['best_miou'])} | {fmt(row['final_miou'])} | {row['complete']} | "
            f"`{row['total_time'] or 'pending'}` |"
        )

    lines.extend(["", "## Aggregate", ""])
    for metric in ["best_miou", "final_miou"]:
        off_values = [row[metric] for row in rows if row["covar"] == "off" and row["complete"]]
        on_values = [row[metric] for row in rows if row["covar"] == "on" and row["complete"]]
        off_mean, off_std = mean_std(off_values)
        on_mean, on_std = mean_std(on_values)
        delta = None if off_mean is None or on_mean is None else on_mean - off_mean
        lines.append(
            f"- `{metric}`: off `{fmt(off_mean)} +/- {fmt(off_std)}`; "
            f"on `{fmt(on_mean)} +/- {fmt(on_std)}`; delta `{fmt(delta)}`."
        )

    lines.extend(["", "## Log Paths", ""])
    for row in rows:
        lines.append(f"- `{row['variant']}`: `{row['log_path']}`")

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
