#!/usr/bin/env python3
import argparse
import datetime as dt
import re
from pathlib import Path


LOG_NAME = "deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"


CELLS = [
    ("1.0", "off", "phaseD_cirkd_no_covar_tout1", "phaseD"),
    ("1.0", "on", "phaseD_covar_newton_gamma2_tout1", "phaseD"),
    ("3.0", "off", "phaseC_lc_no_covar_tout3", "phaseC"),
    ("3.0", "on", "phaseC_lc_newton_gamma2_repro", "phaseC"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize the Tout x CoVar NPU main table.")
    parser.add_argument("--phase-d-root", required=True)
    parser.add_argument("--phase-c-root", required=True)
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
        "last_temp": None,
    }
    if not path.exists():
        return result

    iter_re = re.compile(r"Iters:\s+(\d+)/(\d+)")
    val_re = re.compile(r"Overall validation pixAcc:\s+([0-9.]+), mIoU:\s+([0-9.]+)")
    total_re = re.compile(r"Total training time:\s+(.+)$")
    temp_re = re.compile(r"T_mean:\s+([0-9.]+).*T_min:\s+([0-9.]+).*T_max:\s+([0-9.]+)")

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
            match = temp_re.search(line)
            if match:
                result["last_temp"] = tuple(float(match.group(i)) for i in range(1, 4))

    result["validations"] = len(vals)
    if vals:
        result["best_miou"] = max(vals)
        result["final_miou"] = vals[-1]
    return result


def fmt(value):
    return "pending" if value is None else f"{value:.4f}"


def main():
    args = parse_args()
    roots = {
        "phaseD": Path(args.phase_d_root),
        "phaseC": Path(args.phase_c_root),
    }
    rows = []
    for tout, covar, variant, root_key in CELLS:
        log_path = roots[root_key] / variant / LOG_NAME
        parsed = parse_log(log_path)
        rows.append((tout, covar, variant, log_path, parsed))

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Phase D NPU Main Table Report",
        "",
        f"- Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- Max iterations: `{args.max_iterations}`",
        f"- Phase D root: `{roots['phaseD']}`",
        f"- Phase C 80k root: `{roots['phaseC']}`",
        "",
        "## Tout x CoVar Main Table",
        "",
        "| Teacher output temp | CoVar | Variant | Last iter | Validations | Best mIoU | Final mIoU | Total time | Last T mean/min/max |",
        "|---:|---|---|---:|---:|---:|---:|---|---|",
    ]

    for tout, covar, variant, log_path, parsed in rows:
        temp = "n/a"
        if parsed["last_temp"] is not None:
            temp = "{:.4f}/{:.4f}/{:.4f}".format(*parsed["last_temp"])
        last_iter = "pending" if parsed["last_iter"] is None else str(parsed["last_iter"])
        lines.append(
            f"| {tout} | {covar} | `{variant}` | {last_iter} | {parsed['validations']} | "
            f"{fmt(parsed['best_miou'])} | {fmt(parsed['final_miou'])} | "
            f"`{parsed['total_time'] or 'pending'}` | {temp} |"
        )

    lines.extend([
        "",
        "## Interpretation Guide",
        "",
        "- `Tout=1.0, CoVar=off` is the same-code original CIRKD control.",
        "- `Tout=1.0, CoVar=on` tests whether CoVar helps without teacher softening.",
        "- `Tout=3.0, CoVar=off` isolates teacher softening.",
        "- `Tout=3.0, CoVar=on` is the current main method.",
        "- The key claim is strongest if CoVar improves over no-CoVar at both `Tout=1.0` and `Tout=3.0`.",
        "",
        "## Log Paths",
        "",
    ])

    for _, _, variant, log_path, _ in rows:
        lines.append(f"- `{variant}`: `{log_path}`")

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
