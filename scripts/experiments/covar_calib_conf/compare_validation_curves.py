#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


ITER_RE = re.compile(r"Iters:\s+(\d+)/(\d+)")
VAL_START_RE = re.compile(r"Start validation, Total sample:\s+(\d+)")
VAL_RE = re.compile(r"Sample:\s+(\d+), Validation pixAcc:\s+([0-9.]+), mIoU:\s+([0-9.]+)")


def parse_validations(path, variant=None):
    current_step = ""
    active = None
    rows = []

    def finalize():
        nonlocal active
        if active and active.get("complete") and active.get("miou") is not None:
            rows.append({
                "variant": variant or path.parent.name or path.stem,
                "step": int(active["step"]) if active["step"] else -1,
                "pixacc": active["pixacc"],
                "miou": active["miou"],
                "log_path": str(path),
            })
        active = None

    with path.open("r", errors="ignore") as f:
        for line in f:
            m = ITER_RE.search(line)
            if m:
                if active and active.get("complete"):
                    finalize()
                current_step = m.group(1)

            m = VAL_START_RE.search(line)
            if m:
                finalize()
                active = {
                    "step": current_step,
                    "total": int(m.group(1)),
                    "sample": 0,
                    "pixacc": None,
                    "miou": None,
                    "complete": False,
                }

            m = VAL_RE.search(line)
            if not m:
                continue
            sample = int(m.group(1))
            pixacc = float(m.group(2))
            miou = float(m.group(3))
            if active is None:
                active = {
                    "step": current_step,
                    "total": sample,
                    "sample": sample,
                    "pixacc": pixacc,
                    "miou": miou,
                    "complete": False,
                }
            else:
                active["sample"] = sample
                active["pixacc"] = pixacc
                active["miou"] = miou
            if sample >= int(active.get("total") or sample):
                active["complete"] = True

    if active and active.get("complete"):
        finalize()
    return rows


def find_logs(root):
    root = Path(root)
    logs = sorted(root.glob("*/*_log.txt"))
    logs += sorted(root.glob("*/*.log"))
    return sorted(set(logs))


def main():
    parser = argparse.ArgumentParser(description="Compare complete validation curves at matched iteration steps.")
    parser.add_argument("--root", default="data/winycg/checkpoints/cirkd_checkpoints/voc/covar_calib_conf")
    parser.add_argument(
        "--logs",
        nargs="*",
        default=None,
        help="Optional explicit log files to include; variants use the log stem.",
    )
    parser.add_argument("--baseline", default="01_baseline_no_covar")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows = []
    for path in find_logs(args.root):
        rows.extend(parse_validations(path))
    if args.logs:
        for path in args.logs:
            log_path = Path(path)
            rows.extend(parse_validations(log_path, variant=log_path.stem))

    baseline_by_step = {
        row["step"]: row["miou"]
        for row in rows
        if row["variant"] == args.baseline and row["step"] >= 0
    }

    out_rows = []
    for row in sorted(rows, key=lambda r: (r["step"], r["variant"])):
        base = baseline_by_step.get(row["step"])
        delta = "" if base is None else row["miou"] - base
        out_rows.append({
            "step": row["step"],
            "variant": row["variant"],
            "pixacc": row["pixacc"],
            "miou": row["miou"],
            "baseline_miou": "" if base is None else base,
            "delta_vs_baseline": delta,
        })

    fieldnames = ["step", "variant", "pixacc", "miou", "baseline_miou", "delta_vs_baseline"]
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"Wrote {len(out_rows)} rows to {output}")
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


if __name__ == "__main__":
    main()
