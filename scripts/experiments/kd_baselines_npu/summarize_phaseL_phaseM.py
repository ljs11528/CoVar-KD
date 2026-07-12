#!/usr/bin/env python3
import argparse
import datetime as dt
import math
import re
import statistics
from pathlib import Path


ITER_RE = re.compile(r"Iters:\s+(\d+)/(\d+)")
SAMPLE_VAL_RE = re.compile(r"Sample:\s+1449,.*mIoU:\s+([0-9.]+)")
OVERALL_VAL_RE = re.compile(r"Overall validation .*mIoU:\s+([0-9.]+)")
TIME_RE = re.compile(r"Total training time:\s+(.+)$")


def parse_training_log(path):
    path = Path(path)
    result = {
        "path": path,
        "last_iter": 0,
        "max_iter": 0,
        "validations": 0,
        "best": math.nan,
        "best_iter": 0,
        "final": math.nan,
        "runtime": "missing",
        "complete": False,
    }
    if not path.is_file():
        return result

    best = -math.inf
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            iter_match = ITER_RE.search(line)
            if iter_match:
                result["last_iter"] = int(iter_match.group(1))
                result["max_iter"] = int(iter_match.group(2))

            val_match = SAMPLE_VAL_RE.search(line) or OVERALL_VAL_RE.search(line)
            if val_match:
                value = float(val_match.group(1))
                if value > 1.0:
                    value /= 100.0
                result["validations"] += 1
                result["final"] = value
                if value > best:
                    best = value
                    result["best"] = value
                    result["best_iter"] = result["last_iter"]

            time_match = TIME_RE.search(line)
            if time_match:
                result["runtime"] = time_match.group(1).strip()

    result["complete"] = (
        result["max_iter"] > 0
        and result["last_iter"] >= result["max_iter"]
        and result["runtime"] != "missing"
    )
    return result


def format_score(value):
    return "missing" if math.isnan(value) else f"{value:.4f}"


def mean_sd(values):
    return statistics.mean(values), statistics.pstdev(values)


def write_report(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def summarize_phase_l(root, report):
    log_name = "deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"
    cwd_logs = {
        1234: root / "runs/logs/kd_baselines_npu/phaseK_voc_80k/cwd_80k" / log_name,
        2025: root / "runs/logs/kd_baselines_npu/phaseL_cwd_seed_stability/cwd_80k_seed2025" / log_name,
        3407: root / "runs/logs/kd_baselines_npu/phaseL_cwd_seed_stability/cwd_80k_seed3407" / log_name,
    }
    covar_logs = {
        1234: root / "data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseC_80k/phaseC_lc_newton_gamma2_repro" / log_name,
        2025: root / "data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseE_tout3_seed_stability/phaseE_seed2025_covar_newton_gamma2_tout3" / log_name,
        3407: root / "data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseE_tout3_seed_stability/phaseE_seed3407_covar_newton_gamma2_tout3" / log_name,
    }
    cwd = {seed: parse_training_log(path) for seed, path in cwd_logs.items()}
    covar = {seed: parse_training_log(path) for seed, path in covar_logs.items()}

    lines = [
        "# Phase L CWD Seed Stability Report",
        "",
        f"- Generated: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        "- Seeds: `1234, 2025, 3407`",
        "- Budget: `80000` iterations",
        "- CWD recipe: task CE + KD + adversarial KD + CWD feature/logit",
        "",
        "## Per-Seed Results",
        "",
        "| Seed | CWD best | Best iter | CWD final | CoVar/CIRKD best | CoVar/CIRKD final | CWD-CoVar best | CWD-CoVar final | Complete |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for seed in (1234, 2025, 3407):
        c = cwd[seed]
        v = covar[seed]
        best_delta = c["best"] - v["best"]
        final_delta = c["final"] - v["final"]
        lines.append(
            f"| {seed} | {format_score(c['best'])} | {c['best_iter']} | {format_score(c['final'])} "
            f"| {format_score(v['best'])} | {format_score(v['final'])} | {best_delta:+.4f} "
            f"| {final_delta:+.4f} | {c['complete']} |"
        )

    complete_cwd = [cwd[seed] for seed in (1234, 2025, 3407) if cwd[seed]["complete"]]
    lines.extend(["", "## Aggregate", ""])
    if len(complete_cwd) == 3:
        best_mean, best_sd = mean_sd([item["best"] for item in complete_cwd])
        final_mean, final_sd = mean_sd([item["final"] for item in complete_cwd])
        lines.extend([
            f"- CWD best mIoU: `{best_mean:.4f} +/- {best_sd:.4f}`.",
            f"- CWD final mIoU: `{final_mean:.4f} +/- {final_sd:.4f}`.",
            "- Compare method-level means cautiously: CWD and CoVar use different base recipes.",
        ])
    else:
        lines.append(f"- Incomplete: `{len(complete_cwd)}/3` CWD runs are complete.")

    lines.extend(["", "## Log Paths", ""])
    for seed, path in cwd_logs.items():
        lines.append(f"- CWD seed {seed}: `{path}`")
    write_report(report, lines)


def summarize_phase_m(root, report):
    log_name = "deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"
    log_root = root / "runs/logs/kd_baselines_npu/phaseM_cwd_covar/triage_20k"
    fixed_path = log_root / "cwd_tout3_fixed_20k_seed1234" / log_name
    covar_path = log_root / "cwd_covar_newton_tout3_20k_seed1234" / log_name
    baseline_path = root / "runs/logs/kd_baselines_npu/phaseJ_voc_20k/cwd_20k" / log_name
    fixed = parse_training_log(fixed_path)
    covar = parse_training_log(covar_path)
    baseline = parse_training_log(baseline_path)

    lines = [
        "# Phase M CWD + CoVar 20k Triage Report",
        "",
        f"- Generated: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        "- Seed: `1234`",
        "- Budget: `20000` iterations",
        "- Controlled change: only the logit KD temperature mechanism differs between fixed and CoVar rows.",
        "",
        "## Results",
        "",
        "| Variant | Best mIoU | Best iter | Final mIoU | Validations | Runtime | Complete |",
        "|---|---:|---:|---:|---:|---|---|",
        f"| CWD, Tout=1 historical | {format_score(baseline['best'])} | {baseline['best_iter']} | {format_score(baseline['final'])} | {baseline['validations']} | {baseline['runtime']} | {baseline['complete']} |",
        f"| CWD, Tout=3 fixed | {format_score(fixed['best'])} | {fixed['best_iter']} | {format_score(fixed['final'])} | {fixed['validations']} | {fixed['runtime']} | {fixed['complete']} |",
        f"| CWD, Tout=3 Newton CoVar | {format_score(covar['best'])} | {covar['best_iter']} | {format_score(covar['final'])} | {covar['validations']} | {covar['runtime']} | {covar['complete']} |",
        "",
        "## Controlled Delta",
        "",
    ]
    if fixed["complete"] and covar["complete"]:
        lines.extend([
            f"- Best mIoU delta, CoVar - fixed: `{covar['best'] - fixed['best']:+.4f}`.",
            f"- Final mIoU delta, CoVar - fixed: `{covar['final'] - fixed['final']:+.4f}`.",
        ])
    else:
        lines.append("- Incomplete pair; no controlled conclusion yet.")
    lines.extend([
        "",
        "## Log Paths",
        "",
        f"- Historical CWD: `{baseline_path}`",
        f"- Fixed Tout=3: `{fixed_path}`",
        f"- Newton CoVar: `{covar_path}`",
    ])
    write_report(report, lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("phase-l", "phase-m"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    if args.phase == "phase-l":
        summarize_phase_l(root, args.report)
    else:
        summarize_phase_m(root, args.report)


if __name__ == "__main__":
    main()
