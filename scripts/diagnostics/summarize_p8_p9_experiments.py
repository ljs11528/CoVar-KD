#!/usr/bin/env python3
"""Summarize the P8 CE-only baseline and P9 second-pair temperature response."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from statistics import mean, stdev


SEEDS = (1234, 2025, 3407)
MILESTONES = (20000, 40000, 60000, 80000)
P7_TEMPERATURES = ("0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "2.0")
P9_STAGE1_TEMPERATURES = ("0.25", "0.5", "1.0", "1.5", "2.0")
P9_STAGE2_TEMPERATURES = ("0.75", "1.25")
P9_ALL_TEMPERATURES = P7_TEMPERATURES
P9_INTERIOR_TEMPERATURES = ("0.5", "1.0", "1.5")
SMALL_LOG_NAME = (
    "deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"
)
LARGE_LOG_NAME = (
    "deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_large_log.txt"
)
ITERATION_RE = re.compile(r"Iters:\s*(?P<iteration>\d+)/(?P<maximum>\d+)")
VALIDATION_RE = re.compile(
    r"Overall validation pixAcc: (?P<pixacc>[-+0-9.eE]+), "
    r"mIoU: (?P<miou>[-+0-9.eE]+)"
)
TIME_RE = re.compile(r"Total training time: (?P<time>[^\n(]+)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p7-json",
        type=Path,
        default=Path(
            "reports/covar_match/"
            "P7_fixed_temperature_response_lower_boundary.json"
        ),
    )
    parser.add_argument(
        "--covar-json",
        type=Path,
        default=Path("reports/covar_match/P7A_covar_statistics.json"),
    )
    parser.add_argument(
        "--p8-root",
        type=Path,
        default=Path("runs/covar_match/P8_ce_only_baseline"),
    )
    parser.add_argument(
        "--p9-root",
        type=Path,
        default=Path("runs/covar_match/P9_pair2_temperature_response"),
    )
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument(
        "--p8-output-json",
        type=Path,
        default=Path("reports/covar_match/P8_ce_only_baseline.json"),
    )
    parser.add_argument(
        "--p8-output-markdown",
        type=Path,
        default=Path("reports/covar_match/P8_ce_only_baseline.md"),
    )
    parser.add_argument(
        "--p9-output-json",
        type=Path,
        default=Path(
            "reports/covar_match/P9_pair2_temperature_response.json"
        ),
    )
    parser.add_argument(
        "--p9-output-markdown",
        type=Path,
        default=Path(
            "reports/covar_match/P9_pair2_temperature_response.md"
        ),
    )
    return parser.parse_args()


def parse_log(text):
    current_iteration = None
    maximum = None
    trajectory = {}
    for line in text.splitlines():
        iteration_match = ITERATION_RE.search(line)
        if iteration_match:
            current_iteration = int(iteration_match.group("iteration"))
            maximum = int(iteration_match.group("maximum"))
        validation_match = VALIDATION_RE.search(line)
        if validation_match:
            if current_iteration is None:
                raise RuntimeError("validation appeared before a logged iteration")
            if current_iteration in trajectory:
                raise RuntimeError(
                    f"duplicate validation at iteration {current_iteration}"
                )
            trajectory[current_iteration] = {
                "pixacc_percent": float(validation_match.group("pixacc")),
                "miou_percent": float(validation_match.group("miou")),
            }
    return maximum, trajectory


def sample_statistics(values):
    values = [float(value) for value in values]
    if len(values) < 2:
        raise ValueError("sample standard deviation requires at least two values")
    return {
        "values": values,
        "mean": mean(values),
        "sample_std": stdev(values),
    }


def p8_variant(seed):
    if seed not in SEEDS:
        raise ValueError(f"unexpected P8 seed: {seed}")
    return f"ce_only_80k_seed{seed}"


def temperature_tag(temperature):
    tags = {
        "0.25": "T0p25",
        "0.5": "T0p5",
        "0.75": "T0p75",
        "1.0": "T1p0",
        "1.25": "T1p25",
        "1.5": "T1p5",
        "2.0": "T2p0",
    }
    try:
        return tags[str(temperature)]
    except KeyError as error:
        raise ValueError(f"unexpected temperature: {temperature}") from error


def p9_variant(temperature, seed):
    if seed not in SEEDS:
        raise ValueError(f"unexpected P9 seed: {seed}")
    return f"fixed_{temperature_tag(temperature)}_80k_seed{seed}"


def read_run(
    root,
    variant,
    log_name,
    seed,
    student_backbone,
    lambda_kd,
    temperature,
):
    log_path = root / "logs" / variant / log_name
    state_dir = root / "checkpoints" / variant
    latest_state = state_dir / "training_state_latest.pth"
    if not log_path.is_file() or not latest_state.is_file():
        raise RuntimeError(f"{variant}: missing log or latest training state")
    text = log_path.read_text(encoding="utf-8")
    maximum, trajectory = parse_log(text)
    times = TIME_RE.findall(text)
    checks = {
        "maximum_80k": maximum == 80000,
        "iteration_80k": "Iters: 80000/80000" in text,
        "milestones": tuple(sorted(trajectory)) == MILESTONES,
        "training_time": bool(times),
        "lambda_kd": f"lambda_kd={float(lambda_kd):.1f}" in text,
        "teacher_only": "kd_loss_mode='teacher_only'" in text,
        "student_backbone": (
            f"student_backbone='{student_backbone}'" in text
        ),
        "seed": f"seed={seed}" in text,
        "temperature": (
            f"Teacher-only target T: {float(temperature):.4f}" in text
        ),
    }
    for milestone in MILESTONES:
        checks[f"state_{milestone}"] = (
            state_dir / f"training_state_iter{milestone:06d}.pth"
        ).is_file()
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"{variant}: failed checks {failed}")
    values = [trajectory[milestone]["miou_percent"] for milestone in MILESTONES]
    return {
        "variant": variant,
        "trajectory": {str(key): value for key, value in trajectory.items()},
        "best_observed_miou_percent": max(values),
        "final_miou_percent": values[-1],
        "final_pixacc_percent": trajectory[80000]["pixacc_percent"],
        "training_time": times[-1].strip(),
        "contract_pass": True,
        "log_path": str(log_path),
        "checkpoint_path": str(latest_state),
    }


def read_p8_runs(root):
    return {
        seed: read_run(
            root=root,
            variant=p8_variant(seed),
            log_name=SMALL_LOG_NAME,
            seed=seed,
            student_backbone="mobilenetv3_small",
            lambda_kd=0.0,
            temperature="1.0",
        )
        for seed in SEEDS
    }


def read_p9_grid(root, temperatures):
    return {
        seed: {
            temperature: read_run(
                root=root,
                variant=p9_variant(temperature, seed),
                log_name=LARGE_LOG_NAME,
                seed=seed,
                student_backbone="mobilenetv3_large",
                lambda_kd=1.0,
                temperature=temperature,
            )
            for temperature in temperatures
        }
        for seed in SEEDS
    }


def stage2_output_state(root):
    expected = []
    for seed in SEEDS:
        for temperature in P9_STAGE2_TEMPERATURES:
            variant = p9_variant(temperature, seed)
            expected.extend(
                (
                    root / "logs" / variant / LARGE_LOG_NAME,
                    root / "checkpoints" / variant / "training_state_latest.pth",
                )
            )
    present = [path.exists() for path in expected]
    if any(present) and not all(present):
        return "partial"
    return "complete" if all(present) else "absent"


def load_p7(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    if tuple(int(seed) for seed in protocol["seeds"]) != SEEDS:
        raise RuntimeError("P7 seed protocol does not match P8")
    if tuple(str(value) for value in protocol["temperatures"]) != P7_TEMPERATURES:
        raise RuntimeError("P7 temperature grid is not the locked seven-point grid")
    seed_rows = {int(row["seed"]): row for row in payload["seed_rows"]}
    if tuple(sorted(seed_rows)) != SEEDS:
        raise RuntimeError("P7 payload is missing a paired seed")
    return payload, seed_rows


def load_covar(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    missing = [temperature for temperature in P7_TEMPERATURES if temperature not in rows]
    if missing:
        raise RuntimeError(f"CoVar payload is missing temperatures: {missing}")
    return payload


def summarize_p8(p7_payload, p7_seed_rows, ce_runs):
    ce_final = {
        seed: ce_runs[seed]["final_miou_percent"] for seed in SEEDS
    }
    ce_milestones = {
        str(milestone): sample_statistics(
            [
                ce_runs[seed]["trajectory"][str(milestone)]["miou_percent"]
                for seed in SEEDS
            ]
        )
        for milestone in MILESTONES
    }
    comparisons = {}
    for temperature in P7_TEMPERATURES:
        values = [
            p7_seed_rows[seed]["runs"][temperature]["final_miou_percent"]
            - ce_final[seed]
            for seed in SEEDS
        ]
        stats = sample_statistics(values)
        stats["positive_seed_count"] = sum(value > 0.0 for value in values)
        stats["all_seeds_positive"] = all(value > 0.0 for value in values)
        stats["kd_mean_final_miou_percent"] = mean(
            p7_seed_rows[seed]["runs"][temperature]["final_miou_percent"]
            for seed in SEEDS
        )
        stats["ce_mean_final_miou_percent"] = mean(ce_final.values())
        comparisons[temperature] = stats

    best_temperature = max(
        P7_TEMPERATURES,
        key=lambda temperature: (
            comparisons[temperature]["kd_mean_final_miou_percent"],
            -P7_TEMPERATURES.index(temperature),
        ),
    )
    positive_mean_temperatures = [
        temperature
        for temperature in P7_TEMPERATURES
        if comparisons[temperature]["mean"] > 0.0
    ]
    all_pairwise_positive = all(
        value > 0.0
        for comparison in comparisons.values()
        for value in comparison["values"]
    )
    if len(positive_mean_temperatures) == len(P7_TEMPERATURES):
        if all_pairwise_positive:
            interpretation = (
                "all_fixed_temperatures_outperform_ce_for_every_paired_seed"
            )
        else:
            interpretation = (
                "all_temperature_means_exceed_ce_but_some_seed_temperature_"
                "pairs_show_negative_transfer"
            )
    elif positive_mean_temperatures:
        interpretation = (
            "kd_benefit_depends_on_temperature_and_some_temperatures_reduce_"
            "expected_utility"
        )
    else:
        interpretation = "no_fixed_temperature_has_positive_mean_gain_over_ce"

    return {
        "stage": "P8_ce_only_baseline",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {
            "p7_locked": True,
            "objective": "L_CE_exactly; lambda_kd=0",
            "teacher_forward": (
                "retained for trainer/protocol parity but contributes no gradient"
            ),
            "teacher": "DeepLabV3-ResNet101",
            "student": "DeepLabV3-MobileNetV3-Small",
            "seeds": list(SEEDS),
            "max_iterations": 80000,
            "validation_milestones": list(MILESTONES),
            "fresh_from_same_student_initialization_rule": True,
            "p_value_reported": False,
        },
        "runs": {str(seed): ce_runs[seed] for seed in SEEDS},
        "ce_summary": {
            "final": sample_statistics(ce_final.values()),
            "milestones": ce_milestones,
        },
        "paired_kd_minus_ce_pp": comparisons,
        "conclusions": {
            "p7_best_mean_temperature": best_temperature,
            "best_fixed_temperature_gain_over_ce_pp": comparisons[
                best_temperature
            ],
            "positive_mean_temperatures": positive_mean_temperatures,
            "all_seven_temperature_means_above_ce": (
                len(positive_mean_temperatures) == len(P7_TEMPERATURES)
            ),
            "all_21_paired_runs_above_ce": all_pairwise_positive,
            "interpretation": interpretation,
        },
        "source_p7_selection_summary": p7_payload["selection_summary"],
    }


def summarize_grid(runs, temperatures, delta):
    temperature_summaries = {}
    for temperature in temperatures:
        final_values = [
            runs[seed][temperature]["final_miou_percent"] for seed in SEEDS
        ]
        temperature_summaries[temperature] = {
            "final": sample_statistics(final_values),
            "milestones": {
                str(milestone): sample_statistics(
                    [
                        runs[seed][temperature]["trajectory"][str(milestone)][
                            "miou_percent"
                        ]
                        for seed in SEEDS
                    ]
                )
                for milestone in MILESTONES
            },
        }

    mean_winner = max(
        temperatures,
        key=lambda temperature: (
            temperature_summaries[temperature]["final"]["mean"],
            -temperatures.index(temperature),
        ),
    )
    maximum_mean = temperature_summaries[mean_winner]["final"]["mean"]
    near_optimal = [
        temperature
        for temperature in temperatures
        if temperature_summaries[temperature]["final"]["mean"]
        >= maximum_mean - delta
    ]
    per_seed_winners = {}
    for seed in SEEDS:
        per_seed_winners[str(seed)] = max(
            temperatures,
            key=lambda temperature: (
                runs[seed][temperature]["final_miou_percent"],
                -temperatures.index(temperature),
            ),
        )
    return {
        "temperature_summaries": temperature_summaries,
        "mean_winner": mean_winner,
        "mean_winner_miou_percent": maximum_mean,
        "delta_miou_pp": delta,
        "delta_optimal_grid_set": near_optimal,
        "per_seed_winners": per_seed_winners,
    }


def phase2_decision(stage1_summary):
    winner = stage1_summary["mean_winner"]
    near = set(stage1_summary["delta_optimal_grid_set"])
    interior_near = [
        temperature
        for temperature in P9_INTERIOR_TEMPERATURES
        if temperature in near
    ]
    if winner == "1.0":
        return {
            "required": True,
            "reason": (
                "coarse_grid_mean_winner_is_internal_T1p0_and_requires_"
                "T0p75_T1p25_resolution"
            ),
            "interior_near_optimal_points": interior_near,
        }
    if winner in {"0.5", "1.5"} and len(interior_near) >= 2:
        return {
            "required": True,
            "reason": (
                "coarse_grid_winner_is_in_0p5_to_1p5_and_the_practical_"
                "near_optimal_set_is_unresolved_inside_that_interval"
            ),
            "interior_near_optimal_points": interior_near,
        }
    return {
        "required": False,
        "reason": (
            "coarse_grid_does_not_identify_an_unresolved_candidate_peak_"
            "inside_0p5_to_1p5"
        ),
        "interior_near_optimal_points": interior_near,
    }


def identifiability_case(summary):
    winners = set(summary["per_seed_winners"].values())
    near = summary["delta_optimal_grid_set"]
    if len(winners) == 1 and len(near) == 1:
        return "result_B_candidate_stable_pair_specific_grid_optimum"
    return "result_A_no_unique_reproducible_grid_winner"


def covar_coordinates(covar_payload, temperatures):
    return {
        temperature: {
            "r_c_mean": covar_payload["rows"][temperature]["r_c_mean"],
            "r_v_mean": covar_payload["rows"][temperature]["r_v_mean"],
            "r_mean": covar_payload["rows"][temperature]["r_mean"],
        }
        for temperature in temperatures
    }


def summarize_p9(
    p7_payload,
    covar_payload,
    stage1_runs,
    stage2_runs,
    stage2_state,
    delta,
):
    stage1_summary = summarize_grid(
        stage1_runs, P9_STAGE1_TEMPERATURES, delta
    )
    gate = phase2_decision(stage1_summary)
    if stage2_state == "partial":
        raise RuntimeError("P9 stage-2 outputs are partial")
    if stage2_state == "complete":
        merged_runs = {
            seed: {**stage1_runs[seed], **stage2_runs[seed]}
            for seed in SEEDS
        }
        temperatures = P9_ALL_TEMPERATURES
        phase2_status = "complete"
    else:
        merged_runs = stage1_runs
        temperatures = P9_STAGE1_TEMPERATURES
        phase2_status = "required_pending" if gate["required"] else "not_required"

    final_summary = summarize_grid(merged_runs, temperatures, delta)
    pair1_winner = p7_payload["selection_summary"]["best_mean_temperature"]
    pair1_near = set(
        p7_payload["selection_summary"]["delta_optimal_grid_set"]
    )
    pair2_near = set(final_summary["delta_optimal_grid_set"])
    exact_covar_overlap = [
        temperature
        for temperature in temperatures
        if temperature in pair1_near and temperature in pair2_near
    ]
    cross_pair = {
        "pair1_mean_winner": pair1_winner,
        "pair1_delta_optimal_grid_set": sorted(
            pair1_near, key=P7_TEMPERATURES.index
        ),
        "pair2_mean_winner": final_summary["mean_winner"],
        "pair2_delta_optimal_grid_set": final_summary[
            "delta_optimal_grid_set"
        ],
        "mean_winners_differ": pair1_winner != final_summary["mean_winner"],
        "exact_near_optimal_covar_grid_overlap": exact_covar_overlap,
        "result_C_conservative_support": (
            pair1_winner != final_summary["mean_winner"]
            and bool(exact_covar_overlap)
        ),
        "note": (
            "Exact shared temperature points imply identical teacher CoVar "
            "coordinates because the teacher and analysis protocol are fixed; "
            "no unregistered distance threshold is introduced."
        ),
    }

    return {
        "stage": "P9_pair2_temperature_response",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {
            "teacher": "DeepLabV3-ResNet101",
            "student": "DeepLabV3-MobileNetV3-Large",
            "only_student_capacity_changed_from_P7": True,
            "teacher_only_temperature": True,
            "student_temperature": 1.0,
            "t_squared_compensation": False,
            "lambda_ce": 1.0,
            "lambda_kd": 1.0,
            "seeds": list(SEEDS),
            "max_iterations": 80000,
            "validation_milestones": list(MILESTONES),
            "stage1_temperatures": list(P9_STAGE1_TEMPERATURES),
            "conditional_stage2_temperatures": list(P9_STAGE2_TEMPERATURES),
            "delta_miou_pp": delta,
            "p_value_reported": False,
        },
        "stage1": {
            "runs": {
                str(seed): stage1_runs[seed] for seed in SEEDS
            },
            "summary": stage1_summary,
            "phase2_gate": gate,
        },
        "phase2_status": phase2_status,
        "runs": {str(seed): merged_runs[seed] for seed in SEEDS},
        "final_grid_temperatures": list(temperatures),
        "final_summary": final_summary,
        "identifiability_case": identifiability_case(final_summary),
        "covar_coordinates": covar_coordinates(
            covar_payload, temperatures
        ),
        "cross_pair_comparison": cross_pair,
    }


def format_values(values):
    return ", ".join(f"{value:+.6f}" for value in values)


def write_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_p8_markdown(payload, path):
    ce = payload["ce_summary"]["final"]
    conclusion = payload["conclusions"]
    lines = [
        "# P8: CE-only baseline",
        "",
        "## Outcome",
        "",
        (
            f"CE-only final mIoU is {ce['mean']:.6f} ± "
            f"{ce['sample_std']:.6f} pp across the three paired seeds."
        ),
        (
            "The trainer is identical to P7 and uses lambda_kd=0, so the "
            "optimized objective is exactly L_CE. The frozen teacher forward "
            "is retained only for protocol parity and contributes no gradient."
        ),
        (
            f"Interpretation: {conclusion['interpretation']}."
        ),
        "",
        "## Locked protocol",
        "",
        "- Seeds: 1234, 2025, 3407.",
        "- Fresh 80k poly schedule; validation at 20k, 40k, 60k, and 80k.",
        "- Same R101 teacher, MobileNetV3-Small student initialization rule, optimizer, batch size, and augmentation as P7.",
        "- Primary endpoint: final 80k VOC validation mIoU.",
        "- Descriptive paired statistics only; no p-value is emphasized for n=3.",
        "",
        "## Raw CE-only trajectories",
        "",
        "| seed | 20k | 40k | 60k | 80k/final | best observed |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        run = payload["runs"][str(seed)]
        trajectory = run["trajectory"]
        lines.append(
            f"| {seed} | {trajectory['20000']['miou_percent']:.6f} | "
            f"{trajectory['40000']['miou_percent']:.6f} | "
            f"{trajectory['60000']['miou_percent']:.6f} | "
            f"{trajectory['80000']['miou_percent']:.6f} | "
            f"{run['best_observed_miou_percent']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Paired fixed-temperature KD minus CE-only at 80k",
            "",
            "| T | KD mean | CE mean | paired differences by seed (pp) | paired mean | sample SD | positive seeds |",
            "|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for temperature in P7_TEMPERATURES:
        row = payload["paired_kd_minus_ce_pp"][temperature]
        lines.append(
            f"| {temperature} | {row['kd_mean_final_miou_percent']:.6f} | "
            f"{row['ce_mean_final_miou_percent']:.6f} | "
            f"{format_values(row['values'])} | {row['mean']:+.6f} | "
            f"{row['sample_std']:.6f} | {row['positive_seed_count']}/3 |"
        )

    lines.extend(
        [
            "",
            "## Direct answers",
            "",
            (
                "- Fixed-temperature KD has at least one positive-mean setting: "
                f"{bool(conclusion['positive_mean_temperatures'])}."
            ),
            (
                "- All seven temperature means exceed CE-only: "
                f"{conclusion['all_seven_temperature_means_above_ce']}."
            ),
            (
                "- All 21 seed-temperature pairs exceed CE-only: "
                f"{conclusion['all_21_paired_runs_above_ce']}."
            ),
            (
                "- P7 sample-mean winner: "
                f"T={conclusion['p7_best_mean_temperature']}."
            ),
            "",
            "## Evidence boundary",
            "",
            "These are paired three-seed estimates on VOC validation. They answer whether KD improves this locked protocol, but they are not a claim of universal KD benefit.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_p9_markdown(payload, path):
    summary = payload["final_summary"]
    gate = payload["stage1"]["phase2_gate"]
    lines = [
        "# P9: second teacher-student pair fixed-temperature response",
        "",
        "## Outcome",
        "",
        (
            f"Identifiability result: {payload['identifiability_case']}."
        ),
        (
            f"Final grid sample-mean winner: T={summary['mean_winner']} at "
            f"{summary['mean_winner_miou_percent']:.6f} mIoU."
        ),
        (
            f"Delta=0.2 pp near-optimal grid set: "
            f"{', '.join(summary['delta_optimal_grid_set'])}."
        ),
        (
            f"Conditional stage-2 status: {payload['phase2_status']} "
            f"({gate['reason']})."
        ),
        "",
        "## Locked protocol",
        "",
        "- Teacher: DeepLabV3-ResNet101; student: DeepLabV3-MobileNetV3-Large.",
        "- Relative to P7, only student capacity/initialization is changed.",
        "- Teacher-only temperature, student T=1, no T-squared compensation, CE + KD.",
        "- Seeds: 1234, 2025, 3407; fresh 80k poly schedule.",
        "- Validation: 20k, 40k, 60k, 80k; primary endpoint: final 80k mIoU.",
        "",
        "## Raw trajectories",
        "",
        "| seed | T | 20k | 40k | 60k | 80k/final | best observed |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    temperatures = tuple(payload["final_grid_temperatures"])
    for seed in SEEDS:
        for temperature in temperatures:
            run = payload["runs"][str(seed)][temperature]
            trajectory = run["trajectory"]
            lines.append(
                f"| {seed} | {temperature} | "
                f"{trajectory['20000']['miou_percent']:.6f} | "
                f"{trajectory['40000']['miou_percent']:.6f} | "
                f"{trajectory['60000']['miou_percent']:.6f} | "
                f"{trajectory['80000']['miou_percent']:.6f} | "
                f"{run['best_observed_miou_percent']:.6f} |"
            )

    lines.extend(
        [
            "",
            "## Final response estimates",
            "",
            "| T | mean final mIoU | sample SD | values by seed |",
            "|---:|---:|---:|---|",
        ]
    )
    for temperature in temperatures:
        row = summary["temperature_summaries"][temperature]["final"]
        values = ", ".join(f"{value:.6f}" for value in row["values"])
        lines.append(
            f"| {temperature} | {row['mean']:.6f} | "
            f"{row['sample_std']:.6f} | {values} |"
        )

    lines.extend(
        [
            "",
            "## Per-seed winners",
            "",
            "| seed | winner T |",
            "|---:|---:|",
        ]
    )
    for seed in SEEDS:
        lines.append(
            f"| {seed} | {summary['per_seed_winners'][str(seed)]} |"
        )

    cross = payload["cross_pair_comparison"]
    lines.extend(
        [
            "",
            "## CoVar and cross-pair comparison",
            "",
            "| T | mean r_c | mean r_v | mean r |",
            "|---:|---:|---:|---:|",
        ]
    )
    for temperature in temperatures:
        row = payload["covar_coordinates"][temperature]
        lines.append(
            f"| {temperature} | {row['r_c_mean']:.9f} | "
            f"{row['r_v_mean']:.9f} | {row['r_mean']:.9f} |"
        )
    overlap = ", ".join(cross["exact_near_optimal_covar_grid_overlap"]) or "none"
    lines.extend(
        [
            "",
            (
                f"- Pair 1 mean winner: T={cross['pair1_mean_winner']}; "
                f"pair 2 mean winner: T={cross['pair2_mean_winner']}."
            ),
            (
                "- Exact overlap of near-optimal CoVar grid points: "
                f"{overlap}."
            ),
            (
                "- Conservative support for result C: "
                f"{cross['result_C_conservative_support']}."
            ),
            "",
            "CoVar is used only as the fixed teacher-complexity coordinate. No new distance threshold is fitted after seeing the result.",
            "",
            "## Evidence boundary",
            "",
            "The conclusion is limited to this dense-prediction teacher-student pair and locked training protocol. With three seeds, sample means and sample SDs are emphasized rather than p-values.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    if args.delta < 0.0:
        raise ValueError("delta must be non-negative")
    p7_payload, p7_seed_rows = load_p7(args.p7_json)
    covar_payload = load_covar(args.covar_json)

    ce_runs = read_p8_runs(args.p8_root)
    p8_payload = summarize_p8(p7_payload, p7_seed_rows, ce_runs)
    write_json(p8_payload, args.p8_output_json)
    write_p8_markdown(p8_payload, args.p8_output_markdown)

    stage1_runs = read_p9_grid(
        args.p9_root, P9_STAGE1_TEMPERATURES
    )
    state = stage2_output_state(args.p9_root)
    stage2_runs = (
        read_p9_grid(args.p9_root, P9_STAGE2_TEMPERATURES)
        if state == "complete"
        else None
    )
    p9_payload = summarize_p9(
        p7_payload=p7_payload,
        covar_payload=covar_payload,
        stage1_runs=stage1_runs,
        stage2_runs=stage2_runs,
        stage2_state=state,
        delta=args.delta,
    )
    write_json(p9_payload, args.p9_output_json)
    write_p9_markdown(p9_payload, args.p9_output_markdown)

    print(
        "P8 report: {} | P9 report: {} | P9 stage2={}".format(
            args.p8_output_markdown,
            args.p9_output_markdown,
            p9_payload["phase2_status"],
        )
    )


if __name__ == "__main__":
    main()
