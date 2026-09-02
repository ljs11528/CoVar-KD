import pytest

from scripts.diagnostics.summarize_p8_p9_experiments import (
    MILESTONES,
    P9_STAGE1_TEMPERATURES,
    identifiability_case,
    p8_variant,
    p9_variant,
    parse_log,
    phase2_decision,
    sample_statistics,
    summarize_grid,
)


def synthetic_runs(values):
    runs = {}
    for seed, per_temperature in values.items():
        runs[seed] = {}
        for temperature, final in per_temperature.items():
            trajectory = {
                str(milestone): {
                    "miou_percent": final - (80000 - milestone) / 100000.0,
                    "pixacc_percent": 90.0,
                }
                for milestone in MILESTONES
            }
            runs[seed][temperature] = {
                "trajectory": trajectory,
                "final_miou_percent": final,
            }
    return runs


def test_variant_names_are_locked():
    assert p8_variant(1234) == "ce_only_80k_seed1234"
    assert p8_variant(2025) == "ce_only_80k_seed2025_retry1"
    assert p9_variant("0.25", 2025) == "fixed_T0p25_80k_seed2025"
    assert p9_variant("1.0", 3407) == "fixed_T1p0_80k_seed3407"
    assert p9_variant("1.25", 1234) == "fixed_T1p25_80k_seed1234"


def test_parse_log_attaches_exact_milestones():
    lines = []
    for index, milestone in enumerate(MILESTONES):
        lines.extend(
            [
                f"Iters: {milestone}/80000 || Lr: 0.01",
                (
                    "Overall validation pixAcc: "
                    f"{90 + index}, mIoU: {60 + index}"
                ),
            ]
        )
    maximum, trajectory = parse_log("\n".join(lines))
    assert maximum == 80000
    assert tuple(trajectory) == MILESTONES
    assert trajectory[80000]["miou_percent"] == pytest.approx(63.0)


def test_statistics_use_sample_standard_deviation():
    result = sample_statistics([1.0, 2.0, 3.0])
    assert result["mean"] == pytest.approx(2.0)
    assert result["sample_std"] == pytest.approx(1.0)


def test_stage2_gate_is_predeclared_and_minimal():
    internal_peak = {
        "mean_winner": "1.0",
        "delta_optimal_grid_set": ["1.0"],
    }
    unresolved_left = {
        "mean_winner": "0.5",
        "delta_optimal_grid_set": ["0.5", "1.0"],
    }
    resolved_left = {
        "mean_winner": "0.5",
        "delta_optimal_grid_set": ["0.5"],
    }
    boundary_peak = {
        "mean_winner": "0.25",
        "delta_optimal_grid_set": ["0.25", "0.5"],
    }
    assert phase2_decision(internal_peak)["required"] is True
    assert phase2_decision(unresolved_left)["required"] is True
    assert phase2_decision(resolved_left)["required"] is False
    assert phase2_decision(boundary_peak)["required"] is False


def test_grid_summary_and_identifiability_cases():
    stable = synthetic_runs(
        {
            1234: {"0.25": 60.0, "0.5": 61.0, "1.0": 62.0, "1.5": 61.0, "2.0": 60.0},
            2025: {"0.25": 60.1, "0.5": 61.1, "1.0": 62.1, "1.5": 61.1, "2.0": 60.1},
            3407: {"0.25": 59.9, "0.5": 60.9, "1.0": 61.9, "1.5": 60.9, "2.0": 59.9},
        }
    )
    summary = summarize_grid(stable, P9_STAGE1_TEMPERATURES, delta=0.2)
    assert summary["mean_winner"] == "1.0"
    assert summary["delta_optimal_grid_set"] == ["1.0"]
    assert identifiability_case(summary) == (
        "result_B_candidate_stable_pair_specific_grid_optimum"
    )

    mixed = synthetic_runs(
        {
            1234: {"0.25": 60.0, "0.5": 62.0, "1.0": 61.0, "1.5": 60.5, "2.0": 60.0},
            2025: {"0.25": 60.0, "0.5": 61.0, "1.0": 62.0, "1.5": 60.5, "2.0": 60.0},
            3407: {"0.25": 60.0, "0.5": 61.0, "1.0": 60.5, "1.5": 62.0, "2.0": 60.0},
        }
    )
    mixed_summary = summarize_grid(
        mixed, P9_STAGE1_TEMPERATURES, delta=0.2
    )
    assert len(set(mixed_summary["per_seed_winners"].values())) == 3
    assert identifiability_case(mixed_summary) == (
        "result_A_no_unique_reproducible_grid_winner"
    )
