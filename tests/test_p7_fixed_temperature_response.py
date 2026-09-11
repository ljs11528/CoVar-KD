import pytest

from scripts.diagnostics.summarize_p7_fixed_temperature_response import (
    MILESTONES,
    boundary_status,
    classify,
    count_rank_transitions,
    lower_boundary_case,
    parse_log,
    root_for,
    sample_statistics,
    variant_for,
)


def test_variant_mapping_reuses_only_locked_p6_temperatures():
    assert variant_for("0.25", 1234) == "fixed_T0p25_80k_seed1234"
    assert variant_for("0.5", 1234) == "fixed_T0p5_80k_seed1234_retry1"
    assert variant_for("1.0", 2025) == "fixed_T1p0_80k_seed2025"
    assert variant_for("0.75", 3407) == "fixed_T0p75_80k_seed3407"
    assert variant_for("1.25", 1234) == "fixed_T1p25_80k_seed1234"
    assert variant_for("2.0", 2025) == "fixed_T2p0_80k_seed2025"


def test_root_selection_keeps_p6_and_p7_outputs_separate():
    assert root_for("0.5", "p6", "p7") == "p6"
    assert root_for("1.5", "p6", "p7") == "p6"
    assert root_for("0.25", "p6", "p7") == "p7"
    assert root_for("0.75", "p6", "p7") == "p7"


def test_parse_log_attaches_four_validations_to_locked_milestones():
    lines = []
    for index, milestone in enumerate(MILESTONES):
        lines.extend(
            [
                f"Iters: {milestone}/80000 || Lr: 0.01",
                f"Overall validation pixAcc: {90 + index}, mIoU: {60 + index}",
            ]
        )
    maximum, trajectory = parse_log("\n".join(lines))
    assert maximum == 80000
    assert tuple(trajectory) == MILESTONES
    assert trajectory[80000]["miou_percent"] == pytest.approx(63.0)


def test_parse_log_rejects_duplicate_validation():
    text = "\n".join(
        [
            "Iters: 20000/80000",
            "Overall validation pixAcc: 90, mIoU: 60",
            "Overall validation pixAcc: 91, mIoU: 61",
        ]
    )
    with pytest.raises(RuntimeError, match="duplicate validation"):
        parse_log(text)


def test_statistics_use_sample_standard_deviation():
    stats = sample_statistics([1.0, 2.0, 3.0])
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["sample_std"] == pytest.approx(1.0)


def test_predeclared_classification_and_boundary_gate():
    bests_same = {"1234": "1.25", "2025": "1.25", "3407": "1.25"}
    bests_mixed = {"1234": "1.0", "2025": "1.25", "3407": "1.5"}
    assert classify(["1.25"], bests_same) == (
        "case_A_candidate_unique_grid_optimum_requires_confirmation"
    )
    assert classify(["1.0", "1.25"], bests_same) == (
        "case_B_broad_near_optimal_grid_set"
    )
    assert classify(["1.25"], bests_mixed) == (
        "case_C_no_unique_reproducible_optimum"
    )
    assert classify(["0.5", "1.5"], bests_mixed) == (
        "case_C_no_unique_reproducible_optimum"
    )
    assert boundary_status(["0.25", "1.5"]) == "search_boundary_not_closed"
    assert boundary_status(["1.5", "2.0"]) == "search_boundary_not_closed"
    assert boundary_status(["1.0", "1.25"]) == (
        "search_boundary_closed_on_current_grid"
    )


def test_lower_boundary_cases_are_predeclared():
    assert lower_boundary_case(["0.5", "1.5"], "1.5") == (
        "case_1_lower_boundary_closed"
    )
    assert lower_boundary_case(["0.25", "1.5"], "1.5") == (
        "case_2_low_temperature_near_optimal_not_best"
    )
    assert lower_boundary_case(["0.25", "0.5"], "0.25") == (
        "case_3_t0p25_sample_mean_best"
    )


def test_rank_transitions_count_only_adjacent_changes():
    rankings = [
        ["0.5", "1.0"],
        ["0.5", "1.0"],
        ["1.0", "0.5"],
        ["0.5", "1.0"],
    ]
    assert count_rank_transitions(rankings) == 2
