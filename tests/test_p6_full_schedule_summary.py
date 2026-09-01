import pytest

from scripts.diagnostics.summarize_p6_full_schedule import (
    MILESTONES,
    paired_statistics,
    parse_log,
    rank_transition_count,
    sign_change_count,
    terminal_decision,
    variant_for,
)


def test_parse_log_attaches_each_validation_to_its_iteration():
    lines = []
    for index, milestone in enumerate(MILESTONES):
        lines.extend(
            [
                f"Iters: {milestone}/80000 || Lr: 0.01",
                f"Overall validation pixAcc: {90.0 + index}, mIoU: {60.0 + index}",
            ]
        )
    maximum, trajectory = parse_log("\n".join(lines))
    assert maximum == 80000
    assert tuple(trajectory) == MILESTONES
    assert trajectory[60000]["miou_percent"] == pytest.approx(62.0)


def test_parse_log_rejects_duplicate_milestone_validation():
    text = "\n".join(
        [
            "Iters: 20000/80000",
            "Overall validation pixAcc: 90, mIoU: 60",
            "Overall validation pixAcc: 91, mIoU: 61",
        ]
    )
    with pytest.raises(RuntimeError, match="duplicate validation"):
        parse_log(text)


def test_paired_statistics_use_sample_standard_deviation():
    result = paired_statistics([1.0, 2.0, 3.0])
    assert result["mean_pp"] == pytest.approx(2.0)
    assert result["sample_std_pp"] == pytest.approx(1.0)


def test_failed_infrastructure_attempt_is_replaced_by_fresh_retry():
    assert variant_for("fixed_t0p5", 1234) == "fixed_T0p5_80k_seed1234_retry1"
    assert variant_for("fixed_t0p5", 2025) == "fixed_T0p5_80k_seed2025"


def test_transition_and_sign_change_counts_are_explicit():
    assert rank_transition_count([["a", "b"], ["a", "b"], ["b", "a"]]) == 1
    assert sign_change_count([-1.0, 0.0, 2.0, 3.0, -4.0]) == 2


def _row(seed, adaptive_final, p4a, fixed_values):
    methods = {
        method: {"final_miou_percent": value}
        for method, value in zip(
            ("fixed_t0p5", "fixed_t1p0", "fixed_t1p5"), fixed_values
        )
    }
    methods["p4a_r8"] = {"final_miou_percent": p4a}
    return {
        "seed": seed,
        "methods": methods,
        "differences_pp": {"adaptive_final": adaptive_final},
    }


def test_terminal_decision_uses_predeclared_full_schedule_cases():
    case_b = [
        _row(1234, 0.2, 61.2, (60.0, 60.5, 61.0)),
        _row(2025, 0.2, 62.2, (61.0, 61.5, 62.0)),
        _row(3407, 0.2, 63.2, (62.0, 62.5, 63.0)),
    ]
    assert terminal_decision(case_b) == "case_B_p4a_above_all_fixed_all_seeds"

    case_a = [
        _row(1234, -0.2, 60.8, (60.0, 60.5, 61.0)),
        _row(2025, 0.2, 61.2, (60.0, 60.5, 61.0)),
        _row(3407, -0.1, 60.9, (60.0, 60.5, 61.0)),
    ]
    assert terminal_decision(case_a) == "case_A_mixed_adaptive_final"

    all_nonpositive = [
        _row(seed, value, 60.0 + value, (59.0, 59.5, 60.0))
        for seed, value in zip((1234, 2025, 3407), (-0.1, 0.0, -0.2))
    ]
    assert terminal_decision(all_nonpositive) == "no_reliable_adaptive_improvement"
