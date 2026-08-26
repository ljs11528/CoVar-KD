import math

from scripts.diagnostics.covar_gap_teachability import (
    audit_raw_cost_direction,
    average_ranks,
    evaluate_candidate_group,
    spearman,
)


def test_raw_cost_direction_contract_is_locked():
    result = audit_raw_cost_direction(
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
    )
    assert result["pred_idx"] == 0
    assert result["oracle_idx"] == 0
    assert math.isclose(result["raw_rho"], -1.0, abs_tol=1e-12)
    assert math.isclose(result["aligned_rho"], 1.0, abs_tol=1e-12)
    assert math.isclose(
        result["aligned_rho"],
        -result["raw_rho"],
        abs_tol=1e-12,
    )


def test_average_ranks_and_spearman_handle_ties():
    assert average_ranks([1.0, 1.0, 3.0]).tolist() == [1.5, 1.5, 3.0]
    assert math.isclose(spearman([1, 2, 3], [3, 2, 1]), -1.0)


def test_vector_gap_selects_matching_oracle_candidate():
    rows = []
    for temperature, teacher_rc, teacher_rv, gain in (
        (0.5, 0.0, 0.0, 0.1),
        (1.0, 0.2, 0.1, 0.3),
        (1.5, 0.5, 0.4, 0.2),
    ):
        rows.append(
            {
                "stage": "early",
                "temperature": temperature,
                "one_step_gain": gain,
                "teacher_r_c": teacher_rc,
                "teacher_r_v": teacher_rv,
                "teacher_r": teacher_rc + teacher_rv,
                "student_r_c": 0.2,
                "student_r_v": 0.1,
                "student_r": 0.3,
                "teacher_student_kl": abs(temperature - 1.0),
            }
        )
    result = evaluate_candidate_group(
        rows, {"r_c": 1.0, "r_v": 1.0}, (0.5, 1.0, 1.5)
    )
    vector = result["scores"]["vector_covar_gap"]
    assert vector["prediction_temperature"] == 1.0
    assert vector["exact"] == 1.0
    assert vector["regret"] == 0.0
