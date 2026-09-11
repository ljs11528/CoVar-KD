import math

import pytest

from scripts.diagnostics.finalize_p7a_fixed_temperature_response import (
    QUANTILE_NAMES,
    SEEDS,
    TEMPERATURES,
    leave_one_seed_out_selection,
    quantile_summary,
    report_figure_link,
    summarize_temperature_response,
    temperature_key,
)


FINAL_MIOU = {
    1234: {
        "0.25": 62.045145,
        "0.5": 62.508112,
        "0.75": 62.521303,
        "1.0": 61.444885,
        "1.25": 63.023609,
        "1.5": 63.194788,
        "2.0": 62.340027,
    },
    2025: {
        "0.25": 63.150424,
        "0.5": 63.093483,
        "0.75": 61.310536,
        "1.0": 62.740374,
        "1.25": 62.069303,
        "1.5": 62.807024,
        "2.0": 62.948352,
    },
    3407: {
        "0.25": 62.215215,
        "0.5": 62.346804,
        "0.75": 63.048047,
        "1.0": 62.482047,
        "1.25": 61.768013,
        "1.5": 62.243474,
        "2.0": 62.196857,
    },
}


def test_locked_grid_and_temperature_keys():
    assert SEEDS == (1234, 2025, 3407)
    assert tuple(temperature_key(value) for value in TEMPERATURES) == (
        "0.25",
        "0.5",
        "0.75",
        "1.0",
        "1.25",
        "1.5",
        "2.0",
    )


def test_quantiles_use_predeclared_linear_percentiles():
    summary = quantile_summary([0.0, 1.0, 2.0, 3.0, 4.0])
    assert tuple(summary) == QUANTILE_NAMES
    assert summary == pytest.approx(
        {"q10": 0.4, "q25": 1.0, "q50": 2.0, "q75": 3.0, "q90": 3.6}
    )
    normalized_zero = quantile_summary([-0.0] * 5)["q50"]
    assert normalized_zero == 0.0
    assert math.copysign(1.0, normalized_zero) == 1.0


def test_response_reproduces_p7_winner_and_delta_set():
    response = summarize_temperature_response(FINAL_MIOU, delta=0.2)
    assert response["mean_winner"] == "1.5"
    assert response["temperatures"]["1.5"]["mean"] == pytest.approx(
        62.74842866666667
    )
    assert response["delta_optimal_grid_set"] == ["0.5", "1.5"]


def test_leave_one_seed_out_selection_matches_preregistered_audit():
    result = leave_one_seed_out_selection(FINAL_MIOU)
    rows = {row["held_out_seed"]: row for row in result["rows"]}
    assert rows[1234]["selected_temperature"] == "0.5"
    assert rows[1234]["held_out_best_temperature"] == "1.5"
    assert rows[1234]["selection_regret_pp"] == pytest.approx(0.686676)
    assert rows[2025]["selected_temperature"] == "0.75"
    assert rows[2025]["held_out_best_temperature"] == "0.25"
    assert rows[2025]["selection_regret_pp"] == pytest.approx(1.839888)
    assert rows[3407]["selected_temperature"] == "1.5"
    assert rows[3407]["held_out_best_temperature"] == "0.75"
    assert rows[3407]["selection_regret_pp"] == pytest.approx(0.804573)
    assert result["mean_selection_regret_pp"] == pytest.approx(1.110379)
    assert result["interpretation"] == (
        "post_hoc_descriptive_not_a_strict_generalization_estimate"
    )


def test_negative_delta_is_rejected():
    with pytest.raises(ValueError, match="delta"):
        summarize_temperature_response(FINAL_MIOU, delta=-0.1)


def test_report_figure_links_support_repo_relative_and_absolute_paths():
    assert report_figure_link("figures/covar_match/chart.png") == (
        "../../figures/covar_match/chart.png"
    )
    assert report_figure_link("/tmp/chart.png") == "/tmp/chart.png"
