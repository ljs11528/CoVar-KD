import math

import numpy as np

from scripts.diagnostics.p3a_audit import (
    STAGE_NAMES,
    STEP_SIZES,
    bootstrap_oracle_metrics,
    oracle_point_metrics,
    summarize_direction_rhos,
)


def test_oracle_point_metrics_use_regions_but_keep_image_groups():
    image_gains = {
        "image_a": np.asarray([[1.0, 0.0], [0.0, 2.0]]),
        "image_b": np.asarray([[3.0, 1.0]]),
    }
    result = oracle_point_metrics(image_gains, (0.5, 1.0))
    assert result["image_count"] == 2
    assert result["region_count"] == 3
    assert math.isclose(
        result["mean_gain_by_temperature"]["0.5"], 4.0 / 3.0
    )
    assert math.isclose(
        result["mean_gain_by_temperature"]["1.0"], 1.0
    )
    assert result["best_fixed_temperature"] == 0.5
    assert math.isclose(result["mean_oracle_gain"], 2.0)
    assert math.isclose(
        result["oracle_uplift_over_best_fixed"], 2.0 / 3.0
    )
    assert math.isclose(result["relative_oracle_uplift"], 0.5)
    assert math.isclose(result["median_top1_second_margin"], 2.0)
    assert result["p_margin_lt_1e_4"] == 0.0
    assert result["p_margin_lt_1pct_abs_oracle_gain"] == 0.0


def test_bootstrap_is_deterministic_and_resamples_images():
    image_gains = {
        "image_a": np.asarray([[1.0, 0.0], [0.0, 2.0]]),
        "image_b": np.asarray([[3.0, 1.0]]),
    }
    first = bootstrap_oracle_metrics(
        image_gains, (0.5, 1.0), replicates=100, seed=7
    )
    second = bootstrap_oracle_metrics(
        image_gains, (0.5, 1.0), replicates=100, seed=7
    )
    assert first == second
    assert first["resampling_unit"] == "image"
    assert first["replicates"] == 100
    assert all(
        math.isfinite(value)
        for interval in first[
            "mean_gain_by_temperature_ci95"
        ].values()
        for value in interval
    )


def test_direction_summary_checks_decreasing_eta_trend():
    rhos = {
        stage: {
            STEP_SIZES[0]: [0.5, 0.6],
            STEP_SIZES[1]: [0.8, 0.9],
            STEP_SIZES[2]: [1.0, 1.0],
        }
        for stage in STAGE_NAMES
    }
    summary, trend = summarize_direction_rhos(rhos, 2)
    assert summary["overall"]["0.01"]["region_count"] == 6
    assert all(item["pass"] for item in trend.values())
