import pytest

from scripts.diagnostics.summarize_paired_multiseed import (
    adaptive_decision,
    paired_statistics,
)


def test_paired_statistics_use_sample_standard_deviation():
    result = paired_statistics([1.0, 2.0, 3.0])
    assert result["mean_pp"] == pytest.approx(2.0)
    assert result["sample_std_pp"] == pytest.approx(1.0)


def test_adaptive_decision_follows_predeclared_cases():
    assert adaptive_decision({1234: -0.3, 2025: -0.1, 3407: 0.0}) == (
        "no_reliable_improvement"
    )
    assert adaptive_decision({1234: -0.3, 2025: 0.2, 3407: 0.1}) == (
        "both_new_seeds_positive_reassess"
    )
    assert adaptive_decision({1234: -0.3, 2025: 0.2, 3407: -0.1}) == (
        "inconsistent_improvement"
    )
