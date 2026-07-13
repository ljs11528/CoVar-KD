from types import SimpleNamespace

import pytest

from scripts.diagnostics.diagnose_rtc_routing import (
    pre_registered_config_matches,
    risk_quantile_pairwise_monotonic_agreement,
    risk_quantile_spearman,
    validate_args,
)


def make_rows(rates, counts=None):
    if counts is None:
        counts = [100] * len(rates)
    return [
        {
            'bin': index,
            'count': counts[index],
            'teacher_wrong_rate_native_proxy': rate,
        }
        for index, rate in enumerate(rates)
    ]


def make_args(**overrides):
    values = {
        'phase': 'O1.1',
        'split': 'train',
        'seed': 2025,
        'ranking_seed': 3407,
        'teacher_model': 'deeplabv3',
        'teacher_backbone': 'resnet101',
        'crop_size': [512, 512],
        'no_scale': False,
        'no_mirror': False,
        'batch_size': 4,
        'workers': 0,
        'max_images': 0,
        'ranking_max_pixels_per_image': 1024,
        'num_classes': 21,
        'assess_temperature': 1.0,
        'coefficient_a': 0.0,
        'teacher_output_temp': 3.0,
        'max_fallback_rate': 1e-4,
        'reliability_mode': 'confidence',
        'route_quantile': 0.8,
        'route_width': 0.05,
        'temp_reliable': 0.5,
        'temp_neutral': 1.0,
        'temp_unreliable': 2.0,
        'alpha_reliable': 1.0,
        'alpha_unreliable': 1.0,
        'bisection_iters': 16,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_risk_quantile_spearman_is_tie_aware():
    increasing = risk_quantile_spearman(
        make_rows([0.0, 0.0, 0.01, 0.01, 0.02, 0.03, 0.05, 0.08, 0.2, 0.4])
    )
    reversed_score = risk_quantile_spearman(
        make_rows([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    )
    assert increasing is not None and increasing > 0.98
    assert reversed_score == pytest.approx(-1.0)


def test_risk_quantile_spearman_requires_all_nonempty_bins():
    assert risk_quantile_spearman(make_rows([0.1] * 10, counts=[100] * 9 + [0])) is None
    assert risk_quantile_spearman(make_rows([0.1] * 9)) is None


def test_pairwise_monotonic_agreement_accepts_equal_plateaus():
    plateau = [0.0] * 8 + [0.03, 0.30]
    assert risk_quantile_pairwise_monotonic_agreement(
        make_rows(plateau)
    ) == pytest.approx(1.0)

    one_inversion = [0.0] * 6 + [0.0020, 0.0019, 0.03, 0.30]
    agreement = risk_quantile_pairwise_monotonic_agreement(
        make_rows(one_inversion)
    )
    assert agreement == pytest.approx(44.0 / 45.0)
    assert risk_quantile_pairwise_monotonic_agreement(
        make_rows(plateau, counts=[100] * 9 + [0])
    ) is None


def test_o11_profile_requires_confidence_and_explicit_zero_coefficient():
    args = make_args()
    validate_args(args)
    assert pre_registered_config_matches(args)

    with pytest.raises(ValueError, match='reliability-mode confidence'):
        validate_args(make_args(reliability_mode='full'))
    with pytest.raises(ValueError, match='coefficient-a 0'):
        validate_args(make_args(coefficient_a=None))
    with pytest.raises(ValueError, match='coefficient-a 0'):
        validate_args(make_args(coefficient_a=200.0))


def test_o11_profile_rejects_seed_cap_and_batch_drift():
    for name, value in [('seed', 1234), ('ranking_seed', 2025),
                        ('ranking_max_pixels_per_image', 2048)]:
        assert not pre_registered_config_matches(make_args(**{name: value}))
    assert pre_registered_config_matches(make_args(split='val', batch_size=1))
    assert not pre_registered_config_matches(make_args(split='val', batch_size=4))


def test_o1_profile_remains_backward_compatible():
    args = make_args(
        phase='O1',
        reliability_mode='full',
        coefficient_a=200.0,
    )
    validate_args(args)
    assert pre_registered_config_matches(args)
