import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

from utils.rtc_temperature import (
    FrozenReliabilityCDF,
    RTCConfig,
    build_rtc_temperature_map,
    build_target_log_odds,
    collect_rtc_diagnostics,
    compute_reference_reliability,
    compute_reliability_gates,
    compute_top_vs_rest_log_odds,
    invert_target_by_bisection,
    load_frozen_reliability_cdf,
    masked_temperature_kd_loss,
    query_frozen_cdf,
    reliability_definition_metadata,
    save_frozen_reliability_cdf,
    shuffle_temperature_map,
)


def make_cdf(values, probabilities=None):
    values = torch.as_tensor(values, dtype=torch.float32)
    if probabilities is None:
        probabilities = torch.linspace(0.0, 1.0, values.numel())
    return FrozenReliabilityCDF(
        values=values,
        probabilities=torch.as_tensor(probabilities, dtype=torch.float32),
        metadata={},
        checksum_sha256="test",
    )


class RTCConfigTest(unittest.TestCase):
    def test_validation_rejects_targets_outside_reachable_endpoints(self):
        RTCConfig().validate()
        invalid = (
            RTCConfig(route_quantile=0.0),
            RTCConfig(route_width=0.0),
            RTCConfig(reliable_temperature=1.5),
            RTCConfig(alpha_reliable=1.01),
            RTCConfig(alpha_unreliable=-0.01),
            RTCConfig(bisection_iterations=0),
            RTCConfig(reliability_mode="confidence-only"),
        )
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    config.validate()


class FrozenCDFTest(unittest.TestCase):
    def test_round_trip_checksum_and_boundaries(self):
        samples = torch.tensor([0.1, 0.2, 0.2, 0.4, 0.8])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cdf.pt"
            checksum = save_frozen_reliability_cdf(
                path,
                samples,
                {"num_classes": 21, "assess_temperature": 1.0},
                num_quantiles=17,
            )
            cdf = load_frozen_reliability_cdf(path)
        self.assertEqual(cdf.checksum_sha256, checksum)
        self.assertEqual(cdf.metadata["sample_count"], samples.numel())
        self.assertTrue(torch.all(cdf.values[1:] >= cdf.values[:-1]))
        result = cdf.query(torch.tensor([-1.0, 0.1, 0.2, 2.0]))
        self.assertEqual(float(result[0]), 0.0)
        self.assertGreater(float(result[2]), float(result[1]))
        self.assertEqual(float(result[-1]), 1.0)

    def test_save_uses_numpy_quantiles(self):
        samples = torch.tensor([0.0, 1.0, 2.0, 3.0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cdf.pt"
            with mock.patch.object(
                torch,
                "quantile",
                side_effect=AssertionError("torch.quantile must not be used"),
            ):
                save_frozen_reliability_cdf(
                    path,
                    samples,
                    {"num_classes": 3},
                    num_quantiles=5,
                )
            cdf = load_frozen_reliability_cdf(path)
        torch.testing.assert_close(
            cdf.values, torch.tensor([0.0, 0.75, 1.5, 2.25, 3.0])
        )

    def test_duplicate_knots_have_right_continuous_semantics(self):
        values = torch.tensor([0.0, 1.0, 1.0, 2.0])
        probabilities = torch.tensor([0.0, 0.25, 0.75, 1.0])
        result = query_frozen_cdf(
            torch.tensor([0.5, 1.0, 1.5]), values, probabilities
        )
        torch.testing.assert_close(result, torch.tensor([0.0, 0.75, 0.75]))


class ReliabilityComputationTest(unittest.TestCase):
    def test_high_confidence_nonmax_variance_is_stable(self):
        probabilities = torch.tensor(
            [
                [0.9999, 0.00009, 0.00001],
                [0.99999, 0.000009, 0.000001],
            ],
            dtype=torch.float32,
        )
        logits = probabilities.log().reshape(2, 3, 1, 1)
        valid = torch.ones(2, 1, 1, dtype=torch.bool)
        coefficient_a = 200.0

        maps = compute_reference_reliability(
            logits,
            valid,
            coefficient_a=coefficient_a,
        )
        evaluated_probability = F.softmax(logits, dim=1)
        confidence = evaluated_probability[:, 0]
        nonmax_probability = evaluated_probability[:, 1:]
        expected_variance = (
            (nonmax_probability - nonmax_probability.mean(dim=1, keepdim=True))
            .square()
            .mean(dim=1)
        )
        expected_reliability = -torch.log(confidence) + (
            coefficient_a
            * expected_variance
            / (1.0 - confidence).clamp_min(1e-8)
        )

        self.assertTrue(torch.all(maps.variance[:, 0, 0] > 0))
        torch.testing.assert_close(
            maps.variance[:, 0, 0], expected_variance[:, 0, 0], rtol=1e-4, atol=1e-15
        )
        torch.testing.assert_close(
            maps.reliability[:, 0, 0], expected_reliability[:, 0, 0],
            rtol=1e-4, atol=1e-8,
        )

    def test_confidence_mode_is_exact_neg_log_top1_and_ignores_coefficient(self):
        probabilities = torch.tensor(
            [[0.70, 0.20, 0.09, 0.01]],
            dtype=torch.float32,
        )
        logits = probabilities.log().reshape(1, 4, 1, 1)
        valid = torch.ones(1, 1, 1, dtype=torch.bool)

        confidence_a0 = compute_reference_reliability(
            logits,
            valid,
            coefficient_a=0.0,
            reliability_mode="confidence",
        )
        confidence_a200 = compute_reference_reliability(
            logits,
            valid,
            coefficient_a=200.0,
            reliability_mode="confidence",
        )
        full = compute_reference_reliability(
            logits,
            valid,
            coefficient_a=200.0,
            reliability_mode="full",
        )

        expected = -torch.log(torch.tensor(0.70))
        torch.testing.assert_close(
            confidence_a0.reliability[0, 0, 0], expected, rtol=1e-6, atol=1e-7
        )
        torch.testing.assert_close(
            confidence_a0.reliability,
            confidence_a200.reliability,
            rtol=0,
            atol=0,
        )
        self.assertGreater(
            float(full.reliability[0, 0, 0]),
            float(confidence_a200.reliability[0, 0, 0]),
        )

    def test_confidence_definition_metadata_is_explicit(self):
        metadata = reliability_definition_metadata("confidence", 0.0)
        self.assertEqual(
            metadata["reliability_definition_id"],
            "neg_log_top1_confidence_v1",
        )
        self.assertEqual(metadata["active_terms"], ["confidence"])
        self.assertIs(metadata["coefficient_a_active"], False)
        self.assertEqual(metadata["reliability_epsilon"], 1e-8)
        with self.assertRaises(ValueError):
            reliability_definition_metadata("confidence-only", 0.0)


class RoutingAndTargetTest(unittest.TestCase):
    def test_gates_are_mutually_exclusive_and_switchable(self):
        quantile = torch.tensor([[0.1, 0.8, 0.9]])
        valid = torch.ones_like(quantile, dtype=torch.bool)
        gate_r, gate_u = compute_reliability_gates(
            quantile, valid, route_quantile=0.8, route_width=0.05
        )
        self.assertTrue(torch.all((gate_r * gate_u) == 0))
        self.assertGreater(float(gate_r[0, 0]), 0.0)
        self.assertEqual(float(gate_r[0, 1]), 0.0)
        self.assertEqual(float(gate_u[0, 1]), 0.0)
        self.assertGreater(float(gate_u[0, 2]), 0.0)

        gate_r, gate_u = compute_reliability_gates(
            quantile,
            valid,
            route_quantile=0.8,
            route_width=0.05,
            enable_unreliable=False,
        )
        self.assertTrue(torch.all(gate_u == 0))
        self.assertGreater(float(gate_r[0, 0]), 0.0)

    def test_log_odds_matches_direct_probability_and_is_monotonic(self):
        logits = torch.tensor(
            [[[[3.0]], [[1.0]], [[-1.0]]]], dtype=torch.float64
        )
        values = []
        for temperature in (0.5, 1.0, 2.0):
            actual = compute_top_vs_rest_log_odds(logits, temperature)
            probability = F.softmax(logits / temperature, dim=1)
            confidence = probability.max(dim=1).values
            expected = torch.log(confidence / (1.0 - confidence))
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            values.append(float(actual))
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])

    def test_target_stays_inside_endpoints(self):
        torch.manual_seed(1)
        logits = torch.randn(1, 5, 2, 3)
        gate_r = torch.tensor([[[1.0, 0.4, 0.0], [0.0, 0.0, 0.0]]])
        gate_u = torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.7, 1.0]]])
        target, endpoint_r, endpoint_0, endpoint_u = build_target_log_odds(
            logits, gate_r, gate_u, RTCConfig()
        )
        reliable = gate_r > 0
        unreliable = gate_u > 0
        self.assertTrue(torch.all(target[reliable] >= endpoint_0[reliable]))
        self.assertTrue(torch.all(target[reliable] <= endpoint_r[reliable]))
        self.assertTrue(torch.all(target[unreliable] <= endpoint_0[unreliable]))
        self.assertTrue(torch.all(target[unreliable] >= endpoint_u[unreliable]))

    def test_bisection_recovers_binary_analytic_temperature(self):
        logits = torch.tensor([[[[2.0]], [[0.0]]]])
        target_temperature = 0.75
        target = torch.tensor([[[2.0 / target_temperature]]])
        gate_r = torch.ones_like(target)
        gate_u = torch.zeros_like(target)
        valid = torch.ones_like(target, dtype=torch.bool)
        config = RTCConfig(bisection_iterations=20)
        temperature, residual, fallback, _, _ = invert_target_by_bisection(
            logits, target, gate_r, gate_u, valid, config
        )
        torch.testing.assert_close(
            temperature, torch.full_like(temperature, target_temperature), atol=2e-6, rtol=0
        )
        self.assertLess(float(residual.max()), 1e-5)
        self.assertFalse(bool(fallback.any()))

    def test_tie_and_nonfinite_logits_fall_back_to_neutral(self):
        logits = torch.tensor(
            [[
                [[3.0, 2.0, float("nan")]],
                [[1.0, 2.0, 0.0]],
                [[0.0, 0.0, 0.0]],
            ]]
        )
        gate_r = torch.ones((1, 1, 3))
        gate_u = torch.zeros_like(gate_r)
        valid = torch.ones_like(gate_r, dtype=torch.bool)
        target = torch.ones_like(gate_r)
        temperature, residual, fallback, tie, finite = invert_target_by_bisection(
            logits, target, gate_r, gate_u, valid, RTCConfig()
        )
        self.assertEqual(float(temperature[0, 0, 1]), 1.0)
        self.assertEqual(float(temperature[0, 0, 2]), 1.0)
        self.assertTrue(bool(tie[0, 0, 1]))
        self.assertFalse(bool(finite[0, 0, 2]))
        self.assertTrue(bool(fallback[0, 0, 1]))
        self.assertTrue(bool(fallback[0, 0, 2]))
        self.assertTrue(torch.isfinite(residual).all())

    def test_alpha_zero_and_disabled_branch_are_exactly_neutral(self):
        torch.manual_seed(4)
        raw_logits = torch.randn(1, 4, 3, 5)
        valid = torch.ones(1, 3, 5, dtype=torch.bool)
        cdf = make_cdf(torch.linspace(0.0, 2.0, 65))

        alpha_zero = RTCConfig(alpha_reliable=0.0, alpha_unreliable=0.0)
        maps = build_rtc_temperature_map(
            raw_logits, raw_logits / 3.0, valid, cdf, alpha_zero
        )
        self.assertTrue(torch.equal(maps.temperature, torch.ones_like(maps.temperature)))

        disabled = RTCConfig(enable_reliable=False, enable_unreliable=False)
        maps = build_rtc_temperature_map(
            raw_logits, raw_logits / 3.0, valid, cdf, disabled
        )
        self.assertTrue(torch.equal(maps.temperature, torch.ones_like(maps.temperature)))

        reliable_only = RTCConfig(enable_reliable=True, enable_unreliable=False)
        maps = build_rtc_temperature_map(
            raw_logits, raw_logits / 3.0, valid, cdf, reliable_only
        )
        high_risk = maps.reliability_quantile > reliable_only.route_quantile
        self.assertTrue(torch.all(maps.temperature[high_risk] == 1.0))
        self.assertTrue(torch.all(maps.temperature <= 1.0))


class DiagnosticsTest(unittest.TestCase):
    def test_all_ignore_returns_complete_schema(self):
        torch.manual_seed(5)
        logits = torch.randn(1, 3, 2, 2)
        cdf = make_cdf(torch.linspace(0.0, 2.0, 17))
        config = RTCConfig()

        empty_maps = build_rtc_temperature_map(
            logits,
            logits / 3.0,
            torch.zeros(1, 2, 2, dtype=torch.bool),
            cdf,
            config,
        )
        populated_maps = build_rtc_temperature_map(
            logits,
            logits / 3.0,
            torch.ones(1, 2, 2, dtype=torch.bool),
            cdf,
            config,
        )
        empty = collect_rtc_diagnostics(empty_maps, config)
        populated = collect_rtc_diagnostics(populated_maps, config)

        self.assertEqual(set(empty), set(populated))
        self.assertEqual(empty["valid_count"], 0.0)
        self.assertEqual(empty["temperature_mean"], config.neutral_temperature)
        self.assertEqual(empty["finite_rate"], 1.0)
        self.assertEqual(empty["residual_is_applicable"], 1.0)

    def test_bisection_resolution_counts_reached_endpoints(self):
        logits = torch.tensor([[[[3.0]], [[1.0]], [[0.0]]]])
        valid = torch.ones(1, 1, 1, dtype=torch.bool)
        config = RTCConfig(route_width=0.005)

        reliable_maps = build_rtc_temperature_map(
            logits,
            logits / 3.0,
            valid,
            make_cdf([100.0, 101.0]),
            config,
        )
        unreliable_maps = build_rtc_temperature_map(
            logits,
            logits / 3.0,
            valid,
            make_cdf([-2.0, -1.0]),
            config,
        )
        reliable = collect_rtc_diagnostics(reliable_maps, config)
        unreliable = collect_rtc_diagnostics(unreliable_maps, config)

        self.assertEqual(reliable["temperature_reliable_endpoint_rate"], 1.0)
        self.assertEqual(
            unreliable["temperature_unreliable_endpoint_rate"], 1.0
        )

    def test_shuffled_residual_is_explicitly_not_applicable(self):
        torch.manual_seed(6)
        logits = torch.randn(1, 3, 2, 3)
        valid = torch.ones(1, 2, 3, dtype=torch.bool)
        cdf = make_cdf(torch.linspace(0.0, 2.0, 17))
        config = RTCConfig()
        maps = build_rtc_temperature_map(
            logits,
            logits / 3.0,
            valid,
            cdf,
            config,
            shuffle=True,
            shuffle_generator=torch.Generator().manual_seed(123),
        )
        diagnostics = collect_rtc_diagnostics(maps, config)

        self.assertEqual(diagnostics["residual_is_applicable"], 0.0)
        self.assertTrue(math.isnan(diagnostics["target_residual_mean"]))
        self.assertTrue(math.isnan(diagnostics["target_residual_p95"]))
        self.assertTrue(math.isnan(diagnostics["target_residual_max"]))


class MaskedKDLossTest(unittest.TestCase):
    def test_constant_all_valid_gamma2_matches_scalar_kd(self):
        torch.manual_seed(7)
        student = torch.randn(2, 5, 3, 4, requires_grad=True)
        teacher = torch.randn(2, 5, 3, 4)
        temperature = 0.6
        temperature_map = torch.full((2, 3, 4), temperature)
        valid = torch.ones_like(temperature_map, dtype=torch.bool)
        actual = masked_temperature_kd_loss(
            student,
            teacher,
            temperature_map,
            valid,
            temperature_power=2.0,
        )
        expected = F.kl_div(
            F.log_softmax(student / temperature, dim=1),
            F.softmax(teacher / temperature, dim=1),
            reduction="batchmean",
        ) * temperature**2 / (student.shape[2] * student.shape[3])
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_random_mask_gamma0_matches_manual_valid_pixels(self):
        torch.manual_seed(8)
        student = torch.randn(2, 4, 3, 5, requires_grad=True)
        teacher = torch.randn(2, 4, 3, 5) / 3.0
        temperature_map = torch.rand(2, 3, 5) * 1.5 + 0.5
        valid = torch.rand(2, 3, 5) > 0.3
        actual = masked_temperature_kd_loss(
            student, teacher, temperature_map, valid, temperature_power=0.0
        )
        temperature = temperature_map.unsqueeze(1)
        manual_map = F.kl_div(
            F.log_softmax(student / temperature, dim=1),
            F.softmax(teacher / temperature, dim=1),
            reduction="none",
        ).sum(dim=1)
        expected = manual_map[valid].mean()
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        actual.backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_all_ignore_returns_differentiable_zero(self):
        student = torch.randn(1, 3, 2, 2, requires_grad=True)
        teacher = torch.randn_like(student)
        temperature = torch.ones(1, 2, 2)
        valid = torch.zeros_like(temperature, dtype=torch.bool)
        loss = masked_temperature_kd_loss(student, teacher, temperature, valid)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(student.grad, torch.zeros_like(student.grad)))


class ShuffleTest(unittest.TestCase):
    def test_shuffle_preserves_each_images_valid_multiset(self):
        temperature = torch.tensor(
            [
                [[0.5, 0.7, 1.0], [1.1, 1.5, 2.0]],
                [[0.6, 0.8, 1.0], [1.2, 1.7, 1.9]],
            ]
        )
        valid = torch.tensor(
            [
                [[True, True, False], [True, True, True]],
                [[True, False, False], [False, False, False]],
            ]
        )
        generator = torch.Generator().manual_seed(123)
        shuffled = shuffle_temperature_map(temperature, valid, generator)
        for index in range(temperature.shape[0]):
            torch.testing.assert_close(
                torch.sort(shuffled[index][valid[index]]).values,
                torch.sort(temperature[index][valid[index]]).values,
            )
        torch.testing.assert_close(shuffled[~valid], temperature[~valid])


if __name__ == "__main__":
    unittest.main()
