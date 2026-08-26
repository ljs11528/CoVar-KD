import unittest

import torch
import torch.nn.functional as F

from utils.covar_metrics import (
    covar_coefficient,
    covar_components_from_probabilities,
    covar_components_from_sorted_logits,
    covar_derivatives_from_sorted_logits,
)


class CoVarMetricsTest(unittest.TestCase):
    def test_population_variance_and_stable_decomposition_match_legacy_formula(self):
        torch.manual_seed(17)
        logits = torch.randn(2, 21, 4, 5, dtype=torch.float64)
        probability = F.softmax(logits, dim=1)
        actual = covar_components_from_probabilities(probability, class_dim=1)

        sorted_probability = torch.sort(probability.movedim(1, -1), dim=-1, descending=True).values
        confidence = sorted_probability[..., 0]
        nonmax = sorted_probability[..., 1:]
        mean = nonmax.mean(dim=-1, keepdim=True)
        variance = torch.mean((nonmax - mean) ** 2, dim=-1)
        coefficient = covar_coefficient(21)
        expected_rv = coefficient * variance / (1.0 - confidence)

        self.assertEqual(coefficient, 200.0)
        torch.testing.assert_close(actual["residual_variance"], variance, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual["r_v"], expected_rv, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(actual["r"], actual["r_c"] + actual["r_v"], rtol=0.0, atol=0.0)

    def test_closed_derivatives_match_autograd_and_finite_difference(self):
        torch.manual_seed(23)
        logits = torch.sort(torch.randn(8, 7, dtype=torch.float64), dim=-1, descending=True).values
        temperature = torch.linspace(0.55, 1.95, 8, dtype=torch.float64, requires_grad=True)
        components = covar_components_from_sorted_logits(logits, temperature)
        autograd_first = torch.autograd.grad(components["r"].sum(), temperature, create_graph=True)[0]
        autograd_second = torch.autograd.grad(autograd_first.sum(), temperature)[0]
        closed = covar_derivatives_from_sorted_logits(logits, temperature.detach())

        torch.testing.assert_close(closed["dr_dT"], autograd_first.detach(), rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(closed["d2r_dT2"], autograd_second.detach(), rtol=1e-9, atol=1e-10)

        step = 1e-4
        center = covar_components_from_sorted_logits(logits, temperature.detach())["r"]
        plus = covar_components_from_sorted_logits(logits, temperature.detach() + step)["r"]
        minus = covar_components_from_sorted_logits(logits, temperature.detach() - step)["r"]
        finite_first = (plus - minus) / (2.0 * step)
        finite_second = (plus - 2.0 * center + minus) / (step ** 2)
        torch.testing.assert_close(closed["dr_dT"], finite_first, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(closed["d2r_dT2"], finite_second, rtol=1e-5, atol=1e-6)

    def test_documented_nonmonotonic_counterexample_changes_derivative_sign(self):
        logits = torch.tensor([[1.5, -1.2, 1.48]], dtype=torch.float64)
        sorted_logits = torch.sort(logits, dim=-1, descending=True).values
        low = covar_derivatives_from_sorted_logits(sorted_logits, torch.tensor([0.5], dtype=torch.float64))
        middle = covar_derivatives_from_sorted_logits(sorted_logits, torch.tensor([1.0], dtype=torch.float64))
        high = covar_derivatives_from_sorted_logits(sorted_logits, torch.tensor([2.0], dtype=torch.float64))
        self.assertGreater(float(low["dr_dT"].item()), 0.0)
        self.assertLess(float(middle["dr_dT"].item()), 0.0)
        self.assertGreater(float(high["dr_dT"].item()), 0.0)

    def test_binary_case_has_zero_residual_complexity(self):
        logits = torch.tensor([[2.0, -0.5], [0.7, 0.2]], dtype=torch.float64)
        actual = covar_components_from_sorted_logits(logits, temperature=1.0)
        torch.testing.assert_close(actual["r_v"], torch.zeros_like(actual["r_v"]))
        torch.testing.assert_close(
            actual["normalized_residual_variance"],
            torch.zeros_like(actual["r_v"]),
        )


if __name__ == "__main__":
    unittest.main()
