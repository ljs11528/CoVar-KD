import types
import unittest

import torch
import torch.nn.functional as F

from utils.covar_temperature import (
    NewtonCoVarConfig,
    covar_temperature_kd_loss,
    newton_covar_temperature_map,
)


class NewtonCoVarTemperatureTest(unittest.TestCase):
    def test_matches_cirkdv2_newton_implementation(self):
        from train_cirkdv2 import Trainer as CirkdTrainer

        torch.manual_seed(7)
        teacher_logits = torch.randn(2, 5, 4, 3)
        valid_mask = torch.ones(2, 8, 6, dtype=torch.bool)
        valid_mask[:, :2, :2] = False

        config = NewtonCoVarConfig(
            base_temperature=1.0,
            min_temperature=0.5,
            max_temperature=8.0,
            kd_temperature_power=2.0,
            eta=0.6,
            max_iterations=8,
            hessian_epsilon=1e-5,
            max_step=0.25,
            coefficient_a=None,
            reliability_mode="full",
        )
        actual_temp, actual_r, actual_mask, _, _ = newton_covar_temperature_map(
            teacher_logits,
            valid_mask,
            config,
        )

        reference = CirkdTrainer.__new__(CirkdTrainer)
        reference.device = torch.device("cpu")
        reference.args = types.SimpleNamespace(
            covar_reliability_mode="full",
            covar_temp_base=1.0,
            covar_grad_eta=0.6,
            covar_grad_max_iter=8,
            covar_temp_mode="newton",
            covar_a=None,
            covar_temp_min=0.5,
            covar_temp_max=8.0,
            covar_grad_converge_thresh=1e-2,
            covar_newton_hessian_eps=1e-5,
            covar_newton_max_step=0.25,
        )
        reference_mask = F.interpolate(
            valid_mask.float().unsqueeze(1),
            size=teacher_logits.shape[-2:],
            mode="nearest",
        ).squeeze(1) > 0.5
        expected_temp, expected_r, _ = reference.get_gradient_temperature_map(
            teacher_logits,
            reference_mask,
        )

        self.assertTrue(torch.equal(actual_mask, reference_mask))
        torch.testing.assert_close(actual_temp, expected_temp, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actual_r, expected_r, rtol=1e-6, atol=1e-6)

    def test_constant_temperature_matches_scalar_kd(self):
        torch.manual_seed(11)
        student_logits = torch.randn(2, 4, 3, 5, requires_grad=True)
        teacher_logits = torch.randn(2, 4, 3, 5)
        temperature = 2.0
        temperature_map = torch.full((2, 3, 5), temperature)
        valid_mask = torch.ones(2, 3, 5, dtype=torch.bool)

        actual = covar_temperature_kd_loss(
            student_logits,
            teacher_logits,
            temperature_map,
            valid_mask,
            temperature_power=2.0,
        )
        student_flat = student_logits.permute(0, 2, 3, 1).reshape(-1, 4)
        teacher_flat = teacher_logits.permute(0, 2, 3, 1).reshape(-1, 4)
        expected = F.kl_div(
            F.log_softmax(student_flat / temperature, dim=1),
            F.softmax(teacher_flat / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature ** 2)

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        actual.backward()
        self.assertTrue(torch.isfinite(student_logits.grad).all())

    def test_temperature_bounds_and_invalid_pixels(self):
        torch.manual_seed(17)
        teacher_logits = torch.randn(1, 21, 5, 7)
        valid_mask = torch.ones(1, 5, 7, dtype=torch.bool)
        valid_mask[:, 0, 0] = False
        config = NewtonCoVarConfig()

        temperature, reliability, resized_mask, _, _ = newton_covar_temperature_map(
            teacher_logits,
            valid_mask,
            config,
        )

        self.assertTrue(torch.isfinite(temperature).all())
        self.assertTrue(torch.isfinite(reliability).all())
        self.assertGreaterEqual(float(temperature[resized_mask].min()), config.min_temperature)
        self.assertLessEqual(float(temperature[resized_mask].max()), config.max_temperature)
        self.assertEqual(float(temperature[~resized_mask].item()), config.base_temperature)
        self.assertEqual(float(reliability[~resized_mask].item()), 0.0)


if __name__ == "__main__":
    unittest.main()
