import hashlib
import math
import unittest

import torch
import torch.nn.functional as F

from utils.rtc_o12_calibration import (
    O12_A_STAR,
    O12_B_MAX,
    O12_BISECTION_ITERATIONS,
    O12CalibrationConfig,
    build_o12_teacher_target,
    build_o12_temperature_map,
    compute_o12_gates,
    compute_o12_masked_kl_terms,
    compute_o12_shuffle_seed,
    compute_o12_temperature_statistics,
    normalize_o12_ddp_loss,
    o12_teacher_only_pixel_kl,
    shuffle_o12_temperature_within_images,
    solve_o12_budget_parameter,
)


class O12ConfigAndGateTest(unittest.TestCase):
    def test_frozen_constants_and_config_reject_post_registration_change(self):
        config = O12CalibrationConfig()
        config.validate()
        self.assertEqual(config.a_star, O12_A_STAR)
        self.assertEqual(config.b_max, O12_B_MAX)
        self.assertEqual(config.bisection_iterations, O12_BISECTION_ITERATIONS)
        with self.assertRaisesRegex(ValueError, "q_reliable is frozen"):
            O12CalibrationConfig(q_reliable=0.59).validate()
        with self.assertRaisesRegex(ValueError, "bisection_iterations is frozen"):
            O12CalibrationConfig(bisection_iterations=63).validate()

    def test_gate_boundaries_powers_and_mutual_exclusion(self):
        quantile = torch.tensor(
            [0.0, 0.3, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=torch.float64
        )
        gate_reliable, gate_unreliable = compute_o12_gates(quantile)
        torch.testing.assert_close(
            gate_reliable,
            torch.tensor(
                [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=torch.float64,
            ),
            rtol=0,
            atol=1e-15,
        )
        torch.testing.assert_close(
            gate_unreliable,
            torch.tensor(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 1.0],
                dtype=torch.float64,
            ),
            rtol=0,
            atol=1e-15,
        )
        self.assertTrue(torch.equal(
            gate_reliable * gate_unreliable,
            torch.zeros_like(gate_reliable),
        ))

    def test_invalid_pixels_are_neutral_and_valid_bad_quantiles_fail(self):
        quantile = torch.tensor([0.0, float("nan"), 1.0])
        valid = torch.tensor([True, False, True])
        gate_reliable, gate_unreliable = compute_o12_gates(quantile, valid)
        self.assertEqual(float(gate_reliable[1]), 0.0)
        self.assertEqual(float(gate_unreliable[1]), 0.0)
        temperature = build_o12_temperature_map(
            quantile,
            valid,
            b=0.3,
            branch="full_budgeted",
        )
        self.assertEqual(float(temperature[1]), 1.0)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            compute_o12_gates(quantile, torch.ones(3, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "must be in"):
            compute_o12_gates(torch.tensor([-0.01, 0.5]))
        with self.assertRaises(TypeError):
            compute_o12_gates(torch.tensor([0, 1]))


class O12TemperatureMapTest(unittest.TestCase):
    def test_neutral_and_each_single_branch_have_frozen_coefficients(self):
        quantile = torch.tensor([0.0, 0.7, 1.0])
        b = math.log(1.4)
        neutral = build_o12_temperature_map(quantile, branch="neutral")
        reliable = build_o12_temperature_map(
            quantile, b=0.0, branch="reliable_only"
        )
        unreliable = build_o12_temperature_map(
            quantile, b=b, branch="unreliable_only"
        )
        full = build_o12_temperature_map(
            quantile, b=b, branch="full_budgeted"
        )
        self.assertEqual(neutral.dtype, torch.float32)
        self.assertTrue(torch.equal(neutral, torch.ones_like(neutral)))
        torch.testing.assert_close(
            reliable,
            torch.tensor([0.9, 1.0, 1.0]),
            rtol=1e-6,
            atol=1e-7,
        )
        torch.testing.assert_close(
            unreliable,
            torch.tensor([1.0, 1.0, 1.4]),
            rtol=1e-6,
            atol=1e-7,
        )
        torch.testing.assert_close(
            full,
            torch.tensor([0.9, 1.0, 1.4]),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_full_temperature_is_globally_monotonic_on_checker_grid(self):
        quantile = torch.linspace(0.0, 1.0, 4097, dtype=torch.float64)
        temperature = build_o12_temperature_map(
            quantile,
            b=0.34788,
            branch="full_budgeted",
        )
        self.assertTrue(bool((temperature[1:] >= temperature[:-1]).all()))
        self.assertGreaterEqual(float(temperature.min()), 0.9 - 1e-6)
        self.assertLessEqual(float(temperature.max()), 1.5 + 1e-6)
        middle = (quantile >= 0.6) & (quantile <= 0.8)
        self.assertTrue(torch.equal(
            temperature[middle], torch.ones_like(temperature[middle])
        ))

    def test_high_risk_branches_require_explicit_in_range_b(self):
        quantile = torch.tensor([0.5, 0.9])
        for branch in ("unreliable_only", "full_budgeted"):
            with self.subTest(branch=branch):
                with self.assertRaisesRegex(ValueError, "requires an explicit"):
                    build_o12_temperature_map(quantile, branch=branch)
        with self.assertRaisesRegex(ValueError, "requires b=0"):
            build_o12_temperature_map(
                quantile,
                b=0.1,
                branch="reliable_only",
            )
        with self.assertRaisesRegex(ValueError, "must be in"):
            build_o12_temperature_map(
                quantile,
                b=O12_B_MAX + 1e-6,
                branch="full_budgeted",
            )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            build_o12_temperature_map(quantile, branch="full")

    def test_statistics_use_valid_values_and_harmonic_definition(self):
        temperature = torch.tensor([0.9, 1.0, 1.5, 99.0], dtype=torch.float32)
        valid = torch.tensor([True, True, True, False])
        statistics = compute_o12_temperature_statistics(temperature, valid)
        values64 = temperature[:3].double()
        self.assertEqual(statistics.valid_count, 3)
        self.assertEqual(
            statistics.arithmetic_mean,
            float(values64.sum().item() / 3),
        )
        self.assertEqual(
            statistics.harmonic_mean,
            float(3 / values64.reciprocal().sum().item()),
        )
        with self.assertRaisesRegex(ValueError, "at least one"):
            compute_o12_temperature_statistics(
                temperature,
                torch.zeros_like(valid),
            )


class O12BudgetSolverTest(unittest.TestCase):
    def test_fixed_64_step_solver_reproduces_target_and_theoretical_endpoint(self):
        quantile = torch.linspace(0.0, 1.0, 10001, dtype=torch.float32)
        first = solve_o12_budget_parameter(quantile)
        second = solve_o12_budget_parameter(quantile.clone())
        self.assertEqual(first, second)
        self.assertEqual(first.bisection_iterations, 64)
        self.assertEqual(first.valid_count, quantile.numel())
        self.assertLessEqual(abs(first.residual), 1e-8)
        self.assertGreaterEqual(first.harmonic_mean, 0.98)
        self.assertGreaterEqual(math.exp(first.b), 1.25)
        temperature = build_o12_temperature_map(
            quantile,
            b=first.b,
            branch="full_budgeted",
        )
        statistics = compute_o12_temperature_statistics(temperature)
        self.assertEqual(statistics.arithmetic_mean, first.arithmetic_mean)
        self.assertEqual(statistics.harmonic_mean, first.harmonic_mean)

    def test_solver_rejects_absent_root_and_nonunique_population(self):
        with self.assertRaisesRegex(ValueError, "has no root"):
            solve_o12_budget_parameter(torch.ones(100))
        with self.assertRaisesRegex(ValueError, "not unique"):
            solve_o12_budget_parameter(torch.zeros(100))
        with self.assertRaisesRegex(ValueError, "at least one"):
            solve_o12_budget_parameter(
                torch.zeros(5),
                torch.zeros(5, dtype=torch.bool),
            )

    def test_solver_rejects_root_that_fails_harmonic_gate(self):
        # A root exists near T_high=1.375, but 80% of pixels at T=0.9 makes
        # the harmonic mean smaller than the frozen 0.98 gate.
        quantile = torch.cat((torch.zeros(80), torch.ones(20)))
        with self.assertRaisesRegex(ValueError, "harmonic-mean gate failed"):
            solve_o12_budget_parameter(quantile)


class O12TeacherTargetTest(unittest.TestCase):
    def test_confidence_entropy_argmax_direction_and_neutral_identity(self):
        raw_logits = torch.tensor(
            [[
                [[3.0, 3.0, 3.0]],
                [[1.0, 1.0, 1.0]],
                [[-1.0, -1.0, -1.0]],
            ]],
            dtype=torch.float64,
        )
        temperature = torch.tensor([[[0.9, 1.0, 1.4]]], dtype=torch.float32)
        target = build_o12_teacher_target(raw_logits, temperature)
        baseline = F.softmax(raw_logits / 3.0, dim=1)
        confidence = target.max(dim=1).values
        baseline_confidence = baseline.max(dim=1).values
        entropy = -(target * target.log()).sum(dim=1)
        baseline_entropy = -(baseline * baseline.log()).sum(dim=1)
        self.assertGreaterEqual(float(confidence[0, 0, 0]), float(baseline_confidence[0, 0, 0]))
        self.assertLessEqual(float(entropy[0, 0, 0]), float(baseline_entropy[0, 0, 0]))
        torch.testing.assert_close(target[:, :, :, 1], baseline[:, :, :, 1], rtol=0, atol=0)
        self.assertLessEqual(float(confidence[0, 0, 2]), float(baseline_confidence[0, 0, 2]))
        self.assertGreaterEqual(float(entropy[0, 0, 2]), float(baseline_entropy[0, 0, 2]))
        self.assertTrue(torch.equal(target.argmax(dim=1), raw_logits.argmax(dim=1)))

    def test_target_is_detached_and_teacher_is_divided_only_by_tout_times_map(self):
        torch.manual_seed(12)
        raw_teacher = torch.randn(2, 4, 3, 2, requires_grad=True)
        temperature = torch.rand(2, 3, 2) * 0.5 + 0.9
        target = build_o12_teacher_target(raw_teacher, temperature)
        expected = F.softmax(
            raw_teacher.detach() / (3.0 * temperature.unsqueeze(1)), dim=1
        )
        self.assertFalse(target.requires_grad)
        torch.testing.assert_close(target, expected, rtol=1e-6, atol=1e-7)

    def test_shape_nonfinite_and_nonpositive_temperature_hard_fail(self):
        teacher = torch.randn(1, 3, 2, 2)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            build_o12_teacher_target(teacher, torch.ones(1, 3, 2))
        bad_temperature = torch.ones(1, 2, 2)
        bad_temperature[0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            build_o12_teacher_target(teacher, bad_temperature)
        bad_teacher = teacher.clone()
        bad_teacher[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "must be finite"):
            build_o12_teacher_target(bad_teacher, torch.ones(1, 2, 2))


class O12PixelKLLossTest(unittest.TestCase):
    def test_masked_terms_match_manual_kl_cross_entropy_and_entropy(self):
        torch.manual_seed(21)
        student = torch.randn(2, 5, 3, 4, requires_grad=True)
        raw_teacher = torch.randn(2, 5, 3, 4)
        temperature = torch.rand(2, 3, 4) * 0.5 + 0.9
        valid = torch.rand(2, 3, 4) > 0.3
        target = build_o12_teacher_target(raw_teacher, temperature)
        terms = compute_o12_masked_kl_terms(student, target, valid)
        student_log_probability = F.log_softmax(student, dim=1)
        manual_kl = F.kl_div(
            student_log_probability,
            target,
            reduction="none",
        ).sum(dim=1)[valid].sum()
        manual_ce = -(target * student_log_probability).sum(dim=1)[valid].sum()
        manual_entropy = -(target * target.log()).sum(dim=1)[valid].sum()
        torch.testing.assert_close(terms.kl_sum, manual_kl)
        torch.testing.assert_close(terms.cross_entropy_sum, manual_ce)
        torch.testing.assert_close(terms.teacher_entropy_sum, manual_entropy)
        torch.testing.assert_close(
            terms.kl_sum,
            terms.cross_entropy_sum - terms.teacher_entropy_sum,
            rtol=2e-5,
            atol=2e-6,
        )
        self.assertEqual(int(terms.valid_count), int(valid.sum()))

    def test_neutral_baseline_student_temperature_one_and_teacher_detach(self):
        torch.manual_seed(22)
        student = torch.randn(1, 4, 2, 3, requires_grad=True)
        teacher = torch.randn(1, 4, 2, 3, requires_grad=True)
        valid = torch.ones(1, 2, 3, dtype=torch.bool)
        neutral = torch.ones(1, 2, 3)
        result = o12_teacher_only_pixel_kl(student, teacher, neutral, valid)
        expected_target = F.softmax(teacher.detach() / 3.0, dim=1)
        expected = F.kl_div(
            F.log_softmax(student, dim=1),
            expected_target,
            reduction="none",
        ).sum(dim=1).mean()
        torch.testing.assert_close(result.teacher_target, expected_target)
        torch.testing.assert_close(result.loss, expected)
        self.assertFalse(result.student_probability.requires_grad)
        torch.testing.assert_close(
            result.student_probability,
            F.softmax(student.detach(), dim=1),
            rtol=1e-6,
            atol=1e-7,
        )
        result.loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_spatial_temperature_does_not_enter_student_softmax(self):
        torch.manual_seed(23)
        student = torch.randn(1, 3, 2, 2)
        teacher = torch.randn(1, 3, 2, 2)
        valid = torch.ones(1, 2, 2, dtype=torch.bool)
        low = o12_teacher_only_pixel_kl(
            student,
            teacher,
            torch.full((1, 2, 2), 0.9),
            valid,
        )
        high = o12_teacher_only_pixel_kl(
            student,
            teacher,
            torch.full((1, 2, 2), 1.4),
            valid,
        )
        self.assertTrue(torch.equal(
            low.student_probability, high.student_probability
        ))
        self.assertFalse(low.student_probability.requires_grad)
        self.assertFalse(high.student_probability.requires_grad)
        self.assertFalse(torch.equal(low.teacher_target, high.teacher_target))

    def test_zero_valid_loss_is_differentiable_student_connected_zero(self):
        student = torch.randn(1, 3, 2, 2, requires_grad=True)
        teacher = torch.randn_like(student)
        valid = torch.zeros(1, 2, 2, dtype=torch.bool)
        result = o12_teacher_only_pixel_kl(
            student,
            teacher,
            torch.ones(1, 2, 2),
            valid,
        )
        self.assertEqual(float(result.loss.detach()), 0.0)
        self.assertTrue(result.loss.requires_grad)
        result.loss.backward()
        self.assertTrue(torch.equal(student.grad, torch.zeros_like(student.grad)))

    def test_ddp_rank_formula_averages_to_concatenated_global_mean(self):
        local_sum_0 = torch.tensor(7.0, requires_grad=True)
        local_sum_1 = torch.tensor(11.0, requires_grad=True)
        global_count = torch.tensor(9)
        loss_0 = normalize_o12_ddp_loss(local_sum_0, global_count, world_size=2)
        loss_1 = normalize_o12_ddp_loss(local_sum_1, global_count, world_size=2)
        ddp_averaged = (loss_0 + loss_1) / 2.0
        expected = (local_sum_0 + local_sum_1) / global_count
        torch.testing.assert_close(ddp_averaged, expected, rtol=0, atol=0)
        ddp_averaged.backward()
        self.assertAlmostEqual(float(local_sum_0.grad), 1.0 / 9.0, places=7)
        self.assertAlmostEqual(float(local_sum_1.grad), 1.0 / 9.0, places=7)

    def test_constant_full_map_is_exactly_equivalent_to_scalar_target_and_loss(self):
        torch.manual_seed(24)
        student = torch.randn(1, 4, 2, 3)
        teacher = torch.randn(1, 4, 2, 3)
        valid = torch.tensor([[[True, False, True], [True, True, False]]])
        quantile = torch.ones(1, 2, 3)
        b = math.log(1.4)
        full_map = build_o12_temperature_map(
            quantile,
            b=b,
            branch="full_budgeted",
        )
        scalar_map = torch.full_like(full_map, 1.4)
        self.assertTrue(torch.equal(full_map, scalar_map))
        full = o12_teacher_only_pixel_kl(
            student, teacher, full_map, valid
        )
        scalar = o12_teacher_only_pixel_kl(
            student, teacher, scalar_map, valid
        )
        self.assertTrue(torch.equal(full.teacher_target, scalar.teacher_target))
        self.assertTrue(torch.equal(full.loss, scalar.loss))

    def test_logits_shape_mismatch_hard_fails_without_interpolation(self):
        student = torch.randn(1, 3, 2, 2)
        teacher = torch.randn(1, 3, 3, 2)
        with self.assertRaisesRegex(ValueError, "exactly the same shape"):
            o12_teacher_only_pixel_kl(
                student,
                teacher,
                torch.ones(1, 3, 2),
                torch.ones(1, 2, 2, dtype=torch.bool),
            )


class O12StatelessShuffleTest(unittest.TestCase):
    def test_seed_uses_exact_registered_payload_and_big_endian_prefix(self):
        payload = b"rtc_o12_shuffle_v1|3407|17|23"
        expected = int.from_bytes(
            hashlib.sha256(payload).digest()[:8],
            byteorder="big",
            signed=False,
        ) % ((1 << 63) - 1)
        self.assertEqual(compute_o12_shuffle_seed(17, 23), expected)
        with self.assertRaisesRegex(ValueError, "1-based"):
            compute_o12_shuffle_seed(17, 0)

    def test_exact_mapping_direction_and_invalid_pixels_become_one(self):
        temperature = torch.tensor([[[1.1, 9.0, 1.2, 1.3, 8.0, 1.4, 1.5]]])
        valid = torch.tensor([[[True, False, True, True, False, True, True]]])
        dataset_index = 7
        iteration = 11
        shuffled = shuffle_o12_temperature_within_images(
            temperature,
            valid,
            dataset_index,
            iteration,
        )
        valid_indices = torch.tensor([0, 2, 3, 5, 6])
        generator = torch.Generator(device="cpu").manual_seed(
            compute_o12_shuffle_seed(dataset_index, iteration)
        )
        permutation = torch.randperm(5, generator=generator)
        expected = torch.ones_like(temperature).reshape(-1)
        original_flat = temperature.reshape(-1)
        expected[valid_indices] = original_flat[valid_indices[permutation]]
        torch.testing.assert_close(shuffled.reshape(-1), expected, rtol=0, atol=0)

    def test_each_image_multiset_resume_and_batch_order_are_deterministic(self):
        first_image = torch.arange(1, 21, dtype=torch.float32).reshape(4, 5)
        second_image = torch.arange(101, 121, dtype=torch.float32).reshape(4, 5)
        temperature = torch.stack((first_image, second_image))
        valid = torch.ones_like(temperature, dtype=torch.bool)
        direct = shuffle_o12_temperature_within_images(
            temperature,
            valid,
            [13, 29],
            global_iteration=101,
        )
        resumed = shuffle_o12_temperature_within_images(
            temperature,
            valid,
            torch.tensor([13, 29]),
            global_iteration=101,
        )
        torch.testing.assert_close(direct, resumed, rtol=0, atol=0)
        reversed_batch = shuffle_o12_temperature_within_images(
            temperature.flip(0),
            valid.flip(0),
            [29, 13],
            global_iteration=101,
        ).flip(0)
        torch.testing.assert_close(direct, reversed_batch, rtol=0, atol=0)
        next_iteration = shuffle_o12_temperature_within_images(
            temperature,
            valid,
            [13, 29],
            global_iteration=102,
        )
        self.assertFalse(torch.equal(direct, next_iteration))
        for index in range(2):
            torch.testing.assert_close(
                torch.sort(direct[index][valid[index]]).values,
                torch.sort(temperature[index][valid[index]]).values,
                rtol=0,
                atol=0,
            )

    def test_zero_and_one_valid_pixels_are_identity_on_valid_values(self):
        temperature = torch.tensor(
            [[[7.0, 8.0]], [[1.25, 6.0]]], dtype=torch.float32
        )
        valid = torch.tensor([[[False, False]], [[True, False]]])
        shuffled = shuffle_o12_temperature_within_images(
            temperature,
            valid,
            [1, 2],
            1,
        )
        expected = torch.tensor([[[1.0, 1.0]], [[1.25, 1.0]]])
        torch.testing.assert_close(shuffled, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
