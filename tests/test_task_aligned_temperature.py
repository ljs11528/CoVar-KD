import torch
import torch.nn.functional as F

from utils.task_aligned_temperature import (
    TASK_ALIGNED_FALLBACK_TEMPERATURE,
    TASK_ALIGNED_HIGH_MARGIN_ABSOLUTE,
    TASK_ALIGNED_HIGH_MARGIN_RELATIVE,
    TASK_ALIGNED_MIN_VALID_PIXELS,
    TASK_ALIGNED_REGION_SIZE,
    TASK_ALIGNED_TEMPERATURES,
    build_task_aligned_region_selection,
    select_region_temperature_indices,
    task_aligned_region_kd_loss,
    task_aligned_statistics_dict,
)


def test_p4a_protocol_constants_are_locked():
    assert TASK_ALIGNED_TEMPERATURES == (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    assert TASK_ALIGNED_REGION_SIZE == 8
    assert TASK_ALIGNED_MIN_VALID_PIXELS == 16
    assert TASK_ALIGNED_FALLBACK_TEMPERATURE == 1.5
    assert TASK_ALIGNED_HIGH_MARGIN_ABSOLUTE == 1e-4
    assert TASK_ALIGNED_HIGH_MARGIN_RELATIVE == 0.01


def test_hard_argmax_and_exact_tie_prefer_temperature_nearest_t1p5():
    scores = torch.tensor(
        [
            [
                [[0.1, 2.0, 7.0]],
                [[0.2, 1.0, 7.0]],
                [[0.3, 1.0, 7.0]],
                [[0.4, 0.0, 7.0]],
                [[0.5, 0.0, 7.0]],
                [[0.6, 2.0, 7.0]],
            ]
        ],
        dtype=torch.float64,
    )
    eligible = torch.tensor([[[True, True, False]]])
    selected, margin, exact_tie = select_region_temperature_indices(
        scores, eligible
    )
    assert selected.tolist() == [[[5, 5, 4]]]
    assert torch.allclose(
        margin, torch.tensor([[[0.1, 0.0, 0.0]]], dtype=torch.float64)
    )
    assert exact_tie.tolist() == [[[False, True, False]]]


def test_region_scores_match_direct_task_alignment_definition():
    torch.manual_seed(7)
    student = torch.randn((1, 3, 8, 8), dtype=torch.float64)
    teacher = torch.randn((1, 3, 8, 8), dtype=torch.float64)
    target = torch.randint(0, 3, (1, 8, 8))
    selection = build_task_aligned_region_selection(
        student, teacher, target
    )

    p_s = F.softmax(student, dim=1)
    one_hot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2)
    expected_scores = []
    for temperature in TASK_ALIGNED_TEMPERATURES:
        p_t = F.softmax(teacher / temperature, dim=1)
        kd_gradient = p_s - p_t
        direction = kd_gradient / torch.linalg.vector_norm(
            kd_gradient, dim=1, keepdim=True
        ).clamp_min(1e-12)
        expected_scores.append(((p_s - one_hot) * direction).sum(dim=1).mean())
    expected = torch.stack(expected_scores)
    assert torch.allclose(
        selection.region_scores[0, :, 0, 0],
        expected,
        atol=1e-12,
        rtol=1e-12,
    )
    assert selection.selected_region_index.item() == expected.argmax().item()
    assert torch.all(
        selection.temperature_map
        == TASK_ALIGNED_TEMPERATURES[expected.argmax().item()]
    )


def test_sparse_region_falls_back_to_t1p5_and_is_excluded_from_selection_stats():
    torch.manual_seed(11)
    student = torch.randn((1, 3, 8, 16), dtype=torch.float64)
    teacher = torch.randn((1, 3, 8, 16), dtype=torch.float64)
    target = torch.zeros((1, 8, 16), dtype=torch.long)
    target[:, :, 8:] = -1
    target[:, :2, 8:13] = 1
    selection = build_task_aligned_region_selection(
        student, teacher, target
    )

    assert selection.eligible_region_mask.tolist() == [[[True, False]]]
    assert torch.all(selection.temperature_map[:, :, 8:] == 1.5)
    diagnostics = task_aligned_statistics_dict(selection.statistics)
    assert diagnostics["eligible_regions"] == 1
    assert diagnostics["nonempty_regions"] == 2
    assert diagnostics["fallback_regions"] == 1
    assert sum(diagnostics["temperature_counts"]) == 1
    assert diagnostics["complexity_valid_pixels"] == 64


def test_selected_target_is_detached_and_kl_gradient_has_teacher_to_student_order():
    torch.manual_seed(19)
    student = torch.randn(
        (1, 4, 8, 8), dtype=torch.float64, requires_grad=True
    )
    teacher = torch.randn(
        (1, 4, 8, 8), dtype=torch.float64, requires_grad=True
    )
    target = torch.randint(0, 4, (1, 8, 8))
    target[:, 0, 0] = -1
    selection = build_task_aligned_region_selection(
        student, teacher, target
    )
    assert not selection.teacher_target.requires_grad
    assert not selection.temperature_map.requires_grad
    assert not selection.region_scores.requires_grad

    loss = task_aligned_region_kd_loss(student, selection)
    manual_map = (
        selection.teacher_target
        * (
            torch.log(selection.teacher_target.clamp_min(1e-12))
            - F.log_softmax(student, dim=1)
        )
    ).sum(dim=1)
    expected_loss = manual_map[selection.valid_mask].mean()
    assert torch.allclose(loss, expected_loss, atol=1e-12, rtol=1e-12)

    actual_gradient = torch.autograd.grad(loss, student)[0]
    expected_gradient = (
        (F.softmax(student.detach(), dim=1) - selection.teacher_target)
        * selection.valid_mask.unsqueeze(1)
        / selection.valid_mask.sum()
    )
    assert torch.allclose(
        actual_gradient,
        expected_gradient,
        atol=1e-12,
        rtol=1e-12,
    )
    assert teacher.grad is None


def test_statistics_preserve_covar_decomposition_and_nonnegative_argmax_gain():
    torch.manual_seed(23)
    student = torch.randn((2, 3, 8, 8), dtype=torch.float64)
    teacher = torch.randn((2, 3, 8, 8), dtype=torch.float64)
    target = torch.randint(0, 3, (2, 8, 8))
    selection = build_task_aligned_region_selection(
        student, teacher, target
    )
    diagnostics = task_aligned_statistics_dict(selection.statistics)
    assert diagnostics["eligible_regions"] == 2
    assert sum(diagnostics["temperature_counts"]) == 2
    assert diagnostics["mean_delta_alignment_vs_t1p5"] >= -1e-14
    assert abs(
        diagnostics["mean_selected_r"]
        - diagnostics["mean_selected_r_c"]
        - diagnostics["mean_selected_r_v"]
    ) < 1e-12
    assert 0.0 <= diagnostics["high_margin_absolute_fraction"] <= 1.0
    assert 0.0 <= diagnostics["high_margin_relative_fraction"] <= 1.0
