import pytest
import torch
import torch.nn.functional as F

from utils.teacher_only_kd import teacher_target_kd_loss


def test_teacher_temperature_does_not_scale_student_or_add_t_squared():
    student = torch.tensor(
        [[[[1.2]], [[-0.4]], [[0.3]]]], dtype=torch.float64, requires_grad=True
    )
    teacher = torch.tensor(
        [[[[2.0]], [[0.5]], [[-1.0]]]], dtype=torch.float64
    )
    temperature = 2.0

    actual = teacher_target_kd_loss(student, teacher, temperature)
    target = F.softmax(teacher / temperature, dim=1)
    expected = F.kl_div(
        F.log_softmax(student, dim=1), target, reduction="none"
    ).sum(dim=1).mean()
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)

    gradient = torch.autograd.grad(actual, student)[0]
    expected_gradient = F.softmax(student.detach(), dim=1) - target
    assert torch.allclose(
        gradient, expected_gradient, atol=1e-12, rtol=1e-12
    )


def test_valid_mask_is_resized_and_excludes_ignore_pixels():
    student = torch.zeros((1, 2, 1, 2), dtype=torch.float64)
    teacher = torch.tensor(
        [[[[2.0, -2.0]], [[-2.0, 2.0]]]], dtype=torch.float64
    )
    high_resolution_mask = torch.tensor(
        [[[True, True, False, False], [True, True, False, False]]]
    )
    actual = teacher_target_kd_loss(
        student, teacher, 1.0, high_resolution_mask
    )
    target = F.softmax(teacher[:, :, :, :1], dim=1)
    expected = F.kl_div(
        F.log_softmax(student[:, :, :, :1], dim=1),
        target,
        reduction="none",
    ).sum()
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_non_positive_temperature_is_rejected(temperature):
    logits = torch.zeros((1, 2, 1, 1))
    with pytest.raises(ValueError):
        teacher_target_kd_loss(logits, logits, temperature)


def test_empty_valid_mask_is_rejected():
    logits = torch.zeros((1, 2, 1, 1))
    with pytest.raises(ValueError):
        teacher_target_kd_loss(
            logits, logits, 1.0, torch.zeros((1, 1, 1), dtype=torch.bool)
        )
