import torch
import torch.nn.functional as F

from utils.region_teachability import (
    aggregate_gradient_cosine,
    normalized_teacher_step_maps,
)


def test_normalized_step_gain_matches_direct_logit_update():
    student = torch.tensor(
        [[[[0.4]], [[-0.2]], [[0.1]]]], dtype=torch.float64
    )
    teacher = torch.tensor(
        [[[[1.5]], [[0.3]], [[-0.7]]]], dtype=torch.float64
    )
    target = torch.tensor([[[0]]])
    temperature = 1.5
    step_size = 0.1

    maps = normalized_teacher_step_maps(
        student, teacher, target, temperature, step_size
    )
    probability = F.softmax(student, dim=1)
    teacher_target = F.softmax(teacher / temperature, dim=1)
    gradient = probability - teacher_target
    updated = student - step_size * gradient / torch.linalg.vector_norm(
        gradient, dim=1, keepdim=True
    )
    expected_gain = (
        F.cross_entropy(student, target)
        - F.cross_entropy(updated, target)
    )
    assert torch.allclose(
        maps["ce_gain"].sum(), expected_gain, atol=1e-12, rtol=1e-12
    )


def test_gradient_cosine_uses_region_flattening_and_ignore_mask():
    student = torch.zeros((1, 2, 1, 2), dtype=torch.float64)
    teacher = torch.tensor(
        [[[[2.0, -2.0]], [[-2.0, 2.0]]]], dtype=torch.float64
    )
    target = torch.tensor([[[0, -1]]])
    maps = normalized_teacher_step_maps(student, teacher, target, 1.0)
    cosine = aggregate_gradient_cosine(
        maps["gradient_dot"],
        maps["supervised_gradient_sq"],
        maps["kd_gradient_sq"],
    )
    assert torch.allclose(
        cosine, torch.tensor(1.0, dtype=torch.float64), atol=1e-12
    )
    assert maps["ce_gain"][0, 0, 1].item() == 0.0
    assert maps["teacher_student_kl"][0, 0, 1].item() == 0.0


def test_wrong_teacher_direction_has_negative_gain():
    student = torch.zeros((1, 2, 1, 1), dtype=torch.float64)
    teacher = torch.tensor(
        [[[[-4.0]], [[4.0]]]], dtype=torch.float64
    )
    target = torch.tensor([[[0]]])
    maps = normalized_teacher_step_maps(student, teacher, target, 1.0)
    assert maps["ce_gain"].item() < 0
    cosine = aggregate_gradient_cosine(
        maps["gradient_dot"],
        maps["supervised_gradient_sq"],
        maps["kd_gradient_sq"],
    )
    assert cosine.item() < 0
