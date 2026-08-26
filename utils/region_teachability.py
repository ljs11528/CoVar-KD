"""Offline teachability diagnostics for teacher-target-only KD."""

import torch
import torch.nn.functional as F


def normalized_teacher_step_maps(
    student_logits,
    teacher_logits,
    targets,
    target_temperature,
    step_size=0.1,
    ignore_label=-1,
    epsilon=1e-12,
):
    """Return per-pixel CE gain and gradient-alignment sufficient statistics.

    The KD direction is normalized independently at each pixel before taking a
    logit-space step of length step_size. This removes temperature-dependent
    gradient magnitude from the one-step teachability oracle.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if student_logits.ndim != 4 or targets.ndim != 3:
        raise ValueError("expected BCHW logits and BHW targets")
    if student_logits.shape[0] != targets.shape[0]:
        raise ValueError("batch dimensions do not match")
    if student_logits.shape[-2:] != targets.shape[-2:]:
        raise ValueError("targets must already be aligned to the logit grid")
    if not float(target_temperature) > 0:
        raise ValueError("target_temperature must be positive")
    if not float(step_size) > 0:
        raise ValueError("step_size must be positive")

    valid = targets != int(ignore_label)
    safe_targets = targets.masked_fill(~valid, 0)
    student_probability = F.softmax(student_logits, dim=1)
    teacher_target = F.softmax(
        teacher_logits.detach() / float(target_temperature), dim=1
    )
    one_hot = F.one_hot(
        safe_targets, num_classes=student_logits.shape[1]
    ).permute(0, 3, 1, 2).to(student_logits.dtype)

    supervised_gradient = student_probability - one_hot
    kd_gradient = student_probability - teacher_target
    kd_norm = torch.linalg.vector_norm(
        kd_gradient, ord=2, dim=1, keepdim=True
    )
    normalized_kd_gradient = kd_gradient / kd_norm.clamp_min(float(epsilon))
    updated_logits = student_logits - float(step_size) * normalized_kd_gradient

    ce_before = F.cross_entropy(
        student_logits, safe_targets, reduction="none"
    )
    ce_after = F.cross_entropy(
        updated_logits, safe_targets, reduction="none"
    )
    kl_map = F.kl_div(
        F.log_softmax(student_logits, dim=1),
        teacher_target,
        reduction="none",
    ).sum(dim=1)

    valid_float = valid.to(student_logits.dtype)
    return {
        "valid": valid,
        "ce_gain": (ce_before - ce_after) * valid_float,
        "gradient_dot": (
            supervised_gradient * kd_gradient
        ).sum(dim=1) * valid_float,
        "supervised_gradient_sq": (
            supervised_gradient.square().sum(dim=1) * valid_float
        ),
        "kd_gradient_sq": kd_gradient.square().sum(dim=1) * valid_float,
        "teacher_student_kl": kl_map * valid_float,
    }


def aggregate_gradient_cosine(
    gradient_dot,
    supervised_gradient_sq,
    kd_gradient_sq,
    epsilon=1e-12,
):
    """Aggregate flattened gradient cosine over an arbitrary region."""
    denominator = torch.sqrt(
        supervised_gradient_sq.sum() * kd_gradient_sq.sum()
    ).clamp_min(float(epsilon))
    return gradient_dot.sum() / denominator


__all__ = [
    "aggregate_gradient_cosine",
    "normalized_teacher_step_maps",
]
