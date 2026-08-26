"""Teacher-target-only logit KD used by the CoVar Match experiments."""

import torch.nn.functional as F


def teacher_target_kd_loss(
    student_logits,
    teacher_logits,
    target_temperature,
    valid_mask=None,
):
    """Return mean KL(softmax(z_t/T) || softmax(z_s)).

    Temperature acts only on the teacher target. The student temperature is
    fixed at one and no ``T**2`` compensation is applied.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student and teacher logits must have identical shapes, got "
            f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )
    if student_logits.ndim != 4:
        raise ValueError("expected BCHW logits")
    if not float(target_temperature) > 0:
        raise ValueError("target_temperature must be positive")

    teacher_target = F.softmax(
        teacher_logits.detach() / float(target_temperature), dim=1
    )
    student_log_probability = F.log_softmax(student_logits, dim=1)
    loss_map = F.kl_div(
        student_log_probability,
        teacher_target,
        reduction="none",
    ).sum(dim=1)

    if valid_mask is None:
        return loss_map.mean()
    if valid_mask.ndim != 3 or valid_mask.shape[0] != loss_map.shape[0]:
        raise ValueError("valid_mask must have shape BHW")
    if valid_mask.shape[-2:] != loss_map.shape[-2:]:
        valid_mask = F.interpolate(
            valid_mask.float().unsqueeze(1),
            size=loss_map.shape[-2:],
            mode="nearest",
        ).squeeze(1) > 0.5
    else:
        valid_mask = valid_mask.bool()
    valid_count = valid_mask.sum()
    if int(valid_count.item()) == 0:
        raise ValueError("valid_mask contains no valid pixels")
    return loss_map[valid_mask].sum() / valid_count.to(loss_map.dtype)


__all__ = ["teacher_target_kd_loss"]
