"""Task-aligned region temperature selection for teacher-target-only KD.

The selector is deliberately discrete and detached.  It evaluates the
first-order supervised alignment of six fixed teacher temperatures on 8x8
regions, chooses a hard argmax, and falls back to T=1.5 for sparse regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from utils.covar_metrics import covar_components_from_probabilities


TASK_ALIGNED_KD_LOSS_MODE = "task_aligned_region"
TASK_ALIGNED_TEMPERATURES: Tuple[float, ...] = (
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
)
TASK_ALIGNED_REGION_SIZE = 8
TASK_ALIGNED_MIN_VALID_PIXELS = 16
TASK_ALIGNED_FALLBACK_TEMPERATURE = 1.5
TASK_ALIGNED_EPSILON = 1e-12
TASK_ALIGNED_HIGH_MARGIN_ABSOLUTE = 1e-4
TASK_ALIGNED_HIGH_MARGIN_RELATIVE = 0.01

_TEMPERATURE_COUNT_START = 0
_ELIGIBLE_REGIONS = len(TASK_ALIGNED_TEMPERATURES)
_NONEMPTY_REGIONS = _ELIGIBLE_REGIONS + 1
_FALLBACK_REGIONS = _NONEMPTY_REGIONS + 1
_EXACT_TIES = _FALLBACK_REGIONS + 1
_MARGIN_SUM = _EXACT_TIES + 1
_HIGH_MARGIN_ABSOLUTE_COUNT = _MARGIN_SUM + 1
_HIGH_MARGIN_RELATIVE_COUNT = _HIGH_MARGIN_ABSOLUTE_COUNT + 1
_DELTA_ALIGNMENT_SUM = _HIGH_MARGIN_RELATIVE_COUNT + 1
_SELECTED_ALIGNMENT_SUM = _DELTA_ALIGNMENT_SUM + 1
_COMPLEXITY_VALID_PIXELS = _SELECTED_ALIGNMENT_SUM + 1
_R_C_SUM = _COMPLEXITY_VALID_PIXELS + 1
_R_V_SUM = _R_C_SUM + 1
_R_SUM = _R_V_SUM + 1
TASK_ALIGNED_STATISTICS_SIZE = _R_SUM + 1


@dataclass(frozen=True)
class TaskAlignedRegionSelection:
    """Detached selector output and its sufficient training diagnostics."""

    teacher_target: torch.Tensor
    temperature_map: torch.Tensor
    valid_mask: torch.Tensor
    eligible_region_mask: torch.Tensor
    selected_region_index: torch.Tensor
    region_scores: torch.Tensor
    margin: torch.Tensor
    statistics: torch.Tensor


def _candidate_tensor(
    temperatures: Sequence[float], like: torch.Tensor
) -> torch.Tensor:
    values = tuple(float(value) for value in temperatures)
    if values != TASK_ALIGNED_TEMPERATURES:
        raise ValueError(
            "task-aligned candidate temperatures are locked to {}".format(
                TASK_ALIGNED_TEMPERATURES
            )
        )
    return torch.as_tensor(values, device=like.device, dtype=like.dtype)


def _fallback_index(temperatures: torch.Tensor, fallback: float) -> int:
    matches = torch.nonzero(
        temperatures == float(fallback), as_tuple=False
    ).flatten()
    if matches.numel() != 1:
        raise ValueError("fallback temperature must occur exactly once")
    return int(matches.item())


def _align_targets(
    targets: torch.Tensor, spatial_shape: Tuple[int, int]
) -> torch.Tensor:
    if targets.ndim != 3:
        raise ValueError("targets must have shape BHW")
    if targets.shape[-2:] == spatial_shape:
        return targets.long()
    return F.interpolate(
        targets.float().unsqueeze(1),
        size=spatial_shape,
        mode="nearest",
    ).squeeze(1).long()


def _region_sum(value: torch.Tensor, region_size: int) -> torch.Tensor:
    """Sum BCHW maps in non-overlapping regions, padding only at boundaries."""
    height, width = value.shape[-2:]
    pad_height = (-height) % int(region_size)
    pad_width = (-width) % int(region_size)
    padded = F.pad(value, (0, pad_width, 0, pad_height), value=0.0)
    return F.avg_pool2d(
        padded,
        kernel_size=int(region_size),
        stride=int(region_size),
    ) * float(region_size * region_size)


def select_region_temperature_indices(
    region_scores: torch.Tensor,
    eligible_region_mask: torch.Tensor,
    temperatures: Sequence[float] = TASK_ALIGNED_TEMPERATURES,
    fallback_temperature: float = TASK_ALIGNED_FALLBACK_TEMPERATURE,
):
    """Hard argmax with exact-tie preference for the candidate nearest 1.5."""
    if region_scores.ndim != 4:
        raise ValueError("region_scores must have shape BKHW")
    if eligible_region_mask.shape != (
        region_scores.shape[0],
        region_scores.shape[2],
        region_scores.shape[3],
    ):
        raise ValueError("eligible_region_mask must have shape BHW")
    temperature_values = _candidate_tensor(temperatures, region_scores)
    if region_scores.shape[1] != temperature_values.numel():
        raise ValueError("candidate dimension does not match temperatures")
    eligible = eligible_region_mask.bool()
    if eligible.any() and not torch.isfinite(
        region_scores.movedim(1, -1)[eligible]
    ).all():
        raise FloatingPointError("eligible region scores must be finite")

    maximum = region_scores.max(dim=1).values
    exact_tie_mask = region_scores == maximum.unsqueeze(1)
    distance = (temperature_values - float(fallback_temperature)).abs().view(
        1, -1, 1, 1
    )
    preference = torch.where(
        exact_tie_mask,
        distance,
        torch.full_like(distance, float("inf")),
    )
    preferred_index = preference.argmin(dim=1)
    fallback_index = _fallback_index(
        temperature_values, fallback_temperature
    )
    selected_index = torch.where(
        eligible,
        preferred_index,
        torch.full_like(preferred_index, fallback_index),
    )
    top_two = torch.topk(region_scores, k=2, dim=1).values
    margin = torch.where(
        eligible,
        top_two[:, 0] - top_two[:, 1],
        torch.zeros_like(maximum),
    )
    exact_tie = eligible & (exact_tie_mask.sum(dim=1) > 1)
    return selected_index, margin, exact_tie


@torch.no_grad()
def build_task_aligned_region_selection(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_label: int = -1,
    temperatures: Sequence[float] = TASK_ALIGNED_TEMPERATURES,
    region_size: int = TASK_ALIGNED_REGION_SIZE,
    min_valid_pixels: int = TASK_ALIGNED_MIN_VALID_PIXELS,
    fallback_temperature: float = TASK_ALIGNED_FALLBACK_TEMPERATURE,
    epsilon: float = TASK_ALIGNED_EPSILON,
) -> TaskAlignedRegionSelection:
    """Build a detached region selector and spatial teacher target.

    The region score is the mean pixelwise dot product
    ``<p_s - e_y, (p_s - p_t(T)) / ||p_s - p_t(T)||_2>``.  No model
    forward, real one-step update, or higher-order gradient is performed.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if student_logits.ndim != 4:
        raise ValueError("expected BCHW logits")
    if student_logits.shape[0] != targets.shape[0]:
        raise ValueError("batch dimensions do not match")
    if int(region_size) != TASK_ALIGNED_REGION_SIZE:
        raise ValueError("task-aligned region size is locked to 8")
    if int(min_valid_pixels) != TASK_ALIGNED_MIN_VALID_PIXELS:
        raise ValueError("task-aligned minimum valid pixels is locked to 16")
    if float(fallback_temperature) != TASK_ALIGNED_FALLBACK_TEMPERATURE:
        raise ValueError("task-aligned fallback temperature is locked to 1.5")
    if float(epsilon) <= 0:
        raise ValueError("epsilon must be positive")

    student = student_logits.detach()
    teacher = teacher_logits.detach()
    temperature_values = _candidate_tensor(temperatures, student)
    aligned_targets = _align_targets(targets, student.shape[-2:])
    valid = aligned_targets != int(ignore_label)
    safe_targets = aligned_targets.masked_fill(~valid, 0)

    student_probability = F.softmax(student, dim=1)
    teacher_probabilities = torch.stack(
        [F.softmax(teacher / temperature, dim=1)
         for temperature in temperature_values],
        dim=1,
    )
    one_hot = F.one_hot(
        safe_targets, num_classes=student.shape[1]
    ).permute(0, 3, 1, 2).to(student.dtype)
    supervised_gradient = student_probability - one_hot
    kd_gradient = student_probability.unsqueeze(1) - teacher_probabilities
    kd_norm = torch.linalg.vector_norm(
        kd_gradient, ord=2, dim=2, keepdim=True
    )
    normalized_kd_gradient = kd_gradient / kd_norm.clamp_min(float(epsilon))
    alignment_map = (
        supervised_gradient.unsqueeze(1) * normalized_kd_gradient
    ).sum(dim=2)
    alignment_map = alignment_map * valid.unsqueeze(1).to(student.dtype)

    valid_count = _region_sum(
        valid.unsqueeze(1).to(student.dtype), int(region_size)
    ).squeeze(1)
    alignment_sum = _region_sum(alignment_map, int(region_size))
    region_scores = alignment_sum / valid_count.unsqueeze(1).clamp_min(1.0)
    eligible = valid_count >= int(min_valid_pixels)
    selected_index, margin, exact_tie = select_region_temperature_indices(
        region_scores,
        eligible,
        temperatures=temperatures,
        fallback_temperature=fallback_temperature,
    )

    height, width = student.shape[-2:]
    pixel_index = selected_index.repeat_interleave(
        int(region_size), dim=1
    ).repeat_interleave(int(region_size), dim=2)[:, :height, :width]
    eligible_pixel = eligible.repeat_interleave(
        int(region_size), dim=1
    ).repeat_interleave(int(region_size), dim=2)[:, :height, :width]
    fallback_index = _fallback_index(
        temperature_values, fallback_temperature
    )
    pixel_index = torch.where(
        valid & eligible_pixel,
        pixel_index,
        torch.full_like(pixel_index, fallback_index),
    )
    temperature_map = temperature_values[pixel_index]
    gather_index = pixel_index[:, None, None].expand(
        -1, 1, student.shape[1], -1, -1
    )
    teacher_target = torch.gather(
        teacher_probabilities, 1, gather_index
    ).squeeze(1)

    selected_score = torch.gather(
        region_scores, 1, selected_index.unsqueeze(1)
    ).squeeze(1)
    baseline_score = region_scores[:, fallback_index]
    selected_valid = valid & eligible_pixel
    components = covar_components_from_probabilities(
        teacher_target, class_dim=1
    )

    statistics = torch.zeros(
        TASK_ALIGNED_STATISTICS_SIZE,
        device=student.device,
        dtype=torch.float64,
    )
    for index in range(len(TASK_ALIGNED_TEMPERATURES)):
        statistics[_TEMPERATURE_COUNT_START + index] = (
            eligible & (selected_index == index)
        ).sum()
    statistics[_ELIGIBLE_REGIONS] = eligible.sum()
    nonempty = valid_count > 0
    statistics[_NONEMPTY_REGIONS] = nonempty.sum()
    statistics[_FALLBACK_REGIONS] = (nonempty & ~eligible).sum()
    statistics[_EXACT_TIES] = exact_tie.sum()
    statistics[_MARGIN_SUM] = margin[eligible].double().sum()
    statistics[_HIGH_MARGIN_ABSOLUTE_COUNT] = (
        eligible & (margin >= TASK_ALIGNED_HIGH_MARGIN_ABSOLUTE)
    ).sum()
    statistics[_HIGH_MARGIN_RELATIVE_COUNT] = (
        eligible
        & (
            margin
            >= TASK_ALIGNED_HIGH_MARGIN_RELATIVE * selected_score.abs()
        )
    ).sum()
    statistics[_DELTA_ALIGNMENT_SUM] = (
        (selected_score - baseline_score)[eligible].double().sum()
    )
    statistics[_SELECTED_ALIGNMENT_SUM] = selected_score[eligible].double().sum()
    statistics[_COMPLEXITY_VALID_PIXELS] = selected_valid.sum()
    statistics[_R_C_SUM] = components["r_c"][selected_valid].double().sum()
    statistics[_R_V_SUM] = components["r_v"][selected_valid].double().sum()
    statistics[_R_SUM] = components["r"][selected_valid].double().sum()

    return TaskAlignedRegionSelection(
        teacher_target=teacher_target,
        temperature_map=temperature_map,
        valid_mask=valid,
        eligible_region_mask=eligible,
        selected_region_index=selected_index,
        region_scores=region_scores,
        margin=margin,
        statistics=statistics,
    )


def task_aligned_region_kd_loss(
    student_logits: torch.Tensor,
    selection: TaskAlignedRegionSelection,
) -> torch.Tensor:
    """Mean KL(selected teacher target || student) over valid pixels."""
    if student_logits.shape != selection.teacher_target.shape:
        raise ValueError("student logits and teacher target must have identical shapes")
    loss_map = F.kl_div(
        F.log_softmax(student_logits, dim=1),
        selection.teacher_target,
        reduction="none",
    ).sum(dim=1)
    valid_count = selection.valid_mask.sum()
    if int(valid_count.item()) == 0:
        raise ValueError("valid mask contains no valid pixels")
    return loss_map[selection.valid_mask].sum() / valid_count.to(loss_map.dtype)


def task_aligned_statistics_dict(statistics: torch.Tensor) -> Dict[str, object]:
    """Convert an all-reduced sufficient-statistics vector to loggable data."""
    if statistics.numel() != TASK_ALIGNED_STATISTICS_SIZE:
        raise ValueError("unexpected task-aligned statistics size")
    values = statistics.detach().cpu().double().tolist()
    eligible = values[_ELIGIBLE_REGIONS]
    nonempty = values[_NONEMPTY_REGIONS]
    complexity_pixels = values[_COMPLEXITY_VALID_PIXELS]

    def ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator > 0 else 0.0

    return {
        "candidate_temperatures": list(TASK_ALIGNED_TEMPERATURES),
        "temperature_counts": [
            int(round(values[index]))
            for index in range(len(TASK_ALIGNED_TEMPERATURES))
        ],
        "eligible_regions": int(round(eligible)),
        "nonempty_regions": int(round(nonempty)),
        "fallback_regions": int(round(values[_FALLBACK_REGIONS])),
        "fallback_region_fraction": ratio(
            values[_FALLBACK_REGIONS], nonempty
        ),
        "exact_ties": int(round(values[_EXACT_TIES])),
        "exact_tie_fraction": ratio(values[_EXACT_TIES], eligible),
        "mean_margin": ratio(values[_MARGIN_SUM], eligible),
        "high_margin_absolute_threshold": TASK_ALIGNED_HIGH_MARGIN_ABSOLUTE,
        "high_margin_absolute_count": int(
            round(values[_HIGH_MARGIN_ABSOLUTE_COUNT])
        ),
        "high_margin_absolute_fraction": ratio(
            values[_HIGH_MARGIN_ABSOLUTE_COUNT], eligible
        ),
        "high_margin_relative_threshold": TASK_ALIGNED_HIGH_MARGIN_RELATIVE,
        "high_margin_relative_count": int(
            round(values[_HIGH_MARGIN_RELATIVE_COUNT])
        ),
        "high_margin_relative_fraction": ratio(
            values[_HIGH_MARGIN_RELATIVE_COUNT], eligible
        ),
        "mean_delta_alignment_vs_t1p5": ratio(
            values[_DELTA_ALIGNMENT_SUM], eligible
        ),
        "mean_selected_alignment": ratio(
            values[_SELECTED_ALIGNMENT_SUM], eligible
        ),
        "complexity_valid_pixels": int(round(complexity_pixels)),
        "mean_selected_r_c": ratio(values[_R_C_SUM], complexity_pixels),
        "mean_selected_r_v": ratio(values[_R_V_SUM], complexity_pixels),
        "mean_selected_r": ratio(values[_R_SUM], complexity_pixels),
        "raw_sums": {
            "margin": values[_MARGIN_SUM],
            "delta_alignment": values[_DELTA_ALIGNMENT_SUM],
            "selected_alignment": values[_SELECTED_ALIGNMENT_SUM],
            "r_c": values[_R_C_SUM],
            "r_v": values[_R_V_SUM],
            "r": values[_R_SUM],
        },
    }


__all__ = [
    "TASK_ALIGNED_EPSILON",
    "TASK_ALIGNED_FALLBACK_TEMPERATURE",
    "TASK_ALIGNED_HIGH_MARGIN_ABSOLUTE",
    "TASK_ALIGNED_HIGH_MARGIN_RELATIVE",
    "TASK_ALIGNED_KD_LOSS_MODE",
    "TASK_ALIGNED_MIN_VALID_PIXELS",
    "TASK_ALIGNED_REGION_SIZE",
    "TASK_ALIGNED_STATISTICS_SIZE",
    "TASK_ALIGNED_TEMPERATURES",
    "TaskAlignedRegionSelection",
    "build_task_aligned_region_selection",
    "select_region_temperature_indices",
    "task_aligned_region_kd_loss",
    "task_aligned_statistics_dict",
]
