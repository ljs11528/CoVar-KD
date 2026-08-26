from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.covar_metrics import (
    covar_coefficient,
    covar_components_from_sorted_logits,
    covar_derivatives_from_sorted_logits,
)


@dataclass(frozen=True)
class NewtonCoVarConfig:
    base_temperature: float = 1.0
    min_temperature: float = 0.5
    max_temperature: float = 8.0
    kd_temperature_power: float = 2.0
    eta: float = 0.6
    max_iterations: int = 8
    hessian_epsilon: float = 1e-5
    max_step: float = 0.25
    coefficient_a: float | None = None
    reliability_mode: str = "full"

    def validate(self):
        if self.base_temperature <= 0:
            raise ValueError("base_temperature must be positive")
        if self.min_temperature <= 0:
            raise ValueError("min_temperature must be positive")
        if self.max_temperature < self.min_temperature:
            raise ValueError("max_temperature must be >= min_temperature")
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if self.reliability_mode not in ("full", "confidence", "variance"):
            raise ValueError(f"unsupported reliability_mode: {self.reliability_mode}")


def _combine_reliability(confidence, variance_term, mode, epsilon):
    confidence_term = -torch.log(confidence.clamp(min=epsilon, max=1.0 - epsilon))
    if mode == "confidence":
        return confidence_term
    if mode == "variance":
        return variance_term
    return confidence_term + variance_term


@torch.no_grad()
def compute_reliability_terms(sorted_logits, temperature_map, coefficient_a, reliability_mode="full",
                              epsilon=1e-8):
    components = covar_components_from_sorted_logits(
        sorted_logits,
        temperature_map,
        coefficient_a=coefficient_a,
        epsilon=epsilon,
    )
    if reliability_mode == "confidence":
        reliability = components["r_c"]
    elif reliability_mode == "variance":
        reliability = components["r_v"]
    elif reliability_mode == "full":
        reliability = components["r"]
    else:
        raise ValueError(f"unsupported reliability_mode: {reliability_mode}")
    return (
        components["probability"],
        components["confidence"],
        components["residual_variance"],
        reliability,
    )


@torch.no_grad()
def compute_reliability_derivatives(sorted_logits, temperature_map, coefficient_a,
                                    reliability_mode="full", epsilon=1e-8):
    closed = covar_derivatives_from_sorted_logits(
        sorted_logits,
        temperature_map,
        coefficient_a=coefficient_a,
        reliability_mode=reliability_mode,
        epsilon=epsilon,
    )
    if reliability_mode == "confidence":
        reliability = closed["r_c"]
    elif reliability_mode == "variance":
        reliability = closed["r_v"]
    else:
        reliability = closed["r"]
    return (
        closed["dr_dT"],
        closed["d2r_dT2"],
        reliability,
        closed["confidence"],
        closed["residual_variance"],
    )

def _resize_valid_mask(valid_mask, output_size):
    if valid_mask.shape[-2:] == output_size:
        return valid_mask.bool()
    resized = F.interpolate(valid_mask.float().unsqueeze(1), size=output_size, mode="nearest")
    return resized.squeeze(1) > 0.5


@torch.no_grad()
def newton_covar_temperature_map(teacher_logits, valid_mask, config, epsilon=1e-8):
    config.validate()
    valid_mask_resized = _resize_valid_mask(valid_mask, teacher_logits.shape[-2:])
    sorted_logits = torch.sort(
        teacher_logits.permute(0, 2, 3, 1).contiguous(),
        dim=-1,
        descending=True,
    ).values

    num_classes = sorted_logits.shape[-1]
    coefficient_a = covar_coefficient(num_classes, config.coefficient_a)

    temperature_map = torch.full(
        sorted_logits.shape[:-1],
        config.base_temperature,
        device=teacher_logits.device,
        dtype=teacher_logits.dtype,
    )

    for _ in range(config.max_iterations):
        first, second, _, _, _ = compute_reliability_derivatives(
            sorted_logits,
            temperature_map,
            coefficient_a,
            reliability_mode=config.reliability_mode,
            epsilon=epsilon,
        )
        valid_hessian = torch.isfinite(second) & (second.abs() >= config.hessian_epsilon) & (second > 0)
        safe_hessian = torch.where(valid_hessian, second, torch.ones_like(second))
        delta = torch.where(valid_hessian, config.eta * first / safe_hessian, config.eta * first)
        if config.max_step > 0:
            delta = torch.clamp(delta, min=-config.max_step, max=config.max_step)
        temperature_map = torch.clamp(
            temperature_map - delta,
            min=config.min_temperature,
            max=config.max_temperature,
        )

    _, _, reliability, confidence, variance = compute_reliability_derivatives(
        sorted_logits,
        temperature_map,
        coefficient_a,
        reliability_mode=config.reliability_mode,
        epsilon=epsilon,
    )
    temperature_map = torch.where(
        valid_mask_resized,
        temperature_map,
        torch.full_like(temperature_map, config.base_temperature),
    )
    reliability = torch.where(valid_mask_resized, reliability, torch.zeros_like(reliability))
    confidence = torch.where(valid_mask_resized, confidence, torch.zeros_like(confidence))
    variance = torch.where(valid_mask_resized, variance, torch.zeros_like(variance))
    return temperature_map, reliability, valid_mask_resized, confidence, variance


def covar_temperature_kd_loss(student_logits, teacher_logits, temperature_map, valid_mask,
                              temperature_power=2.0, epsilon=1e-8):
    temp = temperature_map.unsqueeze(1).clamp_min(epsilon)
    student_log_probability = F.log_softmax(student_logits / temp, dim=1)
    teacher_probability = F.softmax(teacher_logits / temp, dim=1)
    kd_map = F.kl_div(student_log_probability, teacher_probability, reduction="none").sum(dim=1)
    kd_scale = temperature_map.clamp_min(epsilon) ** float(temperature_power)
    kd_map = kd_map * valid_mask.float() * kd_scale
    return kd_map.sum() / valid_mask.sum().clamp_min(1)
