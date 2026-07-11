from dataclasses import dataclass

import torch
import torch.nn.functional as F


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
    temp = temperature_map.clamp_min(epsilon)
    probability = F.softmax(sorted_logits / temp.unsqueeze(-1), dim=-1)

    confidence = probability[..., 0].clamp(min=epsilon, max=1.0 - epsilon)
    nonmax_probability = probability[..., 1:]
    if nonmax_probability.shape[-1] == 0:
        variance = torch.zeros_like(confidence)
    else:
        mean = nonmax_probability.mean(dim=-1, keepdim=True)
        variance = torch.mean((nonmax_probability - mean) ** 2, dim=-1)

    residual_mass = (1.0 - confidence).clamp_min(epsilon)
    variance_term = coefficient_a * variance / residual_mass
    reliability = _combine_reliability(
        confidence,
        variance_term,
        mode=reliability_mode,
        epsilon=epsilon,
    )
    return probability, confidence, variance, reliability


@torch.no_grad()
def compute_reliability_derivatives(sorted_logits, temperature_map, coefficient_a,
                                    reliability_mode="full", epsilon=1e-8):
    temp = temperature_map.clamp_min(epsilon)
    probability, confidence, variance, reliability = compute_reliability_terms(
        sorted_logits,
        temp,
        coefficient_a,
        reliability_mode=reliability_mode,
        epsilon=epsilon,
    )

    nonmax_probability = probability[..., 1:]
    if nonmax_probability.shape[-1] == 0:
        zeros = torch.zeros_like(temp)
        return zeros, zeros, reliability, confidence, variance

    mean_logit = torch.sum(probability * sorted_logits, dim=-1)
    centered_logits = mean_logit.unsqueeze(-1) - sorted_logits
    temp_squared = temp ** 2
    temp_cubed = temp_squared * temp
    temp_fourth = temp_squared * temp_squared
    probability_prime = probability * centered_logits / temp_squared.unsqueeze(-1)
    logit_variance = torch.sum(probability * centered_logits ** 2, dim=-1)
    probability_double_prime = probability * (
        (centered_logits ** 2 - logit_variance.unsqueeze(-1)) / temp_fourth.unsqueeze(-1)
        - 2.0 * centered_logits / temp_cubed.unsqueeze(-1)
    )

    residual_mass = (1.0 - confidence).clamp_min(epsilon)
    num_nonmax = nonmax_probability.shape[-1]
    nonmax_mean = residual_mass / num_nonmax

    confidence_prime = probability_prime[..., 0]
    confidence_double_prime = probability_double_prime[..., 0]
    nonmax_prime = probability_prime[..., 1:]
    nonmax_double_prime = probability_double_prime[..., 1:]
    nonmax_mean_prime = -confidence_prime / num_nonmax
    nonmax_mean_double_prime = -confidence_double_prime / num_nonmax

    variance_prime = (
        (2.0 / num_nonmax) * torch.sum(nonmax_probability * nonmax_prime, dim=-1)
        - 2.0 * nonmax_mean * nonmax_mean_prime
    )
    variance_double_prime = (
        (2.0 / num_nonmax)
        * torch.sum(nonmax_prime ** 2 + nonmax_probability * nonmax_double_prime, dim=-1)
        - 2.0 * (nonmax_mean_prime ** 2 + nonmax_mean * nonmax_mean_double_prime)
    )

    confidence_first = -confidence_prime / confidence
    confidence_second = (
        (confidence_prime ** 2) / (confidence ** 2)
        - confidence_double_prime / confidence
    )
    variance_first = coefficient_a * (
        variance * confidence_prime / (residual_mass ** 2)
        + variance_prime / residual_mass
    )
    variance_second = coefficient_a * (
        variance_double_prime / residual_mass
        + variance * confidence_double_prime / (residual_mass ** 2)
        + 2.0 * confidence_prime * variance_prime / (residual_mass ** 2)
        + 2.0 * variance * (confidence_prime ** 2) / (residual_mass ** 3)
    )

    if reliability_mode == "confidence":
        first_derivative = confidence_first
        second_derivative = confidence_second
    elif reliability_mode == "variance":
        first_derivative = variance_first
        second_derivative = variance_second
    else:
        first_derivative = confidence_first + variance_first
        second_derivative = confidence_second + variance_second

    return first_derivative, second_derivative, reliability, confidence, variance


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
    coefficient_a = config.coefficient_a
    if coefficient_a is None:
        coefficient_a = float((max(num_classes, 1) - 1) ** 2) / 2.0

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
