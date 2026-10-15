"""Theory-consistent Confidence-Residual (CoVar) complexity utilities.

The authoritative decomposition is

    r_c = -log(C)
    r_v = a (1 - C) V
    r   = r_c + r_v

where V is the population variance of the normalized non-maximum class
distribution and a=(K-1)^2/2 by default. The stable r_v expression is
algebraically equivalent to a*v/(1-C), with v the population variance of
the original non-maximum probabilities.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def covar_coefficient(num_classes: int, configured_a: float | None = None) -> float:
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    if configured_a is not None:
        return float(configured_a)
    return float((num_classes - 1) ** 2) / 2.0


def sort_logits_for_covar(logits: torch.Tensor, class_dim: int = 1) -> torch.Tensor:
    """Move classes to the last dimension and sort them in descending order."""
    if logits.ndim < 2:
        raise ValueError("logits must have at least two dimensions")
    return torch.sort(logits.movedim(class_dim, -1).contiguous(), dim=-1, descending=True).values


def _as_temperature(temperature, like: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(temperature, device=like.device, dtype=like.dtype)
    if torch.any(value <= 0):
        raise ValueError("temperature must be positive")
    return value


def _as_coefficient(coefficient_a, num_classes: int, like: torch.Tensor) -> torch.Tensor:
    value = covar_coefficient(num_classes) if coefficient_a is None else coefficient_a
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def covar_components_from_sorted_logits(
    sorted_logits: torch.Tensor,
    temperature=1.0,
    coefficient_a: float | torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Compute the two-dimensional CoVar coordinates from sorted logits.

    sorted_logits has shape [..., K] and is sorted descending on the final
    dimension. temperature may be scalar or have shape [...]. Gradients are
    preserved so the function can be checked with autograd.
    """
    if sorted_logits.ndim < 1 or sorted_logits.shape[-1] < 1:
        raise ValueError("sorted_logits must have a non-empty class dimension")

    temperature_tensor = _as_temperature(temperature, sorted_logits)
    probability = F.softmax(sorted_logits / temperature_tensor.unsqueeze(-1), dim=-1)
    num_classes = probability.shape[-1]
    num_nonmax = num_classes - 1
    coefficient = _as_coefficient(coefficient_a, num_classes, probability)

    confidence_raw = probability[..., 0]
    confidence = confidence_raw.clamp(min=epsilon, max=1.0 - epsilon)
    residual_mass = (1.0 - confidence_raw).clamp_min(epsilon)
    confidence_complexity = -torch.log(confidence)
    entropy = -(probability * torch.log(probability.clamp_min(epsilon))).sum(dim=-1)

    if num_nonmax == 0:
        zeros = torch.zeros_like(confidence)
        return {
            "probability": probability,
            "confidence": confidence,
            "residual_mass": residual_mass,
            "residual_variance": zeros,
            "normalized_residual_variance": zeros,
            "r_c": confidence_complexity,
            "r_v": zeros,
            "r": confidence_complexity,
            "entropy": entropy,
        }

    nonmax_probability = probability[..., 1:]
    nonmax_mean = residual_mass.unsqueeze(-1) / float(num_nonmax)
    residual_variance = torch.mean((nonmax_probability - nonmax_mean) ** 2, dim=-1)

    normalized_nonmax = nonmax_probability / residual_mass.unsqueeze(-1)
    uniform = 1.0 / float(num_nonmax)
    normalized_residual_variance = torch.mean((normalized_nonmax - uniform) ** 2, dim=-1)
    variance_complexity = coefficient * residual_mass * normalized_residual_variance
    complexity = confidence_complexity + variance_complexity

    return {
        "probability": probability,
        "confidence": confidence,
        "residual_mass": residual_mass,
        "residual_variance": residual_variance,
        "normalized_residual_variance": normalized_residual_variance,
        "r_c": confidence_complexity,
        "r_v": variance_complexity,
        "r": complexity,
        "entropy": entropy,
    }


def covar_components_from_probabilities(
    probability: torch.Tensor,
    class_dim: int = 1,
    coefficient_a: float | torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Compute CoVar coordinates from probabilities without pixel loops."""
    probability_last = probability.movedim(class_dim, -1)
    if probability_last.shape[-1] < 1:
        raise ValueError("probability must have a non-empty class dimension")
    num_classes = probability_last.shape[-1]
    num_nonmax = num_classes - 1
    coefficient = _as_coefficient(coefficient_a, num_classes, probability_last)

    confidence_raw, top_index = probability_last.max(dim=-1)
    confidence = confidence_raw.clamp(min=epsilon, max=1.0 - epsilon)
    residual_mass = (1.0 - confidence_raw).clamp_min(epsilon)
    confidence_complexity = -torch.log(confidence)
    entropy = -(probability_last * torch.log(probability_last.clamp_min(epsilon))).sum(dim=-1)

    if num_nonmax == 0:
        zeros = torch.zeros_like(confidence)
        return {
            "confidence": confidence,
            "residual_mass": residual_mass,
            "residual_variance": zeros,
            "normalized_residual_variance": zeros,
            "r_c": confidence_complexity,
            "r_v": zeros,
            "r": confidence_complexity,
            "entropy": entropy,
        }

    top_mask = F.one_hot(top_index, num_classes=num_classes).bool()
    nonmax_mask = ~top_mask
    nonmax_mean = residual_mass.unsqueeze(-1) / float(num_nonmax)
    residual_variance = torch.where(
        nonmax_mask,
        (probability_last - nonmax_mean) ** 2,
        torch.zeros_like(probability_last),
    ).sum(dim=-1) / float(num_nonmax)

    normalized_nonmax = probability_last / residual_mass.unsqueeze(-1)
    uniform = 1.0 / float(num_nonmax)
    normalized_residual_variance = torch.where(
        nonmax_mask,
        (normalized_nonmax - uniform) ** 2,
        torch.zeros_like(probability_last),
    ).sum(dim=-1) / float(num_nonmax)
    variance_complexity = coefficient * residual_mass * normalized_residual_variance
    complexity = confidence_complexity + variance_complexity

    return {
        "confidence": confidence,
        "residual_mass": residual_mass,
        "residual_variance": residual_variance,
        "normalized_residual_variance": normalized_residual_variance,
        "r_c": confidence_complexity,
        "r_v": variance_complexity,
        "r": complexity,
        "entropy": entropy,
    }


def covar_derivatives_from_sorted_logits(
    sorted_logits: torch.Tensor,
    temperature,
    coefficient_a: float | torch.Tensor | None = None,
    reliability_mode: str = "full",
    epsilon: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Closed-form first and second temperature derivatives of CoVar."""
    if reliability_mode not in ("full", "confidence", "variance"):
        raise ValueError(f"unsupported reliability_mode: {reliability_mode}")

    temp = _as_temperature(temperature, sorted_logits)
    components = covar_components_from_sorted_logits(
        sorted_logits,
        temp,
        coefficient_a=coefficient_a,
        epsilon=epsilon,
    )
    probability = components["probability"]
    confidence = components["confidence"]
    variance = components["residual_variance"]
    num_nonmax = probability.shape[-1] - 1
    coefficient = _as_coefficient(coefficient_a, probability.shape[-1], probability)

    if num_nonmax == 0:
        zeros = torch.zeros_like(temp)
        return {**components, "dr_dT": zeros, "d2r_dT2": zeros}

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

    residual_mass = components["residual_mass"]
    nonmax_probability = probability[..., 1:]
    nonmax_mean = residual_mass / float(num_nonmax)
    confidence_prime = probability_prime[..., 0]
    confidence_double_prime = probability_double_prime[..., 0]
    nonmax_prime = probability_prime[..., 1:]
    nonmax_double_prime = probability_double_prime[..., 1:]
    nonmax_mean_prime = -confidence_prime / float(num_nonmax)
    nonmax_mean_double_prime = -confidence_double_prime / float(num_nonmax)

    variance_prime = (
        (2.0 / float(num_nonmax)) * torch.sum(nonmax_probability * nonmax_prime, dim=-1)
        - 2.0 * nonmax_mean * nonmax_mean_prime
    )
    variance_double_prime = (
        (2.0 / float(num_nonmax))
        * torch.sum(nonmax_prime ** 2 + nonmax_probability * nonmax_double_prime, dim=-1)
        - 2.0 * (nonmax_mean_prime ** 2 + nonmax_mean * nonmax_mean_double_prime)
    )

    confidence_first = -confidence_prime / confidence
    confidence_second = confidence_prime ** 2 / confidence ** 2 - confidence_double_prime / confidence
    variance_first = coefficient * (
        variance * confidence_prime / residual_mass ** 2 + variance_prime / residual_mass
    )
    variance_second = coefficient * (
        variance_double_prime / residual_mass
        + variance * confidence_double_prime / residual_mass ** 2
        + 2.0 * confidence_prime * variance_prime / residual_mass ** 2
        + 2.0 * variance * confidence_prime ** 2 / residual_mass ** 3
    )

    if reliability_mode == "confidence":
        first, second = confidence_first, confidence_second
    elif reliability_mode == "variance":
        first, second = variance_first, variance_second
    else:
        first = confidence_first + variance_first
        second = confidence_second + variance_second
    return {**components, "dr_dT": first, "d2r_dT2": second}
