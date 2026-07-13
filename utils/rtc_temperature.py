"""Reliability-targeted confidence temperature maps for pixel-wise KD.

The reliability route is computed from raw teacher logits at a fixed assessment
temperature.  Target confidence and the final KD loss use the (optionally)
softened teacher logits supplied separately by the caller.  Keeping those two
inputs explicit prevents ``teacher_output_temp`` from changing the route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


CDF_SCHEMA_VERSION = 1


def reliability_definition_metadata(
    reliability_mode: str,
    coefficient_a: float,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Return an explicit, serializable definition of the routed risk score."""
    definitions = {
        "confidence": {
            "reliability_definition_id": "neg_log_top1_confidence_v1",
            "reliability_formula": "r=-log(clamp(max(softmax(z/T_assess)),eps,1-eps))",
            "active_terms": ["confidence"],
            "coefficient_a_active": False,
        },
        "variance": {
            "reliability_definition_id": "normalized_nonmax_variance_v1",
            "reliability_formula": "r=a*variance_nonmax/(1-confidence+eps)",
            "active_terms": ["variance"],
            "coefficient_a_active": True,
        },
        "full": {
            "reliability_definition_id": "confidence_plus_normalized_nonmax_variance_v1",
            "reliability_formula": "r=-log(confidence)+a*variance_nonmax/(1-confidence+eps)",
            "active_terms": ["confidence", "variance"],
            "coefficient_a_active": True,
        },
    }
    if reliability_mode not in definitions:
        raise ValueError(f"unsupported reliability_mode: {reliability_mode}")
    if coefficient_a < 0:
        raise ValueError("coefficient_a must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return {
        **definitions[reliability_mode],
        "reliability_epsilon": float(epsilon),
    }


@dataclass(frozen=True)
class RTCConfig:
    assess_temperature: float = 1.0
    route_quantile: float = 0.80
    route_width: float = 0.05
    reliable_temperature: float = 0.5
    neutral_temperature: float = 1.0
    unreliable_temperature: float = 2.0
    alpha_reliable: float = 1.0
    alpha_unreliable: float = 1.0
    enable_reliable: bool = True
    enable_unreliable: bool = True
    bisection_iterations: int = 16
    kd_temperature_power: float = 0.0
    coefficient_a: float | None = None
    reliability_mode: str = "full"

    def validate(self) -> None:
        if self.assess_temperature <= 0:
            raise ValueError("assess_temperature must be positive")
        if not 0.0 < self.route_quantile < 1.0:
            raise ValueError("route_quantile must be in (0, 1)")
        if self.route_width <= 0:
            raise ValueError("route_width must be positive")
        if not (
            0.0 < self.reliable_temperature
            <= self.neutral_temperature
            <= self.unreliable_temperature
        ):
            raise ValueError(
                "temperatures must satisfy 0 < reliable <= neutral <= unreliable"
            )
        if not 0.0 <= self.alpha_reliable <= 1.0:
            raise ValueError("alpha_reliable must be in [0, 1]")
        if not 0.0 <= self.alpha_unreliable <= 1.0:
            raise ValueError("alpha_unreliable must be in [0, 1]")
        if self.bisection_iterations <= 0:
            raise ValueError("bisection_iterations must be positive")
        if not math.isfinite(self.kd_temperature_power):
            raise ValueError("kd_temperature_power must be finite")
        if self.coefficient_a is not None and self.coefficient_a < 0:
            raise ValueError("coefficient_a must be non-negative")
        if self.reliability_mode not in ("full", "confidence", "variance"):
            raise ValueError(
                f"unsupported reliability_mode: {self.reliability_mode}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenReliabilityCDF:
    values: torch.Tensor
    probabilities: torch.Tensor
    metadata: Mapping[str, Any]
    checksum_sha256: str
    path: str | None = None

    def to(self, device: torch.device | str) -> "FrozenReliabilityCDF":
        return FrozenReliabilityCDF(
            values=self.values.to(device=device),
            probabilities=self.probabilities.to(device=device),
            metadata=dict(self.metadata),
            checksum_sha256=self.checksum_sha256,
            path=self.path,
        )

    def query(self, reliability: torch.Tensor) -> torch.Tensor:
        return query_frozen_cdf(
            reliability,
            self.values,
            self.probabilities,
        )


@dataclass(frozen=True)
class ReliabilityMaps:
    reliability: torch.Tensor
    confidence: torch.Tensor
    variance: torch.Tensor
    prediction: torch.Tensor
    valid_mask: torch.Tensor
    finite_mask: torch.Tensor


@dataclass(frozen=True)
class RTCMaps:
    temperature: torch.Tensor
    reliability: torch.Tensor
    reliability_quantile: torch.Tensor
    confidence: torch.Tensor
    variance: torch.Tensor
    gate_reliable: torch.Tensor
    gate_unreliable: torch.Tensor
    target_log_odds: torch.Tensor
    target_residual: torch.Tensor
    valid_mask: torch.Tensor
    fallback_mask: torch.Tensor
    tie_mask: torch.Tensor
    finite_mask: torch.Tensor
    shuffled: bool = False


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA256 of a local artifact used by an RTC experiment."""
    return _file_sha256(path)


def _torch_load_local(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.load(path, map_location="cpu")


def save_frozen_reliability_cdf(
    path: str | Path,
    reliability_samples: torch.Tensor,
    metadata: Mapping[str, Any],
    num_quantiles: int = 4097,
) -> str:
    """Save uniformly spaced inverse-CDF knots and return the file SHA256."""
    path = Path(path)
    samples = reliability_samples.detach().reshape(-1).float().cpu()
    samples = samples[torch.isfinite(samples)]
    if samples.numel() == 0:
        raise ValueError("reliability_samples contains no finite values")
    if num_quantiles < 2:
        raise ValueError("num_quantiles must be at least 2")

    probabilities_array = np.linspace(0.0, 1.0, int(num_quantiles), dtype=np.float64)
    samples_array = samples.numpy()
    try:
        values_array = np.quantile(
            samples_array,
            probabilities_array,
            method="linear",
        )
    except TypeError:  # pragma: no cover - NumPy < 1.22 compatibility
        values_array = np.quantile(
            samples_array,
            probabilities_array,
            interpolation="linear",
        )
    probabilities = torch.from_numpy(probabilities_array.astype(np.float32))
    values = torch.from_numpy(np.asarray(values_array, dtype=np.float32))
    payload = {
        "schema_version": CDF_SCHEMA_VERSION,
        "kind": "rtc_reliability_cdf",
        "quantile_values": values,
        "quantile_probabilities": probabilities,
        "metadata": {
            **dict(metadata),
            "sample_count": int(samples.numel()),
            "num_quantiles": int(num_quantiles),
            "query_semantics": "right-continuous step CDF over inverse-CDF knots",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return _file_sha256(path)


def load_frozen_reliability_cdf(
    path: str | Path,
    device: torch.device | str | None = None,
) -> FrozenReliabilityCDF:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"RTC CDF not found: {path}")
    payload = _torch_load_local(path)
    if not isinstance(payload, dict):
        raise ValueError("RTC CDF payload must be a dictionary")
    if payload.get("kind") != "rtc_reliability_cdf":
        raise ValueError("unrecognized RTC CDF kind")
    if int(payload.get("schema_version", -1)) != CDF_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported RTC CDF schema_version: {payload.get('schema_version')}"
        )

    values = torch.as_tensor(payload.get("quantile_values"), dtype=torch.float32)
    probabilities = torch.as_tensor(
        payload.get("quantile_probabilities"), dtype=torch.float32
    )
    if values.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("RTC CDF knots must be one-dimensional")
    if values.numel() < 2 or values.numel() != probabilities.numel():
        raise ValueError("RTC CDF value/probability knots have invalid lengths")
    if not torch.isfinite(values).all() or not torch.isfinite(probabilities).all():
        raise ValueError("RTC CDF knots must be finite")
    if torch.any(values[1:] < values[:-1]):
        raise ValueError("RTC CDF values must be non-decreasing")
    if torch.any(probabilities[1:] < probabilities[:-1]):
        raise ValueError("RTC CDF probabilities must be non-decreasing")
    if float(probabilities[0]) != 0.0 or float(probabilities[-1]) != 1.0:
        raise ValueError("RTC CDF probabilities must start at 0 and end at 1")

    if device is not None:
        values = values.to(device=device)
        probabilities = probabilities.to(device=device)
    return FrozenReliabilityCDF(
        values=values,
        probabilities=probabilities,
        metadata=dict(payload.get("metadata") or {}),
        checksum_sha256=_file_sha256(path),
        path=str(path.resolve()),
    )


def query_frozen_cdf(
    reliability: torch.Tensor,
    sorted_values: torch.Tensor,
    probabilities: torch.Tensor,
) -> torch.Tensor:
    """Query a right-continuous step CDF using device-portable binary search."""
    if sorted_values.device != reliability.device:
        sorted_values = sorted_values.to(reliability.device)
    if probabilities.device != reliability.device:
        probabilities = probabilities.to(reliability.device)
    if sorted_values.dtype != reliability.dtype:
        sorted_values = sorted_values.to(dtype=reliability.dtype)
    if probabilities.dtype != reliability.dtype:
        probabilities = probabilities.to(dtype=reliability.dtype)

    num_knots = int(sorted_values.numel())
    low = torch.zeros_like(reliability, dtype=torch.long)
    high = torch.full_like(low, num_knots)
    for _ in range(max(1, num_knots.bit_length())):
        active = low < high
        mid = torch.div(low + high, 2, rounding_mode="floor")
        safe_mid = mid.clamp(max=num_knots - 1)
        mid_value = sorted_values[safe_mid]
        move_right = active & (reliability >= mid_value)
        low = torch.where(move_right, mid + 1, low)
        high = torch.where(active & ~move_right, mid, high)

    right = low
    lower_index = (right - 1).clamp(min=0, max=num_knots - 1)
    result = probabilities[lower_index]
    result = torch.where(right == 0, torch.zeros_like(result), result)
    result = torch.where(
        reliability >= sorted_values[-1], torch.ones_like(result), result
    )
    return result.clamp(min=0.0, max=1.0)


def _resize_valid_mask(valid_mask: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    if valid_mask.ndim != 3:
        raise ValueError("valid_mask must have shape [B, H, W]")
    if valid_mask.shape[-2:] == output_size:
        return valid_mask.bool()
    resized = F.interpolate(
        valid_mask.float().unsqueeze(1), size=output_size, mode="nearest"
    )
    return resized.squeeze(1) > 0.5


@torch.no_grad()
def compute_reference_reliability(
    raw_teacher_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    assess_temperature: float = 1.0,
    coefficient_a: float | None = None,
    reliability_mode: str = "full",
    epsilon: float = 1e-8,
) -> ReliabilityMaps:
    if raw_teacher_logits.ndim != 4:
        raise ValueError("raw_teacher_logits must have shape [B, C, H, W]")
    if raw_teacher_logits.shape[1] < 2:
        raise ValueError("RTC requires at least two classes")
    if assess_temperature <= 0:
        raise ValueError("assess_temperature must be positive")
    if reliability_mode not in ("full", "confidence", "variance"):
        raise ValueError(f"unsupported reliability_mode: {reliability_mode}")

    resized_mask = _resize_valid_mask(valid_mask, raw_teacher_logits.shape[-2:])
    finite_logits = torch.isfinite(raw_teacher_logits).all(dim=1)
    safe_logits = torch.where(
        finite_logits.unsqueeze(1), raw_teacher_logits, torch.zeros_like(raw_teacher_logits)
    )
    probability = F.softmax(safe_logits / float(assess_temperature), dim=1)
    confidence, prediction = probability.max(dim=1)
    confidence = confidence.clamp(min=epsilon, max=1.0 - epsilon)

    num_classes = probability.shape[1]
    num_nonmax = float(num_classes - 1)
    residual_mass = (1.0 - confidence).clamp_min(epsilon)
    maximum_index = prediction.unsqueeze(1)
    nonmax_probability = probability.scatter(1, maximum_index, 0.0)
    nonmax_mean = nonmax_probability.sum(dim=1) / num_nonmax
    centered_nonmax = (nonmax_probability - nonmax_mean.unsqueeze(1)).scatter(
        1, maximum_index, 0.0
    )
    variance = (centered_nonmax * centered_nonmax).sum(dim=1) / num_nonmax

    if coefficient_a is None:
        coefficient_a = float((num_classes - 1) ** 2) / 2.0
    confidence_term = -torch.log(confidence)
    variance_term = float(coefficient_a) * variance / residual_mass
    if reliability_mode == "confidence":
        reliability = confidence_term
    elif reliability_mode == "variance":
        reliability = variance_term
    else:
        reliability = confidence_term + variance_term

    finite_stats = (
        finite_logits
        & torch.isfinite(confidence)
        & torch.isfinite(variance)
        & torch.isfinite(reliability)
    )
    effective_valid = resized_mask & finite_stats
    zeros = torch.zeros_like(reliability)
    return ReliabilityMaps(
        reliability=torch.where(effective_valid, reliability, zeros),
        confidence=torch.where(effective_valid, confidence, zeros),
        variance=torch.where(effective_valid, variance, zeros),
        prediction=torch.where(
            effective_valid, prediction, torch.zeros_like(prediction)
        ),
        valid_mask=resized_mask,
        finite_mask=finite_stats,
    )


@torch.no_grad()
def compute_reliability_gates(
    reliability_quantile: torch.Tensor,
    valid_mask: torch.Tensor,
    route_quantile: float,
    route_width: float,
    enable_reliable: bool = True,
    enable_unreliable: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if route_width <= 0:
        raise ValueError("route_width must be positive")
    direction = torch.tanh(
        (float(route_quantile) - reliability_quantile) / float(route_width)
    )
    gate_reliable = torch.clamp(direction, min=0.0)
    gate_unreliable = torch.clamp(-direction, min=0.0)
    if not enable_reliable:
        gate_reliable = torch.zeros_like(gate_reliable)
    if not enable_unreliable:
        gate_unreliable = torch.zeros_like(gate_unreliable)
    gate_reliable = torch.where(valid_mask, gate_reliable, torch.zeros_like(gate_reliable))
    gate_unreliable = torch.where(valid_mask, gate_unreliable, torch.zeros_like(gate_unreliable))
    return gate_reliable, gate_unreliable


def _safe_sorted_logits(
    teacher_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    finite_mask = torch.isfinite(teacher_logits).all(dim=1)
    safe_logits = torch.where(
        finite_mask.unsqueeze(1), teacher_logits, torch.zeros_like(teacher_logits)
    )
    sorted_logits = torch.sort(
        safe_logits.permute(0, 2, 3, 1).contiguous(),
        dim=-1,
        descending=True,
    ).values
    tie_mask = sorted_logits[..., 0] == sorted_logits[..., 1]
    return sorted_logits, finite_mask, tie_mask


@torch.no_grad()
def compute_top_vs_rest_log_odds(
    teacher_logits: torch.Tensor,
    temperature: float | torch.Tensor,
) -> torch.Tensor:
    if teacher_logits.ndim != 4 or teacher_logits.shape[1] < 2:
        raise ValueError("teacher_logits must have shape [B, C>=2, H, W]")
    sorted_logits, _, _ = _safe_sorted_logits(teacher_logits)
    if torch.is_tensor(temperature):
        temp = temperature.to(
            device=teacher_logits.device, dtype=teacher_logits.dtype
        )
    else:
        temp = torch.full(
            sorted_logits.shape[:-1],
            float(temperature),
            device=teacher_logits.device,
            dtype=teacher_logits.dtype,
        )
    scaled = sorted_logits / temp.clamp_min(1e-8).unsqueeze(-1)
    return scaled[..., 0] - torch.logsumexp(scaled[..., 1:], dim=-1)


def _log_odds_from_sorted(
    sorted_logits: torch.Tensor,
    temperature: float | torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if torch.is_tensor(temperature):
        temp = temperature.to(
            device=sorted_logits.device, dtype=sorted_logits.dtype
        )
    else:
        temp = torch.full(
            sorted_logits.shape[:-1],
            float(temperature),
            device=sorted_logits.device,
            dtype=sorted_logits.dtype,
        )
    scaled = sorted_logits / temp.clamp_min(epsilon).unsqueeze(-1)
    return scaled[..., 0] - torch.logsumexp(scaled[..., 1:], dim=-1)


@torch.no_grad()
def build_target_log_odds(
    teacher_kd_logits: torch.Tensor,
    gate_reliable: torch.Tensor,
    gate_unreliable: torch.Tensor,
    config: RTCConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    config.validate()
    sorted_logits, _, _ = _safe_sorted_logits(teacher_kd_logits)
    reliable_endpoint = _log_odds_from_sorted(
        sorted_logits, config.reliable_temperature
    )
    neutral_endpoint = _log_odds_from_sorted(
        sorted_logits, config.neutral_temperature
    )
    unreliable_endpoint = _log_odds_from_sorted(
        sorted_logits, config.unreliable_temperature
    )
    target = (
        neutral_endpoint
        + float(config.alpha_reliable)
        * gate_reliable
        * (reliable_endpoint - neutral_endpoint)
        + float(config.alpha_unreliable)
        * gate_unreliable
        * (unreliable_endpoint - neutral_endpoint)
    )
    return target, reliable_endpoint, neutral_endpoint, unreliable_endpoint


@torch.no_grad()
def invert_target_by_bisection(
    teacher_kd_logits: torch.Tensor,
    target_log_odds: torch.Tensor,
    gate_reliable: torch.Tensor,
    gate_unreliable: torch.Tensor,
    valid_mask: torch.Tensor,
    config: RTCConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    config.validate()
    sorted_logits, finite_logits, tie_mask = _safe_sorted_logits(teacher_kd_logits)
    finite_target = torch.isfinite(target_log_odds)
    reliable_active = gate_reliable > 0
    unreliable_active = gate_unreliable > 0
    if config.alpha_reliable == 0.0:
        reliable_active = torch.zeros_like(reliable_active)
    if config.alpha_unreliable == 0.0:
        unreliable_active = torch.zeros_like(unreliable_active)
    route_active = reliable_active | unreliable_active
    solve_mask = valid_mask & finite_logits & finite_target & ~tie_mask & route_active

    neutral = torch.full_like(target_log_odds, config.neutral_temperature)
    lower = torch.where(
        reliable_active,
        torch.full_like(target_log_odds, config.reliable_temperature),
        neutral,
    )
    upper = torch.where(
        unreliable_active,
        torch.full_like(target_log_odds, config.unreliable_temperature),
        neutral,
    )

    for _ in range(config.bisection_iterations):
        middle = (lower + upper) * 0.5
        middle_log_odds = _log_odds_from_sorted(sorted_logits, middle)
        temperature_too_low = middle_log_odds > target_log_odds
        lower = torch.where(
            solve_mask & temperature_too_low, middle, lower
        )
        upper = torch.where(
            solve_mask & ~temperature_too_low, middle, upper
        )

    temperature = torch.where(solve_mask, (lower + upper) * 0.5, neutral)
    achieved = _log_odds_from_sorted(sorted_logits, temperature)
    residual = torch.where(
        solve_mask,
        torch.abs(achieved - target_log_odds),
        torch.zeros_like(target_log_odds),
    )
    finite_result = torch.isfinite(temperature) & torch.isfinite(residual)
    fallback_mask = valid_mask & (~finite_logits | ~finite_target | tie_mask | ~finite_result)
    temperature = torch.where(
        valid_mask & finite_result & ~fallback_mask, temperature, neutral
    )
    residual = torch.where(
        valid_mask & finite_result & ~fallback_mask,
        residual,
        torch.zeros_like(residual),
    )
    return temperature, residual, fallback_mask, tie_mask, finite_logits


@torch.no_grad()
def shuffle_temperature_map(
    temperature_map: torch.Tensor,
    valid_mask: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Shuffle valid temperatures independently within each image."""
    shuffled = temperature_map.clone()
    for batch_index in range(temperature_map.shape[0]):
        flat_valid = valid_mask[batch_index].reshape(-1)
        valid_indices = torch.nonzero(flat_valid, as_tuple=False).squeeze(1)
        if valid_indices.numel() <= 1:
            continue
        permutation = torch.randperm(
            valid_indices.numel(),
            device=valid_indices.device,
            generator=generator,
        )
        flat_source = temperature_map[batch_index].reshape(-1)
        flat_target = shuffled[batch_index].reshape(-1)
        flat_target[valid_indices] = flat_source[valid_indices[permutation]]
    return shuffled


@torch.no_grad()
def build_rtc_temperature_map(
    raw_teacher_logits: torch.Tensor,
    teacher_kd_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    frozen_cdf: FrozenReliabilityCDF,
    config: RTCConfig,
    *,
    shuffle: bool = False,
    reverse_routing: bool = False,
    shuffle_generator: torch.Generator | None = None,
) -> RTCMaps:
    config.validate()
    if raw_teacher_logits.shape != teacher_kd_logits.shape:
        raise ValueError("raw_teacher_logits and teacher_kd_logits must have the same shape")
    reliability_maps = compute_reference_reliability(
        raw_teacher_logits,
        valid_mask,
        assess_temperature=config.assess_temperature,
        coefficient_a=config.coefficient_a,
        reliability_mode=config.reliability_mode,
    )
    effective_valid = reliability_maps.valid_mask & reliability_maps.finite_mask
    quantile = frozen_cdf.query(reliability_maps.reliability)
    quantile = torch.where(effective_valid, quantile, torch.zeros_like(quantile))
    gate_reliable, gate_unreliable = compute_reliability_gates(
        quantile,
        effective_valid,
        config.route_quantile,
        config.route_width,
        enable_reliable=config.enable_reliable,
        enable_unreliable=config.enable_unreliable,
    )
    if reverse_routing:
        gate_reliable, gate_unreliable = gate_unreliable, gate_reliable

    target, _, _, _ = build_target_log_odds(
        teacher_kd_logits,
        gate_reliable,
        gate_unreliable,
        config,
    )
    temperature, residual, fallback, tie_mask, finite_logits = invert_target_by_bisection(
        teacher_kd_logits,
        target,
        gate_reliable,
        gate_unreliable,
        reliability_maps.valid_mask,
        config,
    )
    fallback = fallback | (reliability_maps.valid_mask & ~effective_valid)
    if shuffle:
        temperature = shuffle_temperature_map(
            temperature,
            reliability_maps.valid_mask,
            generator=shuffle_generator,
        )

    return RTCMaps(
        temperature=temperature,
        reliability=reliability_maps.reliability,
        reliability_quantile=quantile,
        confidence=reliability_maps.confidence,
        variance=reliability_maps.variance,
        gate_reliable=gate_reliable,
        gate_unreliable=gate_unreliable,
        target_log_odds=target,
        target_residual=residual,
        valid_mask=reliability_maps.valid_mask,
        fallback_mask=fallback,
        tie_mask=tie_mask & reliability_maps.valid_mask,
        finite_mask=finite_logits & reliability_maps.finite_mask,
        shuffled=shuffle,
    )


def masked_temperature_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature_map: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature_power: float = 0.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Masked pixel KL with DDP-correct global valid-pixel normalization."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have the same shape")
    if temperature_map.shape != student_logits.shape[:1] + student_logits.shape[2:]:
        raise ValueError("temperature_map must have shape [B, H, W]")
    if valid_mask.shape != temperature_map.shape:
        raise ValueError("valid_mask and temperature_map must have the same shape")

    temperature = temperature_map.unsqueeze(1).clamp_min(epsilon)
    student_log_probability = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probability = F.softmax(teacher_logits / temperature, dim=1)
    kd_map = F.kl_div(
        student_log_probability,
        teacher_probability,
        reduction="none",
    ).sum(dim=1)
    scale = temperature_map.clamp_min(epsilon) ** float(temperature_power)
    local_sum = (kd_map * scale * valid_mask.to(kd_map.dtype)).sum()
    local_count = valid_mask.to(kd_map.dtype).sum().detach()

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        global_count = local_count.clone()
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        if float(global_count.item()) == 0.0:
            return student_logits.sum() * 0.0
        return local_sum * float(dist.get_world_size()) / global_count
    if float(local_count.item()) == 0.0:
        return student_logits.sum() * 0.0
    return local_sum / local_count


@torch.no_grad()
def collect_rtc_diagnostics(
    maps: RTCMaps,
    config: RTCConfig,
    teacher_output_temperature: float = 1.0,
) -> dict[str, float]:
    valid = maps.valid_mask
    count = int(valid.sum().item())
    if count == 0:
        residual_is_applicable = not maps.shuffled
        residual_value = 0.0 if residual_is_applicable else float("nan")
        return {
            "valid_count": 0.0,
            "reliability_mean": 0.0,
            "quantile_mean": 0.0,
            "reliable_coverage": 0.0,
            "unreliable_coverage": 0.0,
            "reliable_gate_mean": 0.0,
            "unreliable_gate_mean": 0.0,
            "temperature_mean": float(config.neutral_temperature),
            "temperature_harmonic_mean": float(config.neutral_temperature),
            "temperature_q10": float(config.neutral_temperature),
            "temperature_q50": float(config.neutral_temperature),
            "temperature_q90": float(config.neutral_temperature),
            "temperature_p95": float(config.neutral_temperature),
            "temperature_reliable_endpoint_rate": 0.0,
            "temperature_neutral_rate": 0.0,
            "temperature_unreliable_endpoint_rate": 0.0,
            "effective_teacher_temperature_mean": float(
                config.neutral_temperature * teacher_output_temperature
            ),
            "target_residual_mean": residual_value,
            "target_residual_p95": residual_value,
            "target_residual_max": residual_value,
            "residual_is_applicable": float(residual_is_applicable),
            "fallback_rate": 0.0,
            "tie_rate": 0.0,
            "finite_rate": 1.0,
            "shuffled": float(maps.shuffled),
        }

    temperature = maps.temperature[valid].detach().float().cpu()
    reliability = maps.reliability[valid].detach().float().cpu()
    quantile = maps.reliability_quantile[valid].detach().float().cpu()
    reliable_gate = maps.gate_reliable[valid].detach().float().cpu()
    unreliable_gate = maps.gate_unreliable[valid].detach().float().cpu()
    residual = maps.target_residual[valid].detach().float().cpu()
    fallback = maps.fallback_mask[valid].detach().float().cpu()
    tie = maps.tie_mask[valid].detach().float().cpu()

    quantile_points = torch.tensor([0.10, 0.50, 0.90, 0.95])
    temp_quantiles = torch.quantile(temperature, quantile_points)
    residual_is_applicable = not maps.shuffled
    if residual_is_applicable:
        residual_mean = float(residual.mean())
        residual_p95 = float(torch.quantile(residual, torch.tensor([0.95]))[0])
        residual_max = float(residual.max())
    else:
        residual_mean = float("nan")
        residual_p95 = float("nan")
        residual_max = float("nan")

    bisection_scale = math.ldexp(1.0, -(config.bisection_iterations + 1))
    reliable_resolution = (
        config.neutral_temperature - config.reliable_temperature
    ) * bisection_scale
    unreliable_resolution = (
        config.unreliable_temperature - config.neutral_temperature
    ) * bisection_scale
    endpoint_scale = max(
        1.0,
        abs(config.reliable_temperature),
        abs(config.neutral_temperature),
        abs(config.unreliable_temperature),
    )
    floating_tolerance = (
        float(torch.finfo(temperature.dtype).eps) * endpoint_scale * 4.0
    )
    reliable_tolerance = reliable_resolution + floating_tolerance
    unreliable_tolerance = unreliable_resolution + floating_tolerance
    neutral_tolerance = (
        max(reliable_resolution, unreliable_resolution) + floating_tolerance
    )
    diagnostics = {
        "valid_count": float(count),
        "reliability_mean": float(reliability.mean()),
        "quantile_mean": float(quantile.mean()),
        "reliable_coverage": float((reliable_gate > 0).float().mean()),
        "unreliable_coverage": float((unreliable_gate > 0).float().mean()),
        "reliable_gate_mean": float(reliable_gate.mean()),
        "unreliable_gate_mean": float(unreliable_gate.mean()),
        "temperature_mean": float(temperature.mean()),
        "temperature_harmonic_mean": float(1.0 / (1.0 / temperature).mean()),
        "temperature_q10": float(temp_quantiles[0]),
        "temperature_q50": float(temp_quantiles[1]),
        "temperature_q90": float(temp_quantiles[2]),
        "temperature_p95": float(temp_quantiles[3]),
        "temperature_reliable_endpoint_rate": float(
            (
                torch.abs(temperature - config.reliable_temperature)
                <= reliable_tolerance
            )
            .float()
            .mean()
        ),
        "temperature_neutral_rate": float(
            (torch.abs(temperature - config.neutral_temperature) <= neutral_tolerance)
            .float()
            .mean()
        ),
        "temperature_unreliable_endpoint_rate": float(
            (
                torch.abs(temperature - config.unreliable_temperature)
                <= unreliable_tolerance
            )
            .float()
            .mean()
        ),
        "effective_teacher_temperature_mean": float(
            temperature.mean() * float(teacher_output_temperature)
        ),
        "target_residual_mean": residual_mean,
        "target_residual_p95": residual_p95,
        "target_residual_max": residual_max,
        "residual_is_applicable": float(residual_is_applicable),
        "fallback_rate": float(fallback.mean()),
        "tie_rate": float(tie.mean()),
        "finite_rate": float(maps.finite_mask[valid].float().mean().item()),
        "shuffled": float(maps.shuffled),
    }
    return diagnostics
