"""Frozen Phase O1.2 confidence calibration primitives.

This module starts from the confidence-CDF quantile ``u`` produced by the
frozen O1.1 implementation.  It deliberately does not load or rebuild that
CDF: callers must validate the O1.1 artifact and pass the queried quantiles.

The module also keeps distributed communication outside the loss primitive.
Callers all-reduce the valid-pixel count, then pass it to
``normalize_o12_ddp_loss``.  This makes the frozen DDP normalization formula
explicit and independently testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F


O12_Q_RELIABLE = 0.6
O12_Q_UNRELIABLE = 0.8
O12_P_RELIABLE = 1.0
O12_P_UNRELIABLE = 2.0
O12_TEMPERATURE_MIN = 0.9
O12_TEMPERATURE_MAX = 1.5
O12_A_STAR = -math.log(O12_TEMPERATURE_MIN)
O12_B_MAX = math.log(O12_TEMPERATURE_MAX)
O12_TARGET_MEAN = 0.995
O12_MIN_HARMONIC_MEAN = 0.98
O12_BISECTION_ITERATIONS = 64
O12_TEACHER_OUTPUT_TEMPERATURE = 3.0
O12_SHUFFLE_SEED = 3407

O12_BRANCHES = (
    "neutral",
    "reliable_only",
    "unreliable_only",
    "full_budgeted",
)


@dataclass(frozen=True)
class O12CalibrationConfig:
    """The frozen Phase O1.2 mechanism constants."""

    q_reliable: float = O12_Q_RELIABLE
    q_unreliable: float = O12_Q_UNRELIABLE
    p_reliable: float = O12_P_RELIABLE
    p_unreliable: float = O12_P_UNRELIABLE
    temperature_min: float = O12_TEMPERATURE_MIN
    temperature_max: float = O12_TEMPERATURE_MAX
    target_mean: float = O12_TARGET_MEAN
    min_harmonic_mean: float = O12_MIN_HARMONIC_MEAN
    bisection_iterations: int = O12_BISECTION_ITERATIONS
    teacher_output_temperature: float = O12_TEACHER_OUTPUT_TEMPERATURE

    @property
    def a_star(self) -> float:
        return -math.log(self.temperature_min)

    @property
    def b_max(self) -> float:
        return math.log(self.temperature_max)

    def validate(self) -> None:
        """Reject any post-registration change to the O1.2 constants."""
        expected = O12CalibrationConfig()
        for field_name in (
            "q_reliable",
            "q_unreliable",
            "p_reliable",
            "p_unreliable",
            "temperature_min",
            "temperature_max",
            "target_mean",
            "min_harmonic_mean",
            "teacher_output_temperature",
        ):
            actual = float(getattr(self, field_name))
            frozen = float(getattr(expected, field_name))
            if not math.isfinite(actual) or actual != frozen:
                raise ValueError(
                    f"O1.2 {field_name} is frozen at {frozen}, got {actual}"
                )
        if int(self.bisection_iterations) != O12_BISECTION_ITERATIONS:
            raise ValueError(
                "O1.2 bisection_iterations is frozen at "
                f"{O12_BISECTION_ITERATIONS}, got {self.bisection_iterations}"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["a_star"] = self.a_star
        payload["b_max"] = self.b_max
        return payload


@dataclass(frozen=True)
class O12TemperatureStatistics:
    valid_count: int
    arithmetic_mean: float
    harmonic_mean: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class O12BudgetSolution:
    a: float
    b: float
    b_max: float
    target_mean: float
    arithmetic_mean: float
    harmonic_mean: float
    residual: float
    valid_count: int
    bisection_iterations: int
    lower_endpoint_mean: float
    upper_endpoint_mean: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class O12KLLocalTerms:
    """Unnormalized per-rank sums used by the O1.2 pixel-KL branch."""

    kl_sum: torch.Tensor
    cross_entropy_sum: torch.Tensor
    teacher_entropy_sum: torch.Tensor
    valid_count: torch.Tensor


@dataclass(frozen=True)
class O12PixelKLResult:
    loss: torch.Tensor
    teacher_target: torch.Tensor
    local_terms: O12KLLocalTerms
    student_probability: torch.Tensor


def _validate_quantiles_and_mask(
    reliability_quantile: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(reliability_quantile, torch.Tensor):
        raise TypeError("reliability_quantile must be a torch.Tensor")
    if not torch.is_floating_point(reliability_quantile):
        raise TypeError("reliability_quantile must have a floating dtype")
    if valid_mask is None:
        valid = torch.ones_like(reliability_quantile, dtype=torch.bool)
    else:
        if not isinstance(valid_mask, torch.Tensor):
            raise TypeError("valid_mask must be a torch.Tensor")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have dtype torch.bool")
        if valid_mask.shape != reliability_quantile.shape:
            raise ValueError(
                "valid_mask and reliability_quantile must have the same shape"
            )
        valid = valid_mask.to(device=reliability_quantile.device)

    selected = reliability_quantile[valid]
    if selected.numel() > 0:
        if not bool(torch.isfinite(selected).all().item()):
            raise ValueError("valid reliability_quantile values must be finite")
        if bool(((selected < 0.0) | (selected > 1.0)).any().item()):
            raise ValueError("valid reliability_quantile values must be in [0, 1]")
    return reliability_quantile, valid


def compute_o12_gates(
    reliability_quantile: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    config: O12CalibrationConfig = O12CalibrationConfig(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the frozen mutually-exclusive reliable/unreliable gates."""
    config.validate()
    quantile, valid = _validate_quantiles_and_mask(
        reliability_quantile, valid_mask
    )
    safe_quantile = torch.where(valid, quantile, torch.zeros_like(quantile))
    gate_reliable = (
        ((config.q_reliable - safe_quantile) / config.q_reliable)
        .clamp(min=0.0, max=1.0)
        .pow(config.p_reliable)
    )
    gate_unreliable = (
        ((safe_quantile - config.q_unreliable) / (1.0 - config.q_unreliable))
        .clamp(min=0.0, max=1.0)
        .pow(config.p_unreliable)
    )
    zeros = torch.zeros_like(quantile)
    return (
        torch.where(valid, gate_reliable, zeros),
        torch.where(valid, gate_unreliable, zeros),
    )


def _branch_coefficients(
    branch: str,
    b: float | None,
    config: O12CalibrationConfig,
) -> tuple[float, float]:
    if branch not in O12_BRANCHES:
        raise ValueError(
            f"unsupported O1.2 branch {branch!r}; expected one of {O12_BRANCHES}"
        )
    if branch in ("unreliable_only", "full_budgeted"):
        if b is None:
            raise ValueError(f"branch {branch!r} requires an explicit frozen b")
        coefficient_b = float(b)
    else:
        if b is not None and float(b) != 0.0:
            raise ValueError(f"branch {branch!r} requires b=0")
        coefficient_b = 0.0
    if not math.isfinite(coefficient_b):
        raise ValueError("b must be finite")
    if not 0.0 <= coefficient_b <= config.b_max:
        raise ValueError(f"b must be in [0, {config.b_max}]")
    coefficient_a = config.a_star if branch in (
        "reliable_only",
        "full_budgeted",
    ) else 0.0
    return coefficient_a, coefficient_b


def build_o12_temperature_map(
    reliability_quantile: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    b: float | None = None,
    branch: str,
    config: O12CalibrationConfig = O12CalibrationConfig(),
) -> torch.Tensor:
    """Build a float32 O1.2 temperature map; invalid pixels are exactly one."""
    config.validate()
    quantile, valid = _validate_quantiles_and_mask(
        reliability_quantile, valid_mask
    )
    quantile_float = quantile.to(dtype=torch.float32)
    valid_float_device = valid.to(device=quantile_float.device)
    gate_reliable, gate_unreliable = compute_o12_gates(
        quantile_float,
        valid_float_device,
        config,
    )
    coefficient_a, coefficient_b = _branch_coefficients(branch, b, config)
    log_temperature = (
        -coefficient_a * gate_reliable + coefficient_b * gate_unreliable
    )
    temperature = torch.exp(log_temperature)
    return torch.where(
        valid_float_device,
        temperature,
        torch.ones_like(temperature),
    )


def compute_o12_temperature_statistics(
    temperature_map: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> O12TemperatureStatistics:
    """Compute temperature moments with float64 accumulators."""
    if not isinstance(temperature_map, torch.Tensor):
        raise TypeError("temperature_map must be a torch.Tensor")
    if not torch.is_floating_point(temperature_map):
        raise TypeError("temperature_map must have a floating dtype")
    if valid_mask is None:
        valid = torch.ones_like(temperature_map, dtype=torch.bool)
    else:
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have dtype torch.bool")
        if valid_mask.shape != temperature_map.shape:
            raise ValueError("valid_mask and temperature_map must have the same shape")
        valid = valid_mask.to(device=temperature_map.device)
    selected = temperature_map[valid]
    if selected.numel() == 0:
        raise ValueError("temperature statistics require at least one valid pixel")
    if not bool(torch.isfinite(selected).all().item()):
        raise ValueError("valid temperatures must be finite")
    if bool((selected <= 0).any().item()):
        raise ValueError("valid temperatures must be positive")
    selected64 = selected.detach().to(device="cpu", dtype=torch.float64)
    count = int(selected64.numel())
    arithmetic = float(selected64.sum(dtype=torch.float64).item() / count)
    harmonic = float(
        count / selected64.reciprocal().sum(dtype=torch.float64).item()
    )
    return O12TemperatureStatistics(
        valid_count=count,
        arithmetic_mean=arithmetic,
        harmonic_mean=harmonic,
        minimum=float(selected64.min().item()),
        maximum=float(selected64.max().item()),
    )


def _budget_moments_from_gates(
    gate_reliable: torch.Tensor,
    gate_unreliable: torch.Tensor,
    a: float,
    b: float,
) -> tuple[float, float]:
    # The formal temperature mechanism is float32; only reductions are float64.
    temperature = torch.exp(-a * gate_reliable + b * gate_unreliable)
    temperature64 = temperature.to(dtype=torch.float64)
    count = int(temperature64.numel())
    arithmetic = float(temperature64.sum(dtype=torch.float64).item() / count)
    harmonic = float(
        count / temperature64.reciprocal().sum(dtype=torch.float64).item()
    )
    return arithmetic, harmonic


def solve_o12_budget_parameter(
    reliability_quantile: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    config: O12CalibrationConfig = O12CalibrationConfig(),
) -> O12BudgetSolution:
    """Solve the unique frozen ``b`` using exactly 64 bisection iterations."""
    config.validate()
    quantile, valid = _validate_quantiles_and_mask(
        reliability_quantile, valid_mask
    )
    # Budget solving is a no-grad CPU operation over the exact queried u values.
    quantile_cpu = quantile.detach().to(device="cpu", dtype=torch.float32)
    valid_cpu = valid.detach().to(device="cpu")
    if int(valid_cpu.sum().item()) == 0:
        raise ValueError("budget solving requires at least one valid pixel")
    gate_reliable, gate_unreliable = compute_o12_gates(
        quantile_cpu,
        valid_cpu,
        config,
    )
    gate_reliable = gate_reliable[valid_cpu]
    gate_unreliable = gate_unreliable[valid_cpu]
    if not bool((gate_unreliable > 0).any().item()):
        raise ValueError("budget root is not unique without high-risk pixels")

    lower_mean, _ = _budget_moments_from_gates(
        gate_reliable, gate_unreliable, config.a_star, 0.0
    )
    upper_mean, _ = _budget_moments_from_gates(
        gate_reliable, gate_unreliable, config.a_star, config.b_max
    )
    if not lower_mean <= config.target_mean <= upper_mean:
        raise ValueError(
            "O1.2 target arithmetic mean has no root in [0, log(1.5)]: "
            f"A(0)={lower_mean}, target={config.target_mean}, "
            f"A(b_max)={upper_mean}"
        )

    left = 0.0
    right = config.b_max
    for _ in range(config.bisection_iterations):
        middle = (left + right) / 2.0
        middle_mean, _ = _budget_moments_from_gates(
            gate_reliable, gate_unreliable, config.a_star, middle
        )
        if middle_mean < config.target_mean:
            left = middle
        else:
            right = middle
    coefficient_b = (left + right) / 2.0
    arithmetic, harmonic = _budget_moments_from_gates(
        gate_reliable, gate_unreliable, config.a_star, coefficient_b
    )
    if harmonic < config.min_harmonic_mean:
        raise ValueError(
            "O1.2 harmonic-mean gate failed: "
            f"H={harmonic} < {config.min_harmonic_mean}"
        )
    return O12BudgetSolution(
        a=config.a_star,
        b=coefficient_b,
        b_max=config.b_max,
        target_mean=config.target_mean,
        arithmetic_mean=arithmetic,
        harmonic_mean=harmonic,
        residual=arithmetic - config.target_mean,
        valid_count=int(gate_reliable.numel()),
        bisection_iterations=config.bisection_iterations,
        lower_endpoint_mean=lower_mean,
        upper_endpoint_mean=upper_mean,
    )


@torch.no_grad()
def build_o12_teacher_target(
    raw_teacher_logits: torch.Tensor,
    temperature_map: torch.Tensor,
    teacher_output_temperature: float = O12_TEACHER_OUTPUT_TEMPERATURE,
) -> torch.Tensor:
    """Build detached ``softmax(raw_teacher / (T_out * T_pixel))`` targets."""
    if not isinstance(raw_teacher_logits, torch.Tensor):
        raise TypeError("raw_teacher_logits must be a torch.Tensor")
    if raw_teacher_logits.ndim != 4 or not torch.is_floating_point(
        raw_teacher_logits
    ):
        raise ValueError("raw_teacher_logits must be floating [B, C, H, W]")
    expected_shape = (
        raw_teacher_logits.shape[0],
        raw_teacher_logits.shape[2],
        raw_teacher_logits.shape[3],
    )
    if temperature_map.shape != expected_shape:
        raise ValueError(
            "temperature_map must exactly match teacher logits [B, H, W]"
        )
    if not torch.is_floating_point(temperature_map):
        raise TypeError("temperature_map must have a floating dtype")
    output_temperature = float(teacher_output_temperature)
    if not math.isfinite(output_temperature) or output_temperature <= 0:
        raise ValueError("teacher_output_temperature must be finite and positive")
    if not bool(torch.isfinite(raw_teacher_logits).all().item()):
        raise ValueError("raw_teacher_logits must be finite")
    if not bool(torch.isfinite(temperature_map).all().item()):
        raise ValueError("temperature_map must be finite")
    if bool((temperature_map <= 0).any().item()):
        raise ValueError("temperature_map must be positive")
    temperature = temperature_map.to(
        device=raw_teacher_logits.device,
        dtype=raw_teacher_logits.dtype,
    ).unsqueeze(1)
    target = F.softmax(
        raw_teacher_logits.detach() / (output_temperature * temperature),
        dim=1,
    )
    return target.detach()


def _compute_o12_kl_terms_and_student_probability(
    student_logits: torch.Tensor,
    teacher_target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[O12KLLocalTerms, torch.Tensor]:
    if student_logits.ndim != 4 or not torch.is_floating_point(student_logits):
        raise ValueError("student_logits must be floating [B, C, H, W]")
    if teacher_target.shape != student_logits.shape:
        raise ValueError(
            "student_logits and teacher_target must have exactly the same shape"
        )
    expected_mask_shape = (
        student_logits.shape[0],
        student_logits.shape[2],
        student_logits.shape[3],
    )
    if valid_mask.shape != expected_mask_shape:
        raise ValueError("valid_mask must exactly match logits [B, H, W]")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must have dtype torch.bool")
    if not torch.is_floating_point(teacher_target):
        raise TypeError("teacher_target must have a floating dtype")
    if not bool(torch.isfinite(student_logits).all().item()):
        raise ValueError("student_logits must be finite")
    if not bool(torch.isfinite(teacher_target).all().item()):
        raise ValueError("teacher_target must be finite")
    if bool((teacher_target < 0).any().item()):
        raise ValueError("teacher_target probabilities must be non-negative")
    target = teacher_target.detach().to(
        device=student_logits.device,
        dtype=student_logits.dtype,
    )
    probability_sum = target.sum(dim=1)
    if not bool(
        torch.allclose(
            probability_sum,
            torch.ones_like(probability_sum),
            rtol=1e-5,
            atol=1e-6,
        )
    ):
        raise ValueError("teacher_target must sum to one along the class axis")

    student_log_probability = F.log_softmax(student_logits, dim=1)
    kl_map = F.kl_div(
        student_log_probability,
        target,
        reduction="none",
    ).sum(dim=1)
    cross_entropy_map = -(target * student_log_probability).sum(dim=1)
    target_log_probability = torch.where(
        target > 0,
        target.log(),
        torch.zeros_like(target),
    )
    teacher_entropy_map = -(target * target_log_probability).sum(dim=1)
    mask = valid_mask.to(device=student_logits.device, dtype=student_logits.dtype)
    terms = O12KLLocalTerms(
        kl_sum=(kl_map * mask).sum(),
        cross_entropy_sum=(cross_entropy_map * mask).sum(),
        teacher_entropy_sum=(teacher_entropy_map * mask).sum().detach(),
        valid_count=valid_mask.to(
            device=student_logits.device,
            dtype=torch.int64,
        ).sum().detach(),
    )
    return terms, student_log_probability.exp().detach()


def compute_o12_masked_kl_terms(
    student_logits: torch.Tensor,
    teacher_target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> O12KLLocalTerms:
    """Return local masked KL, cross-entropy, entropy, and valid count sums."""
    terms, _ = _compute_o12_kl_terms_and_student_probability(
        student_logits, teacher_target, valid_mask
    )
    return terms



def normalize_o12_ddp_loss(
    local_kl_sum: torch.Tensor,
    global_valid_count: torch.Tensor | int | float,
    world_size: int,
) -> torch.Tensor:
    """Apply ``world_size * local_sum / global_count`` without communication."""
    if not isinstance(local_kl_sum, torch.Tensor) or local_kl_sum.ndim != 0:
        raise ValueError("local_kl_sum must be a scalar torch.Tensor")
    if isinstance(world_size, bool) or int(world_size) != world_size or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    count = torch.as_tensor(
        global_valid_count,
        device=local_kl_sum.device,
    ).detach()
    if count.numel() != 1 or not bool(torch.isfinite(count).all().item()):
        raise ValueError("global_valid_count must be one finite scalar")
    count_value = float(count.item())
    if count_value < 0:
        raise ValueError("global_valid_count must be non-negative")
    if count_value == 0:
        return local_kl_sum * 0.0
    return local_kl_sum * float(world_size) / count.to(local_kl_sum.dtype)


def o12_teacher_only_pixel_kl(
    student_logits: torch.Tensor,
    raw_teacher_logits: torch.Tensor,
    temperature_map: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    teacher_output_temperature: float = O12_TEACHER_OUTPUT_TEMPERATURE,
    global_valid_count: torch.Tensor | int | float | None = None,
    world_size: int = 1,
) -> O12PixelKLResult:
    """Build the detached target and return the globally normalized pixel KL."""
    if student_logits.shape != raw_teacher_logits.shape:
        raise ValueError(
            "student_logits and raw_teacher_logits must have exactly the same shape"
        )
    teacher_target = build_o12_teacher_target(
        raw_teacher_logits,
        temperature_map,
        teacher_output_temperature,
    )
    terms, student_probability = _compute_o12_kl_terms_and_student_probability(
        student_logits,
        teacher_target,
        valid_mask,
    )
    count = terms.valid_count if global_valid_count is None else global_valid_count
    loss = normalize_o12_ddp_loss(terms.kl_sum, count, world_size)
    return O12PixelKLResult(
        loss=loss,
        teacher_target=teacher_target,
        local_terms=terms,
        student_probability=student_probability,
    )


def compute_o12_shuffle_seed(
    dataset_index: int,
    global_iteration: int,
) -> int:
    """Return the frozen SHA256-derived 63-bit per-image shuffle seed."""
    if isinstance(dataset_index, bool) or not isinstance(dataset_index, int):
        raise TypeError("dataset_index must be an integer")
    if isinstance(global_iteration, bool) or not isinstance(global_iteration, int):
        raise TypeError("global_iteration must be an integer")
    if dataset_index < 0:
        raise ValueError("dataset_index must be non-negative")
    if global_iteration < 1:
        raise ValueError("global_iteration is 1-based and must be positive")
    payload = (
        f"rtc_o12_shuffle_v1|{O12_SHUFFLE_SEED}|"
        f"{dataset_index}|{global_iteration}"
    ).encode("ascii")
    digest = hashlib.sha256(payload).digest()
    seed64 = int.from_bytes(digest[0:8], byteorder="big", signed=False)
    return seed64 % ((1 << 63) - 1)


def _normalize_dataset_indices(
    dataset_indices: Sequence[int] | torch.Tensor | int,
    batch_size: int,
) -> list[int]:
    if isinstance(dataset_indices, torch.Tensor):
        if dataset_indices.ndim != 1:
            raise ValueError("dataset_indices tensor must be one-dimensional")
        values = dataset_indices.detach().to(device="cpu").tolist()
    elif isinstance(dataset_indices, int) and not isinstance(dataset_indices, bool):
        values = [dataset_indices]
    else:
        values = list(dataset_indices)
    if len(values) != batch_size:
        raise ValueError(
            f"dataset_indices length {len(values)} does not match batch {batch_size}"
        )
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("every dataset index must be an integer")
        if value < 0:
            raise ValueError("dataset indices must be non-negative")
        normalized.append(value)
    return normalized


def shuffle_o12_temperature_within_images(
    temperature_map: torch.Tensor,
    valid_mask: torch.Tensor,
    dataset_indices: Sequence[int] | torch.Tensor | int,
    global_iteration: int,
) -> torch.Tensor:
    """Apply the frozen stateless within-image valid-temperature permutation."""
    if temperature_map.ndim != 3 or not torch.is_floating_point(temperature_map):
        raise ValueError("temperature_map must be floating [B, H, W]")
    if valid_mask.shape != temperature_map.shape:
        raise ValueError("valid_mask and temperature_map must have the same shape")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must have dtype torch.bool")
    mask_on_device = valid_mask.to(device=temperature_map.device)
    if not bool(torch.isfinite(temperature_map[mask_on_device]).all().item()):
        raise ValueError("valid temperatures must be finite")
    indices = _normalize_dataset_indices(
        dataset_indices,
        int(temperature_map.shape[0]),
    )
    # Validate the 1-based iteration even for an empty batch.
    if isinstance(global_iteration, bool) or not isinstance(global_iteration, int):
        raise TypeError("global_iteration must be an integer")
    if global_iteration < 1:
        raise ValueError("global_iteration is 1-based and must be positive")

    original = torch.where(
        mask_on_device,
        temperature_map,
        torch.ones_like(temperature_map),
    )
    shuffled = original.clone()
    for batch_index, dataset_index in enumerate(indices):
        mask_flat = mask_on_device[batch_index].reshape(-1)
        valid_indices = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
        num_valid = int(valid_indices.numel())
        if num_valid <= 1:
            continue
        generator = torch.Generator(device="cpu").manual_seed(
            compute_o12_shuffle_seed(dataset_index, global_iteration)
        )
        permutation = torch.randperm(
            num_valid,
            generator=generator,
            device="cpu",
        ).to(device=temperature_map.device)
        original_flat = original[batch_index].reshape(-1)
        shuffled_flat = shuffled[batch_index].reshape(-1)
        shuffled_flat[valid_indices] = original_flat[
            valid_indices[permutation]
        ]
    return shuffled


__all__ = [
    "O12_A_STAR",
    "O12_B_MAX",
    "O12_BISECTION_ITERATIONS",
    "O12_BRANCHES",
    "O12_MIN_HARMONIC_MEAN",
    "O12_Q_RELIABLE",
    "O12_Q_UNRELIABLE",
    "O12_SHUFFLE_SEED",
    "O12_TARGET_MEAN",
    "O12_TEACHER_OUTPUT_TEMPERATURE",
    "O12_TEMPERATURE_MAX",
    "O12_TEMPERATURE_MIN",
    "O12BudgetSolution",
    "O12CalibrationConfig",
    "O12KLLocalTerms",
    "O12PixelKLResult",
    "O12TemperatureStatistics",
    "build_o12_teacher_target",
    "build_o12_temperature_map",
    "compute_o12_gates",
    "compute_o12_masked_kl_terms",
    "compute_o12_shuffle_seed",
    "compute_o12_temperature_statistics",
    "normalize_o12_ddp_loss",
    "o12_teacher_only_pixel_kl",
    "shuffle_o12_temperature_within_images",
    "solve_o12_budget_parameter",
]
