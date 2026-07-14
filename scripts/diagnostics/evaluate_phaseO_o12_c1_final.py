#!/usr/bin/env python3
"""Evaluate a Phase O1.2-C1 final checkpoint on the native KD grid.

The formal path is intentionally independent from training.  It loads the
iteration-20,000 ``training_state_latest.pth`` student, the frozen teacher and
confidence CDF, walks the canonical VOC validation list without augmentation,
and emits one self-contained packed cache plus a fail-closed JSON summary.

The array/count helpers are deliberately NPU-free so the statistical contract
can be unit-tested without importing model or accelerator code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "reports/2026-07-13_phaseO_rtc_o12_budgeted_routing_plan.md"
PLAN_SHA256 = "c6ac659aea7019d8c2faed88ccdd596678e909129e3a468942cde921d6a6d8b9"

PHASE = "O1.2-C1"
CACHE_SCHEMA_VERSION = 1
SUMMARY_SCHEMA_VERSION = 1
EXPECTED_VARIANTS = ("neutral", "unreliable_only")
EXPECTED_IMAGE_COUNT = 1449
EXPECTED_ITERATION = 20000
EXPECTED_ORDER_SHA256 = (
    "10e600fd87537bba4329a7a90473bd71931e5fe2c775345af4dc483c3b9f8c5c"
)
EXPECTED_NUM_CLASSES = 21
IGNORE_LABEL = -1
EPSILON = 1e-8
HIGH_RISK_THRESHOLD = 0.8

TEACHER_SHA256 = "ac49b2c7720b21d565072e974e4404fcb009ba106288d95d1a4bb25f09c3fe58"
STUDENT_INIT_SHA256 = "47085aa164b2977003221458a2a5fdf5f46f434f5539b164dc71969b4dd4cd75"
CDF_SHA256 = "8ba8376a03c2f93835339d98670fa357b381b23a22cbf2a7fc48b0c2441d7e69"
PARAMETERS_SHA256 = "a9006972e165ed6c95e7a966c526927bb9f846fb6c7ec6528f5507dd21b8b5df"
GATE_SHA256 = "c1961cfec8f232c1fdf7fc8b90db5a6f4855901317ea2705536eea88cbc1be82"
O11_GATE_SHA256 = "47ff2f1f2ea68a4e50375bfa7efc7221c8197703372d5d8f22dfec9032088d3a"
TRAIN_LIST_SHA256 = "d1326bd532648d73bc4b1bd275434eba81930982dc7401a65a8cecb26c028e24"
VAL_LIST_SHA256 = "cdc1326d12f69ce5153aa5da04a4d8783e146868d42d19f82f44e74b97ac907d"
TRAIN_ENTRY_SHA256 = "f50a130c17ca3b5f368f848b96c59fd58c408952e75bda447c051589505b316d"
O12_MODULE_SHA256 = "5e23f3e1bd526017aa2046faff621dd284093d91dead9088dd4d7d9ddb641d9e"

DEFAULT_DATA_ROOT = ROOT / "dataset/VOCAug"
DEFAULT_VAL_LIST = ROOT / "dataset/list/voc/val.txt"
DEFAULT_TRAIN_LIST = ROOT / "dataset/list/voc/train_aug.txt"
DEFAULT_TEACHER = (
    ROOT / "data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth"
)
DEFAULT_STUDENT_INIT = (
    ROOT / "data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth"
)
DEFAULT_CDF = ROOT / "runs/diagnostics/phaseO_o11/voc_train_rtc_confidence_cdf.pt"
DEFAULT_PARAMETERS = ROOT / "runs/diagnostics/phaseO_o12/o12_budget_parameters.json"
DEFAULT_GATE = ROOT / "runs/diagnostics/phaseO_o12/o12_joint_gate.json"
DEFAULT_O11_GATE = ROOT / "runs/diagnostics/phaseO_o11/o11_confidence_gate.json"

METRIC_NAMES = (
    "student_rescue_U",
    "error_imitation_U",
    "teacher_correct_retention_U",
)
CACHE_KEYS = frozenset(
    {
        "schema_version",
        "names",
        "shapes",
        "offsets",
        "student_flat",
        "teacher_flat",
        "gt_flat",
        "valid_flat",
        "u_flat",
        "metric_names",
        "metric_counts",
    }
)


class C1EvaluationError(RuntimeError):
    """A fail-closed C1 evaluation contract violation."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_names_sha256(names: Sequence[str]) -> str:
    payload = "".join(f"{name}\n" for name in names).encode("utf-8")
    return bytes_sha256(payload)


def read_canonical_names(path: str | Path) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in raw):
        raise C1EvaluationError("canonical val list contains a blank line")
    names = [line.strip() for line in raw]
    if len(names) != len(set(names)):
        raise C1EvaluationError("canonical val list contains duplicate names")
    return names


def require_file_sha(path: str | Path, expected: str, label: str) -> str:
    path = Path(path).resolve()
    if not path.is_file():
        raise C1EvaluationError(f"{label} is missing: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise C1EvaluationError(
            f"{label} SHA256 mismatch: expected={expected} observed={observed}"
        )
    return observed


def _as_2d(name: str, value: Any) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 2:
        raise C1EvaluationError(f"{name} must be two-dimensional")
    return result


def compute_image_counts(
    student_prediction: Any,
    teacher_prediction: Any,
    ground_truth: Any,
    valid_mask: Any,
    reliability_quantile: Any,
) -> np.ndarray:
    """Return int64 ``[metric, (numerator, denominator)]`` counts.

    W is strictly ``valid & (u > .8) & (teacher != GT)``.  A third student
    error (neither teacher nor GT) contributes to the shared W denominator but
    to neither rescue nor imitation numerator.
    """

    student = _as_2d("student_prediction", student_prediction)
    teacher = _as_2d("teacher_prediction", teacher_prediction)
    gt = _as_2d("ground_truth", ground_truth)
    valid = _as_2d("valid_mask", valid_mask)
    u = _as_2d("reliability_quantile", reliability_quantile)
    shape = gt.shape
    for name, array in (
        ("student_prediction", student),
        ("teacher_prediction", teacher),
        ("valid_mask", valid),
        ("reliability_quantile", u),
    ):
        if array.shape != shape:
            raise C1EvaluationError(
                f"{name} shape {array.shape} differs from GT shape {shape}"
            )

    valid = valid.astype(bool, copy=False)
    if not np.array_equal(valid, gt != IGNORE_LABEL):
        raise C1EvaluationError("valid mask is not exactly (GT != -1)")
    if np.any(~np.isfinite(u[valid])):
        raise C1EvaluationError("u is non-finite on native-valid pixels")
    if np.any((u[valid] < 0.0) | (u[valid] > 1.0)):
        raise C1EvaluationError("u is outside [0,1] on native-valid pixels")
    for label, array in (("student", student), ("teacher", teacher), ("GT", gt)):
        values = array[valid]
        if np.any((values < 0) | (values >= EXPECTED_NUM_CLASSES)):
            raise C1EvaluationError(f"{label} contains an invalid VOC class")

    high_risk = valid & (u > HIGH_RISK_THRESHOLD)
    teacher_wrong = high_risk & (teacher != gt)
    teacher_correct = high_risk & (teacher == gt)
    wrong_denominator = int(np.count_nonzero(teacher_wrong))
    correct_denominator = int(np.count_nonzero(teacher_correct))
    counts = np.asarray(
        [
            [
                int(np.count_nonzero(teacher_wrong & (student == gt))),
                wrong_denominator,
            ],
            [
                int(np.count_nonzero(teacher_wrong & (student == teacher))),
                wrong_denominator,
            ],
            [
                int(np.count_nonzero(teacher_correct & (student == gt))),
                correct_denominator,
            ],
        ],
        dtype=np.int64,
    )
    if np.any(counts[:, 0] < 0) or np.any(counts[:, 0] > counts[:, 1]):
        raise C1EvaluationError("metric numerator/denominator relationship is invalid")
    return counts


def metric_summary(metric_counts: np.ndarray) -> dict[str, dict[str, Any]]:
    counts = np.asarray(metric_counts)
    if counts.ndim != 3 or counts.shape[1:] != (len(METRIC_NAMES), 2):
        raise C1EvaluationError("metric_counts must have shape [N,3,2]")
    if counts.dtype != np.int64:
        raise C1EvaluationError("metric_counts must use int64")
    totals = counts.sum(axis=0, dtype=np.int64)
    result: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(METRIC_NAMES):
        numerator = int(totals[index, 0])
        denominator = int(totals[index, 1])
        if denominator <= 0:
            raise C1EvaluationError(f"{name} has a zero full-val denominator")
        if not 0 <= numerator <= denominator:
            raise C1EvaluationError(f"{name} has invalid aggregate counts")
        result[name] = {
            "numerator": numerator,
            "denominator": denominator,
            "value": numerator / denominator,
        }
    return result


def make_cache_payload(records: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    if not records:
        raise C1EvaluationError("cache has no image records")
    names: list[str] = []
    shapes: list[tuple[int, int]] = []
    offsets = [0]
    students: list[np.ndarray] = []
    teachers: list[np.ndarray] = []
    ground_truths: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    quantiles: list[np.ndarray] = []
    counts: list[np.ndarray] = []

    for record in records:
        name = str(record["name"])
        if not name or "\n" in name or "\r" in name:
            raise C1EvaluationError("cache contains an invalid sample name")
        student = _as_2d("student_prediction", record["student_prediction"])
        teacher = _as_2d("teacher_prediction", record["teacher_prediction"])
        gt = _as_2d("ground_truth", record["ground_truth"])
        valid = _as_2d("valid_mask", record["valid_mask"])
        u = _as_2d("reliability_quantile", record["reliability_quantile"])
        image_counts = compute_image_counts(student, teacher, gt, valid, u)
        if "metric_counts" in record and not np.array_equal(
            image_counts, np.asarray(record["metric_counts"], dtype=np.int64)
        ):
            raise C1EvaluationError(f"precomputed metric counts differ for {name}")

        names.append(name)
        shapes.append((int(gt.shape[0]), int(gt.shape[1])))
        size = int(gt.size)
        offsets.append(offsets[-1] + size)
        students.append(student.astype(np.int16, copy=False).reshape(-1))
        teachers.append(teacher.astype(np.int16, copy=False).reshape(-1))
        ground_truths.append(gt.astype(np.int16, copy=False).reshape(-1))
        valids.append(valid.astype(np.uint8, copy=False).reshape(-1))
        quantiles.append(u.astype(np.float32, copy=False).reshape(-1))
        counts.append(image_counts)

    if len(names) != len(set(names)):
        raise C1EvaluationError("cache contains duplicate sample names")
    max_name_length = max(1, max(len(name) for name in names))
    payload = {
        "schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64),
        "names": np.asarray(names, dtype=f"<U{max_name_length}"),
        "shapes": np.asarray(shapes, dtype=np.int64),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "student_flat": np.concatenate(students).astype(np.int16, copy=False),
        "teacher_flat": np.concatenate(teachers).astype(np.int16, copy=False),
        "gt_flat": np.concatenate(ground_truths).astype(np.int16, copy=False),
        "valid_flat": np.concatenate(valids).astype(np.uint8, copy=False),
        "u_flat": np.concatenate(quantiles).astype(np.float32, copy=False),
        "metric_names": np.asarray(METRIC_NAMES, dtype="<U32"),
        "metric_counts": np.stack(counts, axis=0).astype(np.int64, copy=False),
    }
    validate_cache_payload(payload, recompute=True)
    return payload


def validate_cache_payload(
    payload: Mapping[str, np.ndarray], *, recompute: bool = True
) -> None:
    keys = frozenset(payload.keys())
    if keys != CACHE_KEYS:
        raise C1EvaluationError(
            f"cache keys differ: missing={sorted(CACHE_KEYS-keys)} "
            f"extra={sorted(keys-CACHE_KEYS)}"
        )
    if int(np.asarray(payload["schema_version"]).item()) != CACHE_SCHEMA_VERSION:
        raise C1EvaluationError("cache schema_version mismatch")
    names = np.asarray(payload["names"])
    shapes = np.asarray(payload["shapes"])
    offsets = np.asarray(payload["offsets"])
    metric_names = tuple(str(value) for value in np.asarray(payload["metric_names"]))
    counts = np.asarray(payload["metric_counts"])
    image_count = len(names)
    if names.ndim != 1 or names.dtype.kind != "U":
        raise C1EvaluationError("cache names must be a one-dimensional Unicode array")
    if image_count == 0 or len(set(names.tolist())) != image_count:
        raise C1EvaluationError("cache names are empty or duplicated")
    if shapes.dtype != np.int64 or shapes.shape != (image_count, 2):
        raise C1EvaluationError("cache shapes must be int64 [N,2]")
    if np.any(shapes <= 0):
        raise C1EvaluationError("cache contains a non-positive native shape")
    if offsets.dtype != np.int64 or offsets.shape != (image_count + 1,):
        raise C1EvaluationError("cache offsets must be int64 [N+1]")
    expected_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(np.prod(shapes, axis=1))]
    )
    if not np.array_equal(offsets, expected_offsets):
        raise C1EvaluationError("cache offsets do not match native shapes")
    if metric_names != METRIC_NAMES:
        raise C1EvaluationError("cache metric_names mismatch")
    if counts.dtype != np.int64 or counts.shape != (image_count, 3, 2):
        raise C1EvaluationError("cache metric_counts must be int64 [N,3,2]")

    total = int(offsets[-1])
    expected_flat = {
        "student_flat": np.int16,
        "teacher_flat": np.int16,
        "gt_flat": np.int16,
        "valid_flat": np.uint8,
        "u_flat": np.float32,
    }
    for key, dtype in expected_flat.items():
        array = np.asarray(payload[key])
        if array.dtype != dtype or array.shape != (total,):
            raise C1EvaluationError(
                f"cache {key} must be {np.dtype(dtype)} with shape [{total}]"
            )
    valid_flat = np.asarray(payload["valid_flat"])
    if np.any((valid_flat != 0) & (valid_flat != 1)):
        raise C1EvaluationError("cache valid_flat is not binary")

    if recompute:
        for index in range(image_count):
            start, end = int(offsets[index]), int(offsets[index + 1])
            shape = tuple(int(value) for value in shapes[index])
            observed = compute_image_counts(
                np.asarray(payload["student_flat"])[start:end].reshape(shape),
                np.asarray(payload["teacher_flat"])[start:end].reshape(shape),
                np.asarray(payload["gt_flat"])[start:end].reshape(shape),
                np.asarray(payload["valid_flat"])[start:end].reshape(shape),
                np.asarray(payload["u_flat"])[start:end].reshape(shape),
            )
            if not np.array_equal(observed, counts[index]):
                raise C1EvaluationError(
                    f"cache metric_counts mismatch for image index {index}"
                )


def save_packed_cache(path: str | Path, payload: Mapping[str, np.ndarray]) -> str:
    validate_cache_payload(payload, recompute=True)
    path = Path(path).resolve()
    if path.exists():
        raise C1EvaluationError(f"refusing to overwrite cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return file_sha256(path)


def load_packed_cache(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path).resolve()
    if not path.is_file():
        raise C1EvaluationError(f"cache is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    validate_cache_payload(payload, recompute=True)
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json_new(path: str | Path, payload: Mapping[str, Any]) -> str:
    path = Path(path).resolve()
    if path.exists():
        raise C1EvaluationError(f"refusing to overwrite JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return file_sha256(path)


def _torch_load(path: str | Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - old torch compatibility
        return torch.load(path, map_location="cpu")


def _load_state_dict_strict(module: Any, state: Mapping[str, Any], label: str) -> None:
    if not isinstance(state, Mapping) or not state:
        raise C1EvaluationError(f"{label} state_dict is missing or empty")
    cleaned = {
        key[7:] if str(key).startswith("module.") else str(key): value
        for key, value in state.items()
    }
    try:
        module.load_state_dict(cleaned, strict=True)
    except Exception as error:
        raise C1EvaluationError(f"strict {label} state load failed: {error}") from error


def canonical_state_sha256(state: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            raise C1EvaluationError(f"state_dict entry is not a tensor: {key}")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def validate_final_checkpoint(checkpoint: Any, variant: str) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise C1EvaluationError("checkpoint root is not an object")
    if checkpoint.get("checkpoint_type") != "train_kd_training_state":
        raise C1EvaluationError("checkpoint_type mismatch")
    if int(checkpoint.get("checkpoint_version", -1)) != 4:
        raise C1EvaluationError("C1 requires checkpoint version 4")
    if int(checkpoint.get("iteration", -1)) != EXPECTED_ITERATION:
        raise C1EvaluationError("C1 requires the iteration-20,000 final checkpoint")
    if int(checkpoint.get("world_size", -1)) != 1:
        raise C1EvaluationError("checkpoint world_size must be 1")
    student = checkpoint.get("student")
    if not isinstance(student, Mapping) or not student:
        raise C1EvaluationError("checkpoint student state is missing")

    o12 = checkpoint.get("rtc_o12")
    if not isinstance(o12, Mapping):
        raise C1EvaluationError("checkpoint rtc_o12 metadata is missing")
    expected_o12 = {
        "phase": "O1.2",
        "variant": variant,
        "max_iterations": EXPECTED_ITERATION,
        "skip_val": False,
        "world_size": 1,
    }
    for key, expected in expected_o12.items():
        if o12.get(key) != expected:
            raise C1EvaluationError(
                f"checkpoint rtc_o12.{key} mismatch: {o12.get(key)!r} != {expected!r}"
            )
    target = o12.get("teacher_target_contract")
    expected_target = {
        "formula": "softmax(raw_teacher/(teacher_output_temperature*T_pixel))",
        "teacher_output_temperature": 3.0,
        "student_temperature": 1.0,
        "teacher_target_detached": True,
        "temperature_loss_power": None,
    }
    if target != expected_target:
        raise C1EvaluationError("checkpoint teacher-target contract mismatch")
    artifact = o12.get("artifact_contract")
    if not isinstance(artifact, Mapping):
        raise C1EvaluationError("checkpoint artifact contract is missing")
    expected_artifacts = {
        "cdf_sha256": CDF_SHA256,
        "parameters_sha256": PARAMETERS_SHA256,
        "gate_sha256": GATE_SHA256,
        "o11_gate_sha256": O11_GATE_SHA256,
        "teacher_sha256": TEACHER_SHA256,
        "student_init_sha256": STUDENT_INIT_SHA256,
        "train_list_sha256": TRAIN_LIST_SHA256,
    }
    for key, expected in expected_artifacts.items():
        if artifact.get(key) != expected:
            raise C1EvaluationError(f"checkpoint artifact {key} mismatch")
    sources = artifact.get("source_sha256") or {}
    if sources.get("train_entry") != TRAIN_ENTRY_SHA256:
        raise C1EvaluationError("checkpoint train entry SHA mismatch")
    if sources.get("rtc_o12_calibration") != O12_MODULE_SHA256:
        raise C1EvaluationError("checkpoint O1.2 module SHA mismatch")

    order = checkpoint.get("rtc_o12_data_order")
    if not isinstance(order, Mapping):
        raise C1EvaluationError("checkpoint data-order state is missing")
    contract = order.get("contract")
    if not isinstance(contract, Mapping):
        raise C1EvaluationError("checkpoint data-order contract is missing")
    expected_order = {
        "algorithm": "rtc_o12_canonical_order_v1",
        "canonical_population": 10582,
        "batch_size": 16,
        "max_iterations": EXPECTED_ITERATION,
        "total_canonical_indices": 320000,
        "seed": 1234,
        "complete_order_sha256": EXPECTED_ORDER_SHA256,
    }
    for key, expected in expected_order.items():
        if contract.get(key) != expected:
            raise C1EvaluationError(f"checkpoint order {key} mismatch")
    if order.get("completed_iteration") != EXPECTED_ITERATION:
        raise C1EvaluationError("checkpoint completed_iteration mismatch")
    if order.get("next_global_iteration") is not None:
        raise C1EvaluationError("completed C1 checkpoint has a next iteration")
    if order.get("next_canonical_index_offset") != 320000:
        raise C1EvaluationError("checkpoint next canonical offset mismatch")

    args = checkpoint.get("args")
    if not isinstance(args, Mapping):
        raise C1EvaluationError("checkpoint argv metadata is missing")
    exact_args = {
        "dataset": "voc",
        "teacher_model": "deeplabv3",
        "teacher_backbone": "resnet101",
        "student_model": "deeplabv3_mobilenet_ssseg",
        "student_backbone": "mobilenetv3_small",
        "ignore_label": -1,
        "batch_size": 16,
        "workers": 8,
        "seed": 1234,
        "log_iter": 20,
        "save_per_iters": 800,
        "val_per_iters": 800,
        "max_iterations": 20000,
        "skip_val": False,
        "lr": 0.02,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "kd_loss_mode": "rtc_o12_teacher_target",
        "rtc_o12_variant": variant,
        "teacher_output_temp": 3.0,
        "kd_temperature": 1.0,
        "lambda_kd": 1.0,
        "lambda_adv": 0.001,
        "lambda_d": 0.1,
        "lambda_cwd_fea": 50.0,
        "lambda_cwd_logit": 3.0,
        "lambda_skd": 0.0,
        "lambda_ifv": 0.0,
        "lambda_fitnet": 0.0,
        "lambda_at": 0.0,
        "lambda_psd": 0.0,
        "lambda_csd": 0.0,
        "distributed": False,
        "device": "npu",
        "device_type": "npu",
        "resume": None,
    }
    for key, expected in exact_args.items():
        if args.get(key) != expected:
            raise C1EvaluationError(f"checkpoint args.{key} mismatch")
    if list(args.get("crop_size") or []) != [512, 512]:
        raise C1EvaluationError("checkpoint crop_size mismatch")
    return student


def _build_models(
    checkpoint: Mapping[str, Any], teacher_path: Path, device: Any
) -> tuple[Any, Any, str]:
    import torch
    import torch.nn as nn

    from models.model_zoo import get_segmentation_model

    teacher = get_segmentation_model(
        model="deeplabv3",
        backbone="resnet101",
        local_rank=None,
        pretrained_base="None",
        pretrained="None",
        aux=True,
        norm_layer=nn.BatchNorm2d,
        num_class=EXPECTED_NUM_CLASSES,
    )
    student = get_segmentation_model(
        model="deeplabv3_mobilenet_ssseg",
        backbone="mobilenetv3_small",
        local_rank=None,
        pretrained_base="None",
        pretrained="None",
        aux=False,
        norm_layer=nn.BatchNorm2d,
        num_class=EXPECTED_NUM_CLASSES,
    )
    teacher_state = _torch_load(teacher_path)
    if isinstance(teacher_state, Mapping) and "state_dict" in teacher_state:
        teacher_state = teacher_state["state_dict"]
    _load_state_dict_strict(teacher, teacher_state, "teacher")
    student_state = checkpoint["student"]
    _load_state_dict_strict(student, student_state, "student")
    student_state_sha = canonical_state_sha256(student_state)
    teacher.to(device).eval()
    student.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    return student, teacher, student_state_sha


def _resolve_formal_npu(physical_npu: int) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch_npu  # noqa: F401 - registers the NPU backend

    expected = str(int(physical_npu))
    for key in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"):
        if os.environ.get(key) != expected:
            raise C1EvaluationError(
                f"{key} must expose exactly physical NPU {expected}"
            )
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise C1EvaluationError("C1 evaluator requires WORLD_SIZE=1")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise C1EvaluationError("Ascend NPU is unavailable")
    count = int(torch.npu.device_count())
    if count != 1:
        raise C1EvaluationError(
            f"C1 evaluator must see exactly one logical NPU, observed {count}"
        )
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    environment = {
        "physical_npu": int(physical_npu),
        "logical_device": "npu:0",
        "visible_device_count": count,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
        "pid": os.getpid(),
    }
    return device, environment


def run_formal_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from dataset.voc import VOCDataValSet
    from utils.rtc_temperature import load_frozen_reliability_cdf

    if args.variant not in EXPECTED_VARIANTS:
        raise C1EvaluationError(f"unsupported C1 variant: {args.variant}")
    inputs = {
        "plan": Path(args.plan).resolve(),
        "teacher": Path(args.teacher).resolve(),
        "student_init": Path(args.student_init).resolve(),
        "cdf": Path(args.cdf).resolve(),
        "parameters": Path(args.parameters).resolve(),
        "gate": Path(args.gate).resolve(),
        "o11_gate": Path(args.o11_gate).resolve(),
        "train_list": Path(args.train_list).resolve(),
        "val_list": Path(args.val_list).resolve(),
        "checkpoint": Path(args.checkpoint).resolve(),
    }
    expected_hashes = {
        "plan": PLAN_SHA256,
        "teacher": TEACHER_SHA256,
        "student_init": STUDENT_INIT_SHA256,
        "cdf": CDF_SHA256,
        "parameters": PARAMETERS_SHA256,
        "gate": GATE_SHA256,
        "o11_gate": O11_GATE_SHA256,
        "train_list": TRAIN_LIST_SHA256,
        "val_list": VAL_LIST_SHA256,
    }
    input_hashes = {
        key: require_file_sha(inputs[key], expected, key)
        for key, expected in expected_hashes.items()
    }
    if not inputs["checkpoint"].is_file():
        raise C1EvaluationError(f"checkpoint is missing: {inputs['checkpoint']}")
    checkpoint_sha = file_sha256(inputs["checkpoint"])

    canonical_names = read_canonical_names(inputs["val_list"])
    if len(canonical_names) != EXPECTED_IMAGE_COUNT:
        raise C1EvaluationError(
            f"canonical val population mismatch: {len(canonical_names)}"
        )
    data_root = Path(args.data).resolve()
    if not (data_root / "JPEGImages").is_dir() or not (
        data_root / "SegmentationClass"
    ).is_dir():
        raise C1EvaluationError("VOCDataValSet root is missing validation directories")

    device, environment = _resolve_formal_npu(args.physical_npu)
    checkpoint = _torch_load(inputs["checkpoint"])
    validate_final_checkpoint(checkpoint, args.variant)
    student, teacher, student_state_sha = _build_models(
        checkpoint, inputs["teacher"], device
    )
    cdf = load_frozen_reliability_cdf(inputs["cdf"], device=device)
    if cdf.checksum_sha256 != CDF_SHA256:
        raise C1EvaluationError("loaded CDF checksum mismatch")
    metadata = dict(cdf.metadata)
    if metadata.get("teacher_sha256") != TEACHER_SHA256:
        raise C1EvaluationError("CDF teacher SHA metadata mismatch")
    if metadata.get("train_list_sha256") != TRAIN_LIST_SHA256:
        raise C1EvaluationError("CDF train-list SHA metadata mismatch")
    if metadata.get("reliability_mode") != "confidence":
        raise C1EvaluationError("CDF reliability mode is not confidence-only")

    dataset = VOCDataValSet(
        str(data_root), str(inputs["val_list"]), ignore_label=IGNORE_LABEL
    )
    if list(dataset.img_ids) != canonical_names:
        raise C1EvaluationError("VOCDataValSet order differs from canonical val list")

    records: list[dict[str, Any]] = []
    start = time.time()
    with torch.no_grad():
        for index, expected_name in enumerate(canonical_names):
            image_array, gt_array, returned = dataset[index]
            if not isinstance(returned, (tuple, list)) or len(returned) != 2:
                raise C1EvaluationError(f"unexpected sample identity at index {index}")
            returned_name = str(returned[1])
            if returned_name != expected_name:
                raise C1EvaluationError(
                    f"sample order mismatch at {index}: {returned_name} != {expected_name}"
                )
            image = torch.from_numpy(np.asarray(image_array)).unsqueeze(0).to(device)
            student_output = student(image)
            teacher_output = teacher(image)
            if not isinstance(student_output, (tuple, list)) or not student_output:
                raise C1EvaluationError("student did not return raw logits")
            if not isinstance(teacher_output, (tuple, list)) or not teacher_output:
                raise C1EvaluationError("teacher did not return raw logits")
            student_logits = student_output[0]
            teacher_logits = teacher_output[0]
            if student_logits.shape != teacher_logits.shape:
                raise C1EvaluationError(
                    f"raw logit shape mismatch for {expected_name}: "
                    f"student={tuple(student_logits.shape)} "
                    f"teacher={tuple(teacher_logits.shape)}"
                )
            if (
                student_logits.ndim != 4
                or student_logits.shape[0] != 1
                or student_logits.shape[1] != EXPECTED_NUM_CLASSES
            ):
                raise C1EvaluationError("native logits have an invalid B/C shape")
            native_shape = tuple(int(value) for value in student_logits.shape[-2:])
            gt_tensor = torch.as_tensor(
                np.asarray(gt_array), dtype=torch.float32, device=device
            ).unsqueeze(0).unsqueeze(0)
            native_gt = F.interpolate(
                gt_tensor, size=native_shape, mode="nearest"
            ).squeeze(0).squeeze(0).to(dtype=torch.long)
            valid = native_gt != IGNORE_LABEL
            if not bool(valid.any().item()):
                raise C1EvaluationError(f"{expected_name} has no native-valid pixels")
            if not bool(
                torch.isfinite(student_logits).all(dim=1).squeeze(0)[valid].all().item()
            ):
                raise C1EvaluationError(
                    f"student logits are non-finite on V for {expected_name}"
                )
            if not bool(
                torch.isfinite(teacher_logits).all(dim=1).squeeze(0)[valid].all().item()
            ):
                raise C1EvaluationError(
                    f"teacher logits are non-finite on V for {expected_name}"
                )

            probability = torch.softmax(teacher_logits, dim=1)
            confidence, teacher_prediction = probability.max(dim=1)
            confidence = confidence.clamp(min=EPSILON, max=1.0 - EPSILON)
            risk = -torch.log(confidence)
            u = cdf.query(risk)
            student_prediction = student_logits.argmax(dim=1)
            for label, value in (("c", confidence), ("r", risk), ("u", u)):
                if not bool(torch.isfinite(value.squeeze(0)[valid]).all().item()):
                    raise C1EvaluationError(
                        f"{label} is non-finite on V for {expected_name}"
                    )
            if not bool(
                ((u.squeeze(0)[valid] >= 0.0) & (u.squeeze(0)[valid] <= 1.0))
                .all()
                .item()
            ):
                raise C1EvaluationError(f"u is outside [0,1] for {expected_name}")

            valid_np = valid.cpu().numpy().astype(np.uint8, copy=False)
            u_np = torch.where(valid.unsqueeze(0), u, torch.zeros_like(u))
            record = {
                "name": expected_name,
                "student_prediction": student_prediction.squeeze(0)
                .cpu()
                .numpy()
                .astype(np.int16, copy=False),
                "teacher_prediction": teacher_prediction.squeeze(0)
                .cpu()
                .numpy()
                .astype(np.int16, copy=False),
                "ground_truth": native_gt.cpu().numpy().astype(np.int16, copy=False),
                "valid_mask": valid_np,
                "reliability_quantile": u_np.squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32, copy=False),
            }
            record["metric_counts"] = compute_image_counts(
                record["student_prediction"],
                record["teacher_prediction"],
                record["ground_truth"],
                record["valid_mask"],
                record["reliability_quantile"],
            )
            records.append(record)

    if len(records) != EXPECTED_IMAGE_COUNT:
        raise C1EvaluationError("evaluator did not process all canonical val images")
    payload = make_cache_payload(records)
    if payload["names"].tolist() != canonical_names:
        raise C1EvaluationError("packed cache order differs from canonical val list")
    cache_sha = save_packed_cache(args.cache, payload)
    reloaded = load_packed_cache(args.cache)
    if file_sha256(args.cache) != cache_sha:
        raise C1EvaluationError("cache SHA changed after validation")
    metrics = metric_summary(reloaded["metric_counts"])
    elapsed = time.time() - start
    source_sha = file_sha256(Path(__file__).resolve())
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "phase": PHASE,
        "kind": "final_native_grid_evaluation",
        "variant": args.variant,
        "pass": True,
        "errors": [],
        "warnings": [],
        "plan": {"path": str(inputs["plan"]), "sha256": PLAN_SHA256},
        "source_sha256": source_sha,
        "checkpoint": {
            "path": str(inputs["checkpoint"]),
            "sha256": checkpoint_sha,
            "checkpoint_type": "train_kd_training_state",
            "version": 4,
            "iteration": EXPECTED_ITERATION,
            "student_state_sha256": student_state_sha,
            "sample_order_sha256": EXPECTED_ORDER_SHA256,
        },
        "inputs": {
            key: {"path": str(inputs[key]), "sha256": input_hashes[key]}
            for key in expected_hashes
        },
        "canonical_val": {
            "image_count": EXPECTED_IMAGE_COUNT,
            "names_sha256": canonical_names_sha256(canonical_names),
            "first_name": canonical_names[0],
            "last_name": canonical_names[-1],
            "inference_order": "VOCDataValSet canonical list order",
            "augmentation": "none",
            "gt_resize": "nearest to exact raw-logit native grid",
        },
        "cache": {
            "path": str(Path(args.cache).resolve()),
            "sha256": cache_sha,
            "schema_version": CACHE_SCHEMA_VERSION,
            "image_count": int(len(reloaded["names"])),
            "flat_pixel_count": int(len(reloaded["u_flat"])),
            "u_dtype": str(reloaded["u_flat"].dtype),
        },
        "metric_contract": {
            "high_risk": "valid and u > 0.8",
            "student_rescue_U": "sum(W and s==g)/sum(W)",
            "error_imitation_U": "sum(W and s==t)/sum(W)",
            "teacher_correct_retention_U": "sum(C and s==g)/sum(C)",
            "third_wrong": "shared W denominator; neither rescue nor imitation numerator",
        },
        "metrics": metrics,
        "population": {
            "native_valid": int(reloaded["valid_flat"].sum(dtype=np.int64)),
            "high_risk": int(
                np.count_nonzero(
                    (reloaded["valid_flat"] == 1)
                    & (reloaded["u_flat"] > HIGH_RISK_THRESHOLD)
                )
            ),
        },
        "environment": environment,
        "runtime": {"wall_seconds": elapsed, "images_per_second": len(records) / elapsed},
    }
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one formal Phase O1.2-C1 final checkpoint"
    )
    parser.add_argument("--variant", required=True, choices=EXPECTED_VARIANTS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--physical-npu", required=True, type=int, choices=(0, 1))
    parser.add_argument("--data", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--val-list", default=str(DEFAULT_VAL_LIST))
    parser.add_argument("--train-list", default=str(DEFAULT_TRAIN_LIST))
    parser.add_argument("--teacher", default=str(DEFAULT_TEACHER))
    parser.add_argument("--student-init", default=str(DEFAULT_STUDENT_INIT))
    parser.add_argument("--cdf", default=str(DEFAULT_CDF))
    parser.add_argument("--parameters", default=str(DEFAULT_PARAMETERS))
    parser.add_argument("--gate", default=str(DEFAULT_GATE))
    parser.add_argument("--o11-gate", default=str(DEFAULT_O11_GATE))
    parser.add_argument("--plan", default=str(PLAN_PATH))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = Path(args.summary).resolve()
    if summary_path.exists():
        print(f"refusing to overwrite summary: {summary_path}", file=sys.stderr)
        return 2
    try:
        summary = run_formal_evaluation(args)
    except BaseException as error:
        failure = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "phase": PHASE,
            "kind": "final_native_grid_evaluation",
            "variant": args.variant,
            "pass": False,
            "errors": [f"{type(error).__name__}: {error}"],
            "warnings": [],
            "plan": {"path": str(Path(args.plan).resolve()), "sha256": PLAN_SHA256},
            "source_sha256": file_sha256(Path(__file__).resolve()),
            "checkpoint": {"path": str(Path(args.checkpoint).resolve())},
            "cache": {"path": str(Path(args.cache).resolve())},
            "traceback": traceback.format_exc(),
        }
        try:
            write_json_new(summary_path, failure)
        except BaseException as write_error:
            print(f"failed to seal evaluator failure: {write_error}", file=sys.stderr)
        print(f"C1 evaluator failed: {error}", file=sys.stderr)
        return 1
    write_json_new(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
