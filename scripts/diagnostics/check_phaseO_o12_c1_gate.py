#!/usr/bin/env python3
"""Fail-closed paired gate for the Phase O1.2-C1 20k comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import evaluate_phaseO_o12_c1_final as evaluator  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PHASE = "O1.2-C1"
SCHEMA_VERSION = 1
PLAN_PATH = ROOT / "reports/2026-07-13_phaseO_rtc_o12_budgeted_routing_plan.md"
PLAN_SHA256 = "c6ac659aea7019d8c2faed88ccdd596678e909129e3a468942cde921d6a6d8b9"
BOOTSTRAP_PATH = (
    ROOT
    / "runs/diagnostics/phaseO_o12_c1/bootstrap_indices_pcg64_3407.npy"
)
BOOTSTRAP_SHA256 = "de2b18873dcd9f05f2d1d7acd9c0d94088680fb009441a501b8ba31ee8ce10b5"
BOOTSTRAP_SEED = 3407
BOOTSTRAP_REPLICATES = 10000
IMAGE_COUNT = 1449
EXPECTED_STEPS = list(range(800, 20001, 800))
LAST10_STEPS = list(range(12800, 20001, 800))
GATE_FLOAT_TOLERANCE = 1e-12


class C1GateError(RuntimeError):
    """A paired C1 structural or numerical contract violation."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path, label: str) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise C1GateError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise C1GateError(f"{label} is invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise C1GateError(f"{label} root is not an object")
    return payload


def _new_npy(path: str | Path, array: np.ndarray) -> str:
    path = Path(path).resolve()
    if path.exists():
        raise C1GateError(f"refusing to overwrite NPY: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return file_sha256(path)


def generate_bootstrap_indices(
    image_count: int = IMAGE_COUNT,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    if image_count <= 0 or replicates <= 0:
        raise C1GateError("bootstrap dimensions must be positive")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    result = generator.integers(
        0, int(image_count), size=(int(replicates), int(image_count)), dtype=np.int32
    )
    if result.dtype != np.int32 or result.shape != (replicates, image_count):
        raise C1GateError("bootstrap generator returned the wrong dtype/shape")
    return result


def load_or_create_formal_bootstrap(path: str | Path) -> tuple[np.ndarray, str]:
    path = Path(path).resolve()
    if path != BOOTSTRAP_PATH.resolve():
        raise C1GateError(
            f"formal bootstrap path must be canonical: {BOOTSTRAP_PATH.resolve()}"
        )
    if not path.exists():
        observed = _new_npy(path, generate_bootstrap_indices())
    else:
        observed = file_sha256(path)
    if observed != BOOTSTRAP_SHA256:
        raise C1GateError(
            f"bootstrap SHA mismatch: expected={BOOTSTRAP_SHA256} observed={observed}"
        )
    try:
        indices = np.load(path, allow_pickle=False)
    except Exception as error:
        raise C1GateError(f"bootstrap matrix cannot be loaded: {error}") from error
    if indices.dtype != np.int32 or indices.shape != (BOOTSTRAP_REPLICATES, IMAGE_COUNT):
        raise C1GateError("formal bootstrap dtype/shape mismatch")
    if np.any(indices < 0) or np.any(indices >= IMAGE_COUNT):
        raise C1GateError("formal bootstrap index is outside [0,1449)")
    return indices, observed


def paired_bootstrap_deltas(
    neutral_counts: np.ndarray,
    unreliable_counts: np.ndarray,
    indices: np.ndarray,
    *,
    chunk_size: int = 128,
) -> np.ndarray:
    """Aggregate per-image ratios first, then return U-N paired deltas."""

    neutral = np.asarray(neutral_counts)
    unreliable = np.asarray(unreliable_counts)
    indices = np.asarray(indices)
    if neutral.dtype != np.int64 or unreliable.dtype != np.int64:
        raise C1GateError("metric counts must use int64")
    if neutral.shape != unreliable.shape or neutral.ndim != 3 or neutral.shape[1:] != (3, 2):
        raise C1GateError("paired counts must both have shape [N,3,2]")
    if indices.dtype != np.int32 or indices.ndim != 2 or indices.shape[1] != neutral.shape[0]:
        raise C1GateError("bootstrap matrix shape/dtype is incompatible with counts")
    if np.any(indices < 0) or np.any(indices >= neutral.shape[0]):
        raise C1GateError("bootstrap contains an out-of-range image index")
    if chunk_size <= 0:
        raise C1GateError("bootstrap chunk_size must be positive")

    deltas = np.empty((indices.shape[0], 3), dtype=np.float64)
    for start in range(0, indices.shape[0], chunk_size):
        stop = min(start + chunk_size, indices.shape[0])
        selected = indices[start:stop]
        neutral_totals = neutral[selected].sum(axis=1, dtype=np.int64)
        unreliable_totals = unreliable[selected].sum(axis=1, dtype=np.int64)
        for label, totals in (
            ("neutral", neutral_totals),
            ("unreliable_only", unreliable_totals),
        ):
            if np.any(totals[:, :, 1] <= 0):
                bad = np.argwhere(totals[:, :, 1] <= 0)[0]
                raise C1GateError(
                    f"zero bootstrap denominator for {label} at "
                    f"replicate={start+int(bad[0])} metric={int(bad[1])}"
                )
            if np.any(totals[:, :, 0] < 0) or np.any(
                totals[:, :, 0] > totals[:, :, 1]
            ):
                raise C1GateError(f"invalid bootstrap counts for {label}")
        neutral_ratio = neutral_totals[:, :, 0] / neutral_totals[:, :, 1]
        unreliable_ratio = (
            unreliable_totals[:, :, 0] / unreliable_totals[:, :, 1]
        )
        deltas[start:stop] = unreliable_ratio - neutral_ratio
    if not np.isfinite(deltas).all():
        raise C1GateError("paired bootstrap produced non-finite deltas")
    return deltas


def bootstrap_intervals(deltas: np.ndarray) -> dict[str, dict[str, float]]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise C1GateError("paired delta array must have shape [B,3]")
    if not np.isfinite(values).all():
        raise C1GateError("paired deltas are non-finite")
    result: dict[str, dict[str, float]] = {}
    for index, name in enumerate(evaluator.METRIC_NAMES):
        low, high = np.quantile(
            values[:, index], [0.025, 0.975], method="linear"
        )
        result[name] = {
            "lower_2p5": float(low),
            "upper_97p5": float(high),
            "mean": float(values[:, index].mean()),
        }
    return result


def _finite_unit(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise C1GateError(f"{label} is not numeric") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise C1GateError(f"{label} is not finite in [0,1]")
    return result


def parse_training_acceptance(
    payload: Mapping[str, Any], variant: str
) -> dict[str, Any]:
    expected_top = {
        "schema_version": 1,
        "phase": PHASE,
        "stage": "final",
        "mode": "fresh",
        "variant": variant,
        "pass": True,
        "errors": [],
        "warnings": [],
    }
    for key, expected in expected_top.items():
        if payload.get(key) != expected:
            raise C1GateError(f"{variant} acceptance {key} mismatch")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise C1GateError(f"{variant} acceptance checkpoint is missing")
    checkpoint_expected = {
        "iteration": 20000,
        "version": 4,
        "sample_order_sha256": evaluator.EXPECTED_ORDER_SHA256,
    }
    for key, expected in checkpoint_expected.items():
        if checkpoint.get(key) != expected:
            raise C1GateError(f"{variant} acceptance checkpoint.{key} mismatch")
    if not isinstance(checkpoint.get("path"), str) or not Path(
        checkpoint["path"]
    ).is_absolute():
        raise C1GateError(f"{variant} checkpoint path is not absolute")
    if not isinstance(checkpoint.get("sha256"), str) or len(checkpoint["sha256"]) != 64:
        raise C1GateError(f"{variant} checkpoint SHA is invalid")

    artifacts = payload.get("artifact_hashes")
    if not isinstance(artifacts, Mapping):
        raise C1GateError(f"{variant} artifact_hashes is missing")
    expected_artifacts = {
        "teacher": evaluator.TEACHER_SHA256,
        "student_init": evaluator.STUDENT_INIT_SHA256,
        "cdf": evaluator.CDF_SHA256,
        "parameters": evaluator.PARAMETERS_SHA256,
        "gate": evaluator.GATE_SHA256,
        "o11_gate": evaluator.O11_GATE_SHA256,
        "train_list": evaluator.TRAIN_LIST_SHA256,
        "val_list": evaluator.VAL_LIST_SHA256,
        "bootstrap_indices": BOOTSTRAP_SHA256,
        "plan": PLAN_SHA256,
    }
    for key, expected in expected_artifacts.items():
        if artifacts.get(key) != expected:
            raise C1GateError(f"{variant} acceptance artifact {key} mismatch")
    sources = artifacts.get("sources")
    if not isinstance(sources, Mapping):
        raise C1GateError(f"{variant} acceptance sources are missing")
    if sources.get("train_entry") != evaluator.TRAIN_ENTRY_SHA256:
        raise C1GateError(f"{variant} train entry SHA mismatch")
    if sources.get("rtc_o12_calibration") != evaluator.O12_MODULE_SHA256:
        raise C1GateError(f"{variant} O1.2 source SHA mismatch")

    validation = payload.get("validation")
    if not isinstance(validation, Mapping):
        raise C1GateError(f"{variant} validation summary is missing")
    if validation.get("source") != "single_npu_sample_1449_cumulative_mIoU":
        raise C1GateError(f"{variant} validation source mismatch")
    if validation.get("expected_steps") != EXPECTED_STEPS:
        raise C1GateError(f"{variant} validation expected_steps mismatch")
    blocks = validation.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != len(EXPECTED_STEPS):
        raise C1GateError(f"{variant} must have exactly 25 validation blocks")
    parsed_blocks: list[dict[str, float | int]] = []
    for expected_step, block in zip(EXPECTED_STEPS, blocks):
        if not isinstance(block, Mapping):
            raise C1GateError(f"{variant} validation block is not an object")
        if block.get("step") != expected_step or block.get("next_sample") != 1450:
            raise C1GateError(f"{variant} validation block order/completeness mismatch")
        parsed_blocks.append(
            {
                "step": expected_step,
                "pix_acc": _finite_unit(
                    block.get("final_pix_acc"), f"{variant} step {expected_step} pixAcc"
                ),
                "miou": _finite_unit(
                    block.get("final_miou"), f"{variant} step {expected_step} mIoU"
                ),
            }
        )
    if validation.get("final_step") != 20000:
        raise C1GateError(f"{variant} final validation step mismatch")
    final = _finite_unit(validation.get("final_mIoU"), f"{variant} final_mIoU")
    best = _finite_unit(validation.get("best_mIoU"), f"{variant} best_mIoU")
    last10_mean = _finite_unit(
        validation.get("last10_mean_mIoU"), f"{variant} last10 mean"
    )
    recomputed_final = float(parsed_blocks[-1]["miou"])
    recomputed_best = max(float(row["miou"]) for row in parsed_blocks)
    recomputed_last10 = sum(float(row["miou"]) for row in parsed_blocks[-10:]) / 10
    tolerance = 5e-13
    for label, observed, expected in (
        ("final", final, recomputed_final),
        ("best", best, recomputed_best),
        ("last10", last10_mean, recomputed_last10),
    ):
        if abs(observed - expected) > tolerance:
            raise C1GateError(f"{variant} {label} summary does not recompute")
    last10 = validation.get("last10")
    if not isinstance(last10, list) or len(last10) != 10:
        raise C1GateError(f"{variant} last10 detail is incomplete")
    for expected_step, row, block in zip(LAST10_STEPS, last10, parsed_blocks[-10:]):
        if not isinstance(row, Mapping) or row.get("step") != expected_step:
            raise C1GateError(f"{variant} last10 step mismatch")
        value = _finite_unit(row.get("mIoU"), f"{variant} last10 {expected_step}")
        if abs(value - float(block["miou"])) > tolerance:
            raise C1GateError(f"{variant} last10 value mismatch")
    if abs(_finite_unit(checkpoint.get("best_pred"), f"{variant} best_pred") - best) > 5.1e-7:
        raise C1GateError(f"{variant} checkpoint best_pred mismatch")
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping) or int(runtime.get("optimizer_steps", -1)) != 20000:
        raise C1GateError(f"{variant} optimizer step count mismatch")
    return {
        "checkpoint_path": str(Path(checkpoint["path"]).resolve()),
        "checkpoint_sha256": checkpoint["sha256"],
        "validation_blocks": parsed_blocks,
        "final_mIoU": final,
        "best_mIoU": best,
        "last10_mean_mIoU": last10_mean,
    }


def validate_evaluation_summary(
    summary: Mapping[str, Any],
    variant: str,
    cache_path: str | Path,
    training: Mapping[str, Any],
) -> None:
    expected = {
        "schema_version": evaluator.SUMMARY_SCHEMA_VERSION,
        "phase": PHASE,
        "kind": "final_native_grid_evaluation",
        "variant": variant,
        "pass": True,
        "errors": [],
        "warnings": [],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise C1GateError(f"{variant} evaluator summary {key} mismatch")
    if summary.get("plan", {}).get("sha256") != PLAN_SHA256:
        raise C1GateError(f"{variant} evaluator plan SHA mismatch")
    expected_source = file_sha256(Path(evaluator.__file__).resolve())
    if summary.get("source_sha256") != expected_source:
        raise C1GateError(f"{variant} evaluator source SHA mismatch")
    cache = summary.get("cache")
    if not isinstance(cache, Mapping):
        raise C1GateError(f"{variant} evaluator cache metadata is missing")
    resolved_cache = str(Path(cache_path).resolve())
    if str(Path(cache.get("path", "")).resolve()) != resolved_cache:
        raise C1GateError(f"{variant} evaluator cache path mismatch")
    if cache.get("sha256") != file_sha256(resolved_cache):
        raise C1GateError(f"{variant} evaluator cache SHA mismatch")
    if cache.get("image_count") != IMAGE_COUNT:
        raise C1GateError(f"{variant} evaluator cache image count mismatch")
    checkpoint = summary.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise C1GateError(f"{variant} evaluator checkpoint metadata missing")
    if str(Path(checkpoint.get("path", "")).resolve()) != training["checkpoint_path"]:
        raise C1GateError(f"{variant} evaluator/training checkpoint path mismatch")
    if checkpoint.get("sha256") != training["checkpoint_sha256"]:
        raise C1GateError(f"{variant} evaluator/training checkpoint SHA mismatch")
    if checkpoint.get("iteration") != 20000 or checkpoint.get("version") != 4:
        raise C1GateError(f"{variant} evaluator checkpoint identity mismatch")
    if checkpoint.get("sample_order_sha256") != evaluator.EXPECTED_ORDER_SHA256:
        raise C1GateError(f"{variant} evaluator order SHA mismatch")


def validate_paired_caches(
    neutral: Mapping[str, np.ndarray], unreliable: Mapping[str, np.ndarray]
) -> None:
    pair_equal_keys = (
        "names",
        "shapes",
        "offsets",
        "teacher_flat",
        "gt_flat",
        "valid_flat",
        "u_flat",
        "metric_names",
    )
    for key in pair_equal_keys:
        if not np.array_equal(neutral[key], unreliable[key]):
            raise C1GateError(f"paired cache mismatch: {key}")
    if len(neutral["names"]) != IMAGE_COUNT:
        raise C1GateError("paired cache does not contain 1,449 images")
    neutral_denominators = neutral["metric_counts"][:, :, 1]
    unreliable_denominators = unreliable["metric_counts"][:, :, 1]
    if not np.array_equal(neutral_denominators, unreliable_denominators):
        raise C1GateError("paired cache metric denominators differ")


def _direct_metrics(counts: np.ndarray) -> dict[str, dict[str, Any]]:
    return evaluator.metric_summary(np.asarray(counts, dtype=np.int64))


def build_gate_result(
    neutral_cache: Mapping[str, np.ndarray],
    unreliable_cache: Mapping[str, np.ndarray],
    neutral_training: Mapping[str, Any],
    unreliable_training: Mapping[str, Any],
    bootstrap_indices: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    validate_paired_caches(neutral_cache, unreliable_cache)
    neutral_metrics = _direct_metrics(neutral_cache["metric_counts"])
    unreliable_metrics = _direct_metrics(unreliable_cache["metric_counts"])
    metric_deltas = {
        name: unreliable_metrics[name]["value"] - neutral_metrics[name]["value"]
        for name in evaluator.METRIC_NAMES
    }
    deltas = paired_bootstrap_deltas(
        neutral_cache["metric_counts"],
        unreliable_cache["metric_counts"],
        bootstrap_indices,
    )
    intervals = bootstrap_intervals(deltas)
    performance = {
        "neutral": {
            key: neutral_training[key]
            for key in ("final_mIoU", "best_mIoU", "last10_mean_mIoU")
        },
        "unreliable_only": {
            key: unreliable_training[key]
            for key in ("final_mIoU", "best_mIoU", "last10_mean_mIoU")
        },
    }
    final_delta = (
        unreliable_training["final_mIoU"] - neutral_training["final_mIoU"]
    )
    checks = {
        "delta_student_rescue_at_least_0p005": (
            metric_deltas["student_rescue_U"]
            >= 0.005 - GATE_FLOAT_TOLERANCE
        ),
        "student_rescue_ci_lower_strictly_positive": (
            intervals["student_rescue_U"]["lower_2p5"] > 0.0
        ),
        "delta_error_imitation_nonpositive": (
            metric_deltas["error_imitation_U"] <= GATE_FLOAT_TOLERANCE
        ),
        "final_miou_delta_at_least_minus_0p002": (
            final_delta >= -0.002 - GATE_FLOAT_TOLERANCE
        ),
        "delta_teacher_correct_retention_at_least_minus_0p005": (
            metric_deltas["teacher_correct_retention_U"]
            >= -0.005 - GATE_FLOAT_TOLERANCE
        ),
        "numerical_runtime_configuration_integrity": True,
    }
    return (
        {
            "metrics": {
                "neutral": neutral_metrics,
                "unreliable_only": unreliable_metrics,
                "delta_unreliable_minus_neutral": metric_deltas,
            },
            "bootstrap_intervals": intervals,
            "performance": performance,
            "performance_delta_unreliable_minus_neutral": {
                "final_mIoU": final_delta,
                "best_mIoU": (
                    unreliable_training["best_mIoU"]
                    - neutral_training["best_mIoU"]
                ),
                "last10_mean_mIoU": (
                    unreliable_training["last10_mean_mIoU"]
                    - neutral_training["last10_mean_mIoU"]
                ),
            },
            "checks": checks,
            "joint_gate_pass": all(checks.values()),
        },
        deltas,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the paired O1.2-C1 gate")
    parser.add_argument("--neutral-summary", required=True)
    parser.add_argument("--neutral-cache", required=True)
    parser.add_argument("--neutral-acceptance", required=True)
    parser.add_argument("--unreliable-summary", required=True)
    parser.add_argument("--unreliable-cache", required=True)
    parser.add_argument("--unreliable-acceptance", required=True)
    parser.add_argument("--bootstrap-indices", default=str(BOOTSTRAP_PATH))
    parser.add_argument("--paired-deltas", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def run_gate(args: argparse.Namespace) -> tuple[dict[str, Any], np.ndarray]:
    if file_sha256(PLAN_PATH) != PLAN_SHA256:
        raise C1GateError("C1 plan SHA mismatch")
    paths = {
        "neutral_summary": Path(args.neutral_summary).resolve(),
        "neutral_cache": Path(args.neutral_cache).resolve(),
        "neutral_acceptance": Path(args.neutral_acceptance).resolve(),
        "unreliable_summary": Path(args.unreliable_summary).resolve(),
        "unreliable_cache": Path(args.unreliable_cache).resolve(),
        "unreliable_acceptance": Path(args.unreliable_acceptance).resolve(),
    }
    neutral_acceptance = load_json(paths["neutral_acceptance"], "neutral acceptance")
    unreliable_acceptance = load_json(
        paths["unreliable_acceptance"], "unreliable acceptance"
    )
    neutral_training = parse_training_acceptance(neutral_acceptance, "neutral")
    unreliable_training = parse_training_acceptance(
        unreliable_acceptance, "unreliable_only"
    )
    neutral_summary = load_json(paths["neutral_summary"], "neutral evaluator summary")
    unreliable_summary = load_json(
        paths["unreliable_summary"], "unreliable evaluator summary"
    )
    validate_evaluation_summary(
        neutral_summary, "neutral", paths["neutral_cache"], neutral_training
    )
    validate_evaluation_summary(
        unreliable_summary,
        "unreliable_only",
        paths["unreliable_cache"],
        unreliable_training,
    )
    neutral_cache = evaluator.load_packed_cache(paths["neutral_cache"])
    unreliable_cache = evaluator.load_packed_cache(paths["unreliable_cache"])
    # Summary metrics are not trusted; independently recompute and compare.
    for variant, summary, cache in (
        ("neutral", neutral_summary, neutral_cache),
        ("unreliable_only", unreliable_summary, unreliable_cache),
    ):
        recomputed = evaluator.metric_summary(cache["metric_counts"])
        if summary.get("metrics") != recomputed:
            raise C1GateError(f"{variant} summary metrics do not match its cache")

    bootstrap_indices, bootstrap_sha = load_or_create_formal_bootstrap(
        args.bootstrap_indices
    )
    result, deltas = build_gate_result(
        neutral_cache,
        unreliable_cache,
        neutral_training,
        unreliable_training,
        bootstrap_indices,
    )
    result.update(
        {
            "schema_version": SCHEMA_VERSION,
            "phase": PHASE,
            "kind": "paired_final_gate",
            "pass": bool(result["joint_gate_pass"]),
            "errors": [],
            "warnings": [],
            "plan": {"path": str(PLAN_PATH.resolve()), "sha256": PLAN_SHA256},
            "source_sha256": file_sha256(Path(__file__).resolve()),
            "evaluator_source_sha256": file_sha256(Path(evaluator.__file__).resolve()),
            "inputs": {
                key: {"path": str(path), "sha256": file_sha256(path)}
                for key, path in paths.items()
            },
            "bootstrap": {
                "path": str(Path(args.bootstrap_indices).resolve()),
                "sha256": bootstrap_sha,
                "algorithm": "numpy.random.Generator(PCG64(3407)).integers",
                "shape": [BOOTSTRAP_REPLICATES, IMAGE_COUNT],
                "dtype": "int32",
                "quantile_method": "linear",
            },
            "paired_delta_artifact": {
                "path": str(Path(args.paired_deltas).resolve()),
                "shape": [BOOTSTRAP_REPLICATES, len(evaluator.METRIC_NAMES)],
                "dtype": "float64",
                "columns": list(evaluator.METRIC_NAMES),
            },
            "decision": {
                "next_stage_authorized": False,
                "automatic_c2_launch": False,
                "action": "stop_for_manual_review",
                "scope": "C1 signal only; never launches C2",
            },
        }
    )
    failed = [name for name, passed in result["checks"].items() if not passed]
    if failed:
        result["errors"] = [f"C1 continuation check failed: {name}" for name in failed]
    return result, deltas


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output).resolve()
    paired_path = Path(args.paired_deltas).resolve()
    if output.exists() or paired_path.exists():
        print("refusing to overwrite paired gate output", file=sys.stderr)
        return 2
    try:
        result, deltas = run_gate(args)
        paired_sha = _new_npy(paired_path, deltas.astype(np.float64, copy=False))
        result["paired_delta_artifact"]["sha256"] = paired_sha
    except BaseException as error:
        result = {
            "schema_version": SCHEMA_VERSION,
            "phase": PHASE,
            "kind": "paired_final_gate",
            "pass": False,
            "joint_gate_pass": False,
            "errors": [f"{type(error).__name__}: {error}"],
            "warnings": [],
            "plan": {"path": str(PLAN_PATH.resolve()), "sha256": PLAN_SHA256},
            "source_sha256": file_sha256(Path(__file__).resolve()),
            "decision": {
                "next_stage_authorized": False,
                "automatic_c2_launch": False,
                "action": "stop_for_manual_review",
            },
            "traceback": traceback.format_exc(),
        }
    evaluator.write_json_new(output, result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result.get("pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
