from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / "scripts/diagnostics"
if str(DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS))

import check_phaseO_o12_c1_gate as gate
import evaluate_phaseO_o12_c1_final as evaluator


def _record(name: str, student, teacher, gt, u):
    gt = np.asarray(gt, dtype=np.int16)
    return {
        "name": name,
        "student_prediction": np.asarray(student, dtype=np.int16),
        "teacher_prediction": np.asarray(teacher, dtype=np.int16),
        "ground_truth": gt,
        "valid_mask": (gt != -1).astype(np.uint8),
        "reliability_quantile": np.asarray(u, dtype=np.float32),
    }


def test_counts_use_strict_high_risk_and_preserve_third_wrong():
    # Pixels 0..2 are W: rescue, imitation, and a third-label error. Pixel 3
    # is exactly u=.8 and must not enter U. Pixels 4..5 are C.
    record = _record(
        "sample",
        student=[[0, 1, 2, 0, 0, 1, 0, 0]],
        teacher=[[1, 1, 1, 1, 0, 0, 1, 0]],
        gt=[[0, 0, 0, 0, 0, 0, -1, 0]],
        u=[[0.9, 0.9, 0.9, 0.8, 0.9, 0.9, 0.99, 0.2]],
    )
    counts = evaluator.compute_image_counts(
        record["student_prediction"],
        record["teacher_prediction"],
        record["ground_truth"],
        record["valid_mask"],
        record["reliability_quantile"],
    )
    assert counts.dtype == np.int64
    assert counts.tolist() == [[1, 3], [1, 3], [1, 2]]


def test_counts_reject_mask_not_exactly_gt_valid():
    with pytest.raises(evaluator.C1EvaluationError, match="exactly"):
        evaluator.compute_image_counts(
            np.zeros((1, 2), dtype=np.int16),
            np.zeros((1, 2), dtype=np.int16),
            np.asarray([[0, -1]], dtype=np.int16),
            np.ones((1, 2), dtype=np.uint8),
            np.asarray([[0.9, 0.9]], dtype=np.float32),
        )


def test_cache_roundtrip_recomputes_counts(tmp_path):
    records = [
        _record(
            "a",
            student=[[0, 1, 2]],
            teacher=[[1, 1, 1]],
            gt=[[0, 0, 0]],
            u=[[0.9, 0.9, 0.9]],
        ),
        _record(
            "b",
            student=[[0, 1], [0, 0]],
            teacher=[[0, 0], [1, 0]],
            gt=[[0, 0], [0, -1]],
            u=[[0.95, 0.81], [0.99, 0.99]],
        ),
    ]
    payload = evaluator.make_cache_payload(records)
    path = tmp_path / "cache.npz"
    observed_sha = evaluator.save_packed_cache(path, payload)
    loaded = evaluator.load_packed_cache(path)
    assert observed_sha == evaluator.file_sha256(path)
    assert loaded["names"].tolist() == ["a", "b"]
    assert loaded["shapes"].tolist() == [[1, 3], [2, 2]]
    assert loaded["offsets"].tolist() == [0, 3, 7]
    assert np.array_equal(loaded["metric_counts"], payload["metric_counts"])
    assert loaded["u_flat"].dtype == np.float32


def test_cache_detects_tampered_counts():
    payload = evaluator.make_cache_payload(
        [
            _record(
                "a",
                student=[[0]],
                teacher=[[1]],
                gt=[[0]],
                u=[[0.9]],
            )
        ]
    )
    payload["metric_counts"][0, 0, 0] = 0
    with pytest.raises(evaluator.C1EvaluationError, match="metric_counts mismatch"):
        evaluator.validate_cache_payload(payload, recompute=True)


def test_formal_bootstrap_is_reproducible_and_matches_frozen_npy_sha(tmp_path):
    first = gate.generate_bootstrap_indices()
    second = gate.generate_bootstrap_indices()
    assert first.dtype == np.int32
    assert first.shape == (10000, 1449)
    assert np.array_equal(first, second)
    path = tmp_path / "bootstrap.npy"
    with path.open("wb") as handle:
        np.save(handle, first, allow_pickle=False)
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    assert observed == gate.BOOTSTRAP_SHA256


def test_paired_bootstrap_is_reproducible():
    neutral = np.asarray(
        [
            [[1, 2], [1, 2], [2, 2]],
            [[0, 1], [1, 1], [1, 2]],
        ],
        dtype=np.int64,
    )
    unreliable = np.asarray(
        [
            [[2, 2], [0, 2], [2, 2]],
            [[1, 1], [0, 1], [1, 2]],
        ],
        dtype=np.int64,
    )
    indices = gate.generate_bootstrap_indices(image_count=2, replicates=200, seed=3407)
    first = gate.paired_bootstrap_deltas(neutral, unreliable, indices)
    second = gate.paired_bootstrap_deltas(neutral, unreliable, indices)
    assert first.shape == (200, 3)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_any_complete_bootstrap_zero_denominator_fails():
    counts = np.asarray(
        [
            [[0, 0], [0, 0], [1, 1]],
            [[1, 1], [1, 1], [0, 0]],
        ],
        dtype=np.int64,
    )
    # The first replicate selects only image 0, so W is zero; the second
    # selects only image 1, so C is zero. Either is a structural failure.
    indices = np.asarray([[0, 0], [1, 1]], dtype=np.int32)
    with pytest.raises(gate.C1GateError, match="zero bootstrap denominator"):
        gate.paired_bootstrap_deltas(counts, counts, indices)


def _minimal_cache(counts: np.ndarray) -> dict[str, np.ndarray]:
    image_count = gate.IMAGE_COUNT
    names = np.asarray([f"x{index:04d}" for index in range(image_count)])
    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "names": names,
        "shapes": np.ones((image_count, 2), dtype=np.int64),
        "offsets": np.arange(image_count + 1, dtype=np.int64),
        "student_flat": np.zeros(image_count, dtype=np.int16),
        "teacher_flat": np.zeros(image_count, dtype=np.int16),
        "gt_flat": np.zeros(image_count, dtype=np.int16),
        "valid_flat": np.ones(image_count, dtype=np.uint8),
        "u_flat": np.full(image_count, 0.9, dtype=np.float32),
        "metric_names": np.asarray(evaluator.METRIC_NAMES, dtype="<U32"),
        "metric_counts": np.repeat(
            np.asarray(counts, dtype=np.int64).reshape(1, 3, 2),
            image_count,
            axis=0,
        ),
    }


def test_pair_invariants_reject_teacher_or_denominator_drift():
    counts = np.asarray([[1, 2], [1, 2], [2, 3]], dtype=np.int64)
    neutral = _minimal_cache(counts)
    unreliable = _minimal_cache(counts.copy())
    gate.validate_paired_caches(neutral, unreliable)
    unreliable["teacher_flat"][0] = 1
    with pytest.raises(gate.C1GateError, match="teacher_flat"):
        gate.validate_paired_caches(neutral, unreliable)
    unreliable = _minimal_cache(counts.copy())
    unreliable["metric_counts"][0, 0, 1] += 1
    with pytest.raises(gate.C1GateError, match="denominators"):
        gate.validate_paired_caches(neutral, unreliable)


def test_gate_threshold_boundaries_are_inclusive_but_ci_is_strict():
    neutral_counts = np.asarray(
        [[0, 200], [100, 200], [200, 200]], dtype=np.int64
    )
    unreliable_counts = np.asarray(
        [[1, 200], [100, 200], [199, 200]], dtype=np.int64
    )
    neutral = _minimal_cache(neutral_counts)
    unreliable = _minimal_cache(unreliable_counts)
    training_neutral = {
        "final_mIoU": 0.500,
        "best_mIoU": 0.510,
        "last10_mean_mIoU": 0.505,
    }
    training_unreliable = {
        "final_mIoU": 0.498,
        "best_mIoU": 0.511,
        "last10_mean_mIoU": 0.506,
    }
    indices = np.zeros((100, gate.IMAGE_COUNT), dtype=np.int32)
    result, deltas = gate.build_gate_result(
        neutral, unreliable, training_neutral, training_unreliable, indices
    )
    assert result["joint_gate_pass"] is True
    assert all(result["checks"].values())
    assert result["metrics"]["delta_unreliable_minus_neutral"][
        "student_rescue_U"
    ] == pytest.approx(0.005)
    assert result["bootstrap_intervals"]["student_rescue_U"]["lower_2p5"] > 0
    assert deltas.shape == (100, 3)


def test_gate_rejects_nonpositive_rescue_ci():
    counts = np.asarray([[0, 200], [0, 200], [200, 200]], dtype=np.int64)
    neutral = _minimal_cache(counts)
    unreliable = _minimal_cache(counts.copy())
    training = {
        "final_mIoU": 0.5,
        "best_mIoU": 0.5,
        "last10_mean_mIoU": 0.5,
    }
    result, _ = gate.build_gate_result(
        neutral,
        unreliable,
        training,
        training,
        np.zeros((10, gate.IMAGE_COUNT), dtype=np.int32),
    )
    assert result["joint_gate_pass"] is False
    assert result["checks"]["student_rescue_ci_lower_strictly_positive"] is False


def _acceptance(variant: str) -> dict:
    blocks = [
        {
            "step": step,
            "next_sample": 1450,
            "final_pix_acc": 0.8,
            "final_miou": 0.4 + index * 0.001,
        }
        for index, step in enumerate(gate.EXPECTED_STEPS)
    ]
    final = blocks[-1]["final_miou"]
    best = max(row["final_miou"] for row in blocks)
    last10 = [
        {"step": row["step"], "mIoU": row["final_miou"]}
        for row in blocks[-10:]
    ]
    return {
        "schema_version": 1,
        "phase": "O1.2-C1",
        "stage": "final",
        "mode": "fresh",
        "variant": variant,
        "pass": True,
        "errors": [],
        "warnings": [],
        "checkpoint": {
            "path": f"/tmp/{variant}/training_state_latest.pth",
            "sha256": "a" * 64,
            "iteration": 20000,
            "version": 4,
            "best_pred": best,
            "sample_order_sha256": evaluator.EXPECTED_ORDER_SHA256,
        },
        "artifact_hashes": {
            "teacher": evaluator.TEACHER_SHA256,
            "student_init": evaluator.STUDENT_INIT_SHA256,
            "cdf": evaluator.CDF_SHA256,
            "parameters": evaluator.PARAMETERS_SHA256,
            "gate": evaluator.GATE_SHA256,
            "o11_gate": evaluator.O11_GATE_SHA256,
            "train_list": evaluator.TRAIN_LIST_SHA256,
            "val_list": evaluator.VAL_LIST_SHA256,
            "bootstrap_indices": gate.BOOTSTRAP_SHA256,
            "plan": gate.PLAN_SHA256,
            "sources": {
                "train_entry": evaluator.TRAIN_ENTRY_SHA256,
                "rtc_o12_calibration": evaluator.O12_MODULE_SHA256,
            },
        },
        "validation": {
            "source": "single_npu_sample_1449_cumulative_mIoU",
            "expected_steps": gate.EXPECTED_STEPS,
            "blocks": blocks,
            "final_step": 20000,
            "final_mIoU": final,
            "best_mIoU": best,
            "last10": last10,
            "last10_mean_mIoU": sum(row["mIoU"] for row in last10) / 10,
        },
        "runtime": {"optimizer_steps": 20000},
    }


def test_training_acceptance_requires_25_ordered_final_blocks():
    payload = _acceptance("neutral")
    parsed = gate.parse_training_acceptance(payload, "neutral")
    assert parsed["final_mIoU"] == payload["validation"]["final_mIoU"]
    payload["validation"]["blocks"][3]["step"] = 999
    with pytest.raises(gate.C1GateError, match="order"):
        gate.parse_training_acceptance(payload, "neutral")
