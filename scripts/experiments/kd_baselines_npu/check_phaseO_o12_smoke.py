#!/usr/bin/env python3
"""Fail-closed verifier for Phase O1.2-B 20-step smoke artifacts."""

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path("/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python")
ASCEND_ENV = Path("/usr/local/Ascend/cann-8.5.0/set_env.sh")
EXPECTED_ORDER_SHA256 = (
    "99326472a2e5e2bd42428d4709ff9f8049d2c906c7a6b9e8fa3068cb0439d564"
)
EXPECTED_CDF_SHA256 = (
    "8ba8376a03c2f93835339d98670fa357b381b23a22cbf2a7fc48b0c2441d7e69"
)
EXPECTED_PARAMETERS_SHA256 = (
    "a9006972e165ed6c95e7a966c526927bb9f846fb6c7ec6528f5507dd21b8b5df"
)
EXPECTED_GATE_SHA256 = (
    "c1961cfec8f232c1fdf7fc8b90db5a6f4855901317ea2705536eea88cbc1be82"
)
EXPECTED_O11_GATE_SHA256 = (
    "47ff2f1f2ea68a4e50375bfa7efc7221c8197703372d5d8f22dfec9032088d3a"
)
EXPECTED_TEACHER_SHA256 = (
    "ac49b2c7720b21d565072e974e4404fcb009ba106288d95d1a4bb25f09c3fe58"
)
EXPECTED_STUDENT_INIT_SHA256 = (
    "47085aa164b2977003221458a2a5fdf5f46f434f5539b164dc71969b4dd4cd75"
)
EXPECTED_TRAIN_LIST_SHA256 = (
    "d1326bd532648d73bc4b1bd275434eba81930982dc7401a65a8cecb26c028e24"
)
EXPECTED_SOURCES = {
    "rtc_o12_calibration": (
        "5e23f3e1bd526017aa2046faff621dd284093d91dead9088dd4d7d9ddb641d9e"
    ),
    "diagnose_rtc_o12_budget": (
        "cc391388f64505abae4cded5ac7b36122018a131b3c90f480290ff275a4cee50"
    ),
    "check_rtc_o12_gate": (
        "805a19625d496d3c3864d529e314a49d75584afad69fedf69e28cadc431ce085"
    ),
    "train_entry": (
        "f50a130c17ca3b5f368f848b96c59fd58c408952e75bda447c051589505b316d"
    ),
}
EXPECTED_O11_SOURCES = {
    "rtc_temperature": (
        "01b7b6e6aa0d513561332510347b52ea9411330dfb0f2da54abdc36f2375fe59"
    ),
    "build_rtc_cdf": (
        "c88bdfcb885cde01cbf437e2e7751c8eab510067aacf530b469df808ae6604dd"
    ),
    "diagnose_rtc_routing": (
        "f838f25b70b8eadfc982873057e5fb68c54c81089a32e9121fd664e02916f9ef"
    ),
    "check_rtc_o11_gate": (
        "55553ec523ac2c2a979470b8542f31878462bbe5a919e0c52132edfbba4eb256"
    ),
}
EXPECTED_CONFIGURATION = {
    "phase": "O1.2",
    "reliability_mode": "confidence",
    "reliability_definition_id": "neg_log_top1_confidence_v1",
    "assess_temperature": 1.0,
    "epsilon": 1e-8,
    "q_reliable": 0.6,
    "q_unreliable": 0.8,
    "p_reliable": 1.0,
    "p_unreliable": 2.0,
    "a": 0.10536051565782628,
    "b_min": 0.0,
    "b_max": 0.4054651081081644,
    "target_arithmetic_mean": 0.995,
    "minimum_harmonic_mean": 0.98,
    "bisection_iterations": 64,
    "teacher_output_temperature": 3.0,
    "temperature_map_dtype": "float32",
    "budget_accumulator_dtype": "float64",
    "temperature_minimum": 0.9,
    "temperature_maximum": 1.5,
    "tau_temperature": 1e-6,
    "tau_probability": 1e-6,
    "tau_entropy": 1e-6,
    "tau_student": 1e-7,
    "tau_formula_monotonic": 1e-12,
}
EXPECTED_CALIBRATION = {
    "q_reliable": 0.6,
    "q_unreliable": 0.8,
    "p_reliable": 1.0,
    "p_unreliable": 2.0,
    "temperature_min": 0.9,
    "temperature_max": 1.5,
    "target_mean": 0.995,
    "min_harmonic_mean": 0.98,
    "bisection_iterations": 64,
    "teacher_output_temperature": 3.0,
    "a_star": 0.10536051565782628,
    "b_max": 0.4054651081081644,
}
EXPECTED_SCALARS = {
    "unreliable_only": {
        "arithmetic": 1.0256612287700149,
        "harmonic": 1.0212646673765022,
    },
    "full_budgeted": {
        "arithmetic": 0.9950000002788227,
        "harmonic": 0.9880688428627586,
    },
}
EXPECTED_ORDER = {
    "schema_version": 1,
    "algorithm": "rtc_o12_canonical_order_v1",
    "seed": 1234,
    "seed_derivation": (
        "seed63=SHA256(ASCII(algorithm|seed|sampling_epoch))[:8] "
        "big-endian mod (2^63-1)"
    ),
    "permutation": "torch.randperm(canonical_population, CPU generator)",
    "canonical_population": 10582,
    "batch_size": 16,
    "max_iterations": 20,
    "total_canonical_indices": 320,
    "complete_order_sha256": EXPECTED_ORDER_SHA256,
    "complete_order_sha256_scope": (
        "full canonical-index sequence for current max_iterations"
    ),
    "resume_offset": "completed_iteration * batch_size",
}
EXPECTED_DIRTY_STATUS = {
    "?? scripts/experiments/kd_baselines_npu/check_phaseO_rtc_runs.py",
    "?? scripts/experiments/kd_baselines_npu/launch_phaseO_rtc.sh",
    "?? scripts/experiments/kd_baselines_npu/run_phaseO_rtc.sh",
    "?? scripts/experiments/kd_baselines_npu/run_phaseO_rtc_variant.sh",
}
LOGGER_NAME = "deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt"
MODEL_NAME = "kd_deeplabv3_mobilenet_ssseg_mobilenetv3_small_voc.pth"
FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_fingerprint(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_checkpoint(path):
    kwargs = {"map_location": "cpu"}
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def require(condition, message, errors):
    if not condition:
        errors.append(message)


def require_exact_mapping(actual, expected, label, errors):
    if not isinstance(actual, dict):
        errors.append(f"{label} is not a dict")
        return
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        errors.append(f"{label} key mismatch: missing={missing} extra={extra}")
    for key, expected_value in expected.items():
        if key in actual and actual[key] != expected_value:
            errors.append(
                f"{label}.{key} mismatch: {actual[key]!r} != {expected_value!r}"
            )


def compare_nested(left, right, path="root"):
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            raise AssertionError(f"{path}: tensor/type mismatch")
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            raise AssertionError(f"{path}: tensor metadata mismatch")
        if not torch.equal(left.cpu(), right.cpu()):
            raise AssertionError(f"{path}: tensor value mismatch")
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and np.array_equal(left, right)
        ):
            raise AssertionError(f"{path}: ndarray mismatch")
        return
    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict)):
            raise AssertionError(f"{path}: dict/type mismatch")
        if set(left) != set(right):
            raise AssertionError(f"{path}: key mismatch")
        for key in sorted(left, key=str):
            compare_nested(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, type(right)) or len(left) != len(right):
            raise AssertionError(f"{path}: sequence mismatch")
        for index, (item_left, item_right) in enumerate(zip(left, right)):
            compare_nested(item_left, item_right, f"{path}[{index}]")
        return
    if left != right:
        raise AssertionError(f"{path}: value mismatch {left!r} != {right!r}")


def check_finite_nested(value, path, errors, counters):
    if torch.is_tensor(value):
        counters["tensors"] += 1
        counters["tensor_elements"] += int(value.numel())
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all().item()):
                errors.append(f"{path}: non-finite tensor")
        return
    if isinstance(value, np.ndarray):
        counters["arrays"] += 1
        counters["array_elements"] += int(value.size)
        if np.issubdtype(value.dtype, np.inexact) and not np.isfinite(value).all():
            errors.append(f"{path}: non-finite ndarray")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            check_finite_nested(item, f"{path}.{key}", errors, counters)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            check_finite_nested(item, f"{path}[{index}]", errors, counters)
        return
    if isinstance(value, (float, np.floating)):
        counters["float_scalars"] += 1
        if not math.isfinite(float(value)):
            errors.append(f"{path}: non-finite scalar")


def parse_provenance(path, errors):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            errors.append(f"malformed provenance line: {line!r}")
            continue
        key, value = line.split("=", 1)
        if key in rows:
            errors.append(f"duplicate provenance key: {key}")
        rows[key] = value
    return rows


def parse_iso(value, label, errors):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        errors.append(f"invalid {label}: {value!r}")
        return None


def finite_float(value, label, errors):
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} is not numeric: {value!r}")
        return None
    if not math.isfinite(number):
        errors.append(f"{label} is non-finite: {number}")
    return number


def read_argv_nul(path, errors):
    raw = Path(path).read_bytes()
    require(raw.endswith(b"\0"), "argv.nul is not NUL terminated", errors)
    chunks = raw.split(b"\0")
    if chunks and chunks[-1] == b"":
        chunks.pop()
    try:
        return [chunk.decode("utf-8") for chunk in chunks]
    except UnicodeDecodeError as error:
        errors.append(f"argv.nul is not UTF-8: {error}")
        return []


def canonical_paths():
    return {
        "teacher": (ROOT / "data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth").resolve(),
        "student_init": (ROOT / "data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth").resolve(),
        "cdf": (ROOT / "runs/diagnostics/phaseO_o11/voc_train_rtc_confidence_cdf.pt").resolve(),
        "parameters": (ROOT / "runs/diagnostics/phaseO_o12/o12_budget_parameters.json").resolve(),
        "gate": (ROOT / "runs/diagnostics/phaseO_o12/o12_joint_gate.json").resolve(),
        "o11_gate": (ROOT / "runs/diagnostics/phaseO_o11/o11_confidence_gate.json").resolve(),
        "train_list": (ROOT / "dataset/list/voc/train_aug.txt").resolve(),
        "train_entry": (ROOT / "train_kd.py").resolve(),
        "o12_module": (ROOT / "utils/rtc_o12_calibration.py").resolve(),
        "o12_diagnose": (ROOT / "scripts/diagnostics/diagnose_rtc_o12_budget.py").resolve(),
        "o12_gate_checker": (ROOT / "scripts/diagnostics/check_rtc_o12_gate.py").resolve(),
    }


def expected_training_argv(mode, variant, save_dir, log_dir, source_checkpoint):
    paths = canonical_paths()
    data_dir = str((ROOT / "dataset/VOCAug").resolve()) + "/"
    argv = [
        str(PYTHON),
        str(paths["train_entry"]),
        "--device-type", "npu",
        "--local-rank", "0",
        "--seed", "1234",
        "--teacher-model", "deeplabv3",
        "--teacher-backbone", "resnet101",
        "--student-model", "deeplabv3_mobilenet_ssseg",
        "--student-backbone", "mobilenetv3_small",
        "--dataset", "voc",
        "--data", data_dir,
        "--crop-size", "512", "512",
        "--batch-size", "16",
        "--workers", "8",
        "--ignore-label", "-1",
        "--start_epoch", "0",
        "--max-iterations", "20",
        "--lr", "0.02",
        "--momentum", "0.9",
        "--weight-decay", "0.0001",
        "--kd-loss-mode", "rtc_o12_teacher_target",
        "--rtc-o12-variant", variant,
        "--rtc-o12-cdf-path", str(paths["cdf"]),
        "--rtc-o12-parameters-path", str(paths["parameters"]),
        "--rtc-o12-gate-path", str(paths["gate"]),
        "--teacher-output-temp", "3.0",
        "--kd-temperature", "1.0",
        "--lambda-kd", "1.0",
        "--lambda-adv", "0.001",
        "--lambda-d", "0.1",
        "--lambda-cwd-fea", "50.0",
        "--lambda-cwd-logit", "3.0",
        "--lambda-skd", "0.0",
        "--lambda-ifv", "0.0",
        "--lambda-fitnet", "0.0",
        "--lambda-at", "0.0",
        "--lambda-psd", "0.0",
        "--lambda-csd", "0.0",
        "--teacher-pretrained-base", "None",
        "--teacher-pretrained", str(paths["teacher"]),
        "--student-pretrained-base", str(paths["student_init"]),
        "--student-pretrained", "None",
        "--log-iter", "20",
        "--save-per-iters", "800",
        "--val-per-iters", "800",
        "--skip-val",
        "--save-dir", str(save_dir),
        "--log-dir", str(log_dir),
    ]
    if mode == "resume_audit":
        argv.extend(["--resume", str(source_checkpoint)])
    return argv


def expected_saved_args(mode, variant, save_dir, log_dir, source_checkpoint):
    paths = canonical_paths()
    return {
        "teacher_model": "deeplabv3",
        "student_model": "deeplabv3_mobilenet_ssseg",
        "student_backbone": "mobilenetv3_small",
        "teacher_backbone": "resnet101",
        "dataset": "voc",
        "data": str((ROOT / "dataset/VOCAug").resolve()) + "/",
        "crop_size": [512, 512],
        "workers": 8,
        "ignore_label": -1,
        "aux": False,
        "batch_size": 16,
        "start_epoch": 0,
        "max_iterations": 20,
        "lr": 0.02,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "kd_temperature": 1.0,
        "kd_loss_mode": "rtc_o12_teacher_target",
        "lambda_kd": 1.0,
        "lambda_adv": 0.001,
        "lambda_d": 0.1,
        "lambda_skd": 0.0,
        "lambda_cwd_fea": 50.0,
        "lambda_cwd_logit": 3.0,
        "lambda_ifv": 0.0,
        "lambda_fitnet": 0.0,
        "lambda_at": 0.0,
        "lambda_psd": 0.0,
        "lambda_csd": 0.0,
        "use_covar": False,
        "covar_temp_mode": "newton",
        "covar_alpha": 2.0,
        "teacher_output_temp": 3.0,
        "covar_temp_base": 1.0,
        "covar_temp_min": 0.5,
        "covar_temp_max": 8.0,
        "covar_kd_temp_power": 2.0,
        "covar_grad_eta": 0.6,
        "covar_grad_max_iter": 8,
        "covar_a": None,
        "covar_reliability_mode": "full",
        "covar_newton_hessian_eps": 1e-5,
        "covar_newton_max_step": 0.25,
        "rtc_cdf_path": None,
        "rtc_assess_temperature": 1.0,
        "rtc_route_quantile": 0.8,
        "rtc_route_width": 0.05,
        "rtc_temp_reliable": 0.5,
        "rtc_temp_neutral": 1.0,
        "rtc_temp_unreliable": 2.0,
        "rtc_alpha_reliable": 1.0,
        "rtc_alpha_unreliable": 1.0,
        "rtc_enable_reliable": None,
        "rtc_enable_unreliable": None,
        "rtc_bisection_iters": 16,
        "rtc_shuffle": None,
        "rtc_reverse_routing": None,
        "rtc_o12_variant": variant,
        "rtc_o12_cdf_path": str(paths["cdf"]),
        "rtc_o12_parameters_path": str(paths["parameters"]),
        "rtc_o12_gate_path": str(paths["gate"]),
        "device_type": "npu",
        "seed": 1234,
        "no_cuda": False,
        "local_rank": 0,
        "resume": None if mode == "fresh" else str(source_checkpoint),
        "save_dir": str(save_dir),
        "save_epoch": 10,
        "log_dir": str(log_dir),
        "log_iter": 20,
        "save_per_iters": 800,
        "val_per_iters": 800,
        "teacher_pretrained_base": "None",
        "teacher_pretrained": str(paths["teacher"]),
        "student_pretrained_base": str(paths["student_init"]),
        "student_pretrained": "None",
        "val_epoch": 1,
        "skip_val": True,
        "num_gpus": 1,
        "distributed": False,
        "device": "npu",
        "rtc_o12_cdf_sha256": EXPECTED_CDF_SHA256,
        "rtc_o12_parameters_sha256": EXPECTED_PARAMETERS_SHA256,
        "rtc_o12_gate_sha256": EXPECTED_GATE_SHA256,
        "rtc_o12_complete_order_sha256": EXPECTED_ORDER_SHA256,
    }


def expected_artifact_contract():
    paths = canonical_paths()
    return {
        "configuration": EXPECTED_CONFIGURATION,
        "b": 0.3476499170064926,
        "branch_scalar_temperatures": EXPECTED_SCALARS,
        "cdf_path": str(paths["cdf"]),
        "cdf_sha256": EXPECTED_CDF_SHA256,
        "parameters_path": str(paths["parameters"]),
        "parameters_sha256": EXPECTED_PARAMETERS_SHA256,
        "gate_path": str(paths["gate"]),
        "gate_sha256": EXPECTED_GATE_SHA256,
        "o11_gate_path": str(paths["o11_gate"]),
        "o11_gate_sha256": EXPECTED_O11_GATE_SHA256,
        "train_list_path": str(paths["train_list"]),
        "train_list_sha256": EXPECTED_TRAIN_LIST_SHA256,
        "teacher_sha256": EXPECTED_TEACHER_SHA256,
        "student_init_sha256": EXPECTED_STUDENT_INIT_SHA256,
        "source_sha256": EXPECTED_SOURCES,
    }


def expected_rtc_o12(variant):
    return {
        "phase": "O1.2",
        "variant": variant,
        "variant_spec": {
            "variant": variant,
            "branch": variant,
            "scalar_moment": None,
            "shuffled": False,
        },
        "calibration_config": EXPECTED_CALIBRATION,
        "artifact_contract": expected_artifact_contract(),
        "teacher_target_contract": {
            "formula": "softmax(raw_teacher/(teacher_output_temperature*T_pixel))",
            "teacher_output_temperature": 3.0,
            "student_temperature": 1.0,
            "teacher_target_detached": True,
            "temperature_loss_power": None,
        },
        "scalar_temperature": None,
        "data_order_contract": EXPECTED_ORDER,
        "world_size": 1,
        "max_iterations": 20,
        "skip_val": True,
    }


def validate_cdf(path, errors):
    payload = load_checkpoint(path)
    require(isinstance(payload, dict), "CDF root is not a dict", errors)
    if not isinstance(payload, dict):
        return
    require(
        set(payload) == {
            "schema_version", "kind", "metadata",
            "quantile_probabilities", "quantile_values",
        },
        "CDF top-level keyset mismatch",
        errors,
    )
    require(payload.get("schema_version") == 1, "CDF schema mismatch", errors)
    require(payload.get("kind") == "rtc_reliability_cdf", "CDF kind mismatch", errors)
    metadata = payload.get("metadata", {})
    expected_metadata = {
        "phase": "O1.1",
        "dataset": "voc",
        "split": "train_aug",
        "num_classes": 21,
        "processed_images": 10582,
        "dataset_size": 10582,
        "full_dataset_scan": True,
        "batch_size": 4,
        "workers": 0,
        "max_images": 0,
        "max_pixels_per_image": 4096,
        "num_quantiles": 4097,
        "crop_size": [512, 512],
        "scale": True,
        "mirror": True,
        "seed": 1234,
        "teacher_output_grid": "native",
        "valid_mask_resize": "nearest",
        "assess_temperature": 1.0,
        "coefficient_a": 0.0,
        "coefficient_a_active": False,
        "reliability_mode": "confidence",
        "reliability_definition_id": "neg_log_top1_confidence_v1",
        "reliability_epsilon": 1e-8,
        "active_terms": ["confidence"],
        "teacher_sha256": EXPECTED_TEACHER_SHA256,
        "train_list_sha256": EXPECTED_TRAIN_LIST_SHA256,
        "nonfinite_valid_pixels": 0,
        "valid_native_pixels": 32298651,
        "finite_valid_pixels": 32298651,
        "source_sha256": EXPECTED_O11_SOURCES,
    }
    if not isinstance(metadata, dict):
        errors.append("CDF metadata is not a dict")
    else:
        for key, expected in expected_metadata.items():
            require(
                metadata.get(key) == expected,
                f"CDF metadata mismatch: {key}",
                errors,
            )
    for key in ("quantile_probabilities", "quantile_values"):
        tensor = payload.get(key)
        require(torch.is_tensor(tensor), f"CDF {key} is not a tensor", errors)
        if torch.is_tensor(tensor):
            require(tuple(tensor.shape) == (4097,), f"CDF {key} shape mismatch", errors)
            require(tensor.dtype == torch.float32, f"CDF {key} dtype mismatch", errors)
            require(bool(torch.isfinite(tensor).all().item()), f"CDF {key} non-finite", errors)
            require(
                bool((tensor[1:] >= tensor[:-1]).all().item()),
                f"CDF {key} is not monotone",
                errors,
            )
    probs = payload.get("quantile_probabilities")
    if torch.is_tensor(probs) and probs.numel() == 4097:
        require(float(probs[0]) == 0.0, "CDF probability start mismatch", errors)
        require(float(probs[-1]) == 1.0, "CDF probability end mismatch", errors)


def parse_npu_evidence(npu_text, mapping_text, training_pid, requested, errors):
    requested_int = int(requested)
    mapping_rows = []
    for line in mapping_text.splitlines():
        tokens = line.split()
        if len(tokens) >= 5 and all(token.isdigit() for token in tokens[:4]):
            mapping_rows.append(tuple(int(token) for token in tokens[:4]))
    require(
        (0, requested_int, requested_int, requested_int) in mapping_rows,
        "requested physical NPU is not exact in npu-smi mapping",
        errors,
    )

    current_kind = None
    process_rows = []
    hbm_rows = []
    for line in npu_text.splitlines():
        if line.startswith("snapshot_kind="):
            current_kind = line.split("=", 1)[1]
            continue
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) == 4:
            device_tokens = cells[0].split()
            if (
                len(device_tokens) == 2
                and all(token.isdigit() for token in device_tokens)
                and cells[1].isdigit()
            ):
                memory_match = re.search(r"(\d+)", cells[3])
                process_rows.append(
                    {
                        "kind": current_kind,
                        "npu": int(device_tokens[0]),
                        "chip": int(device_tokens[1]),
                        "pid": cells[1],
                        "name": cells[2],
                        "memory_mb": (
                            int(memory_match.group(1)) if memory_match else None
                        ),
                    }
                )
        elif len(cells) == 3:
            device_tokens = cells[0].split()
            if (
                len(device_tokens) == 2
                and all(token.isdigit() for token in device_tokens)
            ):
                hbm_match = re.search(r"(\d+)\s*/\s*(\d+)\s*$", cells[2])
                if hbm_match:
                    hbm_rows.append(
                        {
                            "kind": current_kind,
                            "chip": int(device_tokens[0]),
                            "physical": int(device_tokens[1]),
                            "used_mb": int(hbm_match.group(1)),
                            "total_mb": int(hbm_match.group(2)),
                        }
                    )

    own_rows = [row for row in process_rows if row["pid"] == training_pid]
    correct_rows = [row for row in own_rows if row["chip"] == requested_int]
    require(bool(own_rows), "training PID absent from exact npu-smi process rows", errors)
    require(
        bool(correct_rows),
        "training PID was not observed on requested physical NPU",
        errors,
    )
    require(
        all(row["chip"] == requested_int for row in own_rows),
        "training PID appeared on an unexpected NPU chip",
        errors,
    )
    memory_values = [
        row["memory_mb"] for row in correct_rows if row["memory_mb"] is not None
    ]
    require(bool(memory_values), "training process memory evidence missing", errors)
    if memory_values:
        require(max(memory_values) > 0, "training process memory must be positive", errors)

    requested_hbm = [
        row for row in hbm_rows
        if row["chip"] == requested_int and row["physical"] == requested_int
    ]
    baseline = [row["used_mb"] for row in requested_hbm if row["kind"] == "before_training"]
    running = [row["used_mb"] for row in requested_hbm if row["kind"] == "running"]
    require(bool(baseline), "pre-training HBM sample missing", errors)
    require(bool(running), "running HBM samples missing", errors)
    baseline_mb = baseline[0] if baseline else None
    peak_mb = max(running) if running else None
    if baseline_mb is not None and peak_mb is not None:
        require(peak_mb >= baseline_mb, "running HBM peak below baseline", errors)
    return {
        "mapping_rows": mapping_rows,
        "process_row_count": len(process_rows),
        "own_process_observations": len(own_rows),
        "own_process_peak_memory_mb": max(memory_values) if memory_values else None,
        "device_hbm_baseline_mb": baseline_mb,
        "device_hbm_peak_mb": peak_mb,
        "device_hbm_delta_mb": (
            peak_mb - baseline_mb
            if baseline_mb is not None and peak_mb is not None else None
        ),
        "process_names": sorted({row["name"] for row in correct_rows}),
    }


def parse_fresh_diagnostics(logger_text, variant, errors, warnings):
    pattern = re.compile(
        rf"^.*?Iters:\s*20/20\s*\|\|\s*Lr:\s*(?P<lr>{FLOAT_PATTERN})"
        rf"\s*\|\|\s*Task Loss:\s*(?P<task>{FLOAT_PATTERN})"
        rf"\s*\|\|\s*KD Loss:\s*(?P<kd>{FLOAT_PATTERN})"
        rf"\s*\|\|\s*Adv_G Loss:\s*(?P<adv_g>{FLOAT_PATTERN})"
        rf"\s*\|\|\s*Adv_D Loss:\s*(?P<adv_d>{FLOAT_PATTERN})"
        rf"\s*\|\|\s*skd_loss:\s*(?P<skd>{FLOAT_PATTERN})"
        rf"\s*\|\|\s*cwd_fea_loss:\s*(?P<cwd_fea>{FLOAT_PATTERN})"
        rf"\s*\|\|\s*cwd_logit_loss:\s*(?P<cwd_logit>{FLOAT_PATTERN})"
        rf"\s*\|\|\s*ifv_loss:\s*(?P<ifv>{FLOAT_PATTERN})"
        rf"\s*\|\|\s*at_loss:\s*(?P<at>{FLOAT_PATTERN})"
        rf"\s*\|\|\s*fitnet_loss:\s*(?P<fitnet>{FLOAT_PATTERN})"
        rf"\s*\|\|\s*psd_loss:\s*(?P<psd>{FLOAT_PATTERN})"
        rf"\s*\|\|\s*csd_loss:\s*(?P<csd>{FLOAT_PATTERN})"
        r"\s*\|\|\s*Cost Time:\s*(?P<cost>[^|]+?)"
        r"\s*\|\|\s*Estimated Time:\s*(?P<eta>[^|]+?)"
        r"\s*\|\|\s*O1\.2 variant:\s*(?P<variant>neutral|unreliable_only)"
        rf"\s*\|\|\s*O1\.2 branch KL mean:\s*(?P<kl>{FLOAT_PATTERN})"
        rf"\s*\|\|\s*O1\.2 cross-entropy mean:\s*(?P<ce>{FLOAT_PATTERN})"
        rf"\s*\|\|\s*O1\.2 teacher entropy mean:\s*(?P<teacher_entropy>{FLOAT_PATTERN})"
        rf"\s*\|\|\s*O1\.2 KD-only student-logit grad L2:\s*(?P<grad>{FLOAT_PATTERN})"
        r"\s*\|\|\s*O1\.2 valid pixels:\s*(?P<valid>\d+)"
        rf"\s*\|\|\s*Teacher output T:\s*(?P<teacher_output_t>{FLOAT_PATTERN})\s*$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(logger_text))
    require(logger_text.count("Iters:") == 1, "fresh logger must have one Iters line", errors)
    require(len(matches) == 1, "fresh logger full 20/20 line did not match", errors)
    if not matches:
        return None
    groups = matches[0].groupdict()
    require(groups["variant"] == variant, "logger variant mismatch", errors)
    numeric_keys = (
        "lr", "task", "kd", "adv_g", "adv_d", "skd", "cwd_fea",
        "cwd_logit", "ifv", "at", "fitnet", "psd", "csd", "kl",
        "ce", "teacher_entropy", "grad", "teacher_output_t",
    )
    values = {
        key: finite_float(groups[key], f"logged {key}", errors)
        for key in numeric_keys
    }
    values["valid_pixels"] = int(groups["valid"])
    values["cost_time"] = groups["cost"].strip()
    values["estimated_time"] = groups["eta"].strip()
    if values["lr"] is not None:
        require(abs(values["lr"] - 0.001349) <= 5e-7, "iteration-20 LR mismatch", errors)
    if values["teacher_output_t"] is not None:
        require(values["teacher_output_t"] == 3.0,
                "logged teacher output temperature mismatch", errors)
    for key in ("task", "kd", "cwd_fea", "cwd_logit", "kl", "ce", "teacher_entropy"):
        if values[key] is not None:
            require(values[key] >= 0.0, f"logged {key} must be nonnegative", errors)
    for key in ("skd", "ifv", "at", "fitnet", "psd", "csd"):
        if values[key] is not None:
            require(values[key] == 0.0, f"disabled loss {key} must be zero", errors)
    require(values["valid_pixels"] > 0, "valid pixel count must be positive", errors)
    if values["grad"] is not None and values["grad"] == 0.0:
        warnings.append("KD-only student-logit gradient printed as zero; finite gate still passes")
    if all(values[key] is not None for key in ("kl", "ce", "teacher_entropy")):
        closure = values["ce"] - values["teacher_entropy"]
        require(abs(values["kl"] - closure) <= 2e-5, "KL/CE/entropy closure failed", errors)
    values["gradient_nonzero_observed"] = (
        values["grad"] is not None and values["grad"] > 0.0
    )
    return values


def parse_runtime(logger_text, provenance, mode, errors):
    pattern = re.compile(
        rf"Total training time:\s*([^\n]+?)\s*\(({FLOAT_PATTERN})s / it\)"
    )
    matches = pattern.findall(logger_text)
    require(len(matches) == 1, "exactly one total training time record required", errors)
    seconds_per_iteration = None
    if matches:
        seconds_per_iteration = finite_float(matches[0][1], "seconds per iteration", errors)
        if seconds_per_iteration is not None:
            require(seconds_per_iteration > 0.0, "seconds per iteration must be positive", errors)
    start = parse_iso(provenance.get("start_utc"), "start_utc", errors)
    training_end = parse_iso(
        provenance.get("training_end_utc"), "training_end_utc", errors
    )
    wall_seconds = None
    if start is not None and training_end is not None:
        require(start <= training_end, "training timestamps out of order", errors)
        wall_seconds = (training_end - start).total_seconds()
        require(wall_seconds > 0.0, "training wall time must be positive", errors)
    return {
        "optimizer_steps": 20 if mode == "fresh" else 0,
        "reported_total_time": matches[0][0].strip() if matches else None,
        "seconds_per_iteration": seconds_per_iteration,
        "samples_per_second": (
            16.0 / seconds_per_iteration
            if mode == "fresh" and seconds_per_iteration else None
        ),
        "training_wall_seconds": wall_seconds,
    }


def validate_provenance(
    provenance, stage, mode, variant, save_dir, log_dir, runtime_dir,
    source_checkpoint, errors,
):
    paths = canonical_paths()
    expected = {
        "phase": "O1.2-B",
        "run_kind": "20_step_smoke",
        "variant": variant,
        "mode": mode,
        "world_size": "1",
        "rank": "0",
        "local_rank": "0",
        "seed": "1234",
        "python_path": str(PYTHON),
        "python_realpath": str(PYTHON.resolve()),
        "ascend_env_path": str(ASCEND_ENV),
        "teacher_sha256": EXPECTED_TEACHER_SHA256,
        "student_init_sha256": EXPECTED_STUDENT_INIT_SHA256,
        "cdf_sha256": EXPECTED_CDF_SHA256,
        "parameters_sha256": EXPECTED_PARAMETERS_SHA256,
        "gate_sha256": EXPECTED_GATE_SHA256,
        "o11_gate_sha256": EXPECTED_O11_GATE_SHA256,
        "train_list_sha256": EXPECTED_TRAIN_LIST_SHA256,
        "train_entry_sha256": EXPECTED_SOURCES["train_entry"],
        "rtc_o12_calibration_sha256": EXPECTED_SOURCES["rtc_o12_calibration"],
        "diagnose_rtc_o12_budget_sha256": EXPECTED_SOURCES["diagnose_rtc_o12_budget"],
        "check_rtc_o12_gate_sha256": EXPECTED_SOURCES["check_rtc_o12_gate"],
        "resume_source_sha256": (
            "" if mode == "fresh" else sha256(source_checkpoint)
        ),
        "save_dir": str(save_dir),
        "log_dir": str(log_dir),
        "runtime_dir": str(runtime_dir),
        "training_exit_status": "0",
    }
    for key, expected_value in expected.items():
        require(
            provenance.get(key) == expected_value,
            f"provenance mismatch: {key}",
            errors,
        )
    require(provenance.get("requested_physical_npu") in {"0", "1"},
            "invalid requested_physical_npu", errors)
    require(provenance.get("training_pid", "").isdigit(),
            "invalid training_pid", errors)
    require(provenance.get("runner_pid", "").isdigit(),
            "invalid runner_pid", errors)
    require(provenance.get("launcher_pid", "").isdigit(),
            "invalid launcher_pid", errors)
    require(Path(provenance.get("launcher_path", "")) == (
        ROOT / "scripts/experiments/kd_baselines_npu/run_phaseO_o12_smoke_variant.sh"
    ).resolve(), "launcher path mismatch", errors)
    require(Path(provenance.get("checker_path", "")) == Path(__file__).resolve(),
            "checker path mismatch", errors)
    if Path(provenance.get("launcher_path", "")).is_file():
        require(sha256(provenance["launcher_path"]) == provenance.get("launcher_sha256"),
                "launcher actual SHA mismatch", errors)
    require(sha256(Path(__file__).resolve()) == provenance.get("checker_sha256"),
            "checker actual SHA mismatch", errors)
    require(sha256(PYTHON.resolve()) == provenance.get("python_sha256"),
            "Python binary SHA mismatch", errors)
    require(sha256(ASCEND_ENV) == provenance.get("ascend_env_sha256"),
            "Ascend env SHA mismatch", errors)
    for label, path, expected_sha in (
        ("teacher", paths["teacher"], EXPECTED_TEACHER_SHA256),
        ("student init", paths["student_init"], EXPECTED_STUDENT_INIT_SHA256),
        ("CDF", paths["cdf"], EXPECTED_CDF_SHA256),
        ("parameters", paths["parameters"], EXPECTED_PARAMETERS_SHA256),
        ("gate", paths["gate"], EXPECTED_GATE_SHA256),
        ("O1.1 gate", paths["o11_gate"], EXPECTED_O11_GATE_SHA256),
        ("train list", paths["train_list"], EXPECTED_TRAIN_LIST_SHA256),
        ("train entry", paths["train_entry"], EXPECTED_SOURCES["train_entry"]),
        ("O1.2 module", paths["o12_module"], EXPECTED_SOURCES["rtc_o12_calibration"]),
        ("O1.2 diagnose", paths["o12_diagnose"], EXPECTED_SOURCES["diagnose_rtc_o12_budget"]),
        ("O1.2 gate checker", paths["o12_gate_checker"], EXPECTED_SOURCES["check_rtc_o12_gate"]),
    ):
        require(path.is_file(), f"{label} missing", errors)
        if path.is_file():
            require(sha256(path) == expected_sha, f"{label} SHA drift", errors)

    try:
        runtime_env = json.loads(provenance.get("runtime_env_json", ""))
    except json.JSONDecodeError as error:
        errors.append(f"runtime_env_json invalid: {error}")
        runtime_env = {}
    require(runtime_env.get("python_executable") == str(PYTHON),
            "runtime Python executable mismatch", errors)
    require(runtime_env.get("python_version") == "3.11.10",
            "runtime Python version mismatch", errors)
    require(runtime_env.get("torch_version") == "2.8.0+cpu",
            "runtime torch version mismatch", errors)
    require(runtime_env.get("torch_npu_version") == "2.8.0.post2",
            "runtime torch_npu version mismatch", errors)
    require(runtime_env.get("npu_available") is True,
            "runtime NPU unavailable", errors)
    require(isinstance(runtime_env.get("npu_device_count"), int)
            and runtime_env["npu_device_count"] >= 1,
            "runtime NPU device count invalid", errors)

    try:
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        errors.append(f"cannot read current git commit: {error}")
        current_commit = None
    if current_commit is not None:
        require(provenance.get("git_commit") == current_commit,
                "git commit changed during run", errors)

    require("--stage prefinal" in provenance.get("checker_argv_shell", ""),
            "prefinal checker argv missing", errors)
    require("--stage final" in provenance.get("final_checker_argv_shell", ""),
            "final checker argv missing", errors)
    if stage == "final":
        require(provenance.get("checker_exit_status") == "0",
                "prefinal checker status is not zero", errors)
        require(provenance.get("exit_status") == "0",
                "sealed exit status is not zero", errors)
        require(provenance.get("exit_reason") == "completed_and_checked",
                "sealed exit reason mismatch", errors)
        start = parse_iso(provenance.get("start_utc"), "start_utc", errors)
        train_end = parse_iso(
            provenance.get("training_end_utc"), "training_end_utc", errors
        )
        checker_end = parse_iso(
            provenance.get("prefinal_checker_end_utc"),
            "prefinal_checker_end_utc", errors,
        )
        end = parse_iso(provenance.get("end_utc"), "end_utc", errors)
        if all(value is not None for value in (start, train_end, checker_end, end)):
            require(start <= train_end <= checker_end <= end,
                    "sealed timestamps out of order", errors)
    return runtime_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prefinal", "final"), required=True)
    parser.add_argument("--mode", choices=("fresh", "resume_audit"), required=True)
    parser.add_argument(
        "--variant", choices=("neutral", "unreliable_only"), required=True
    )
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--source-checkpoint")
    args = parser.parse_args()

    save_dir = Path(args.save_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    runtime_dir = Path(args.runtime_dir).resolve()
    source_checkpoint = (
        Path(args.source_checkpoint).resolve() if args.source_checkpoint else None
    )
    output_path = runtime_dir / (
        "acceptance.prefinal.json" if args.stage == "prefinal" else "acceptance.json"
    )
    state_path = save_dir / "training_state_latest.pth"
    model_path = save_dir / MODEL_NAME
    logger_path = log_dir / LOGGER_NAME
    console_path = runtime_dir / "console.log"
    provenance_path = runtime_dir / "provenance.txt"
    npu_path = runtime_dir / "npu_smi.log"
    mapping_path = runtime_dir / "npu_mapping.txt"
    argv_path = runtime_dir / "argv.nul"
    git_status_path = runtime_dir / "git_status.txt"
    training_pid_path = runtime_dir / "training.pid"
    checker_log_path = runtime_dir / "checker.log"
    prefinal_path = runtime_dir / "acceptance.prefinal.json"
    errors = []
    warnings = []

    required = [
        ("training state", state_path),
        ("student weights", model_path),
        ("logger log", logger_path),
        ("console log", console_path),
        ("provenance", provenance_path),
        ("NPU snapshots", npu_path),
        ("NPU mapping", mapping_path),
        ("argv.nul", argv_path),
        ("git status", git_status_path),
        ("training PID", training_pid_path),
    ]
    if args.stage == "final":
        required.extend([
            ("prefinal acceptance", prefinal_path),
            ("prefinal checker log", checker_log_path),
        ])
    for label, path in required:
        require(path.is_file(), f"{label} missing: {path}", errors)
    require(torch_npu is not None, "torch_npu import failed", errors)
    require(
        (args.mode == "fresh" and source_checkpoint is None)
        or (
            args.mode == "resume_audit"
            and source_checkpoint is not None
            and source_checkpoint.is_file()
        ),
        "source checkpoint mode contract mismatch",
        errors,
    )
    if errors:
        output_path.write_text(
            json.dumps({
                "schema_version": 2,
                "phase": "O1.2-B",
                "stage": args.stage,
                "mode": args.mode,
                "variant": args.variant,
                "pass": False,
                "errors": errors,
                "warnings": warnings,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1

    logger_text = logger_path.read_text(encoding="utf-8", errors="replace")
    console_text = console_path.read_text(encoding="utf-8", errors="replace")
    npu_text = npu_path.read_text(encoding="utf-8", errors="replace")
    mapping_text = mapping_path.read_text(encoding="utf-8", errors="replace")
    combined_text = logger_text + "\n" + console_text
    provenance = parse_provenance(provenance_path, errors)
    state = load_checkpoint(state_path)
    legacy_state = load_checkpoint(model_path)

    runtime_env = validate_provenance(
        provenance, args.stage, args.mode, args.variant, save_dir, log_dir,
        runtime_dir, source_checkpoint, errors,
    )
    require(training_pid_path.read_text(encoding="utf-8").strip()
            == provenance.get("training_pid"),
            "training.pid does not match provenance", errors)

    argv = read_argv_nul(argv_path, errors)
    expected_argv = expected_training_argv(
        args.mode, args.variant, save_dir, log_dir, source_checkpoint
    )
    require(argv == expected_argv, "training argv.nul mismatch", errors)

    git_status_lines = {
        line for line in git_status_path.read_text(encoding="utf-8").splitlines()
        if line
    }
    require(git_status_lines == EXPECTED_DIRTY_STATUS,
            "git dirty status differs from four quarantined old launchers", errors)
    require(provenance.get("git_dirty_count") == str(len(git_status_lines)),
            "git_dirty_count mismatch", errors)

    paths = canonical_paths()
    validate_cdf(paths["cdf"], errors)

    require(isinstance(state, dict), "checkpoint root is not a dict", errors)
    expected_checkpoint_keys = {
        "checkpoint_type", "checkpoint_version", "student", "criterion_cwd",
        "criterion_fitnet", "D", "optimizer", "D_optimizer", "iteration",
        "best_pred", "rng_state", "rng_state_by_rank", "world_size", "args",
        "rtc", "rtc_o12", "rtc_o12_data_order",
    }
    if isinstance(state, dict):
        require(set(state) == expected_checkpoint_keys,
                "checkpoint top-level keyset mismatch", errors)
        require(state.get("checkpoint_type") == "train_kd_training_state",
                "checkpoint_type mismatch", errors)
        require(state.get("checkpoint_version") == 4,
                "checkpoint_version must be 4", errors)
        require(state.get("iteration") == 20,
                "checkpoint iteration must be 20", errors)
        require(state.get("world_size") == 1,
                "checkpoint world_size must be 1", errors)
        require(state.get("best_pred") == 0.0,
                "skip-val best_pred must remain zero", errors)
        require(state.get("rtc") is None, "legacy RTC metadata must be None", errors)
        require(isinstance(state.get("rng_state_by_rank"), list)
                and len(state["rng_state_by_rank"]) == 1,
                "rng_state_by_rank must contain one rank", errors)

        expected_args = expected_saved_args(
            args.mode, args.variant, save_dir, log_dir, source_checkpoint
        )
        require_exact_mapping(state.get("args"), expected_args,
                              "checkpoint.args", errors)
        require_exact_mapping(state.get("rtc_o12"), expected_rtc_o12(args.variant),
                              "checkpoint.rtc_o12", errors)
        expected_order_state = {
            "contract": EXPECTED_ORDER,
            "completed_iteration": 20,
            "next_global_iteration": None,
            "next_canonical_index_offset": 320,
        }
        require_exact_mapping(
            state.get("rtc_o12_data_order"), expected_order_state,
            "checkpoint.rtc_o12_data_order", errors,
        )
        try:
            compare_nested(legacy_state, state.get("student"), "legacy_student")
        except AssertionError as error:
            errors.append(str(error))

    finite_counters = {
        "tensors": 0,
        "tensor_elements": 0,
        "arrays": 0,
        "array_elements": 0,
        "float_scalars": 0,
    }
    if isinstance(state, dict):
        for key in (
            "student", "criterion_cwd", "criterion_fitnet", "D",
            "optimizer", "D_optimizer",
        ):
            check_finite_nested(
                state.get(key), f"checkpoint.{key}", errors, finite_counters
            )
    check_finite_nested(legacy_state, "legacy_student", errors, finite_counters)

    require("Traceback (most recent call last)" not in combined_text,
            "traceback found in logs", errors)
    require("Overall validation pixAcc" not in combined_text,
            "20-step smoke unexpectedly ran validation", errors)
    require("Using 1 process(es) on device npu" in logger_text,
            "single-process NPU logger marker missing", errors)
    require("Loaded O1.2 frozen inputs" in logger_text,
            "frozen O1.2 input marker missing", errors)

    diagnostics = None
    if args.mode == "fresh":
        require("Resumed full training state" not in logger_text,
                "fresh run unexpectedly resumed", errors)
        diagnostics = parse_fresh_diagnostics(
            logger_text, args.variant, errors, warnings
        )
    else:
        require("Iters:" not in logger_text,
                "resume audit executed an optimizer step", errors)
        require(
            "Resumed full training state" in logger_text
            and "iteration=20" in logger_text,
            "resume audit did not load iteration 20", errors,
        )
        require("Continuing training from iteration 20" in logger_text,
                "resume continuation marker missing", errors)
        source_state = load_checkpoint(source_checkpoint)
        require(sha256(source_checkpoint) == provenance.get("resume_source_sha256"),
                "resume source SHA is not bound", errors)
        for key in (
            "student", "criterion_cwd", "criterion_fitnet", "D",
            "optimizer", "D_optimizer", "rtc_o12", "rtc_o12_data_order",
        ):
            try:
                compare_nested(source_state.get(key), state.get(key), f"resume.{key}")
            except AssertionError as error:
                errors.append(str(error))

    runtime_metrics = parse_runtime(logger_text, provenance, args.mode, errors)
    npu_evidence = parse_npu_evidence(
        npu_text, mapping_text, provenance.get("training_pid", ""),
        provenance.get("requested_physical_npu", "-1"), errors,
    )

    prefinal_sha = None
    checker_log_sha = None
    if args.stage == "final":
        try:
            prefinal = json.loads(prefinal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"prefinal acceptance invalid: {error}")
            prefinal = {}
        require(prefinal.get("pass") is True,
                "prefinal acceptance did not pass", errors)
        require(prefinal.get("stage") == "prefinal",
                "prefinal acceptance stage mismatch", errors)
        require(prefinal.get("mode") == args.mode
                and prefinal.get("variant") == args.variant,
                "prefinal acceptance identity mismatch", errors)
        checker_log_text = checker_log_path.read_text(
            encoding="utf-8", errors="replace"
        )
        require("FAIL:" not in checker_log_text,
                "prefinal checker log contains failure", errors)
        prefinal_sha = sha256(prefinal_path)
        checker_log_sha = sha256(checker_log_path)

    artifact_hashes = {
        "teacher": EXPECTED_TEACHER_SHA256,
        "student_init": EXPECTED_STUDENT_INIT_SHA256,
        "cdf": EXPECTED_CDF_SHA256,
        "parameters": EXPECTED_PARAMETERS_SHA256,
        "gate": EXPECTED_GATE_SHA256,
        "o11_gate": EXPECTED_O11_GATE_SHA256,
        "train_list": EXPECTED_TRAIN_LIST_SHA256,
        "sources": EXPECTED_SOURCES,
    }
    files = {
        "checkpoint": sha256(state_path),
        "student_weights": sha256(model_path),
        "logger": sha256(logger_path),
        "console": sha256(console_path),
        "npu_smi": sha256(npu_path),
        "npu_mapping": sha256(mapping_path),
        "provenance": sha256(provenance_path),
        "argv_nul": sha256(argv_path),
        "git_status": sha256(git_status_path),
        "prefinal_acceptance": prefinal_sha,
        "prefinal_checker_log": checker_log_sha,
    }
    manifest = {
        "schema_version": 2,
        "phase": "O1.2-B",
        "stage": args.stage,
        "mode": args.mode,
        "variant": args.variant,
        "pass": not errors,
        "errors": errors,
        "warnings": warnings,
        "checkpoint": {
            "path": str(state_path),
            "sha256": files["checkpoint"],
            "iteration": state.get("iteration") if isinstance(state, dict) else None,
            "version": state.get("checkpoint_version") if isinstance(state, dict) else None,
            "recipe_fingerprint": json_fingerprint(
                state.get("args", {}) if isinstance(state, dict) else {}
            ),
            "sample_order_sha256": (
                state.get("rtc_o12_data_order", {}).get("contract", {}).get(
                    "complete_order_sha256"
                ) if isinstance(state, dict) else None
            ),
            "finite_scan": finite_counters,
        },
        "diagnostics": diagnostics,
        "runtime": {
            **runtime_metrics,
            "requested_physical_npu": provenance.get("requested_physical_npu"),
            "training_pid": provenance.get("training_pid"),
            "npu_evidence": npu_evidence,
        },
        "environment": runtime_env,
        "artifact_hashes": artifact_hashes,
        "resume_source_sha256": provenance.get("resume_source_sha256"),
        "files_sha256": files,
        "launcher_sha256": provenance.get("launcher_sha256"),
        "checker_sha256": provenance.get("checker_sha256"),
        "git_commit": provenance.get("git_commit"),
        "git_dirty_status": sorted(git_status_lines),
    }
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(
        f"PASS: O1.2-B {args.variant} {args.mode} {args.stage} "
        f"checkpoint={files['checkpoint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
