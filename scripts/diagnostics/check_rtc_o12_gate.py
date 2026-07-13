#!/usr/bin/env python3
"""Independent non-synonymous joint gate for Phase O1.2-A."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / 'runs' / 'diagnostics' / 'phaseO_o12'
EXPECTED_PHASE = 'O1.2'
GATE_PROFILE = 'o12_budgeted_teacher_target_non_synonymous_v1'
EXPECTED_CDF_PATH = (
    ROOT / 'runs' / 'diagnostics' / 'phaseO_o11'
    / 'voc_train_rtc_confidence_cdf.pt'
).resolve()
EXPECTED_CDF_SHA256 = (
    '8ba8376a03c2f93835339d98670fa357b381b23a22cbf2a7fc48b0c2441d7e69'
)
EXPECTED_O11_GATE_PATH = (
    ROOT / 'runs' / 'diagnostics' / 'phaseO_o11'
    / 'o11_confidence_gate.json'
).resolve()
EXPECTED_O11_REPORT_PATH = {
    split: (
        ROOT / 'runs' / 'diagnostics' / 'phaseO_o11'
        / f'rtc_confidence_routing_{split}.json'
    ).resolve()
    for split in ('train', 'val')
}
EXPECTED_O11_REPORT_SHA256 = {
    'train': '312c1a1d309df0406402b07fbb94401d265ccc89dd062314004ac8fdf191c116',
    'val': '1decff5e04b585a2cec3172213a55570ea5bc555eac3a9e00a8802b077049863',
}
EXPECTED_TEACHER_SHA256 = (
    'ac49b2c7720b21d565072e974e4404fcb009ba106288d95d1a4bb25f09c3fe58'
)
EXPECTED_TEACHER_PATH = (
    ROOT / 'data' / 'winycg' / 'cirkd' / 'teachers'
    / 'deeplabv3_resnet101_voc_best_model.pth'
).resolve()
EXPECTED_LIST_PATH = {
    'train': (ROOT / 'dataset' / 'list' / 'voc' / 'train_aug.txt').resolve(),
    'val': (ROOT / 'dataset' / 'list' / 'voc' / 'val.txt').resolve(),
}
EXPECTED_LIST_SHA256 = {
    'train': 'd1326bd532648d73bc4b1bd275434eba81930982dc7401a65a8cecb26c028e24',
    'val': 'cdc1326d12f69ce5153aa5da04a4d8783e146868d42d19f82f44e74b97ac907d',
}
EXPECTED_DATASET_SIZE = {'train': 10582, 'val': 1449}
EXPECTED_NATIVE_VALID = {'train': 32246990, 'val': 3878674}
FROZEN_O11_SOURCE_SHA256 = {
    'rtc_temperature': '01b7b6e6aa0d513561332510347b52ea9411330dfb0f2da54abdc36f2375fe59',
    'build_rtc_cdf': 'c88bdfcb885cde01cbf437e2e7751c8eab510067aacf530b469df808ae6604dd',
    'diagnose_rtc_routing': 'f838f25b70b8eadfc982873057e5fb68c54c81089a32e9121fd664e02916f9ef',
    'check_rtc_o11_gate': '55553ec523ac2c2a979470b8542f31878462bbe5a919e0c52132edfbba4eb256',
}
O11_SOURCE_PATHS = {
    'rtc_temperature': ROOT / 'utils' / 'rtc_temperature.py',
    'build_rtc_cdf': ROOT / 'scripts' / 'diagnostics' / 'build_rtc_cdf.py',
    'diagnose_rtc_routing': ROOT / 'scripts' / 'diagnostics' / 'diagnose_rtc_routing.py',
    'check_rtc_o11_gate': ROOT / 'scripts' / 'diagnostics' / 'check_rtc_o11_gate.py',
}
O12_SOURCE_PATHS = {
    'rtc_o12_calibration': ROOT / 'utils' / 'rtc_o12_calibration.py',
    'diagnose_rtc_o12_budget': ROOT / 'scripts' / 'diagnostics' / 'diagnose_rtc_o12_budget.py',
    'check_rtc_o12_gate': Path(__file__).resolve(),
    'train_entry': ROOT / 'train_kd.py',
}
EXPECTED_CONFIG = {
    'phase': EXPECTED_PHASE,
    'reliability_mode': 'confidence',
    'reliability_definition_id': 'neg_log_top1_confidence_v1',
    'assess_temperature': 1.0,
    'epsilon': 1e-8,
    'q_reliable': 0.6,
    'q_unreliable': 0.8,
    'p_reliable': 1.0,
    'p_unreliable': 2.0,
    'a': 0.10536051565782628,
    'b_min': 0.0,
    'b_max': 0.4054651081081644,
    'target_arithmetic_mean': 0.995,
    'minimum_harmonic_mean': 0.98,
    'bisection_iterations': 64,
    'teacher_output_temperature': 3.0,
    'temperature_map_dtype': 'float32',
    'budget_accumulator_dtype': 'float64',
    'temperature_minimum': 0.9,
    'temperature_maximum': 1.5,
    'tau_temperature': 1e-6,
    'tau_probability': 1e-6,
    'tau_entropy': 1e-6,
    'tau_student': 1e-7,
    'tau_formula_monotonic': 1e-12,
}
CANONICAL_PATHS = {
    'parameters': DEFAULT_DIR / 'o12_budget_parameters.json',
    'train': DEFAULT_DIR / 'o12_budget_train.json',
    'val': DEFAULT_DIR / 'o12_budget_val.json',
    'output': DEFAULT_DIR / 'o12_joint_gate.json',
}
VIOLATION_KEYS = (
    'nonfinite',
    'reliable_temperature_above_one',
    'neutral_temperature_nonunit',
    'unreliable_temperature_below_one',
    'formula_monotonic',
    'reliable_confidence_decrease',
    'unreliable_confidence_increase',
    'reliable_entropy_increase',
    'unreliable_entropy_decrease',
    'teacher_argmax_mismatch',
    'temperature_out_of_bounds',
    'neutral_target_mismatch',
    'student_softmax_changed',
    'population_closure',
    'risk_bin_closure',
    'error_count_closure',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate formal O1.2 train/val budget diagnostics.'
    )
    parser.add_argument('--parameters', default=str(CANONICAL_PATHS['parameters']))
    parser.add_argument('--train-json', default=str(CANONICAL_PATHS['train']))
    parser.add_argument('--val-json', default=str(CANONICAL_PATHS['val']))
    parser.add_argument('--output', default=str(CANONICAL_PATHS['output']))
    parser.add_argument('--strict', action='store_true', default=False)
    return parser.parse_args()


def expected_scan_protocol(stage, split):
    return {
        'stage': stage,
        'split': split,
        'teacher_model': 'deeplabv3',
        'teacher_backbone': 'resnet101',
        'num_classes': 21,
        'ignore_label': -1,
        'crop_size': [512, 512],
        'scale': split == 'train',
        'mirror': split == 'train',
        'batch_size': 4 if split == 'train' else 1,
        'workers': 0,
        'augmentation_seed': 2025,
        'max_images': 0,
        'device_type': 'npu',
        'world_size': 1,
        'rank': 0,
        'process_mode': 'single_process_single_npu',
        'teacher_output_grid': 'native_logits',
        'valid_mask_resize': 'nearest',
        'cdf_query_dtype': 'float32',
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f'JSON root must be an object: {path}')
    return payload


def load_o11_reference(split):
    path = EXPECTED_O11_REPORT_PATH[split]
    if not path.is_file():
        raise FileNotFoundError(f'missing frozen O1.1 {split} report: {path}')
    actual_sha = file_sha256(path)
    if actual_sha != EXPECTED_O11_REPORT_SHA256[split]:
        raise ValueError(f'frozen O1.1 {split} report SHA mismatch')
    return load_json(path)


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), allow_nan=False
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def numeric_equal(left: Any, right: Any, atol: float = 1e-9) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and math.isfinite(float(left))
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)
    )


def parse_utc_timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        return None
    return parsed

def runtime_identity_matches_requested_npu(identity):
    if not isinstance(identity, dict):
        return False
    requested_index = identity.get('requested_npu_index')
    current_index = identity.get('npu_current_device')
    return (
        identity.get('device_type') == 'npu'
        and isinstance(requested_index, int)
        and not isinstance(requested_index, bool)
        and requested_index >= 0
        and identity.get('requested_device') == f'npu:{requested_index}'
        and current_index == requested_index
    )


def runtime_identity_is_single_process_npu(identity):
    if not runtime_identity_matches_requested_npu(identity):
        return False
    return (
        identity.get('world_size') == 1
        and identity.get('rank') == 0
        and identity.get('local_rank') == 0
        and isinstance(identity.get('process_id'), int)
        and not isinstance(identity.get('process_id'), bool)
        and identity.get('process_id') > 0
        and isinstance(identity.get('npu_device_name'), str)
        and bool(identity.get('npu_device_name').strip())
    )



def runtime_provenance_checks(payload):
    started = parse_utc_timestamp(payload.get('started_at_utc'))
    ended = parse_utc_timestamp(payload.get('ended_at_utc'))
    elapsed = payload.get('elapsed_seconds')
    elapsed_valid = (
        isinstance(elapsed, (int, float))
        and not isinstance(elapsed, bool)
        and math.isfinite(float(elapsed))
        and float(elapsed) > 0.0
    )
    wall_elapsed = (
        (ended - started).total_seconds()
        if started is not None and ended is not None else None
    )
    identity = payload.get('runtime_identity')
    identity = identity if isinstance(identity, dict) else {}
    identity_keys = {
        'process_id', 'world_size', 'rank', 'local_rank',
        'requested_device', 'requested_npu_index', 'resolved_device', 'device_type',
        'npu_current_device', 'npu_device_name',
        'ascend_rt_visible_devices',
    }
    process_id = identity.get('process_id')
    visible = identity.get('ascend_rt_visible_devices')
    return {
        'runtime_timestamps_are_utc': started is not None and ended is not None,
        'runtime_end_not_before_start': (
            wall_elapsed is not None and wall_elapsed >= 0.0
        ),
        'runtime_elapsed_positive_finite': elapsed_valid,
        'runtime_elapsed_matches_wall_clock': (
            elapsed_valid and wall_elapsed is not None
            and abs(float(elapsed) - wall_elapsed)
            <= max(5.0, .05 * max(float(elapsed), wall_elapsed))
        ),
        'runtime_identity_keys_exact': set(identity) == identity_keys,
        'runtime_process_id_positive': (
            isinstance(process_id, int) and not isinstance(process_id, bool)
            and process_id > 0
        ),
        'runtime_single_process_rank_zero': (
            identity.get('world_size') == 1
            and identity.get('rank') == 0
            and identity.get('local_rank') == 0
        ),
        'runtime_requested_npu_index_matches_current': (
            runtime_identity_matches_requested_npu(identity)
        ),
        'runtime_requested_device_is_explicit_npu': (
            isinstance(identity.get('requested_npu_index'), int)
            and not isinstance(identity.get('requested_npu_index'), bool)
            and identity.get('requested_npu_index') >= 0
            and identity.get('requested_device')
            == f"npu:{identity.get('requested_npu_index')}"
        ),
        'runtime_npu_name_nonempty': (
            isinstance(identity.get('npu_device_name'), str)
            and bool(identity['npu_device_name'].strip())
        ),
        'runtime_visible_devices_type_valid': (
            visible is None or isinstance(visible, str)
        ),
    }

def all_true(mapping: Any) -> bool:
    return isinstance(mapping, dict) and bool(mapping) and all(
        value is True for value in mapping.values()
    )


def joint_runtime_device_provenance(parameters, train, val):
    artifacts = {
        'solve': parameters,
        'train': train,
        'val': val,
    }
    identities = {
        name: payload.get('runtime_identity', {})
        if isinstance(payload, dict) else {}
        for name, payload in artifacts.items()
    }
    valid = {
        name: runtime_identity_is_single_process_npu(identity)
        for name, identity in identities.items()
    }
    assignments = {
        name: {
            'process_id': identity.get('process_id'),
            'requested_npu_index': identity.get('requested_npu_index'),
            'npu_current_device': identity.get('npu_current_device'),
            'npu_device_name': identity.get('npu_device_name'),
        }
        for name, identity in identities.items()
    }
    actual_indices = [
        row['npu_current_device'] for row in assignments.values()
        if isinstance(row['npu_current_device'], int)
        and not isinstance(row['npu_current_device'], bool)
    ]
    return {
        'assignments': assignments,
        'per_artifact_valid': valid,
        'all_artifacts_single_process_single_npu': all(valid.values()),
        'same_actual_npu': len(actual_indices) == 3 and len(set(actual_indices)) == 1,
        'same_actual_npu_is_gate': False,
        'semantics': 'device index is execution provenance, not a statistical definition',
    }

def source_sha256() -> dict[str, str]:
    result = {}
    for name, path in O12_SOURCE_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f'missing O1.2 source: {path}')
        result[name] = file_sha256(path)
    return result


def frozen_o11_source_checks() -> dict[str, bool]:
    return {
        f'o11_source_{name}_byte_exact': (
            path.is_file() and file_sha256(path) == FROZEN_O11_SOURCE_SHA256[name]
        )
        for name, path in O11_SOURCE_PATHS.items()
    }


def _histogram_add(target: dict[float, int], values: np.ndarray) -> None:
    unique, counts = np.unique(
        values.astype(np.float32, copy=False), return_counts=True
    )
    for value, count in zip(unique.tolist(), counts.tolist()):
        key = float(np.float32(value))
        target[key] = target.get(key, 0) + int(count)


def histogram_from_cache(path: Path, expected_sha: str | None = None):
    checks = {
        'risk_cache_exists': path.is_file(),
        'risk_cache_sha_matches_report': False,
        'risk_cache_numpy_loadable': False,
        'risk_cache_dtype_float32': False,
        'risk_cache_one_dimensional': False,
        'risk_cache_all_finite': False,
        'risk_cache_in_unit_interval': False,
    }
    if not path.is_file():
        return np.empty(0, np.float32), np.empty(0, np.int64), checks, None, 0
    actual_sha = file_sha256(path)
    checks['risk_cache_sha_matches_report'] = actual_sha == expected_sha
    try:
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        checks['risk_cache_numpy_loadable'] = True
    except Exception:
        return np.empty(0, np.float32), np.empty(0, np.int64), checks, actual_sha, 0
    checks['risk_cache_dtype_float32'] = values.dtype == np.float32
    checks['risk_cache_one_dimensional'] = values.ndim == 1
    if values.ndim != 1:
        return np.empty(0, np.float32), np.empty(0, np.int64), checks, actual_sha, 0
    histogram: dict[float, int] = {}
    finite = True
    in_range = True
    for start in range(0, int(values.size), 1_048_576):
        chunk = np.asarray(values[start:start + 1_048_576])
        finite = finite and bool(np.isfinite(chunk).all())
        in_range = in_range and bool(((chunk >= 0.0) & (chunk <= 1.0)).all())
        if finite and in_range:
            _histogram_add(histogram, chunk)
    checks['risk_cache_all_finite'] = finite
    checks['risk_cache_in_unit_interval'] = in_range
    if not (finite and in_range):
        return np.empty(0, np.float32), np.empty(0, np.int64), checks, actual_sha, int(values.size)
    ordered = sorted(histogram.items())
    unique = np.asarray([item[0] for item in ordered], dtype=np.float32)
    counts = np.asarray([item[1] for item in ordered], dtype=np.int64)
    return unique, counts, checks, actual_sha, int(values.size)


def gates_from_unique(unique_u: np.ndarray):
    u = unique_u.astype(np.float32, copy=False)
    g_r = np.clip((np.float32(.6) - u) / np.float32(.6), 0, 1).astype(np.float32)
    base_u = np.clip((u - np.float32(.8)) / np.float32(.2), 0, 1).astype(np.float32)
    return g_r, np.square(base_u).astype(np.float32)


def temperature_from_unique(unique_u: np.ndarray, a: float, b: float):
    g_r, g_u = gates_from_unique(unique_u)
    ell = (-np.float32(a) * g_r + np.float32(b) * g_u).astype(np.float32)
    return np.exp(ell).astype(np.float32)


def weighted_mean(values: np.ndarray, counts: np.ndarray) -> float | None:
    total = int(counts.sum())
    if total <= 0:
        return None
    return float(
        np.dot(values.astype(np.float64), counts.astype(np.float64)) / total
    )


def weighted_quantile(values, counts, quantile):
    total = int(counts.sum())
    if total <= 0:
        return None
    order = np.argsort(values, kind='stable')
    values = values[order].astype(np.float64)
    cumulative = np.cumsum(counts[order].astype(np.int64))
    position = (total - 1) * float(quantile)
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    lo = int(np.searchsorted(cumulative, lower + 1, side='left'))
    hi = int(np.searchsorted(cumulative, upper + 1, side='left'))
    weight = position - lower
    return float((1.0 - weight) * values[lo] + weight * values[hi])


def temperature_statistics(unique_u, counts, a, b, scale=1.0):
    temperature = temperature_from_unique(unique_u, a, b)
    temperature = (
        temperature * np.float32(scale)
    ).astype(np.float32, copy=False)
    total = int(counts.sum())
    inverse_mean = weighted_mean(1.0 / temperature.astype(np.float64), counts)
    result = {
        'count': total,
        'min': float(temperature.min()) if total else None,
        'max': float(temperature.max()) if total else None,
        'mean': weighted_mean(temperature, counts),
        'harmonic_mean': 1.0 / inverse_mean if inverse_mean else None,
    }
    for name, q in (
        ('q01', .01), ('q10', .10), ('q50', .50), ('q80', .80),
        ('q90', .90), ('q95', .95), ('q99', .99),
    ):
        result[name] = weighted_quantile(temperature, counts, q)
    top = unique_u >= np.float32(.9)
    result['top_risk_decile_mean'] = weighted_mean(temperature[top], counts[top])
    result['less_than_0p9_count'] = int(counts[temperature < np.float32(.9)].sum())
    result['equal_1_count'] = int(counts[temperature == np.float32(1.0)].sum())
    result['greater_than_1_count'] = int(counts[temperature > np.float32(1.0)].sum())
    result['greater_than_1p25_count'] = int(counts[temperature > np.float32(1.25)].sum())
    result['greater_than_1p5_count'] = int(counts[temperature > np.float32(1.5)].sum())
    for key in (
        'less_than_0p9_count', 'equal_1_count', 'greater_than_1_count',
        'greater_than_1p25_count', 'greater_than_1p5_count',
    ):
        result[key.replace('_count', '_coverage')] = (
            result[key] / total if total else None
        )
    return result


def solve_budget(unique_u, counts):
    a = float(EXPECTED_CONFIG['a'])
    left, right = 0.0, float(EXPECTED_CONFIG['b_max'])
    target = float(EXPECTED_CONFIG['target_arithmetic_mean'])

    def mean_at(b):
        return weighted_mean(temperature_from_unique(unique_u, a, b), counts)

    mean_left, mean_right = mean_at(left), mean_at(right)
    feasible = (
        mean_left is not None and mean_right is not None
        and mean_left <= target <= mean_right
    )
    if not feasible:
        return {
            'feasible': False, 'a': a, 'b': None, 'left': left,
            'right': right, 'mean_left': mean_left, 'mean_right': mean_right,
        }
    for _ in range(64):
        mid = (left + right) / 2.0
        if mean_at(mid) < target:
            left = mid
        else:
            right = mid
    b = (left + right) / 2.0
    stats = temperature_statistics(unique_u, counts, a, b)
    return {
        'feasible': True, 'a': a, 'b': b, 'left': left, 'right': right,
        'iterations': 64, 'mean_left': mean_left, 'mean_right': mean_right,
        'mean': stats['mean'], 'harmonic_mean': stats['harmonic_mean'],
        'mean_residual': stats['mean'] - target,
        'theoretical_high_risk_endpoint': math.exp(b),
    }


def solve_budget_exact_cache(path: Path):
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    u = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
    if u.numel() == 0:
        return {'feasible': False}
    g_r = ((.6 - u) / .6).clamp(0.0, 1.0).pow(1.0)
    g_u = ((u - .8) / .2).clamp(0.0, 1.0).pow(2.0)
    if not bool((g_u > 0).any().item()):
        return {'feasible': False}
    a = float(EXPECTED_CONFIG['a'])
    target = float(EXPECTED_CONFIG['target_arithmetic_mean'])

    def moments(b):
        temperature = torch.exp(-a * g_r + float(b) * g_u)
        values64 = temperature.to(dtype=torch.float64)
        count = int(values64.numel())
        arithmetic = float(values64.sum(dtype=torch.float64).item() / count)
        harmonic = float(
            count / values64.reciprocal().sum(dtype=torch.float64).item()
        )
        return arithmetic, harmonic

    left, right = 0.0, float(EXPECTED_CONFIG['b_max'])
    mean_left, _ = moments(left)
    mean_right, _ = moments(right)
    if not mean_left <= target <= mean_right:
        return {
            'feasible': False, 'mean_left': mean_left,
            'mean_right': mean_right,
        }
    for _ in range(64):
        middle = (left + right) / 2.0
        if moments(middle)[0] < target:
            left = middle
        else:
            right = middle
    b = (left + right) / 2.0
    mean, harmonic = moments(b)
    return {
        'feasible': True, 'a': a, 'b': b, 'left': left, 'right': right,
        'iterations': 64, 'mean_left': mean_left, 'mean_right': mean_right,
        'mean': mean, 'harmonic_mean': harmonic,
        'mean_residual': mean - target,
        'theoretical_high_risk_endpoint': math.exp(b),
    }


def _compare_stats(reported, recomputed, checks, prefix='temperature'):
    for key in (
        'min', 'max', 'mean', 'harmonic_mean', 'q01', 'q10', 'q50',
        'q80', 'q90', 'q95', 'q99', 'top_risk_decile_mean',
    ):
        checks[f'{prefix}_{key}_matches_cache'] = numeric_equal(
            reported.get(key), recomputed.get(key), 2e-6
        )
    for key in (
        'less_than_0p9_count', 'equal_1_count', 'greater_than_1_count',
        'greater_than_1p25_count', 'greater_than_1p5_count',
    ):
        checks[f'{prefix}_{key}_matches_cache'] = (
            reported.get(key) == recomputed.get(key)
        )
        coverage = key.replace('_count', '_coverage')
        checks[f'{prefix}_{coverage}_matches_cache'] = numeric_equal(
            reported.get(coverage), recomputed.get(coverage), 1e-12
        )
    checks[f'{prefix}_count_matches_cache'] = (
        reported.get('count') == recomputed.get('count')
    )


def evaluate_parameters(payload, actual_sources):
    checks = {
        'schema_version_1': payload.get('schema_version') == 1,
        'phase_o1p2': payload.get('phase') == EXPECTED_PHASE,
        'kind_exact': payload.get('artifact_kind') == 'o12_budget_parameters',
        'split_train': payload.get('split') == 'train',
        'formal_full_run': payload.get('formal_full_run') is True,
        'config_exact': payload.get('configuration') == EXPECTED_CONFIG,
        'config_fingerprint_exact': (
            payload.get('configuration_fingerprint')
            == canonical_fingerprint(EXPECTED_CONFIG)
        ),
        'scan_protocol_exact': (
            payload.get('scan_protocol')
            == expected_scan_protocol('solve', 'train')
        ),
        'scan_protocol_fingerprint_exact': (
            payload.get('scan_protocol_fingerprint')
            == canonical_fingerprint(expected_scan_protocol('solve', 'train'))
        ),
        'source_sha_exact': payload.get('source_sha256') == actual_sources,
        'cdf_path_exact': Path(str(payload.get('cdf_path', ''))).resolve() == EXPECTED_CDF_PATH,
        'cdf_sha_exact': payload.get('cdf_sha256') == EXPECTED_CDF_SHA256,
        'o11_joint_gate_true': payload.get('o11_joint_gate_pass') is True,
        'teacher_sha_exact': payload.get('teacher_sha256') == EXPECTED_TEACHER_SHA256,
        'train_list_sha_exact': payload.get('list_sha256') == EXPECTED_LIST_SHA256['train'],
        'dataset_size_exact': payload.get('dataset_size') == EXPECTED_DATASET_SIZE['train'],
        'population_exact': payload.get('valid_native_pixels') == EXPECTED_NATIVE_VALID['train'],
        'nonfinite_zero': payload.get('nonfinite_native_pixels') == 0,
    }
    checks.update(runtime_provenance_checks(payload))
    cache = payload.get('risk_cache')
    cache = cache if isinstance(cache, dict) else {}
    cache_path = Path(str(cache.get('path', ''))).resolve()
    unique, counts, cache_checks, actual_sha, size = histogram_from_cache(
        cache_path, cache.get('sha256')
    )
    checks.update(cache_checks)
    checks.update({
        'cache_actual_sha_matches': actual_sha == cache.get('sha256'),
        'cache_count_matches_report': size == cache.get('count'),
        'cache_count_matches_population': size == EXPECTED_NATIVE_VALID['train'],
        'histogram_closes': int(counts.sum()) == size,
    })
    solved = (
        solve_budget_exact_cache(cache_path) if size else {'feasible': False}
    )
    reported = payload.get('solution')
    reported = reported if isinstance(reported, dict) else {}
    checks.update({
        'independent_budget_feasible': solved.get('feasible') is True,
        'reported_budget_feasible': reported.get('feasible') is True,
        'a_exact': numeric_equal(reported.get('a'), EXPECTED_CONFIG['a'], 1e-15),
        'b_in_range': (
            isinstance(reported.get('b'), (int, float))
            and 0 <= float(reported['b']) <= EXPECTED_CONFIG['b_max']
        ),
        'independent_b_matches_report_le_1e8': (
            solved.get('b') is not None
            and numeric_equal(reported.get('b'), solved['b'], 1e-8)
        ),
        'iterations_64': reported.get('iterations') == 64,
        'solver_population_exact': reported.get('population') == EXPECTED_NATIVE_VALID['train'],
        'train_mean_budget': solved.get('mean') is not None and abs(solved['mean'] - .995) <= 1e-4,
        'train_harmonic_budget': solved.get('harmonic_mean') is not None and solved['harmonic_mean'] >= .98,
        'high_endpoint_ge_1p25': (
            solved.get('theoretical_high_risk_endpoint') is not None
            and solved['theoretical_high_risk_endpoint'] >= 1.25
        ),
        'reported_b_max_matches': numeric_equal(
            reported.get('b_max'), EXPECTED_CONFIG['b_max'], 1e-15
        ),
        'reported_target_mean_matches': numeric_equal(
            reported.get('target_mean'),
            EXPECTED_CONFIG['target_arithmetic_mean'], 1e-15,
        ),
        'reported_arithmetic_mean_matches_independent': numeric_equal(
            reported.get('arithmetic_mean'), solved.get('mean'), 1e-12
        ),
        'reported_harmonic_mean_matches_independent': numeric_equal(
            reported.get('harmonic_mean'), solved.get('harmonic_mean'), 1e-12
        ),
        'reported_residual_matches_independent': numeric_equal(
            reported.get('residual'), solved.get('mean_residual'), 1e-12
        ),
        'reported_endpoint_matches_independent': numeric_equal(
            reported.get('theoretical_high_risk_endpoint'),
            solved.get('theoretical_high_risk_endpoint'), 1e-12,
        ),
        'reported_valid_count_matches': (
            reported.get('valid_count') == EXPECTED_NATIVE_VALID['train']
        ),
        'reported_bisection_iterations_matches': (
            reported.get('bisection_iterations') == 64
        ),
        'reported_lower_endpoint_mean_matches': numeric_equal(
            reported.get('lower_endpoint_mean'), solved.get('mean_left'), 1e-12
        ),
        'reported_upper_endpoint_mean_matches': numeric_equal(
            reported.get('upper_endpoint_mean'), solved.get('mean_right'), 1e-12
        ),
        'arithmetic_scalar_saved': numeric_equal(
            payload.get('arithmetic_matched_scalar_temperature'),
            solved.get('mean'), 2e-6
        ),
        'harmonic_scalar_saved': numeric_equal(
            payload.get('harmonic_matched_scalar_temperature'),
            solved.get('harmonic_mean'), 2e-6
        ),
    })
    branch_scalars = payload.get('branch_scalar_temperatures')
    branch_scalars = branch_scalars if isinstance(branch_scalars, dict) else {}
    if solved.get('b') is not None:
        full_stats = temperature_statistics(
            unique, counts, EXPECTED_CONFIG['a'], solved['b']
        )
        unreliable_stats = temperature_statistics(
            unique, counts, 0.0, solved['b']
        )
    else:
        full_stats = unreliable_stats = {}
    for branch, stats in (
        ('unreliable_only', unreliable_stats),
        ('full_budgeted', full_stats),
    ):
        row = branch_scalars.get(branch)
        row = row if isinstance(row, dict) else {}
        checks[f'{branch}_arithmetic_scalar_matches'] = numeric_equal(
            row.get('arithmetic'), stats.get('mean'), 2e-6
        )
        checks[f'{branch}_harmonic_scalar_matches'] = numeric_equal(
            row.get('harmonic'), stats.get('harmonic_mean'), 2e-6
        )
    return {
        'pass': all(checks.values()), 'checks': checks,
        'cache_sha256': actual_sha, 'cache_size': size,
        'unique_risk_values': int(unique.size),
        'independent_solution': solved,
    }


def validate_bucket_rows(rows, expected_count, expected_wrong, temperature_mean):
    metrics = (
        'temperature_mean', 'c_base_mean', 'c_target_mean',
        'entropy_base_mean', 'entropy_target_mean',
    )
    structure = True
    numeric = True
    weighted_temperature = 0.0
    total_count = 0
    total_wrong = 0
    for row in rows:
        if not isinstance(row, dict):
            structure = False
            continue
        count = row.get('count')
        wrong = row.get('teacher_wrong_count')
        if (
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            or not isinstance(wrong, int) or isinstance(wrong, bool)
            or wrong < 0 or wrong > count
        ):
            structure = False
            continue
        total_count += count
        total_wrong += wrong
        if count == 0:
            numeric = numeric and all(row.get(key) is None for key in metrics)
            continue
        values = {key: row.get(key) for key in metrics}
        numeric = numeric and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values.values()
        )
        if not numeric:
            continue
        numeric = numeric and (
            .9 - 1e-6 <= values['temperature_mean'] <= 1.5 + 1e-6
            and 0 <= values['c_base_mean'] <= 1
            and 0 <= values['c_target_mean'] <= 1
            and 0 <= values['entropy_base_mean'] <= math.log(21) + 1e-6
            and 0 <= values['entropy_target_mean'] <= math.log(21) + 1e-6
        )
        weighted_temperature += count * values['temperature_mean']
    return {
        'structure_and_wrong_range': structure,
        'finite_metric_ranges': numeric,
        'population_closes': total_count == expected_count,
        'wrong_closes': total_wrong == expected_wrong,
        'temperature_weighted_mean_closes': (
            total_count > 0
            and numeric_equal(
                weighted_temperature / total_count, temperature_mean, 2e-6
            )
        ),
    }


def evaluate_split(payload, split, parameters, parameters_sha, actual_sources):
    expected_size = EXPECTED_DATASET_SIZE[split]
    expected_pixels = EXPECTED_NATIVE_VALID[split]
    solution = parameters.get('solution', {})
    o11_reference = load_o11_reference(split)
    checks = {
        'schema_version_1': payload.get('schema_version') == 1,
        'phase_o1p2': payload.get('phase') == EXPECTED_PHASE,
        'kind_exact': payload.get('artifact_kind') == 'o12_budget_evaluation',
        'split_exact': payload.get('split') == split,
        'formal_full_run': payload.get('formal_full_run') is True,
        'images_complete': (
            payload.get('processed_images') == expected_size
            and payload.get('dataset_size') == expected_size
        ),
        'valid_population_exact': payload.get('valid_native_pixels') == expected_pixels,
        'finite_population_exact': payload.get('finite_native_pixels') == expected_pixels,
        'nonfinite_population_zero': payload.get('nonfinite_native_pixels') == 0,
        'config_exact': payload.get('configuration') == EXPECTED_CONFIG,
        'config_fingerprint_exact': (
            payload.get('configuration_fingerprint')
            == canonical_fingerprint(EXPECTED_CONFIG)
        ),
        'scan_protocol_exact': (
            payload.get('scan_protocol')
            == expected_scan_protocol('evaluate', split)
        ),
        'scan_protocol_fingerprint_exact': (
            payload.get('scan_protocol_fingerprint')
            == canonical_fingerprint(expected_scan_protocol('evaluate', split))
        ),
        'source_sha_exact': payload.get('source_sha256') == actual_sources,
        'cdf_path_exact': Path(str(payload.get('cdf_path', ''))).resolve() == EXPECTED_CDF_PATH,
        'cdf_sha_exact': payload.get('cdf_sha256') == EXPECTED_CDF_SHA256,
        'teacher_sha_exact': payload.get('teacher_sha256') == EXPECTED_TEACHER_SHA256,
        'list_sha_exact': payload.get('list_sha256') == EXPECTED_LIST_SHA256[split],
        'parameters_sha_exact': payload.get('parameters_sha256') == parameters_sha,
        'parameters_not_refit': payload.get('parameters_refit') is False,
        'frozen_a_matches': numeric_equal(payload.get('a'), solution.get('a'), 1e-15),
        'frozen_b_matches': numeric_equal(payload.get('b'), solution.get('b'), 1e-15),
    }
    checks.update(runtime_provenance_checks(payload))
    cache = payload.get('risk_cache')
    cache = cache if isinstance(cache, dict) else {}
    cache_path = Path(str(cache.get('path', ''))).resolve()
    unique, counts, cache_checks, actual_sha, size = histogram_from_cache(
        cache_path, cache.get('sha256')
    )
    checks.update(cache_checks)
    checks.update({
        'cache_actual_sha_matches': actual_sha == cache.get('sha256'),
        'cache_count_matches_report': size == cache.get('count'),
        'cache_count_matches_population': size == expected_pixels,
        'histogram_closes': int(counts.sum()) == size,
    })
    if size and isinstance(solution.get('b'), (int, float)):
        stats = temperature_statistics(
            unique, counts, float(solution['a']), float(solution['b'])
        )
        effective_stats = temperature_statistics(
            unique, counts, float(solution['a']), float(solution['b']),
            scale=EXPECTED_CONFIG['teacher_output_temperature'],
        )
    else:
        stats = {'count': 0}
        effective_stats = {'count': 0}
    reported_temp = payload.get('temperature')
    reported_temp = reported_temp if isinstance(reported_temp, dict) else {}
    _compare_stats(reported_temp, stats, checks)
    reported_effective = payload.get('effective_temperature')
    reported_effective = (
        reported_effective if isinstance(reported_effective, dict) else {}
    )
    _compare_stats(
        reported_effective, effective_stats, checks,
        prefix='effective_temperature',
    )
    checks.update({
        'temperature_min_bound': stats.get('min') is not None and stats['min'] >= .9 - 1e-6,
        'temperature_max_bound': stats.get('max') is not None and stats['max'] <= 1.5 + 1e-6,
        'temperature_q10_ge_0p9': stats.get('q10') is not None and stats['q10'] >= .9,
        'temperature_q50_ge_0p95': stats.get('q50') is not None and stats['q50'] >= .95,
    })
    if split == 'train':
        checks.update({
            'train_mean_0p995': stats.get('mean') is not None and abs(stats['mean'] - .995) <= 1e-4,
            'train_harmonic_ge_0p98': stats.get('harmonic_mean') is not None and stats['harmonic_mean'] >= .98,
            'train_top_decile_mean_ge_1p10': stats.get('top_risk_decile_mean') is not None and stats['top_risk_decile_mean'] >= 1.10,
        })
    else:
        checks.update({
            'val_mean_in_0p98_1p02': stats.get('mean') is not None and .98 <= stats['mean'] <= 1.02,
            'val_harmonic_ge_0p97': stats.get('harmonic_mean') is not None and stats['harmonic_mean'] >= .97,
        })

    violations = payload.get('violations')
    violations = violations if isinstance(violations, dict) else {}
    checks['violation_keys_exact'] = set(violations) == set(VIOLATION_KEYS)
    for key in VIOLATION_KEYS:
        checks[f'violation_{key}_zero'] = violations.get(key) == 0

    maxima = payload.get('target_numeric_maxima')
    maxima = maxima if isinstance(maxima, dict) else {}
    maxima_keys = {
        'neutral_target_max_abs_error',
        'student_softmax_max_abs_error',
        'teacher_target_probe_max_abs_difference',
    }
    checks.update({
        'target_numeric_maxima_keys_exact': set(maxima) == maxima_keys,
        'target_numeric_maxima_all_finite_nonnegative': (
            set(maxima) == maxima_keys
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value)) and value >= 0
                for value in maxima.values()
            )
        ),
        'neutral_target_max_within_tau': (
            maxima.get('neutral_target_max_abs_error', math.inf) <= 1e-6
        ),
        'actual_loss_student_probe_within_tau': (
            maxima.get('student_softmax_max_abs_error', math.inf) <= 1e-7
        ),
        'actual_loss_teacher_target_probe_nontrivial': (
            maxima.get('teacher_target_probe_max_abs_difference', 0.0) > 0
        ),
    })
    regions = payload.get('risk_regions')
    regions = regions if isinstance(regions, dict) else {}
    region_rows = [
        regions.get(name) if isinstance(regions.get(name), dict) else {}
        for name in ('reliable', 'neutral', 'unreliable')
    ]
    region_counts = [row.get('count') for row in region_rows]
    wrong_counts = [row.get('teacher_wrong_count') for row in region_rows]
    total_wrong = payload.get('teacher_wrong_pixels')
    checks.update({
        'risk_region_population_closes': (
            all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in region_counts)
            and sum(region_counts) == expected_pixels
        ),
        'risk_region_wrong_closes': (
            all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in wrong_counts)
            and isinstance(total_wrong, int) and sum(wrong_counts) == total_wrong
        ),
    })
    expected_region_counts = {
        'reliable': int(counts[unique < np.float32(.6)].sum()),
        'neutral': int(counts[
            (unique >= np.float32(.6)) & (unique <= np.float32(.8))
        ].sum()),
        'unreliable': int(counts[unique > np.float32(.8)].sum()),
    }
    cache_bin = np.minimum(
        np.floor(unique * np.float32(10.0)).astype(np.int64), 9
    )
    expected_bin_counts = [
        int(counts[cache_bin == index].sum()) for index in range(10)
    ]
    for name, row in zip(
        ('reliable', 'neutral', 'unreliable'), region_rows
    ):
        checks[f'{name}_population_matches_exact_u_cache'] = (
            row.get('count') == expected_region_counts[name]
        )
    boundary = payload.get('boundary_counts')
    boundary = boundary if isinstance(boundary, dict) else {}
    checks.update({
        'u_eq_0p6_matches_exact_u_cache': (
            boundary.get('u_eq_0p6')
            == int(counts[unique == np.float32(.6)].sum())
        ),
        'u_eq_0p8_matches_exact_u_cache': (
            boundary.get('u_eq_0p8')
            == int(counts[unique == np.float32(.8)].sum())
        ),
    })
    bins = payload.get('risk_quantile_bins')
    bins = bins if isinstance(bins, list) else []
    checks.update({
        'ten_risk_bins_present': len(bins) == 10,
        'risk_bin_indices_exact': (
            len(bins) == 10 and [row.get('bin') for row in bins] == list(range(10))
        ),
        'risk_bin_population_closes': (
            len(bins) == 10 and sum(row.get('count', -expected_pixels) for row in bins) == expected_pixels
        ),
        'risk_bin_wrong_closes': (
            len(bins) == 10 and isinstance(total_wrong, int)
            and sum(row.get('teacher_wrong_count', -expected_pixels) for row in bins) == total_wrong
        ),
        'risk_bin_counts_match_exact_u_cache': (
            len(bins) == 10
            and [row.get('count') for row in bins] == expected_bin_counts
        ),
        'all_risk_bins_nonempty': all(count > 0 for count in expected_bin_counts),
        'boundary_counts_saved': (
            isinstance(payload.get('boundary_counts'), dict)
            and set(payload['boundary_counts']) == {
                'u_eq_0p6', 'u_eq_0p8',
                'u_eq_0p6_teacher_wrong', 'u_eq_0p8_teacher_wrong',
            }
            and isinstance(payload['boundary_counts'].get('u_eq_0p6'), int)
            and isinstance(payload['boundary_counts'].get('u_eq_0p8'), int)
            and isinstance(payload['boundary_counts'].get('u_eq_0p6_teacher_wrong'), int)
            and isinstance(payload['boundary_counts'].get('u_eq_0p8_teacher_wrong'), int)
        ),
        'closure_checks_all_true': all_true(payload.get('closure_checks')),
        'risk_evidence_rechecked': isinstance(payload.get('risk_evidence'), dict),
        'target_deciles_ten_rows': (
            isinstance(payload.get('target_by_risk_decile'), list)
            and len(payload['target_by_risk_decile']) == 10
        ),
        'effective_temperature_reported': isinstance(payload.get('effective_temperature'), dict),
        'stratified_sections_present': (
            isinstance(payload.get('stratified_diagnostics'), dict)
            and all(key in payload['stratified_diagnostics'] for key in (
                'class', 'foreground_background', 'boundary_interior', 'small_object'
            ))
        ),
        'teacher_target_contract_exact': payload.get('teacher_target_contract') == {
            'spatial_temperature_applies_to': 'teacher_target_only',
            'student_temperature': 1.0,
            'teacher_target_detached': True,
            'spatial_temperature_loss_power': 'not_applicable',
        },
    })
    o11_bins = o11_reference.get('risk_quantile_bins')
    o11_bins = o11_bins if isinstance(o11_bins, list) else []
    o11_routing = o11_reference.get('risk_routing_counts')
    o11_routing = o11_routing if isinstance(o11_routing, dict) else {}
    o11_bins_valid = (
        len(o11_bins) == 10
        and all(
            isinstance(row, dict)
            and isinstance(row.get('count'), int)
            and not isinstance(row.get('count'), bool)
            and isinstance(row.get('teacher_wrong_count'), int)
            and not isinstance(row.get('teacher_wrong_count'), bool)
            and 0 <= row['teacher_wrong_count'] <= row['count']
            for row in o11_bins
        )
    )
    o11_count_by_bin = (
        [row['count'] for row in o11_bins] if o11_bins_valid else []
    )
    o11_wrong_by_bin = (
        [row['teacher_wrong_count'] for row in o11_bins]
        if o11_bins_valid else []
    )
    routing_fields = (
        'valid_native_pixels', 'teacher_wrong_pixels',
        'high_risk_pixels', 'high_risk_wrong_pixels',
        'low_risk_pixels', 'low_risk_wrong_pixels',
        'boundary_pixels', 'boundary_wrong_pixels',
    )
    o11_routing_valid = all(
        isinstance(o11_routing.get(key), int)
        and not isinstance(o11_routing.get(key), bool)
        and o11_routing[key] >= 0
        for key in routing_fields
    )
    if o11_bins_valid and o11_routing_valid:
        bottom_count = sum(o11_count_by_bin[:6])
        bottom_wrong = sum(o11_wrong_by_bin[:6])
        expected_o11_region_count = {
            'reliable': bottom_count,
            'neutral': (
                o11_routing['low_risk_pixels'] - bottom_count
                + o11_routing['boundary_pixels']
            ),
            'unreliable': o11_routing['high_risk_pixels'],
        }
        expected_o11_region_wrong = {
            'reliable': bottom_wrong,
            'neutral': (
                o11_routing['low_risk_wrong_pixels'] - bottom_wrong
                + o11_routing['boundary_wrong_pixels']
            ),
            'unreliable': o11_routing['high_risk_wrong_pixels'],
        }
    else:
        expected_o11_region_count = {}
        expected_o11_region_wrong = {}
    checks.update({
        'o11_reference_phase_split_exact': (
            o11_reference.get('phase') == 'O1.1'
            and o11_reference.get('split') == split
        ),
        'o11_routing_semantics_exact': (
            o11_routing.get('semantics')
            == 'exact native-valid micro counts; high u>q, low u<q, boundary u==q'
        ),
        'o11_bins_numeric_and_bounded': o11_bins_valid,
        'o11_routing_numeric_nonnegative': o11_routing_valid,
        'o11_routing_population_closes': (
            o11_routing_valid
            and o11_routing['low_risk_pixels']
            + o11_routing['boundary_pixels']
            + o11_routing['high_risk_pixels']
            == o11_routing['valid_native_pixels']
        ),
        'o11_routing_wrong_closes': (
            o11_routing_valid
            and o11_routing['low_risk_wrong_pixels']
            + o11_routing['boundary_wrong_pixels']
            + o11_routing['high_risk_wrong_pixels']
            == o11_routing['teacher_wrong_pixels']
        ),
        'o11_deciles_close_to_routing_threshold': (
            o11_bins_valid and o11_routing_valid
            and sum(o11_count_by_bin[:8]) == o11_routing['low_risk_pixels']
            and sum(o11_count_by_bin[8:]) == (
                o11_routing['boundary_pixels'] + o11_routing['high_risk_pixels']
            )
            and sum(o11_wrong_by_bin[:8]) == o11_routing['low_risk_wrong_pixels']
            and sum(o11_wrong_by_bin[8:]) == (
                o11_routing['boundary_wrong_pixels']
                + o11_routing['high_risk_wrong_pixels']
            )
        ),
        'teacher_wrong_total_matches_frozen_o11': (
            total_wrong == o11_routing.get('teacher_wrong_pixels')
        ),
        'risk_bin_counts_match_frozen_o11': (
            len(bins) == 10 and o11_bins_valid
            and [row.get('count') for row in bins] == o11_count_by_bin
        ),
        'risk_bin_wrong_counts_match_frozen_o11': (
            len(bins) == 10 and o11_bins_valid
            and [row.get('teacher_wrong_count') for row in bins]
            == o11_wrong_by_bin
        ),
        'u_eq_0p8_count_matches_frozen_o11_boundary': (
            boundary.get('u_eq_0p8') == o11_routing.get('boundary_pixels')
        ),
        'u_eq_0p8_wrong_matches_frozen_o11_boundary': (
            boundary.get('u_eq_0p8_teacher_wrong')
            == o11_routing.get('boundary_wrong_pixels')
        ),
        'u_eq_0p6_wrong_within_boundary_count': (
            isinstance(boundary.get('u_eq_0p6_teacher_wrong'), int)
            and 0 <= boundary['u_eq_0p6_teacher_wrong']
            <= boundary.get('u_eq_0p6', -1)
        ),
    })
    for name, row in zip(
        ('reliable', 'neutral', 'unreliable'), region_rows
    ):
        checks[f'{name}_population_matches_frozen_o11_routing'] = (
            row.get('count') == expected_o11_region_count.get(name)
        )
        checks[f'{name}_wrong_matches_frozen_o11_routing'] = (
            row.get('teacher_wrong_count')
            == expected_o11_region_wrong.get(name)
        )
    effective = reported_effective
    for key in (
        'min', 'max', 'mean', 'harmonic_mean', 'q01', 'q10', 'q50',
        'q80', 'q90', 'q95', 'q99', 'top_risk_decile_mean',
    ):
        checks[f'effective_{key}_is_three_times_temperature'] = numeric_equal(
            effective.get(key),
            reported_temp.get(key) * 3.0
            if isinstance(reported_temp.get(key), (int, float)) else None,
            6e-6,
        )
    checks['effective_temperature_semantics_exact'] = (
        effective.get('semantics')
        == 'T_effective=T_out*T with T_out=3.0'
    )
    target_rows = payload.get('target_by_risk_decile')
    checks['target_deciles_exactly_match_risk_bins'] = target_rows == bins
    stratified = payload.get('stratified_diagnostics')
    stratified = stratified if isinstance(stratified, dict) else {}
    class_rows = stratified.get('class')
    class_rows = class_rows if isinstance(class_rows, list) else []
    checks.update({
        'class_strata_21_rows': (
            len(class_rows) == 21
            and [row.get('class_id') for row in class_rows] == list(range(21))
        ),
        'class_strata_population_closes': (
            len(class_rows) == 21
            and sum(row.get('count', -expected_pixels) for row in class_rows)
            == expected_pixels
        ),
        'class_strata_wrong_closes': (
            len(class_rows) == 21 and isinstance(total_wrong, int)
            and sum(
                row.get('teacher_wrong_count', -expected_pixels)
                for row in class_rows
            ) == total_wrong
        ),
    })
    for section, names in (
        ('foreground_background', ('background', 'foreground')),
        ('boundary_interior', ('boundary', 'interior')),
        ('small_object', ('small_object', 'not_small_object')),
    ):
        rows = stratified.get(section)
        rows = rows if isinstance(rows, dict) else {}
        checks[f'{section}_keys_exact'] = set(rows) == set(names)
        checks[f'{section}_population_closes'] = (
            set(rows) == set(names)
            and sum(
                rows[name].get('count', -expected_pixels) for name in names
            ) == expected_pixels
        )
        checks[f'{section}_wrong_closes'] = (
            set(rows) == set(names) and isinstance(total_wrong, int)
            and sum(
                rows[name].get('teacher_wrong_count', -expected_pixels)
                for name in names
            ) == total_wrong
        )
    bucket_partitions = {
        'risk_regions': region_rows,
        'risk_bins': bins,
        'classes': class_rows,
        'foreground_background': [
            stratified.get('foreground_background', {}).get(name, {})
            for name in ('background', 'foreground')
        ],
        'boundary_interior': [
            stratified.get('boundary_interior', {}).get(name, {})
            for name in ('boundary', 'interior')
        ],
        'small_object': [
            stratified.get('small_object', {}).get(name, {})
            for name in ('small_object', 'not_small_object')
        ],
    }
    for partition_name, rows in bucket_partitions.items():
        row_checks = validate_bucket_rows(
            rows, expected_pixels, total_wrong, stats.get('mean')
        )
        for check_name, passed in row_checks.items():
            checks[f'{partition_name}_{check_name}'] = passed
    evidence = payload.get('risk_evidence')
    evidence = evidence if isinstance(evidence, dict) else {}
    high = region_rows[2]
    global_rate = (
        total_wrong / expected_pixels if isinstance(total_wrong, int) else None
    )
    high_rate = (
        high.get('teacher_wrong_count') / high.get('count')
        if isinstance(high.get('count'), int) and high.get('count') > 0
        else None
    )
    rates = [
        row.get('teacher_wrong_count') / row.get('count')
        if isinstance(row.get('count'), int) and row.get('count') > 0
        else None
        for row in bins
    ]
    comparable = [
        (rates[left], rates[right])
        for left in range(len(rates))
        for right in range(left + 1, len(rates))
        if rates[left] is not None and rates[right] is not None
    ]
    agreement = (
        sum(left <= right for left, right in comparable) / len(comparable)
        if comparable else None
    )
    checks.update({
        'risk_evidence_global_rate_recomputed': numeric_equal(
            evidence.get('global_teacher_wrong_rate'), global_rate, 1e-12
        ),
        'risk_evidence_high_precision_recomputed': numeric_equal(
            evidence.get('high_risk_teacher_wrong_precision'), high_rate, 1e-12
        ),
        'risk_evidence_high_coverage_recomputed': numeric_equal(
            evidence.get('high_risk_coverage'),
            high.get('count') / expected_pixels
            if isinstance(high.get('count'), int) else None,
            1e-12,
        ),
        'risk_evidence_high_recall_recomputed': numeric_equal(
            evidence.get('high_risk_teacher_wrong_recall'),
            high.get('teacher_wrong_count') / total_wrong
            if isinstance(total_wrong, int) and total_wrong > 0 else None,
            1e-12,
        ),
        'risk_evidence_enrichment_recomputed': numeric_equal(
            evidence.get('high_risk_enrichment'),
            high_rate / global_rate
            if high_rate is not None and global_rate not in (None, 0.0) else None,
            1e-12,
        ),
        'risk_evidence_pairwise_agreement_recomputed': numeric_equal(
            evidence.get('risk_quantile_pairwise_monotonic_agreement'),
            agreement, 1e-12,
        ),
    })
    return {
        'pass': all(checks.values()), 'checks': checks,
        'cache_sha256': actual_sha, 'cache_size': size,
        'unique_risk_values': int(unique.size),
        'temperature_recomputed': stats,
    }


def formula_monotonic_check(a: float, b: float):
    grid = np.linspace(0.0, 1.0, 4097, dtype=np.float64)
    g_r = np.clip((.6 - grid) / .6, 0.0, 1.0)
    g_u = np.clip((grid - .8) / .2, 0.0, 1.0) ** 2
    values = np.exp(-float(a) * g_r + float(b) * g_u)
    diff = np.diff(values)
    violations = int((diff < -1e-12).sum())
    return {
        'grid_points': 4097, 'minimum_difference': float(diff.min()),
        'violation_count': violations, 'pass': violations == 0,
    }


def evaluate_joint(
    parameters: dict,
    train: dict,
    val: dict,
    *,
    parameters_path: Path,
    train_path: Path,
    val_path: Path,
    enforce_canonical: bool = True,
):
    actual_sources = source_sha256()
    o11_checks = frozen_o11_source_checks()
    cdf_exists = EXPECTED_CDF_PATH.is_file()
    o11_gate_exists = EXPECTED_O11_GATE_PATH.is_file()
    o11_gate = load_json(EXPECTED_O11_GATE_PATH) if o11_gate_exists else {}
    parameters_sha = file_sha256(parameters_path)
    parameter_eval = evaluate_parameters(parameters, actual_sources)
    train_eval = evaluate_split(
        train, 'train', parameters, parameters_sha, actual_sources
    )
    val_eval = evaluate_split(
        val, 'val', parameters, parameters_sha, actual_sources
    )
    solved = parameter_eval.get('independent_solution', {})
    monotonic = formula_monotonic_check(
        solved.get('a', EXPECTED_CONFIG['a']), solved.get('b') or 0.0
    )
    runtime_devices = joint_runtime_device_provenance(parameters, train, val)
    joint_checks = {
        **o11_checks,
        'canonical_parameters_path': (
            not enforce_canonical
            or parameters_path.resolve() == CANONICAL_PATHS['parameters'].resolve()
        ),
        'canonical_train_path': (
            not enforce_canonical
            or train_path.resolve() == CANONICAL_PATHS['train'].resolve()
        ),
        'canonical_val_path': (
            not enforce_canonical
            or val_path.resolve() == CANONICAL_PATHS['val'].resolve()
        ),
        'cdf_file_exists': cdf_exists,
        'cdf_actual_sha_exact': (
            cdf_exists and file_sha256(EXPECTED_CDF_PATH) == EXPECTED_CDF_SHA256
        ),
        'teacher_file_exists_and_sha_exact': (
            EXPECTED_TEACHER_PATH.is_file()
            and file_sha256(EXPECTED_TEACHER_PATH) == EXPECTED_TEACHER_SHA256
        ),
        'train_list_exists_and_sha_exact': (
            EXPECTED_LIST_PATH['train'].is_file()
            and file_sha256(EXPECTED_LIST_PATH['train'])
            == EXPECTED_LIST_SHA256['train']
        ),
        'val_list_exists_and_sha_exact': (
            EXPECTED_LIST_PATH['val'].is_file()
            and file_sha256(EXPECTED_LIST_PATH['val'])
            == EXPECTED_LIST_SHA256['val']
        ),
        'o11_gate_file_exists': o11_gate_exists,
        'o11_gate_phase_exact': o11_gate.get('phase') == 'O1.1',
        'o11_joint_gate_still_true': o11_gate.get('joint_gate_pass') is True,
        'o11_gate_cdf_sha_exact': (
            o11_gate.get('actual_cdf_sha256') == EXPECTED_CDF_SHA256
        ),
        'parameters_gate_pass': parameter_eval['pass'],
        'train_gate_pass': train_eval['pass'],
        'val_gate_pass': val_eval['pass'],
        'train_solve_and_evaluate_cache_sha_identical': (
            parameter_eval.get('cache_sha256') == train_eval.get('cache_sha256')
        ),
        'train_val_each_single_process_single_npu': (
            runtime_devices['per_artifact_valid']['train']
            and runtime_devices['per_artifact_valid']['val']
        ),
        'solve_train_val_each_single_process_single_npu': (
            runtime_devices['all_artifacts_single_process_single_npu']
        ),
        'formula_monotonic_on_4097_float64_grid': monotonic['pass'],
    }
    return {
        'schema_version': 1,
        'phase': EXPECTED_PHASE,
        'gate_profile': GATE_PROFILE,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'joint_gate_pass': all(joint_checks.values()),
        'checker_sha256': actual_sources['check_rtc_o12_gate'],
        'configuration': EXPECTED_CONFIG,
        'configuration_fingerprint': canonical_fingerprint(EXPECTED_CONFIG),
        'source_sha256': actual_sources,
        'parameters_path': str(parameters_path.resolve()),
        'parameters_sha256': parameters_sha,
        'train_json': str(train_path.resolve()),
        'val_json': str(val_path.resolve()),
        'cdf_path': str(EXPECTED_CDF_PATH),
        'cdf_sha256': EXPECTED_CDF_SHA256,
        'joint_checks': joint_checks,
        'runtime_device_provenance': runtime_devices,
        'formula_monotonicity': monotonic,
        'parameters_evaluation': parameter_eval,
        'train_evaluation': train_eval,
        'val_evaluation': val_eval,
    }


def main():
    args = parse_args()
    parameters_path = Path(args.parameters).resolve()
    train_path = Path(args.train_json).resolve()
    val_path = Path(args.val_json).resolve()
    output_path = Path(args.output).resolve()
    for path in (parameters_path, train_path, val_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(f'refusing to overwrite O1.2 gate: {output_path}')
    result = evaluate_joint(
        load_json(parameters_path),
        load_json(train_path),
        load_json(val_path),
        parameters_path=parameters_path,
        train_path=train_path,
        val_path=val_path,
        enforce_canonical=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8'
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    print(f'wrote={output_path}', flush=True)
    if args.strict and not result['joint_gate_pass']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
