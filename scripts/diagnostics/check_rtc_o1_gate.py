#!/usr/bin/env python3
"""Joint train/val gate for the Phase O1 RTC routing diagnosis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / 'runs' / 'diagnostics' / 'phaseO'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate matching full train/val RTC O1 diagnostic artifacts.'
    )
    parser.add_argument(
        '--train-json',
        default=str(DEFAULT_DIR / 'rtc_routing_train.json'),
    )
    parser.add_argument(
        '--val-json',
        default=str(DEFAULT_DIR / 'rtc_routing_val.json'),
    )
    parser.add_argument(
        '--output',
        default=str(DEFAULT_DIR / 'o1_joint_gate.json'),
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        default=False,
        help='Exit with status 2 when the joint gate does not pass.',
    )
    return parser.parse_args()


def load_json(path):
    with path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f'JSON root must be an object: {path}')
    return payload


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_fingerprint(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def nested_bool(payload, *keys):
    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return current is True


def resolved_reported_path(payload, key):
    value = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(value, str) or not value:
        return None
    return str(Path(value).resolve())


def main():
    args = parse_args()
    train_path = Path(args.train_json).resolve()
    val_path = Path(args.val_json).resolve()
    output_path = Path(args.output).resolve()
    errors = []

    train = None
    val = None
    try:
        train = load_json(train_path)
    except Exception as exc:
        errors.append(f'train_json: {type(exc).__name__}: {exc}')
    try:
        val = load_json(val_path)
    except Exception as exc:
        errors.append(f'val_json: {type(exc).__name__}: {exc}')

    train_config = train.get('critical_config') if isinstance(train, dict) else None
    val_config = val.get('critical_config') if isinstance(val, dict) else None
    configs_are_dicts = isinstance(train_config, dict) and isinstance(val_config, dict)
    configs_match = configs_are_dicts and train_config == val_config
    critical_config = train_config if configs_match else None
    config_fingerprint = (
        canonical_fingerprint(critical_config)
        if critical_config is not None else None
    )

    train_cdf_path = resolved_reported_path(train or {}, 'cdf_path')
    val_cdf_path = resolved_reported_path(val or {}, 'cdf_path')
    cdf_paths_match = (
        train_cdf_path is not None
        and train_cdf_path == val_cdf_path
    )
    cdf_path = train_cdf_path or val_cdf_path
    train_cdf_sha = train.get('cdf_sha256') if isinstance(train, dict) else None
    val_cdf_sha = val.get('cdf_sha256') if isinstance(val, dict) else None
    cdf_shas_match = (
        isinstance(train_cdf_sha, str)
        and train_cdf_sha
        and train_cdf_sha == val_cdf_sha
    )
    cdf_sha = (
        train_cdf_sha
        if isinstance(train_cdf_sha, str) and train_cdf_sha
        else val_cdf_sha
    )

    actual_cdf_sha = None
    actual_cdf_exists = False
    if cdf_path is not None:
        actual_cdf_file = Path(cdf_path)
        actual_cdf_exists = actual_cdf_file.is_file()
        if actual_cdf_exists:
            try:
                actual_cdf_sha = file_sha256(actual_cdf_file)
            except Exception as exc:
                errors.append(f'cdf_sha256: {type(exc).__name__}: {exc}')

    train_teacher_sha = (
        train.get('input_provenance', {}).get('teacher_sha256')
        if isinstance(train, dict) else None
    )
    val_teacher_sha = (
        val.get('input_provenance', {}).get('teacher_sha256')
        if isinstance(val, dict) else None
    )
    checks = {
        'train_json_loaded': train is not None,
        'val_json_loaded': val is not None,
        'train_schema_version_1': (
            isinstance(train, dict) and train.get('schema_version') == 1
        ),
        'val_schema_version_1': (
            isinstance(val, dict) and val.get('schema_version') == 1
        ),
        'train_phase_o1': (
            isinstance(train, dict) and train.get('phase') == 'O1'
        ),
        'val_phase_o1': (
            isinstance(val, dict) and val.get('phase') == 'O1'
        ),
        'train_split_exact': (
            isinstance(train, dict) and train.get('split') == 'train'
        ),
        'val_split_exact': (
            isinstance(val, dict) and val.get('split') == 'val'
        ),
        'train_all_checks_pass': (
            isinstance(train, dict) and train.get('all_checks_pass') is True
        ),
        'val_all_checks_pass': (
            isinstance(val, dict) and val.get('all_checks_pass') is True
        ),
        'train_formal_full_run': (
            isinstance(train, dict) and train.get('formal_full_run') is True
        ),
        'val_formal_full_run': (
            isinstance(val, dict) and val.get('formal_full_run') is True
        ),
        'train_native_nonfinite_zero': nested_bool(
            train or {}, 'checks', 'native_nonfinite_valid_pixels_zero'
        ),
        'val_native_nonfinite_zero': nested_bool(
            val or {}, 'checks', 'native_nonfinite_valid_pixels_zero'
        ),
        'train_route_invariance_pass': nested_bool(
            train or {}, 'checks', 'route_fields_identical_for_tout1_and_tout3'
        ),
        'val_route_invariance_pass': nested_bool(
            val or {}, 'checks', 'route_fields_identical_for_tout1_and_tout3'
        ),
        'cdf_paths_match': cdf_paths_match,
        'cdf_reported_sha256_match': cdf_shas_match,
        'actual_cdf_file_exists': actual_cdf_exists,
        'actual_cdf_sha256_matches_reports': (
            actual_cdf_sha is not None
            and actual_cdf_sha == train_cdf_sha
            and actual_cdf_sha == val_cdf_sha
        ),
        'critical_config_exact_match': configs_match,
        'teacher_sha256_exact_match': (
            isinstance(train_teacher_sha, str)
            and train_teacher_sha
            and train_teacher_sha == val_teacher_sha
        ),
        'native_correctness_semantics_match': (
            isinstance(train, dict)
            and isinstance(val, dict)
            and train.get('native_correctness_semantics')
            == val.get('native_correctness_semantics')
        ),
        'cdf_metadata_exact_match': (
            isinstance(train, dict)
            and isinstance(val, dict)
            and isinstance(train.get('cdf_metadata'), dict)
            and train.get('cdf_metadata') == val.get('cdf_metadata')
        ),
    }
    joint_gate_pass = not errors and all(checks.values())
    result = {
        'schema_version': 1,
        'phase': 'O1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'joint_gate_pass': bool(joint_gate_pass),
        'cdf_path': cdf_path,
        'cdf_sha256': cdf_sha,
        'actual_cdf_sha256': actual_cdf_sha,
        'train_reported_cdf_sha256': train_cdf_sha,
        'val_reported_cdf_sha256': val_cdf_sha,
        'train_json': str(train_path),
        'val_json': str(val_path),
        'config_fingerprint': config_fingerprint,
        'critical_config': critical_config,
        'checks': checks,
        'errors': errors,
        'train_summary': {
            'all_checks_pass': (
                train.get('all_checks_pass') if isinstance(train, dict) else None
            ),
            'processed_images': (
                train.get('processed_images') if isinstance(train, dict) else None
            ),
            'dataset_size': (
                train.get('dataset_size') if isinstance(train, dict) else None
            ),
            'high_risk_coverage': (
                train.get('high_risk_coverage') if isinstance(train, dict) else None
            ),
            'high_risk_wrong_precision': (
                train.get('high_risk_wrong_precision')
                if isinstance(train, dict) else None
            ),
            'high_risk_wrong_recall': (
                train.get('high_risk_wrong_recall')
                if isinstance(train, dict) else None
            ),
        },
        'val_summary': {
            'all_checks_pass': (
                val.get('all_checks_pass') if isinstance(val, dict) else None
            ),
            'processed_images': (
                val.get('processed_images') if isinstance(val, dict) else None
            ),
            'dataset_size': (
                val.get('dataset_size') if isinstance(val, dict) else None
            ),
            'high_risk_coverage': (
                val.get('high_risk_coverage') if isinstance(val, dict) else None
            ),
            'high_risk_wrong_precision': (
                val.get('high_risk_wrong_precision')
                if isinstance(val, dict) else None
            ),
            'high_risk_wrong_recall': (
                val.get('high_risk_wrong_recall')
                if isinstance(val, dict) else None
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f'wrote={output_path}')
    if args.strict and not joint_gate_pass:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
