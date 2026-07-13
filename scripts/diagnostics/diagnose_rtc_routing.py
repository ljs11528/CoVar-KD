#!/usr/bin/env python3
"""Diagnose frozen-CDF routing quality and RTC temperature inversion.

All routing coverage and teacher-error precision/recall values are exact micro
statistics over every native-grid valid pixel. Ranking and quantile summaries
use an explicitly bounded, reproducible per-image sample.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.voc import VOCDataTrainSet, VOCDataValSet
from models.model_zoo import get_segmentation_model
from utils.rtc_temperature import (
    RTCConfig,
    build_rtc_temperature_map,
    file_sha256,
    load_frozen_reliability_cdf,
    reliability_definition_metadata,
)


SOURCE_PATHS = {
    'build_rtc_cdf': ROOT / 'scripts' / 'diagnostics' / 'build_rtc_cdf.py',
    'diagnose_rtc_routing': Path(__file__).resolve(),
    'rtc_temperature': ROOT / 'utils' / 'rtc_temperature.py',
}
NATIVE_CORRECTNESS_SEMANTICS = (
    'teacher argmax on native logits versus ground truth resized to the native '
    'logit grid with nearest-neighbor interpolation; this is a KD-grid proxy, '
    'not standard full-resolution validation accuracy'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate RTC routing on native teacher-logit pixels.'
    )
    parser.add_argument('--phase', choices=['O1', 'O1.1'], default='O1')
    parser.add_argument('--data', default=str(ROOT / 'dataset' / 'VOCAug'))
    parser.add_argument('--split', choices=['train', 'val'], required=True)
    parser.add_argument('--list-path', default=None)
    parser.add_argument('--cdf', required=True)
    parser.add_argument('--teacher-model', default='deeplabv3')
    parser.add_argument('--teacher-backbone', default='resnet101')
    parser.add_argument(
        '--teacher-pretrained',
        default=str(
            ROOT / 'data' / 'winycg' / 'cirkd' / 'teachers'
            / 'deeplabv3_resnet101_voc_best_model.pth'
        ),
    )
    parser.add_argument('--num-classes', type=int, default=21)
    parser.add_argument('--crop-size', nargs=2, type=int, default=[512, 512])
    parser.add_argument('--ignore-label', type=int, default=-1)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument(
        '--max-pixels-per-image',
        '--ranking-max-pixels-per-image',
        dest='ranking_max_pixels_per_image',
        type=int,
        default=1024,
        help=(
            'Maximum native valid pixels sampled per image for AP/AUC and '
            'quantiles only; routing coverage and error counts always use all pixels.'
        ),
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=2025,
        help='Dataset augmentation seed; formal train diagnosis defaults to 2025.',
    )
    parser.add_argument('--ranking-seed', type=int, default=3407)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--no-scale', action='store_true', default=False)
    parser.add_argument('--no-mirror', action='store_true', default=False)
    parser.add_argument('--assess-temperature', type=float, default=1.0)
    parser.add_argument('--coefficient-a', type=float, default=None)
    parser.add_argument(
        '--reliability-mode',
        choices=['full', 'confidence', 'variance'],
        default='full',
    )
    parser.add_argument('--teacher-output-temp', type=float, default=3.0)
    parser.add_argument('--route-quantile', type=float, default=0.80)
    parser.add_argument('--route-width', type=float, default=0.05)
    parser.add_argument('--temp-reliable', type=float, default=0.5)
    parser.add_argument('--temp-neutral', type=float, default=1.0)
    parser.add_argument('--temp-unreliable', type=float, default=2.0)
    parser.add_argument('--alpha-reliable', type=float, default=1.0)
    parser.add_argument('--alpha-unreliable', type=float, default=1.0)
    parser.add_argument('--bisection-iters', type=int, default=16)
    parser.add_argument(
        '--max-fallback-rate',
        type=float,
        default=1e-4,
        help='Strict implementation-integrity limit over all native valid pixels.',
    )
    parser.add_argument('--strict', action='store_true', default=False)
    parser.add_argument(
        '--output',
        default=None,
        help='JSON output; phase-specific defaults are used when omitted.',
    )
    return parser.parse_args()


def resolve_device(requested):
    if requested.startswith('npu'):
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError('NPU requested but unavailable')
        device = torch.device(requested)
        torch.npu.set_device(device)
        return device
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return torch.device(requested)


def git_provenance():
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
        ).strip()
    except Exception:
        commit = 'unknown'
    try:
        status = subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=normal'],
            cwd=ROOT,
            text=True,
        )
        dirty = bool(status.strip())
        dirty_entries = len(status.splitlines())
    except Exception:
        dirty = None
        dirty_entries = None
    return commit, dirty, dirty_entries


def source_sha256(phase='O1'):
    paths = dict(SOURCE_PATHS)
    if phase == 'O1.1':
        paths['check_rtc_o11_gate'] = ROOT / 'scripts' / 'diagnostics' / 'check_rtc_o11_gate.py'
    result = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'Required RTC source is missing: {path}')
        result[name] = file_sha256(path)
    return result


def resolve_list_path(args):
    if args.list_path is not None:
        return Path(args.list_path).resolve()
    filename = 'train_aug.txt' if args.split == 'train' else 'val.txt'
    return (ROOT / 'dataset' / 'list' / 'voc' / filename).resolve()


def build_dataset(args, list_path):
    if args.split == 'train':
        return VOCDataTrainSet(
            args.data,
            str(list_path),
            max_iters=None,
            crop_size=tuple(args.crop_size),
            scale=not args.no_scale,
            mirror=not args.no_mirror,
            ignore_label=args.ignore_label,
        )
    return VOCDataValSet(
        args.data,
        str(list_path),
        crop_size=tuple(args.crop_size),
        ignore_label=args.ignore_label,
    )


def expected_coefficient(args):
    if args.coefficient_a is not None:
        return float(args.coefficient_a)
    return float((args.num_classes - 1) ** 2) / 2.0


def float_matches(left, right, tolerance=1e-12):
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def validate_args(args):
    if args.batch_size <= 0:
        raise ValueError('--batch-size must be positive')
    if args.workers < 0:
        raise ValueError('--workers must be non-negative')
    if args.max_images < 0:
        raise ValueError('--max-images must be non-negative')
    if args.ranking_max_pixels_per_image < 0:
        raise ValueError('--max-pixels-per-image must be non-negative')
    if args.num_classes < 2:
        raise ValueError('--num-classes must be at least 2')
    if args.assess_temperature <= 0:
        raise ValueError('--assess-temperature must be positive')
    if args.coefficient_a is not None and args.coefficient_a < 0:
        raise ValueError('--coefficient-a must be non-negative')
    if args.teacher_output_temp <= 0:
        raise ValueError('--teacher-output-temp must be positive')
    if args.max_fallback_rate < 0 or args.max_fallback_rate >= 1:
        raise ValueError('--max-fallback-rate must be in [0, 1)')
    if args.phase == 'O1.1':
        if args.reliability_mode != 'confidence':
            raise ValueError('O1.1 requires --reliability-mode confidence')
        if args.coefficient_a is None or float(args.coefficient_a) != 0.0:
            raise ValueError('O1.1 requires explicit --coefficient-a 0')


def cdf_source_hash(metadata, name):
    source = metadata.get('source_sha256')
    if not isinstance(source, dict):
        return None
    entry = source.get(name)
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get('sha256')
    return None


def validate_cdf_metadata(cdf, args, list_path, current_sources):
    metadata = dict(cdf.metadata)
    teacher_path = Path(args.teacher_pretrained).resolve()
    teacher_sha = file_sha256(teacher_path)
    current_list_sha = file_sha256(list_path)
    coefficient_a = expected_coefficient(args)
    expected_definition = reliability_definition_metadata(
        args.reliability_mode, coefficient_a
    )

    cdf_train_path_raw = metadata.get('train_list_path')
    cdf_train_path = (
        Path(cdf_train_path_raw).resolve() if isinstance(cdf_train_path_raw, str) else None
    )
    cdf_train_file_sha = (
        file_sha256(cdf_train_path)
        if cdf_train_path is not None and cdf_train_path.is_file()
        else None
    )
    cdf_phase = metadata.get('phase')
    checks = {
        'phase_matches': (
            cdf_phase == args.phase
            if args.phase == 'O1.1'
            else cdf_phase in (None, 'O1')
        ),
        'dataset_is_voc': metadata.get('dataset') == 'voc',
        'cdf_split_is_train_aug': metadata.get('split') == 'train_aug',
        'data_root_matches': (
            Path(str(metadata.get('data_root', ''))).resolve()
            == Path(args.data).resolve()
        ),
        'teacher_model_matches': metadata.get('teacher_model') == args.teacher_model,
        'teacher_backbone_matches': (
            metadata.get('teacher_backbone') == args.teacher_backbone
        ),
        'teacher_sha256_matches': metadata.get('teacher_sha256') == teacher_sha,
        'num_classes_matches': metadata.get('num_classes') == int(args.num_classes),
        'reliability_mode_matches': (
            metadata.get('reliability_mode') == args.reliability_mode
        ),
        'assess_temperature_matches': float_matches(
            metadata.get('assess_temperature'), args.assess_temperature
        ),
        'coefficient_a_matches': float_matches(
            metadata.get('coefficient_a'), coefficient_a
        ),
        'reliability_definition_matches': (
            args.phase != 'O1.1'
            or all(
                metadata.get(key) == value
                for key, value in expected_definition.items()
            )
        ),
        'teacher_output_grid_is_native': (
            metadata.get('teacher_output_grid') == 'native'
        ),
        'valid_mask_resize_is_nearest': (
            metadata.get('valid_mask_resize') == 'nearest'
        ),
        'cdf_train_list_file_exists': (
            cdf_train_path is not None and cdf_train_path.is_file()
        ),
        'cdf_train_list_file_sha_matches': (
            metadata.get('train_list_sha256') == cdf_train_file_sha
        ),
        'diagnostic_train_list_matches_cdf': (
            args.split != 'train'
            or metadata.get('train_list_sha256') == current_list_sha
        ),
        'train_crop_size_matches': (
            args.split != 'train'
            or list(metadata.get('crop_size', [])) == list(args.crop_size)
        ),
        'train_scale_policy_matches': (
            args.split != 'train'
            or bool(metadata.get('scale')) == (not args.no_scale)
        ),
        'train_mirror_policy_matches': (
            args.split != 'train'
            or bool(metadata.get('mirror')) == (not args.no_mirror)
        ),
        'builder_records_valid_native_pixels': (
            isinstance(metadata.get('valid_native_pixels'), int)
            and metadata.get('valid_native_pixels', 0) > 0
        ),
        'builder_records_finite_valid_pixels': (
            isinstance(metadata.get('finite_valid_pixels'), int)
            and metadata.get('finite_valid_pixels', -1) >= 0
        ),
        'builder_records_nonfinite_valid_pixels': (
            isinstance(metadata.get('nonfinite_valid_pixels'), int)
            and metadata.get('nonfinite_valid_pixels', -1) >= 0
        ),
        'builder_valid_partition_is_additive': (
            metadata.get('valid_native_pixels')
            == metadata.get('finite_valid_pixels', -1)
            + metadata.get('nonfinite_valid_pixels', -1)
        ),
    }
    for name in current_sources:
        checks[f'source_{name}_sha_matches'] = (
            cdf_source_hash(metadata, name)
            == current_sources[name]
        )
    # The exact full-scan recipe is a formal-run contract. Partial smoke runs
    # still validate phase/formula/source identity above, but are allowed to use
    # a deliberately truncated CDF; they can never pass ``formal_full_run``.
    if args.phase == 'O1.1' and args.max_images == 0:
        checks.update({
            'o11_cdf_seed_is_1234': metadata.get('seed') == 1234,
            'o11_cdf_batch_size_is_4': metadata.get('batch_size') == 4,
            'o11_cdf_workers_is_0': metadata.get('workers') == 0,
            'o11_cdf_max_images_is_0': metadata.get('max_images') == 0,
            'o11_cdf_full_dataset_scan_is_true': (
                metadata.get('full_dataset_scan') is True
            ),
            'o11_cdf_dataset_complete': (
                metadata.get('processed_images') == metadata.get('dataset_size')
                and metadata.get('dataset_size') == 10582
            ),
            'o11_cdf_max_pixels_per_image_is_4096': (
                metadata.get('max_pixels_per_image') == 4096
            ),
            'o11_cdf_num_quantiles_is_4097': (
                metadata.get('num_quantiles') == 4097
            ),
            'o11_cdf_crop_size_is_512': metadata.get('crop_size') == [512, 512],
            'o11_cdf_scale_is_true': metadata.get('scale') is True,
            'o11_cdf_mirror_is_true': metadata.get('mirror') is True,
            'o11_cdf_nonfinite_valid_pixels_zero': (
                metadata.get('nonfinite_valid_pixels') == 0
            ),
        })
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            'CDF metadata mismatch; rebuild or select the correct artifact: '
            + ', '.join(failed)
        )
    return checks, teacher_sha, current_list_sha, coefficient_a


def sampled_indices(valid, max_pixels, generator):
    indices = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(1).cpu()
    if max_pixels > 0 and indices.numel() > max_pixels:
        permutation = torch.randperm(indices.numel(), generator=generator)[:max_pixels]
        indices = indices[permutation]
    return indices


def tie_aware_ranking(scores, positives):
    """Return grouped-threshold AP/AUC so equal scores receive equal treatment."""
    scores = scores.detach().reshape(-1).float().cpu()
    positives = positives.detach().reshape(-1).bool().cpu()
    finite = torch.isfinite(scores)
    scores = scores[finite]
    positives = positives[finite]
    sample_count = int(scores.numel())
    positive_count = int(positives.sum().item())
    negative_count = sample_count - positive_count
    if sample_count == 0:
        return {
            'average_precision': None,
            'roc_auc': None,
            'sample_count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'score_group_count': 0,
            'tied_score_group_count': 0,
            'largest_tie': 0,
            'tie_semantics': 'grouped thresholds; ties are never broken by input order',
        }

    order = torch.argsort(scores, descending=True)
    sorted_scores = scores[order]
    sorted_positive = positives[order].to(torch.float64)
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    group_ids = torch.repeat_interleave(
        torch.arange(counts.numel(), dtype=torch.long), counts
    )
    group_positive = torch.zeros(counts.numel(), dtype=torch.float64)
    group_positive.scatter_add_(0, group_ids, sorted_positive)
    group_count = counts.to(torch.float64)
    cumulative_positive = torch.cumsum(group_positive, dim=0)
    cumulative_count = torch.cumsum(group_count, dim=0)

    average_precision = None
    if positive_count > 0:
        precision = cumulative_positive / cumulative_count
        recall_increment = group_positive / float(positive_count)
        average_precision = float((precision * recall_increment).sum().item())

    roc_auc = None
    if positive_count > 0 and negative_count > 0:
        cumulative_negative = cumulative_count - cumulative_positive
        true_positive_rate = cumulative_positive / float(positive_count)
        false_positive_rate = cumulative_negative / float(negative_count)
        true_positive_rate = torch.cat(
            (torch.zeros(1, dtype=torch.float64), true_positive_rate)
        )
        false_positive_rate = torch.cat(
            (torch.zeros(1, dtype=torch.float64), false_positive_rate)
        )
        roc_auc = float(torch.trapz(true_positive_rate, false_positive_rate).item())

    return {
        'average_precision': average_precision,
        'roc_auc': roc_auc,
        'sample_count': sample_count,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'score_group_count': int(counts.numel()),
        'tied_score_group_count': int((counts > 1).sum().item()),
        'largest_tie': int(counts.max().item()),
        'tie_semantics': 'grouped thresholds; ties are never broken by input order',
    }


def optional_mean(values):
    if values.numel() == 0:
        return None
    return float(values.float().mean().item())


def summarize_r0_deciles(r0, correct, confidence, variance, bins=10):
    if r0.numel() == 0:
        return [], {'sample_count': 0, 'edges': []}
    quantile_points = torch.linspace(0.0, 1.0, bins + 1)
    edges = torch.quantile(r0.float(), quantile_points)
    bin_index = torch.bucketize(r0.float(), edges[1:-1], right=True)
    rows = []
    wrong = ~correct.bool()
    for index in range(bins):
        mask = bin_index == index
        rows.append({
            'bin': index,
            'sample_population': 'finite controlled ranking sample',
            'count': int(mask.sum().item()),
            'r0_low': float(edges[index].item()),
            'r0_high': float(edges[index + 1].item()),
            'r0_mean': optional_mean(r0[mask]),
            'confidence_mean': optional_mean(confidence[mask]),
            'variance_mean': optional_mean(variance[mask]),
            'teacher_accuracy_native_proxy': optional_mean(correct[mask]),
            'teacher_wrong_rate_native_proxy': optional_mean(wrong[mask]),
        })
    return rows, {
        'sample_count': int(r0.numel()),
        'edges': [float(item) for item in edges.tolist()],
        'mean': float(r0.float().mean().item()),
        'min': float(r0.float().min().item()),
        'max': float(r0.float().max().item()),
    }


def summarize_u_bins(counts, wrong_counts):
    rows = []
    for index in range(10):
        count = int(counts[index])
        wrong = int(wrong_counts[index])
        rows.append({
            'bin': index,
            'population': 'all native valid pixels (micro)',
            'u_low': index / 10.0,
            'u_high': (index + 1) / 10.0,
            'count': count,
            'teacher_wrong_count': wrong,
            'teacher_accuracy_native_proxy': (
                float((count - wrong) / count) if count > 0 else None
            ),
            'teacher_wrong_rate_native_proxy': (
                float(wrong / count) if count > 0 else None
            ),
        })
    return rows


def average_tie_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def risk_quantile_spearman(rows):
    if len(rows) != 10:
        return None
    if any(
        row.get('count', 0) <= 0
        or row.get('teacher_wrong_rate_native_proxy') is None
        for row in rows
    ):
        return None
    x_rank = np.arange(len(rows), dtype=np.float64)
    y_rank = average_tie_ranks([
        row['teacher_wrong_rate_native_proxy'] for row in rows
    ])
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = float(np.sqrt(np.dot(x_rank, x_rank) * np.dot(y_rank, y_rank)))
    if denominator == 0.0:
        return None
    return float(np.dot(x_rank, y_rank) / denominator)


def risk_quantile_pairwise_monotonic_agreement(rows):
    """Fraction of ordered decile pairs without a wrong-rate inversion.

    Equal wrong rates count as monotonic agreement, so a desirable zero-error
    plateau on the reliable side is not penalized.
    """
    if len(rows) != 10:
        return None
    if any(
        row.get('count', 0) <= 0
        or row.get('teacher_wrong_rate_native_proxy') is None
        for row in rows
    ):
        return None
    rates = [row['teacher_wrong_rate_native_proxy'] for row in rows]
    total_pairs = 0
    concordant_pairs = 0
    for left in range(len(rates)):
        for right in range(left + 1, len(rates)):
            total_pairs += 1
            concordant_pairs += int(rates[left] <= rates[right])
    return float(concordant_pairs / total_pairs) if total_pairs else None


def route_field_comparison(maps_t1, maps_t3, native_valid):
    result = {}
    for name in (
        'reliability',
        'reliability_quantile',
        'gate_reliable',
        'gate_unreliable',
    ):
        left = getattr(maps_t1, name)
        right = getattr(maps_t3, name)
        mismatch = native_valid & (left != right)
        finite_pair = native_valid & torch.isfinite(left) & torch.isfinite(right)
        if bool(finite_pair.any().item()):
            max_abs_diff = float(
                torch.abs(left[finite_pair] - right[finite_pair]).max().item()
            )
        else:
            max_abs_diff = None
        result[name] = {
            'mismatch_count': int(mismatch.sum().item()),
            'max_abs_diff': max_abs_diff,
        }
    valid_mismatch = maps_t1.valid_mask != maps_t3.valid_mask
    finite_mismatch = maps_t1.finite_mask != maps_t3.finite_mask
    result['valid_mask'] = {
        'mismatch_count': int(valid_mismatch.sum().item()),
        'max_abs_diff': None,
    }
    result['finite_mask'] = {
        'mismatch_count': int(finite_mismatch.sum().item()),
        'max_abs_diff': None,
    }
    return result


def empty_counters():
    return {
        'valid': 0,
        'finite_valid': 0,
        'nonfinite_valid': 0,
        'correct': 0,
        'wrong': 0,
        'high': 0,
        'low': 0,
        'neutral': 0,
        'high_wrong': 0,
        'low_wrong': 0,
        'neutral_wrong': 0,
        'gate_reliable_positive': 0,
        'gate_unreliable_positive': 0,
        'gate_reliable_sum': 0.0,
        'gate_unreliable_sum': 0.0,
        'fallback': 0,
        'tie': 0,
        'active': 0,
        'active_fallback': 0,
        'solved': 0,
        'residual_sum': 0.0,
        'residual_max': None,
        'temperature_sum': 0.0,
        'temperature_inverse_sum': 0.0,
        'temperature_min': None,
        'temperature_max': None,
        'temperature_reliable_endpoint': 0,
        'temperature_neutral_endpoint': 0,
        'temperature_unreliable_endpoint': 0,
        'direction_violations_reliable': 0,
        'direction_violations_unreliable': 0,
        'argmax_disagreements': 0,
        'route_mismatches': {
            name: 0 for name in (
                'reliability',
                'reliability_quantile',
                'gate_reliable',
                'gate_unreliable',
                'valid_mask',
                'finite_mask',
            )
        },
        'route_max_abs_diff': {
            name: None for name in (
                'reliability',
                'reliability_quantile',
                'gate_reliable',
                'gate_unreliable',
            )
        },
        'u_bin_count': [0] * 10,
        'u_bin_wrong': [0] * 10,
    }


def add_optional_max(current, candidate):
    if candidate is None:
        return current
    if current is None:
        return candidate
    return max(current, candidate)


def update_micro_counters(
    counters,
    maps,
    maps_t1,
    maps_t3,
    prediction,
    softened_prediction,
    target_native,
    config,
):
    native_valid = maps.valid_mask
    finite_valid = native_valid & maps.finite_mask
    nonfinite_valid = native_valid & ~maps.finite_mask
    wrong = native_valid & ((prediction != target_native) | nonfinite_valid)
    correct = native_valid & ~wrong
    high = native_valid & (maps.reliability_quantile > config.route_quantile)
    low = native_valid & (maps.reliability_quantile < config.route_quantile)
    neutral = native_valid & ~(high | low)
    gate_reliable = native_valid & (maps.gate_reliable > 0)
    gate_unreliable = native_valid & (maps.gate_unreliable > 0)
    fallback = native_valid & maps.fallback_mask
    tie = native_valid & maps.tie_mask
    active = native_valid & (gate_reliable | gate_unreliable)
    solved = (
        active
        & ~fallback
        & finite_valid
        & torch.isfinite(maps.target_residual)
    )

    counters['valid'] += int(native_valid.sum().item())
    counters['finite_valid'] += int(finite_valid.sum().item())
    counters['nonfinite_valid'] += int(nonfinite_valid.sum().item())
    counters['correct'] += int(correct.sum().item())
    counters['wrong'] += int(wrong.sum().item())
    counters['high'] += int(high.sum().item())
    counters['low'] += int(low.sum().item())
    counters['neutral'] += int(neutral.sum().item())
    counters['high_wrong'] += int((high & wrong).sum().item())
    counters['low_wrong'] += int((low & wrong).sum().item())
    counters['neutral_wrong'] += int((neutral & wrong).sum().item())
    counters['gate_reliable_positive'] += int(gate_reliable.sum().item())
    counters['gate_unreliable_positive'] += int(gate_unreliable.sum().item())
    counters['gate_reliable_sum'] += float(
        maps.gate_reliable[native_valid].float().sum().item()
    )
    counters['gate_unreliable_sum'] += float(
        maps.gate_unreliable[native_valid].float().sum().item()
    )
    counters['fallback'] += int(fallback.sum().item())
    counters['tie'] += int(tie.sum().item())
    counters['active'] += int(active.sum().item())
    counters['active_fallback'] += int((active & fallback).sum().item())
    counters['solved'] += int(solved.sum().item())

    if bool(solved.any().item()):
        residual = maps.target_residual[solved].detach().float()
        counters['residual_sum'] += float(residual.sum().item())
        counters['residual_max'] = add_optional_max(
            counters['residual_max'], float(residual.max().item())
        )

    temperature = maps.temperature[native_valid].detach().float()
    if temperature.numel() > 0:
        counters['temperature_sum'] += float(temperature.sum().item())
        counters['temperature_inverse_sum'] += float((1.0 / temperature).sum().item())
        counters['temperature_min'] = (
            float(temperature.min().item())
            if counters['temperature_min'] is None
            else min(counters['temperature_min'], float(temperature.min().item()))
        )
        counters['temperature_max'] = add_optional_max(
            counters['temperature_max'], float(temperature.max().item())
        )
        bisection_scale = math.ldexp(
            1.0, -(config.bisection_iterations + 1)
        )
        floating_tolerance = (
            4.0
            * torch.finfo(temperature.dtype).eps
            * max(1.0, config.unreliable_temperature)
        )
        reliable_tolerance = (
            (config.neutral_temperature - config.reliable_temperature)
            * bisection_scale
            + floating_tolerance
        )
        unreliable_tolerance = (
            (config.unreliable_temperature - config.neutral_temperature)
            * bisection_scale
            + floating_tolerance
        )
        neutral_tolerance = max(reliable_tolerance, unreliable_tolerance)
        counters['temperature_reliable_endpoint'] += int(
            (
                torch.abs(temperature - config.reliable_temperature)
                <= reliable_tolerance
            ).sum().item()
        )
        counters['temperature_neutral_endpoint'] += int(
            (
                torch.abs(temperature - config.neutral_temperature)
                <= neutral_tolerance
            ).sum().item()
        )
        counters['temperature_unreliable_endpoint'] += int(
            (
                torch.abs(temperature - config.unreliable_temperature)
                <= unreliable_tolerance
            ).sum().item()
        )

    counters['direction_violations_reliable'] += int(
        (
            gate_reliable
            & (maps.temperature > config.neutral_temperature + 1e-5)
        ).sum().item()
    )
    counters['direction_violations_unreliable'] += int(
        (
            gate_unreliable
            & (maps.temperature < config.neutral_temperature - 1e-5)
        ).sum().item()
    )
    counters['argmax_disagreements'] += int(
        ((prediction != softened_prediction) & finite_valid).sum().item()
    )

    comparison = route_field_comparison(maps_t1, maps_t3, native_valid)
    for name, values in comparison.items():
        counters['route_mismatches'][name] += values['mismatch_count']
        if name in counters['route_max_abs_diff']:
            counters['route_max_abs_diff'][name] = add_optional_max(
                counters['route_max_abs_diff'][name], values['max_abs_diff']
            )

    u_values = maps.reliability_quantile[native_valid].detach().float().cpu()
    wrong_values = wrong[native_valid].detach().cpu()
    if u_values.numel() > 0:
        bin_index = torch.clamp((u_values * 10.0).long(), min=0, max=9)
        counts = torch.bincount(bin_index, minlength=10)
        wrong_counts = torch.bincount(bin_index[wrong_values], minlength=10)
        for index in range(10):
            counters['u_bin_count'][index] += int(counts[index].item())
            counters['u_bin_wrong'][index] += int(wrong_counts[index].item())

    return native_valid, finite_valid, wrong, active, solved


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def quantile_dict(values, points):
    values = values.detach().reshape(-1).float().cpu()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {name: None for name in points}
    quantiles = torch.quantile(values, torch.tensor(list(points.values())))
    return {
        name: float(quantiles[index].item())
        for index, name in enumerate(points)
    }


def pre_registered_config_matches(args):
    if args.phase == 'O1.1':
        score_matches = (
            args.reliability_mode == 'confidence'
            and float_matches(expected_coefficient(args), 0.0)
        )
        profile_matches = (
            args.seed == 2025
            and args.ranking_seed == 3407
            and args.ranking_max_pixels_per_image == 1024
            and args.workers == 0
            and args.max_images == 0
            and args.num_classes == 21
            and args.teacher_model == 'deeplabv3'
            and args.teacher_backbone == 'resnet101'
            and list(args.crop_size) == [512, 512]
            and ((args.split == 'train' and args.batch_size == 4)
                 or (args.split == 'val' and args.batch_size == 1))
            and (args.split != 'train' or (not args.no_scale and not args.no_mirror))
        )
    else:
        profile_matches = True
        score_matches = (
            args.reliability_mode == 'full'
            and float_matches(
                expected_coefficient(args),
                (args.num_classes - 1) ** 2 / 2.0,
            )
        )
    return (
        float_matches(args.assess_temperature, 1.0)
        and score_matches
        and profile_matches
        and float_matches(args.teacher_output_temp, 3.0)
        and float_matches(args.route_quantile, 0.80)
        and float_matches(args.route_width, 0.05)
        and float_matches(args.temp_reliable, 0.5)
        and float_matches(args.temp_neutral, 1.0)
        and float_matches(args.temp_unreliable, 2.0)
        and float_matches(args.alpha_reliable, 1.0)
        and float_matches(args.alpha_unreliable, 1.0)
        and args.bisection_iters == 16
    )


def main():
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == 'npu':
        torch.npu.manual_seed_all(args.seed)
    if args.split == 'val' and args.batch_size != 1:
        print('val images have variable sizes; forcing batch_size=1', flush=True)
        args.batch_size = 1

    list_path = resolve_list_path(args)
    dataset = build_dataset(args, list_path)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    teacher_path = Path(args.teacher_pretrained).resolve()
    teacher = get_segmentation_model(
        model=args.teacher_model,
        backbone=args.teacher_backbone,
        local_rank=0,
        pretrained_base='None',
        pretrained=str(teacher_path),
        aux=True,
        norm_layer=nn.BatchNorm2d,
        num_class=args.num_classes,
    ).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    current_sources = source_sha256(args.phase)
    cdf = load_frozen_reliability_cdf(args.cdf, device=device)
    (
        cdf_metadata_checks,
        teacher_sha,
        current_list_sha,
        coefficient_a,
    ) = validate_cdf_metadata(cdf, args, list_path, current_sources)
    config = RTCConfig(
        assess_temperature=args.assess_temperature,
        route_quantile=args.route_quantile,
        route_width=args.route_width,
        reliable_temperature=args.temp_reliable,
        neutral_temperature=args.temp_neutral,
        unreliable_temperature=args.temp_unreliable,
        alpha_reliable=args.alpha_reliable,
        alpha_unreliable=args.alpha_unreliable,
        enable_reliable=True,
        enable_unreliable=True,
        bisection_iterations=args.bisection_iters,
        kd_temperature_power=0.0,
        coefficient_a=coefficient_a,
        reliability_mode=args.reliability_mode,
    )
    config.validate()

    sample_fields = {
        key: [] for key in (
            'r_primary',
            'r_full',
            'r_confidence',
            'r_variance',
            'u',
            'correct',
            'confidence',
            'variance',
            'finite',
            'fallback',
            'tie',
            'temperature',
            'residual',
            'active',
            'solved',
        )
    }
    generator = torch.Generator(device='cpu').manual_seed(args.ranking_seed)
    counters = empty_counters()
    processed_images = 0
    native_logit_shapes = set()

    with torch.no_grad():
        for images, targets, _ in loader:
            if args.max_images > 0:
                remaining = args.max_images - processed_images
                if remaining <= 0:
                    break
                images = images[:remaining]
                targets = targets[:remaining]
            images = images.to(device)
            targets = targets.long().to(device)
            output = teacher(images)
            raw_logits = output[0] if isinstance(output, (list, tuple)) else output
            native_logit_shapes.add(tuple(int(v) for v in raw_logits.shape[-2:]))
            input_valid_mask = targets != args.ignore_label

            maps_t1 = build_rtc_temperature_map(
                raw_logits,
                raw_logits,
                input_valid_mask,
                cdf,
                config,
            )
            maps_t3 = build_rtc_temperature_map(
                raw_logits,
                raw_logits / 3.0,
                input_valid_mask,
                cdf,
                config,
            )
            if float_matches(args.teacher_output_temp, 1.0):
                maps = maps_t1
            elif float_matches(args.teacher_output_temp, 3.0):
                maps = maps_t3
            else:
                maps = build_rtc_temperature_map(
                    raw_logits,
                    raw_logits / args.teacher_output_temp,
                    input_valid_mask,
                    cdf,
                    config,
                )

            target_native = F.interpolate(
                targets.float().unsqueeze(1),
                size=raw_logits.shape[-2:],
                mode='nearest',
            ).squeeze(1).long()
            prediction = raw_logits.argmax(dim=1)
            softened_prediction = (
                raw_logits / args.teacher_output_temp
            ).argmax(dim=1)

            (
                native_valid,
                finite_valid,
                wrong,
                active,
                solved,
            ) = update_micro_counters(
                counters,
                maps,
                maps_t1,
                maps_t3,
                prediction,
                softened_prediction,
                target_native,
                config,
            )

            residual_mass = (1.0 - maps.confidence).clamp_min(1e-8)
            r_confidence = -torch.log(maps.confidence.clamp_min(1e-8))
            canonical_a = float((args.num_classes - 1) ** 2) / 2.0
            r_variance = canonical_a * maps.variance / residual_mass
            r_full = r_confidence + r_variance
            batch_values = {
                'r_primary': maps.reliability,
                'r_full': r_full,
                'r_confidence': r_confidence,
                'r_variance': r_variance,
                'u': maps.reliability_quantile,
                'correct': ~wrong,
                'confidence': maps.confidence,
                'variance': maps.variance,
                'finite': finite_valid,
                'fallback': maps.fallback_mask,
                'tie': maps.tie_mask,
                'temperature': maps.temperature,
                'residual': maps.target_residual,
                'active': active,
                'solved': solved,
            }
            for image_index in range(images.shape[0]):
                indices = sampled_indices(
                    native_valid[image_index],
                    args.ranking_max_pixels_per_image,
                    generator,
                )
                for key, value in batch_values.items():
                    flattened = value[image_index].reshape(-1).detach().cpu()
                    sample_fields[key].append(flattened[indices])

            processed_images += int(images.shape[0])
            if args.log_every > 0 and processed_images % args.log_every < images.shape[0]:
                sampled_count = sum(item.numel() for item in sample_fields['u'])
                print(
                    f'split={args.split} processed_images={processed_images}/{len(dataset)} '
                    f'native_valid_pixels={counters["valid"]} '
                    f'nonfinite_valid_pixels={counters["nonfinite_valid"]} '
                    f'ranking_sample_pixels={sampled_count}',
                    flush=True,
                )
            if args.max_images > 0 and processed_images >= args.max_images:
                break

    if counters['valid'] == 0:
        raise RuntimeError('No native-grid valid pixels were diagnosed')
    tensors = {
        key: torch.cat(value) if value else torch.empty(0)
        for key, value in sample_fields.items()
    }
    ranking_valid = (
        tensors['finite'].bool()
        & torch.isfinite(tensors['r_primary'].float())
        & torch.isfinite(tensors['r_full'].float())
        & torch.isfinite(tensors['r_confidence'].float())
        & torch.isfinite(tensors['r_variance'].float())
    )
    ranking_wrong = ~tensors['correct'][ranking_valid].bool()
    ranking = {}
    for name in ('r_primary', 'r_full', 'r_confidence', 'r_variance'):
        ranking[name] = tie_aware_ranking(
            tensors[name][ranking_valid].float(),
            ranking_wrong,
        )

    r0_rows, r0_summary = summarize_r0_deciles(
        tensors['r_primary'][ranking_valid].float(),
        tensors['correct'][ranking_valid].bool(),
        tensors['confidence'][ranking_valid].float(),
        tensors['variance'][ranking_valid].float(),
    )
    u_rows = summarize_u_bins(
        counters['u_bin_count'],
        counters['u_bin_wrong'],
    )
    u_spearman = risk_quantile_spearman(u_rows)
    u_pairwise_agreement = risk_quantile_pairwise_monotonic_agreement(u_rows)

    residual_sample_mask = tensors['solved'].bool() & torch.isfinite(
        tensors['residual'].float()
    )
    residual_sample = tensors['residual'][residual_sample_mask].float()
    residual_sample_quantiles = quantile_dict(
        residual_sample,
        {'p95': 0.95},
    )
    temperature_sample = tensors['temperature'][
        torch.isfinite(tensors['temperature'].float())
    ].float()
    temperature_sample_quantiles = quantile_dict(
        temperature_sample,
        {'q10': 0.10, 'q50': 0.50, 'q90': 0.90, 'p95': 0.95},
    )

    valid_count = counters['valid']
    risk_routing_counts = {
        'semantics': 'exact native-valid micro counts; high u>q, low u<q, boundary u==q',
        'valid_native_pixels': int(valid_count),
        'teacher_wrong_pixels': int(counters['wrong']),
        'high_risk_pixels': int(counters['high']),
        'high_risk_wrong_pixels': int(counters['high_wrong']),
        'low_risk_pixels': int(counters['low']),
        'low_risk_wrong_pixels': int(counters['low_wrong']),
        'boundary_pixels': int(counters['neutral']),
        'boundary_wrong_pixels': int(counters['neutral_wrong']),
    }
    wrong_rate = ratio(counters['wrong'], valid_count)
    high_wrong_rate = ratio(counters['high_wrong'], counters['high'])
    low_wrong_rate = ratio(counters['low_wrong'], counters['low'])
    high_wrong_recall = ratio(counters['high_wrong'], counters['wrong'])
    fallback_rate = ratio(counters['fallback'], valid_count)
    tie_rate = ratio(counters['tie'], valid_count)
    active_solved_rate = ratio(counters['solved'], counters['active'])
    residual_mean = ratio(counters['residual_sum'], counters['solved'])
    residual_p95 = residual_sample_quantiles['p95']

    route_invariant = all(
        value == 0 for value in counters['route_mismatches'].values()
    )
    cdf_metadata = dict(cdf.metadata)
    cdf_full_scan = (
        cdf_metadata.get('max_images') == 0
        and cdf_metadata.get('processed_images') == cdf_metadata.get('dataset_size')
        and bool(cdf_metadata.get('full_dataset_scan'))
    )
    cdf_nonfinite_zero = cdf_metadata.get('nonfinite_valid_pixels') == 0
    formal_full_run = args.max_images == 0 and processed_images == len(dataset)
    independent_train_seed = (
        args.split != 'train'
        or int(args.seed) != int(cdf_metadata.get('seed', args.seed))
    )
    full_ap = ranking['r_full']['average_precision']
    confidence_ap = ranking['r_confidence']['average_precision']
    checks = {
        'formal_max_images_zero_and_dataset_complete': formal_full_run,
        'cdf_metadata_all_strong_checks_pass': all(cdf_metadata_checks.values()),
        'cdf_was_full_dataset_scan': cdf_full_scan,
        'cdf_nonfinite_valid_pixels_zero': cdf_nonfinite_zero,
        'independent_train_augmentation_seed': independent_train_seed,
        'pre_registered_phase_config_exact': pre_registered_config_matches(args),
        'native_valid_pixels_positive': valid_count > 0,
        'native_nonfinite_valid_pixels_zero': counters['nonfinite_valid'] == 0,
        'ranking_sample_nonempty': int(ranking_valid.sum().item()) > 0,
        'active_solved_pixels_positive': counters['solved'] > 0,
        'active_solved_residual_p95_le_1e_3': (
            residual_p95 is not None and residual_p95 <= 1e-3
        ),
        'fallback_rate_le_configured_limit': (
            fallback_rate is not None
            and fallback_rate <= args.max_fallback_rate
        ),
        'tie_rate_le_configured_limit': (
            tie_rate is not None
            and tie_rate <= args.max_fallback_rate
        ),
        'direction_violations_zero': (
            counters['direction_violations_reliable'] == 0
            and counters['direction_violations_unreliable'] == 0
        ),
        'route_fields_identical_for_tout1_and_tout3': route_invariant,
        'argmax_invariant_to_requested_positive_tout': (
            counters['argmax_disagreements'] == 0
        ),
    }
    if args.phase == 'O1':
        checks.update({
            'high_risk_coverage_ge_0p10': (
                ratio(counters['high'], valid_count) is not None
                and ratio(counters['high'], valid_count) >= 0.10
            ),
            'high_risk_wrong_recall_ge_0p60': (
                high_wrong_recall is not None and high_wrong_recall >= 0.60
            ),
            'high_risk_wrong_rate_ge_2x_global': (
                high_wrong_rate is not None
                and wrong_rate is not None
                and high_wrong_rate >= 2.0 * wrong_rate
            ),
            'wrong_rate_high_gt_low': (
                high_wrong_rate is not None
                and low_wrong_rate is not None
                and high_wrong_rate > low_wrong_rate
            ),
            'full_ap_not_below_conf_by_0p005': (
                full_ap is not None
                and confidence_ap is not None
                and full_ap >= confidence_ap - 0.005
            ),
        })
    else:
        high_coverage = ratio(counters['high'], valid_count)
        checks.update({
            'high_risk_coverage_in_0p15_0p25': (
                high_coverage is not None and 0.15 <= high_coverage <= 0.25
            ),
            'high_risk_wrong_rate_ge_2x_global': (
                high_wrong_rate is not None
                and wrong_rate is not None
                and high_wrong_rate >= 2.0 * wrong_rate
            ),
            'high_risk_wrong_recall_ge_0p70': (
                high_wrong_recall is not None and high_wrong_recall >= 0.70
            ),
            'low_risk_wrong_rate_lt_global': (
                low_wrong_rate is not None
                and wrong_rate is not None
                and low_wrong_rate < wrong_rate
            ),
            'all_ten_risk_quantile_bins_nonempty': (
                len(u_rows) == 10 and all(row['count'] > 0 for row in u_rows)
            ),
            'risk_quantile_pairwise_monotonic_agreement_ge_0p90': (
                u_pairwise_agreement is not None
                and u_pairwise_agreement >= 0.90
            ),
            'fallback_count_zero': counters['fallback'] == 0,
            'tie_count_zero': counters['tie'] == 0,
            'active_solved_residual_p95_lt_1e_3': (
                residual_p95 is not None and residual_p95 < 1e-3
            ),
        })

    commit, dirty, dirty_entries = git_provenance()
    critical_config = {
        'rtc_config': config.to_dict(),
        'reliability_definition': reliability_definition_metadata(
            args.reliability_mode, coefficient_a
        ),
        'teacher_output_temp': float(args.teacher_output_temp),
        'teacher_model': args.teacher_model,
        'teacher_backbone': args.teacher_backbone,
        'teacher_sha256': teacher_sha,
        'num_classes': int(args.num_classes),
        'teacher_output_grid': 'native',
        'valid_mask_resize': 'nearest',
        'native_correctness_semantics': NATIVE_CORRECTNESS_SEMANTICS,
        'route_invariance_temperatures': [1.0, 3.0],
    }
    summary = {
        'schema_version': 1,
        'phase': args.phase,
        'diagnostic_kind': 'rtc_native_routing',
        'primary_reliability_mode': args.reliability_mode,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'split': args.split,
        'processed_images': int(processed_images),
        'dataset_size': int(len(dataset)),
        'formal_full_run': bool(formal_full_run),
        'native_logit_shapes': [list(shape) for shape in sorted(native_logit_shapes)],
        'native_correctness_semantics': NATIVE_CORRECTNESS_SEMANTICS,
        'teacher_error_positive_class': (
            'native-grid prediction differs from nearest-resized ground truth; '
            'nonfinite teacher output is conservatively counted as wrong'
        ),
        'cdf_path': str(Path(args.cdf).resolve()),
        'cdf_sha256': cdf.checksum_sha256,
        'cdf_metadata': cdf_metadata,
        'cdf_metadata_checks': cdf_metadata_checks,
        'critical_config': critical_config,
        'rtc_config': config.to_dict(),
        'teacher_output_temp': float(args.teacher_output_temp),
        'input_provenance': {
            'data_root': str(Path(args.data).resolve()),
            'list_path': str(list_path),
            'list_sha256': current_list_sha,
            'teacher_path': str(teacher_path),
            'teacher_sha256': teacher_sha,
            'dataset_class': type(dataset).__name__,
            'augmentation_seed': int(args.seed),
            'ranking_seed': int(args.ranking_seed),
            'crop_size': list(args.crop_size),
            'scale': args.split == 'train' and not args.no_scale,
            'mirror': args.split == 'train' and not args.no_mirror,
            'val_protocol': (
                'original variable-resolution image; crop_size is not applied'
                if args.split == 'val' else None
            ),
            'argv': list(sys.argv),
            'command_args': dict(vars(args)),
        },
        'code_provenance': {
            'git_commit': commit,
            'git_dirty': dirty,
            'git_dirty_entry_count': dirty_entries,
            'source_sha256': current_sources,
        },
        'population': {
            'statistics_semantics': (
                'exact micro counts over all native valid pixels; no per-image cap'
            ),
            'valid_native_pixels': int(valid_count),
            'finite_valid_pixels': int(counters['finite_valid']),
            'nonfinite_valid_pixels': int(counters['nonfinite_valid']),
            'nonfinite_valid_rate': ratio(counters['nonfinite_valid'], valid_count),
            'active_pixels': int(counters['active']),
            'active_rate': ratio(counters['active'], valid_count),
            'solved_active_pixels': int(counters['solved']),
            'active_solved_rate': active_solved_rate,
            'active_fallback_pixels': int(counters['active_fallback']),
        },
        'ranking_sample': {
            'sampling_semantics': (
                'uniform without replacement per image among all native valid pixels; '
                'finite-score pixels are used by AP/AUC'
            ),
            'max_pixels_per_image': int(args.ranking_max_pixels_per_image),
            'seed': int(args.ranking_seed),
            'sampled_native_valid_pixels': int(tensors['u'].numel()),
            'finite_ranking_pixels': int(ranking_valid.sum().item()),
            'finite_ranking_wrong_pixels': int(ranking_wrong.sum().item()),
        },
        'teacher_accuracy': ratio(counters['correct'], valid_count),
        'global_wrong_rate': wrong_rate,
        'high_risk_coverage': ratio(counters['high'], valid_count),
        'low_risk_coverage': ratio(counters['low'], valid_count),
        'neutral_quantile_coverage': ratio(counters['neutral'], valid_count),
        'risk_routing_counts': risk_routing_counts,
        'high_risk_wrong_precision': high_wrong_rate,
        'high_risk_wrong_recall': high_wrong_recall,
        'high_risk_enrichment': (
            high_wrong_rate / wrong_rate
            if high_wrong_rate is not None and wrong_rate not in (None, 0.0)
            else None
        ),
        'low_risk_wrong_rate': low_wrong_rate,
        'ranking': ranking,
        'primary_score_sample_summary': r0_summary,
        'r0_sample_summary': r0_summary,
        'risk_quantile_bins': u_rows,
        'risk_quantile_spearman': u_spearman,
        'risk_quantile_pairwise_monotonic_agreement': (
            u_pairwise_agreement
        ),
        'gates': {
            'coverage_reliable': ratio(
                counters['gate_reliable_positive'], valid_count
            ),
            'coverage_unreliable': ratio(
                counters['gate_unreliable_positive'], valid_count
            ),
            'mean_reliable': ratio(counters['gate_reliable_sum'], valid_count),
            'mean_unreliable': ratio(counters['gate_unreliable_sum'], valid_count),
        },
        'temperature': {
            'mean': ratio(counters['temperature_sum'], valid_count),
            'harmonic_mean': (
                valid_count / counters['temperature_inverse_sum']
                if counters['temperature_inverse_sum'] > 0 else None
            ),
            **temperature_sample_quantiles,
            'quantile_population': 'controlled native-valid ranking sample',
            'min': counters['temperature_min'],
            'max': counters['temperature_max'],
            'reliable_endpoint_rate': ratio(
                counters['temperature_reliable_endpoint'], valid_count
            ),
            'neutral_endpoint_rate': ratio(
                counters['temperature_neutral_endpoint'], valid_count
            ),
            'unreliable_endpoint_rate': ratio(
                counters['temperature_unreliable_endpoint'], valid_count
            ),
            'effective_teacher_mean': (
                ratio(counters['temperature_sum'], valid_count)
                * args.teacher_output_temp
            ),
        },
        'target_residual': {
            'population': 'active solved nonfallback native valid pixels only',
            'count': int(counters['solved']),
            'mean': residual_mean,
            'p95': residual_p95,
            'p95_population': 'controlled sample restricted to active solved pixels',
            'sample_count': int(residual_sample.numel()),
            'max': counters['residual_max'],
        },
        'fallback_rate': fallback_rate,
        'fallback_count': int(counters['fallback']),
        'tie_rate': tie_rate,
        'tie_count': int(counters['tie']),
        'direction_violations_reliable': int(
            counters['direction_violations_reliable']
        ),
        'direction_violations_unreliable': int(
            counters['direction_violations_unreliable']
        ),
        'argmax_disagreements_after_teacher_output_temp': int(
            counters['argmax_disagreements']
        ),
        'route_invariance_tout1_vs_tout3': {
            'comparison': (
                'build_rtc_temperature_map(raw, raw/T) is run at T=1 and T=3; '
                'r0, u, and both gates are compared over every native valid pixel'
            ),
            'all_route_fields_identical': route_invariant,
            'mismatch_counts': counters['route_mismatches'],
            'max_abs_diff': counters['route_max_abs_diff'],
        },
        'checks': checks,
        'all_checks_pass': all(checks.values()),
    }

    if args.output:
        output_path = Path(args.output)
    elif args.phase == 'O1.1':
        output_path = (
            ROOT / 'runs' / 'diagnostics' / 'phaseO_o11'
            / f'rtc_confidence_routing_{args.split}.json'
        )
    else:
        output_path = (
            ROOT / 'runs' / 'diagnostics' / 'phaseO'
            / f'rtc_routing_{args.split}.json'
        )
    u_csv_path = output_path.with_suffix('.bins.csv')
    score_csv_path = output_path.with_suffix(
        '.primary_deciles.csv' if args.phase == 'O1.1' else '.r0_deciles.csv'
    )
    existing = [
        str(path) for path in (output_path, u_csv_path, score_csv_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            'Refusing to overwrite routing diagnostic artifacts: '
            + ', '.join(existing)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    with u_csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(u_rows[0].keys()))
        writer.writeheader()
        writer.writerows(u_rows)
    r0_csv_path = score_csv_path
    r0_fieldnames = (
        list(r0_rows[0].keys())
        if r0_rows
        else [
            'bin',
            'sample_population',
            'count',
            'r0_low',
            'r0_high',
            'r0_mean',
            'confidence_mean',
            'variance_mean',
            'teacher_accuracy_native_proxy',
            'teacher_wrong_rate_native_proxy',
        ]
    )
    with r0_csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=r0_fieldnames)
        writer.writeheader()
        writer.writerows(r0_rows)

    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print(f'wrote={output_path}', flush=True)
    print(f'wrote={u_csv_path}', flush=True)
    print(f'wrote={r0_csv_path}', flush=True)
    if args.strict and not summary['all_checks_pass']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
