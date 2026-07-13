#!/usr/bin/env python3
"""Solve and diagnose the frozen Phase O1.2 budgeted calibration mechanism."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import sys
from typing import Any

import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.voc import VOCDataTrainSet, VOCDataValSet
from models.model_zoo import get_segmentation_model
from utils.rtc_temperature import (
    compute_reference_reliability,
    file_sha256,
    load_frozen_reliability_cdf,
)
from utils.rtc_o12_calibration import (
    O12CalibrationConfig,
    build_o12_teacher_target,
    build_o12_temperature_map,
    o12_teacher_only_pixel_kl,
    solve_o12_budget_parameter,
)


PHASE = 'O1.2'
DEFAULT_DIR = ROOT / 'runs' / 'diagnostics' / 'phaseO_o12'
CDF_PATH = (
    ROOT / 'runs' / 'diagnostics' / 'phaseO_o11'
    / 'voc_train_rtc_confidence_cdf.pt'
).resolve()
CDF_SHA256 = (
    '8ba8376a03c2f93835339d98670fa357b381b23a22cbf2a7fc48b0c2441d7e69'
)
O11_GATE_PATH = (
    ROOT / 'runs' / 'diagnostics' / 'phaseO_o11'
    / 'o11_confidence_gate.json'
).resolve()
TEACHER_SHA256 = (
    'ac49b2c7720b21d565072e974e4404fcb009ba106288d95d1a4bb25f09c3fe58'
)
LIST_SHA256 = {
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
SOURCE_PATHS = {
    'rtc_o12_calibration': ROOT / 'utils' / 'rtc_o12_calibration.py',
    'diagnose_rtc_o12_budget': Path(__file__).resolve(),
    'check_rtc_o12_gate': ROOT / 'scripts' / 'diagnostics' / 'check_rtc_o12_gate.py',
    'train_entry': ROOT / 'train_kd.py',
}
CONTRACT = {
    'spatial_temperature_applies_to': 'teacher_target_only',
    'student_temperature': 1.0,
    'teacher_target_detached': True,
    'spatial_temperature_loss_power': 'not_applicable',
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


def configuration() -> dict[str, Any]:
    config = O12CalibrationConfig()
    config.validate()
    return {
        'phase': PHASE,
        'reliability_mode': 'confidence',
        'reliability_definition_id': 'neg_log_top1_confidence_v1',
        'assess_temperature': 1.0,
        'epsilon': 1e-8,
        'q_reliable': config.q_reliable,
        'q_unreliable': config.q_unreliable,
        'p_reliable': config.p_reliable,
        'p_unreliable': config.p_unreliable,
        'a': config.a_star,
        'b_min': 0.0,
        'b_max': config.b_max,
        'target_arithmetic_mean': config.target_mean,
        'minimum_harmonic_mean': config.min_harmonic_mean,
        'bisection_iterations': config.bisection_iterations,
        'teacher_output_temperature': config.teacher_output_temperature,
        'temperature_map_dtype': 'float32',
        'budget_accumulator_dtype': 'float64',
        'temperature_minimum': config.temperature_min,
        'temperature_maximum': config.temperature_max,
        'tau_temperature': 1e-6,
        'tau_probability': 1e-6,
        'tau_entropy': 1e-6,
        'tau_student': 1e-7,
        'tau_formula_monotonic': 1e-12,
    }


def scan_protocol(args):
    return {
        'stage': args.stage,
        'split': args.split,
        'teacher_model': args.teacher_model,
        'teacher_backbone': args.teacher_backbone,
        'num_classes': args.num_classes,
        'ignore_label': args.ignore_label,
        'crop_size': list(args.crop_size),
        'scale': args.split == 'train' and not args.no_scale,
        'mirror': args.split == 'train' and not args.no_mirror,
        'batch_size': args.batch_size,
        'workers': args.workers,
        'augmentation_seed': args.seed,
        'max_images': args.max_images,
        'device_type': 'npu',
        'world_size': int(os.environ.get('WORLD_SIZE', '1')),
        'rank': int(os.environ.get('RANK', '0')),
        'process_mode': 'single_process_single_npu',
        'teacher_output_grid': 'native_logits',
        'valid_mask_resize': 'nearest',
        'cdf_query_dtype': 'float32',
    }


def canonical_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), allow_nan=False
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['solve', 'evaluate'], required=True)
    parser.add_argument('--split', choices=['train', 'val'], required=True)
    parser.add_argument('--data', default=str(ROOT / 'dataset' / 'VOCAug'))
    parser.add_argument('--list-path', default=None)
    parser.add_argument('--cdf', default=str(CDF_PATH))
    parser.add_argument('--o11-gate', default=str(O11_GATE_PATH))
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
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument(
        '--max-images', '--max-samples', dest='max_images', type=int, default=0
    )
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--no-scale', action='store_true', default=False)
    parser.add_argument('--no-mirror', action='store_true', default=False)
    parser.add_argument('--parameters', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--risk-cache', default=None)
    parser.add_argument('--strict', action='store_true', default=False)
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


def runtime_identity(args, device):
    current_device = int(torch.npu.current_device())
    requested_npu_index = device.index
    if requested_npu_index is None:
        raise ValueError('O1.2 scans require an explicit NPU device index')
    requested_npu_index = int(requested_npu_index)
    get_device_name = getattr(torch.npu, 'get_device_name', None)
    device_name = (
        str(get_device_name(current_device))
        if callable(get_device_name) else None
    )
    return {
        'process_id': int(os.getpid()),
        'world_size': int(os.environ.get('WORLD_SIZE', '1')),
        'rank': int(os.environ.get('RANK', '0')),
        'local_rank': int(os.environ.get('LOCAL_RANK', '0')),
        'requested_device': str(args.device),
        'requested_npu_index': requested_npu_index,
        'resolved_device': str(device),
        'device_type': device.type,
        'npu_current_device': current_device,
        'npu_device_name': device_name,
        'ascend_rt_visible_devices': os.environ.get(
            'ASCEND_RT_VISIBLE_DEVICES'
        ),
    }


def git_provenance():
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=normal'],
            cwd=ROOT, text=True,
        )
        return commit, bool(status.strip()), len(status.splitlines())
    except Exception:
        return 'unknown', None, None


def source_sha256():
    result = {}
    for name, path in SOURCE_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f'missing required O1.2 source: {path}')
        result[name] = file_sha256(path)
    return result


def load_json(path):
    with Path(path).open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f'JSON root must be an object: {path}')
    return payload


def resolve_list_path(args):
    if args.list_path:
        return Path(args.list_path).resolve()
    filename = 'train_aug.txt' if args.split == 'train' else 'val.txt'
    return (ROOT / 'dataset' / 'list' / 'voc' / filename).resolve()


def build_dataset(args, list_path):
    if args.split == 'train':
        return VOCDataTrainSet(
            args.data, str(list_path), max_iters=None,
            crop_size=tuple(args.crop_size),
            scale=not args.no_scale, mirror=not args.no_mirror,
            ignore_label=args.ignore_label,
        )
    return VOCDataValSet(
        args.data, str(list_path), crop_size=tuple(args.crop_size),
        ignore_label=args.ignore_label,
    )


def validate_frozen_inputs(args, list_path):
    if args.stage == 'solve' and args.split != 'train':
        raise ValueError('stage solve is defined only for split train')
    if args.max_images < 0:
        raise ValueError('--max-images must be nonnegative')
    if not str(args.device).startswith('npu'):
        raise ValueError('O1.2 scans are frozen to one NPU; CPU/CUDA are forbidden')
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    if world_size != 1 or rank != 0:
        raise ValueError(
            'O1.2 budget scans require a single process (WORLD_SIZE=1,RANK=0)'
        )
    expected_batch = 4 if args.split == 'train' else 1
    if args.batch_size is None:
        args.batch_size = expected_batch
    if args.batch_size != expected_batch:
        raise ValueError(f'{args.split} batch size is frozen at {expected_batch}')
    if (
        args.workers != 0 or args.seed != 2025 or args.num_classes != 21
        or list(args.crop_size) != [512, 512]
        or args.teacher_model != 'deeplabv3'
        or args.teacher_backbone != 'resnet101'
    ):
        raise ValueError('O1.2 frozen scan configuration mismatch')
    if args.split == 'train' and (args.no_scale or args.no_mirror):
        raise ValueError('formal O1.2 train requires scale=true and mirror=true')
    cdf_path = Path(args.cdf).resolve()
    gate_path = Path(args.o11_gate).resolve()
    teacher_path = Path(args.teacher_pretrained).resolve()
    if cdf_path != CDF_PATH or file_sha256(cdf_path) != CDF_SHA256:
        raise ValueError('frozen O1.1 confidence CDF path/SHA mismatch')
    gate = load_json(gate_path)
    if (
        gate_path != O11_GATE_PATH or gate.get('phase') != 'O1.1'
        or gate.get('joint_gate_pass') is not True
        or gate.get('actual_cdf_sha256') != CDF_SHA256
    ):
        raise ValueError('frozen O1.1 joint gate is missing or no longer true')
    if file_sha256(teacher_path) != TEACHER_SHA256:
        raise ValueError('teacher SHA mismatch')
    list_sha = file_sha256(list_path)
    if list_sha != LIST_SHA256[args.split]:
        raise ValueError(f'{args.split} list SHA mismatch')
    for name, path in O11_SOURCE_PATHS.items():
        if file_sha256(path) != FROZEN_O11_SOURCE_SHA256[name]:
            raise ValueError(f'frozen O1.1 source changed: {name}')
    return cdf_path, gate_path, teacher_path, list_sha, gate


def output_paths(args):
    formal = args.max_images == 0
    if formal:
        if args.stage == 'solve':
            output = Path(args.output or DEFAULT_DIR / 'o12_budget_parameters.json')
            cache = Path(args.risk_cache or DEFAULT_DIR / 'o12_budget_train_solve_u.npy')
        else:
            output = Path(args.output or DEFAULT_DIR / f'o12_budget_{args.split}.json')
            cache = Path(
                args.risk_cache
                or DEFAULT_DIR / f'o12_budget_{args.split}_evaluate_u.npy'
            )
    else:
        prefix = Path('/tmp') / f'phaseO_o12_smoke_{args.stage}_{args.split}'
        output = Path(args.output or f'{prefix}.json')
        cache = Path(args.risk_cache or f'{prefix}_u.npy')
        canonical = DEFAULT_DIR.resolve()
        for path in (output.resolve(), cache.resolve()):
            if canonical == path or canonical in path.parents:
                raise ValueError('partial smoke cannot write canonical O1.2 paths')
    return output.resolve(), cache.resolve(), formal


class RiskCacheWriter:
    def __init__(self, path, formal, expected_count):
        self.path = Path(path)
        self.partial = self.path.with_name(self.path.name + '.partial')
        if self.path.exists() or self.partial.exists():
            raise FileExistsError(f'refusing to overwrite risk cache: {self.path}')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.formal = formal
        self.expected_count = expected_count
        self.offset = 0
        self.chunks = []
        self.array = (
            np.lib.format.open_memmap(
                self.partial, mode='w+', dtype=np.float32,
                shape=(expected_count,),
            )
            if formal else None
        )

    def append(self, values):
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if self.array is None:
            self.chunks.append(values.copy())
        else:
            end = self.offset + int(values.size)
            if end > self.expected_count:
                raise RuntimeError('risk cache exceeded frozen native-valid population')
            self.array[self.offset:end] = values
        self.offset += int(values.size)

    def finalize(self):
        if self.formal and self.offset != self.expected_count:
            raise RuntimeError(
                f'native-valid population {self.offset} != {self.expected_count}'
            )
        if self.array is not None:
            self.array.flush()
            del self.array
        else:
            values = np.concatenate(self.chunks) if self.chunks else np.empty(0, np.float32)
            with self.partial.open('wb') as handle:
                np.save(handle, values, allow_pickle=False)
        os.replace(self.partial, self.path)
        return {
            'path': str(self.path), 'sha256': file_sha256(self.path),
            'dtype': 'float32', 'count': self.offset,
            'semantics': 'right-continuous O1.1 confidence CDF u over native-valid pixels in scan order',
        }


def setup_scan(args, list_path, teacher_path, cdf_path):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == 'npu':
        torch.npu.manual_seed_all(args.seed)
    dataset = build_dataset(args, list_path)
    loader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False,
    )
    teacher = get_segmentation_model(
        model=args.teacher_model, backbone=args.teacher_backbone,
        local_rank=0, pretrained_base='None', pretrained=str(teacher_path),
        aux=True, norm_layer=nn.BatchNorm2d, num_class=args.num_classes,
    ).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    cdf = load_frozen_reliability_cdf(cdf_path, device=device)
    metadata = dict(cdf.metadata)
    if (
        metadata.get('phase') != 'O1.1'
        or metadata.get('reliability_mode') != 'confidence'
        or float(metadata.get('coefficient_a', -1)) != 0.0
        or metadata.get('teacher_sha256') != TEACHER_SHA256
    ):
        raise ValueError('O1.1 CDF metadata definition mismatch')
    source_meta = metadata.get('source_sha256')
    if not isinstance(source_meta, dict):
        raise ValueError('O1.1 CDF source metadata missing')
    for name, expected in FROZEN_O11_SOURCE_SHA256.items():
        if source_meta.get(name) != expected:
            raise ValueError(f'O1.1 CDF source metadata mismatch: {name}')
    return device, dataset, loader, teacher, cdf, metadata


def scan_risk(args, loader, teacher, cdf, device, writer):
    processed = valid_count = finite_count = 0
    shapes = set()
    histogram = {}
    with torch.no_grad():
        for images, targets, _ in loader:
            if args.max_images:
                remaining = args.max_images - processed
                if remaining <= 0:
                    break
                images, targets = images[:remaining], targets[:remaining]
            images = images.to(device)
            targets = targets.long().to(device)
            output = teacher(images)
            raw = output[0] if isinstance(output, (list, tuple)) else output
            shapes.add(tuple(int(value) for value in raw.shape[-2:]))
            maps = compute_reference_reliability(
                raw, targets != args.ignore_label,
                assess_temperature=1.0, coefficient_a=0.0,
                reliability_mode='confidence', epsilon=1e-8,
            )
            u = cdf.query(maps.reliability)
            selected = u[maps.valid_mask].detach().float().cpu().numpy()
            writer.append(selected)
            _histogram_add(histogram, selected)
            valid_count += int(maps.valid_mask.sum().item())
            finite_count += int((maps.valid_mask & maps.finite_mask).sum().item())
            processed += int(images.shape[0])
            if args.log_every and processed % args.log_every < images.shape[0]:
                print(
                    f'stage={args.stage} split={args.split} images={processed} '
                    f'native_valid={valid_count}', flush=True
                )
            if args.max_images and processed >= args.max_images:
                break
    return {
        'processed_images': processed,
        'valid_native_pixels': valid_count,
        'finite_native_pixels': finite_count,
        'nonfinite_native_pixels': valid_count - finite_count,
        'native_logit_shapes': [list(shape) for shape in sorted(shapes)],
    }, histogram


def solve_from_cache(cache_path):
    values = np.load(cache_path, mmap_mode='r', allow_pickle=False)
    exact_u = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
    solution = solve_o12_budget_parameter(exact_u)
    payload = solution.to_dict()
    payload.update({
        'feasible': True,
        'iterations': solution.bisection_iterations,
        'population': solution.valid_count,
        'mean': solution.arithmetic_mean,
        'harmonic_mean': solution.harmonic_mean,
        'mean_residual': solution.residual,
        'theoretical_high_risk_endpoint': math.exp(solution.b),
        'solver_input': 'exact float32 risk cache; no histogram approximation',
    })
    return payload


def _histogram_add(histogram, values):
    unique, counts = np.unique(values.astype(np.float32), return_counts=True)
    for value, count in zip(unique.tolist(), counts.tolist()):
        key = float(np.float32(value))
        histogram[key] = histogram.get(key, 0) + int(count)


def weighted_quantile(values, counts, q):
    total = int(counts.sum())
    order = np.argsort(values, kind='stable')
    values, counts = values[order], counts[order]
    cumulative = np.cumsum(counts)
    position = (total - 1) * q
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    lo = int(np.searchsorted(cumulative, lower + 1))
    hi = int(np.searchsorted(cumulative, upper + 1))
    weight = position - lower
    return float((1 - weight) * values[lo] + weight * values[hi])


def temperature_summary(histogram, b, branch='full_budgeted', scale=1.0):
    items = sorted(histogram.items())
    u = torch.tensor([item[0] for item in items], dtype=torch.float32)
    counts = np.asarray([item[1] for item in items], dtype=np.int64)
    temperature = build_o12_temperature_map(
        u, b=b, branch=branch
    ).cpu().numpy().astype(np.float32)
    temperature = (
        temperature * np.float32(scale)
    ).astype(np.float32, copy=False)
    count = int(counts.sum())
    mean = float(np.dot(temperature.astype(np.float64), counts) / count)
    inverse = float(np.dot(1.0 / temperature.astype(np.float64), counts) / count)
    result = {
        'count': count, 'min': float(temperature.min()),
        'max': float(temperature.max()), 'mean': mean,
        'harmonic_mean': 1.0 / inverse,
    }
    for name, q in (
        ('q01', .01), ('q10', .1), ('q50', .5), ('q80', .8),
        ('q90', .9), ('q95', .95), ('q99', .99),
    ):
        result[name] = weighted_quantile(temperature, counts, q)
    u_np = u.numpy()
    top = u_np >= np.float32(.9)
    result['top_risk_decile_mean'] = float(
        np.dot(temperature[top].astype(np.float64), counts[top])
        / int(counts[top].sum())
    )
    result['less_than_0p9_count'] = int(counts[temperature < np.float32(.9)].sum())
    result['equal_1_count'] = int(counts[temperature == np.float32(1)].sum())
    result['greater_than_1_count'] = int(counts[temperature > np.float32(1)].sum())
    result['greater_than_1p25_count'] = int(counts[temperature > np.float32(1.25)].sum())
    result['greater_than_1p5_count'] = int(counts[temperature > np.float32(1.5)].sum())
    for key in (
        'less_than_0p9_count', 'equal_1_count', 'greater_than_1_count',
        'greater_than_1p25_count', 'greater_than_1p5_count',
    ):
        result[key.replace('_count', '_coverage')] = (
            result[key] / count if count else None
        )
    return result


def empty_bucket():
    return {
        'count': 0, 'teacher_wrong_count': 0, 'temperature_sum': 0.0,
        'base_confidence_sum': 0.0, 'target_confidence_sum': 0.0,
        'base_entropy_sum': 0.0, 'target_entropy_sum': 0.0,
    }


def update_bucket(bucket, mask, wrong, temperature, c_base, c_target, h_base, h_target):
    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())
    bucket['count'] += count
    if not count:
        return
    bucket['teacher_wrong_count'] += int(np.asarray(wrong)[mask].sum())
    bucket['temperature_sum'] += float(np.asarray(temperature)[mask].astype(np.float64).sum())
    bucket['base_confidence_sum'] += float(np.asarray(c_base)[mask].astype(np.float64).sum())
    bucket['target_confidence_sum'] += float(np.asarray(c_target)[mask].astype(np.float64).sum())
    bucket['base_entropy_sum'] += float(np.asarray(h_base)[mask].astype(np.float64).sum())
    bucket['target_entropy_sum'] += float(np.asarray(h_target)[mask].astype(np.float64).sum())


def finalize_bucket(bucket):
    count = bucket['count']
    result = {
        'count': count,
        'teacher_wrong_count': bucket['teacher_wrong_count'],
        'teacher_wrong_rate': (
            bucket['teacher_wrong_count'] / count if count else None
        ),
    }
    for source, target in (
        ('temperature_sum', 'temperature_mean'),
        ('base_confidence_sum', 'c_base_mean'),
        ('target_confidence_sum', 'c_target_mean'),
        ('base_entropy_sum', 'entropy_base_mean'),
        ('target_entropy_sum', 'entropy_target_mean'),
    ):
        result[target] = bucket[source] / count if count else None
    return result


def boundary_and_small_masks(target, valid):
    batch, height, width = target.shape
    boundary = np.zeros_like(valid, dtype=bool)
    small = np.zeros_like(valid, dtype=bool)
    structure = np.ones((3, 3), dtype=np.uint8)
    for image in range(batch):
        label = target[image]
        mask = valid[image]
        core = np.zeros((height, width), dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                y0, y1 = max(0, -dy), min(height, height - dy)
                x0, x1 = max(0, -dx), min(width, width - dx)
                ny0, ny1 = y0 + dy, y1 + dy
                nx0, nx1 = x0 + dx, x1 + dx
                center_valid = mask[y0:y1, x0:x1]
                neighbor_valid = mask[ny0:ny1, nx0:nx1]
                different = label[y0:y1, x0:x1] != label[ny0:ny1, nx0:nx1]
                core[y0:y1, x0:x1] |= center_valid & neighbor_valid & different
        boundary[image] = ndimage.binary_dilation(
            core, structure=np.ones((5, 5), dtype=bool)
        ) & mask
        threshold = float(mask.sum()) * .005
        for class_id in np.unique(label[mask]):
            if int(class_id) <= 0:
                continue
            components, number = ndimage.label(
                (label == class_id) & mask, structure=structure
            )
            if number:
                areas = np.bincount(components.reshape(-1))
                ids = np.flatnonzero((areas < threshold) & (np.arange(areas.size) > 0))
                if ids.size:
                    small[image] |= np.isin(components, ids)
    return boundary, small


def scan_evaluation(args, loader, teacher, cdf, device, writer, b):
    config = O12CalibrationConfig()
    histogram = {}
    counters = {
        'processed_images': 0, 'valid': 0, 'finite': 0, 'wrong': 0,
        'regions': {name: empty_bucket() for name in ('reliable', 'neutral', 'unreliable')},
        'bins': [empty_bucket() for _ in range(10)],
        'classes': [empty_bucket() for _ in range(21)],
        'fgbg': {name: empty_bucket() for name in ('background', 'foreground')},
        'boundary': {name: empty_bucket() for name in ('boundary', 'interior')},
        'small': {name: empty_bucket() for name in ('small_object', 'not_small_object')},
        'violations': {name: 0 for name in VIOLATION_KEYS},
        'u_eq_0p6': 0, 'u_eq_0p8': 0,
        'u_eq_0p6_wrong': 0, 'u_eq_0p8_wrong': 0,
        'neutral_max_abs_error': 0.0, 'student_max_abs_error': 0.0,
        'teacher_probe_max_abs_difference': 0.0,
        'shapes': set(),
    }
    with torch.no_grad():
        for images, targets, _ in loader:
            if args.max_images:
                remaining = args.max_images - counters['processed_images']
                if remaining <= 0:
                    break
                images, targets = images[:remaining], targets[:remaining]
            images = images.to(device)
            targets = targets.long().to(device)
            output = teacher(images)
            raw = output[0] if isinstance(output, (list, tuple)) else output
            counters['shapes'].add(tuple(int(v) for v in raw.shape[-2:]))
            maps = compute_reference_reliability(
                raw, targets != args.ignore_label, assess_temperature=1.0,
                coefficient_a=0.0, reliability_mode='confidence', epsilon=1e-8,
            )
            u = cdf.query(maps.reliability)
            valid = maps.valid_mask
            finite = maps.finite_mask
            selected_u = u[valid].detach().float().cpu().numpy()
            writer.append(selected_u)
            _histogram_add(histogram, selected_u)

            temperature = build_o12_temperature_map(
                u, valid, b=b, branch='full_budgeted', config=config
            )
            safe_raw = torch.where(
                torch.isfinite(raw), raw, torch.zeros_like(raw)
            )
            class_bias = torch.linspace(
                -.1, .1, raw.shape[1], device=raw.device, dtype=raw.dtype
            ).view(1, -1, 1, 1)
            surrogate_student_logits = safe_raw * .37 + class_bias
            actual_kl_probe = o12_teacher_only_pixel_kl(
                surrogate_student_logits,
                safe_raw,
                temperature,
                valid,
                teacher_output_temperature=config.teacher_output_temperature,
            )
            neutral_kl_probe = o12_teacher_only_pixel_kl(
                surrogate_student_logits, safe_raw, torch.ones_like(temperature),
                valid,
                teacher_output_temperature=config.teacher_output_temperature,
            )
            target_probability = actual_kl_probe.teacher_target
            base_probability = neutral_kl_probe.teacher_target
            c_target = target_probability.max(dim=1).values
            c_base = base_probability.max(dim=1).values
            entropy_target = -(
                target_probability * target_probability.clamp_min(1e-12).log()
            ).sum(dim=1)
            entropy_base = -(
                base_probability * base_probability.clamp_min(1e-12).log()
            ).sum(dim=1)
            target_native = F.interpolate(
                targets.float().unsqueeze(1), size=raw.shape[-2:], mode='nearest'
            ).squeeze(1).long()
            prediction = raw.argmax(dim=1)
            wrong = valid & ((prediction != target_native) | ~finite)
            reliable = valid & (u < .6)
            neutral = valid & (u >= .6) & (u <= .8)
            unreliable = valid & (u > .8)
            tau_t, tau_p, tau_h = 1e-6, 1e-6, 1e-6
            violation = counters['violations']
            violation['nonfinite'] += int((valid & ~finite).sum().item())
            violation['reliable_temperature_above_one'] += int((reliable & (temperature > 1 + tau_t)).sum().item())
            violation['neutral_temperature_nonunit'] += int((neutral & ((temperature - 1).abs() > tau_t)).sum().item())
            violation['unreliable_temperature_below_one'] += int((unreliable & (temperature < 1 - tau_t)).sum().item())
            violation['reliable_confidence_decrease'] += int((reliable & (c_target < c_base - tau_p)).sum().item())
            violation['unreliable_confidence_increase'] += int((unreliable & (c_target > c_base + tau_p)).sum().item())
            violation['reliable_entropy_increase'] += int((reliable & (entropy_target > entropy_base + tau_h)).sum().item())
            violation['unreliable_entropy_decrease'] += int((unreliable & (entropy_target < entropy_base - tau_h)).sum().item())
            violation['teacher_argmax_mismatch'] += int((valid & (target_probability.argmax(1) != prediction)).sum().item())
            violation['temperature_out_of_bounds'] += int((valid & ((temperature < .9 - tau_t) | (temperature > 1.5 + tau_t))).sum().item())
            neutral_error = (
                (target_probability - base_probability).abs().amax(dim=1)
            )
            violation['neutral_target_mismatch'] += int((neutral & (neutral_error > tau_p)).sum().item())
            if bool(neutral.any().item()):
                counters['neutral_max_abs_error'] = max(
                    counters['neutral_max_abs_error'],
                    float(neutral_error[neutral].max().item()),
                )
            student_a = actual_kl_probe.student_probability
            student_b = neutral_kl_probe.student_probability
            student_error = (student_a - student_b).abs().amax(dim=1)
            teacher_probe_error = (
                actual_kl_probe.teacher_target - neutral_kl_probe.teacher_target
            ).abs().amax(dim=1)
            violation['student_softmax_changed'] += int((valid & (student_error > 1e-7)).sum().item())
            if bool(valid.any().item()):
                counters['student_max_abs_error'] = max(
                    counters['student_max_abs_error'],
                    float(student_error[valid].max().item()),
                )
                counters['teacher_probe_max_abs_difference'] = max(
                    counters['teacher_probe_max_abs_difference'],
                    float(teacher_probe_error[valid].max().item()),
                )

            cpu = {
                'valid': valid.cpu().numpy(), 'finite': finite.cpu().numpy(),
                'wrong': wrong.cpu().numpy(), 'u': u.float().cpu().numpy(),
                'temperature': temperature.float().cpu().numpy(),
                'c_base': c_base.float().cpu().numpy(),
                'c_target': c_target.float().cpu().numpy(),
                'h_base': entropy_base.float().cpu().numpy(),
                'h_target': entropy_target.float().cpu().numpy(),
                'target': target_native.cpu().numpy(),
            }
            boundary, small = boundary_and_small_masks(cpu['target'], cpu['valid'])
            valid_np, u_np, wrong_np = cpu['valid'], cpu['u'], cpu['wrong']
            common = (
                wrong_np, cpu['temperature'], cpu['c_base'], cpu['c_target'],
                cpu['h_base'], cpu['h_target'],
            )
            region_masks = {
                'reliable': valid_np & (u_np < .6),
                'neutral': valid_np & (u_np >= .6) & (u_np <= .8),
                'unreliable': valid_np & (u_np > .8),
            }
            for name, mask in region_masks.items():
                update_bucket(counters['regions'][name], mask, *common)
            bins = np.minimum(np.floor(u_np * 10).astype(np.int64), 9)
            for index in range(10):
                update_bucket(counters['bins'][index], valid_np & (bins == index), *common)
            for class_id in range(21):
                update_bucket(counters['classes'][class_id], valid_np & (cpu['target'] == class_id), *common)
            update_bucket(counters['fgbg']['background'], valid_np & (cpu['target'] == 0), *common)
            update_bucket(counters['fgbg']['foreground'], valid_np & (cpu['target'] != 0), *common)
            update_bucket(counters['boundary']['boundary'], boundary, *common)
            update_bucket(counters['boundary']['interior'], valid_np & ~boundary, *common)
            update_bucket(counters['small']['small_object'], small, *common)
            update_bucket(counters['small']['not_small_object'], valid_np & ~small, *common)
            equal_0p6 = valid_np & (u_np == np.float32(.6))
            equal_0p8 = valid_np & (u_np == np.float32(.8))
            counters['u_eq_0p6'] += int(equal_0p6.sum())
            counters['u_eq_0p8'] += int(equal_0p8.sum())
            counters['u_eq_0p6_wrong'] += int(
                (equal_0p6 & wrong_np).sum()
            )
            counters['u_eq_0p8_wrong'] += int(
                (equal_0p8 & wrong_np).sum()
            )
            counters['valid'] += int(valid.sum().item())
            counters['finite'] += int((valid & finite).sum().item())
            counters['wrong'] += int(wrong.sum().item())
            counters['processed_images'] += int(images.shape[0])
            if args.log_every and counters['processed_images'] % args.log_every < images.shape[0]:
                print(
                    f'stage=evaluate split={args.split} '
                    f'images={counters["processed_images"]} '
                    f'native_valid={counters["valid"]}', flush=True
                )
            if args.max_images and counters['processed_images'] >= args.max_images:
                break
    return counters, histogram


def risk_evidence(regions, bins, total_valid, total_wrong):
    high = regions['unreliable']
    high_count = high['count']
    high_wrong = high['teacher_wrong_count']
    global_rate = total_wrong / total_valid if total_valid else None
    high_rate = high_wrong / high_count if high_count else None
    rates = [
        row['teacher_wrong_count'] / row['count'] if row['count'] else None
        for row in bins
    ]
    pairs = [
        (rates[left], rates[right])
        for left in range(len(rates))
        for right in range(left + 1, len(rates))
        if rates[left] is not None and rates[right] is not None
    ]
    pairwise = (
        sum(left <= right for left, right in pairs) / len(pairs)
        if pairs else None
    )
    finite_rows = [
        (index, rate) for index, rate in enumerate(rates) if rate is not None
    ]
    spearman = (
        float(np.corrcoef(
            [row[0] for row in finite_rows],
            np.argsort(np.argsort(
                [row[1] for row in finite_rows], kind='stable'
            ), kind='stable'),
        )[0, 1])
        if len(finite_rows) >= 2 else None
    )
    return {
        'global_teacher_wrong_rate': global_rate,
        'high_risk_coverage': high_count / total_valid if total_valid else None,
        'high_risk_teacher_wrong_precision': high_rate,
        'high_risk_teacher_wrong_recall': high_wrong / total_wrong if total_wrong else None,
        'high_risk_enrichment': (
            high_rate / global_rate if high_rate is not None and global_rate else None
        ),
        'risk_quantile_wrong_rates': rates,
        'risk_quantile_pairwise_monotonic_agreement': pairwise,
        'risk_quantile_spearman_approx_tie_stable': spearman,
        'semantics': 'O1.1 evidence rechecked for reporting only; not an O1.2 temperature gate',
    }


def common_provenance(args, list_path, list_sha, teacher_path, cdf, metadata):
    commit, dirty, dirty_entries = git_provenance()
    return {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'configuration': configuration(),
        'configuration_fingerprint': canonical_fingerprint(configuration()),
        'scan_protocol': scan_protocol(args),
        'scan_protocol_fingerprint': canonical_fingerprint(
            scan_protocol(args)
        ),
        'cdf_path': str(Path(args.cdf).resolve()),
        'cdf_sha256': cdf.checksum_sha256,
        'cdf_metadata': metadata,
        'teacher_path': str(teacher_path),
        'teacher_sha256': TEACHER_SHA256,
        'list_path': str(list_path),
        'list_sha256': list_sha,
        'source_sha256': source_sha256(),
        'git_commit': commit,
        'git_dirty': dirty,
        'git_dirty_entry_count': dirty_entries,
        'argv': list(sys.argv),
        'command_args': dict(vars(args)),
    }


def run_solve(args, output_path, cache_path, formal, scan_context):
    device, dataset, loader, teacher, cdf, metadata = scan_context
    writer = RiskCacheWriter(
        cache_path, formal, EXPECTED_NATIVE_VALID['train']
    )
    counts, histogram = scan_risk(args, loader, teacher, cdf, device, writer)
    cache = writer.finalize()
    solution = solve_from_cache(cache_path)
    unreliable_stats = temperature_summary(
        histogram, solution['b'], branch='unreliable_only'
    )
    full_stats = temperature_summary(
        histogram, solution['b'], branch='full_budgeted'
    )
    formal_complete = (
        formal
        and counts['processed_images'] == EXPECTED_DATASET_SIZE['train']
        and counts['valid_native_pixels'] == EXPECTED_NATIVE_VALID['train']
    )
    checks = {
        'formal_full_scan': formal_complete,
        'native_nonfinite_zero': counts['nonfinite_native_pixels'] == 0,
        'solution_population_matches_scan': solution['population'] == counts['valid_native_pixels'],
        'mean_target_tol_1e4': abs(solution['mean'] - .995) <= 1e-4,
        'harmonic_mean_ge_0p98': solution['harmonic_mean'] >= .98,
        'temperature_endpoint_ge_1p25': solution['theoretical_high_risk_endpoint'] >= 1.25,
        'cdf_sha_exact': cdf.checksum_sha256 == CDF_SHA256,
    }
    return {
        'schema_version': 1, 'phase': PHASE,
        'artifact_kind': 'o12_budget_parameters', 'split': 'train',
        'stage': 'solve', 'formal_full_run': formal_complete,
        'processed_images': counts['processed_images'],
        'dataset_size': len(dataset),
        'valid_native_pixels': counts['valid_native_pixels'],
        'finite_native_pixels': counts['finite_native_pixels'],
        'nonfinite_native_pixels': counts['nonfinite_native_pixels'],
        'native_logit_shapes': counts['native_logit_shapes'],
        'risk_cache': cache, 'solution': solution,
        'branch_scalar_temperatures': {
            'unreliable_only': {
                'arithmetic': unreliable_stats['mean'],
                'harmonic': unreliable_stats['harmonic_mean'],
            },
            'full_budgeted': {
                'arithmetic': full_stats['mean'],
                'harmonic': full_stats['harmonic_mean'],
            },
        },
        'arithmetic_matched_scalar_temperature': solution['mean'],
        'harmonic_matched_scalar_temperature': solution['harmonic_mean'],
        'o11_joint_gate_pass': True,
        'checks': checks, 'all_checks_pass': all(checks.values()),
    }


def run_evaluate(args, output_path, cache_path, formal, scan_context, parameters):
    device, dataset, loader, teacher, cdf, metadata = scan_context
    parameters_path = Path(args.parameters).resolve()
    parameters_sha = file_sha256(parameters_path)
    solution = parameters.get('solution')
    if not isinstance(solution, dict) or solution.get('feasible') is not True:
        raise ValueError('O1.2 parameter artifact has no feasible frozen solution')
    if (
        parameters.get('phase') != PHASE
        or parameters.get('artifact_kind') != 'o12_budget_parameters'
        or parameters.get('configuration') != configuration()
        or parameters.get('source_sha256') != source_sha256()
    ):
        raise ValueError('O1.2 parameter artifact contract/source mismatch')
    if formal and parameters.get('formal_full_run') is not True:
        raise ValueError('formal evaluation requires formal full-train parameters')
    b = float(solution['b'])
    writer = RiskCacheWriter(
        cache_path, formal, EXPECTED_NATIVE_VALID[args.split]
    )
    counters, histogram = scan_evaluation(
        args, loader, teacher, cdf, device, writer, b
    )
    cache = writer.finalize()
    temperature = temperature_summary(histogram, b)
    regions = {
        name: finalize_bucket(bucket)
        for name, bucket in counters['regions'].items()
    }
    bins = [
        {'bin': index, **finalize_bucket(bucket)}
        for index, bucket in enumerate(counters['bins'])
    ]
    class_rows = [
        {'class_id': index, **finalize_bucket(bucket)}
        for index, bucket in enumerate(counters['classes'])
    ]
    fgbg = {
        name: finalize_bucket(bucket) for name, bucket in counters['fgbg'].items()
    }
    boundary = {
        name: finalize_bucket(bucket) for name, bucket in counters['boundary'].items()
    }
    small = {
        name: finalize_bucket(bucket) for name, bucket in counters['small'].items()
    }
    total_valid = counters['valid']
    total_wrong = counters['wrong']
    region_count = sum(row['count'] for row in regions.values())
    region_wrong = sum(row['teacher_wrong_count'] for row in regions.values())
    bin_count = sum(row['count'] for row in bins)
    bin_wrong = sum(row['teacher_wrong_count'] for row in bins)
    closure = {
        'valid_finite_partition': counters['finite'] + (total_valid - counters['finite']) == total_valid,
        'risk_regions_population': region_count == total_valid,
        'risk_regions_wrong': region_wrong == total_wrong,
        'risk_bins_population': bin_count == total_valid,
        'risk_bins_wrong': bin_wrong == total_wrong,
        'classes_population': sum(row['count'] for row in class_rows) == total_valid,
        'foreground_background_population': sum(row['count'] for row in fgbg.values()) == total_valid,
        'boundary_interior_population': sum(row['count'] for row in boundary.values()) == total_valid,
        'small_object_population': sum(row['count'] for row in small.values()) == total_valid,
    }
    violations = dict(counters['violations'])
    violations['population_closure'] = 0 if closure['risk_regions_population'] else 1
    violations['risk_bin_closure'] = 0 if closure['risk_bins_population'] else 1
    violations['error_count_closure'] = 0 if (
        closure['risk_regions_wrong'] and closure['risk_bins_wrong']
    ) else 1
    grid = np.linspace(0, 1, 4097, dtype=np.float64)
    formula_t = np.exp(
        -configuration()['a'] * np.clip((.6 - grid) / .6, 0, 1)
        + b * np.clip((grid - .8) / .2, 0, 1) ** 2
    )
    violations['formula_monotonic'] = int((np.diff(formula_t) < -1e-12).sum())
    formal_complete = (
        formal
        and counters['processed_images'] == EXPECTED_DATASET_SIZE[args.split]
        and total_valid == EXPECTED_NATIVE_VALID[args.split]
    )
    checks = {
        'formal_full_scan': formal_complete,
        'parameters_not_refit': True,
        'native_nonfinite_zero': counters['finite'] == total_valid,
        'all_direction_numeric_violations_zero': all(value == 0 for value in violations.values()),
        'all_population_closures_true': all(closure.values()),
        'actual_loss_probe_student_invariant': (
            counters['student_max_abs_error'] <= 1e-7
        ),
        'actual_loss_probe_teacher_target_changed': counters['teacher_probe_max_abs_difference'] > 0,
        'temperature_min_bound': temperature['min'] >= .9 - 1e-6,
        'temperature_max_bound': temperature['max'] <= 1.5 + 1e-6,
        'temperature_q10_ge_0p9': temperature['q10'] >= .9,
        'temperature_q50_ge_0p95': temperature['q50'] >= .95,
    }
    if args.split == 'train':
        checks.update({
            'train_mean_0p995': abs(temperature['mean'] - .995) <= 1e-4,
            'train_harmonic_ge_0p98': temperature['harmonic_mean'] >= .98,
            'train_top_decile_mean_ge_1p10': temperature['top_risk_decile_mean'] >= 1.10,
        })
    else:
        checks.update({
            'val_mean_in_0p98_1p02': .98 <= temperature['mean'] <= 1.02,
            'val_harmonic_ge_0p97': temperature['harmonic_mean'] >= .97,
        })
    effective = temperature_summary(
        histogram, b, scale=configuration()['teacher_output_temperature']
    )
    effective['semantics'] = 'T_effective=T_out*T with T_out=3.0'
    return {
        'schema_version': 1, 'phase': PHASE,
        'artifact_kind': 'o12_budget_evaluation',
        'stage': 'evaluate', 'split': args.split,
        'formal_full_run': formal_complete,
        'processed_images': counters['processed_images'],
        'dataset_size': len(dataset),
        'valid_native_pixels': total_valid,
        'finite_native_pixels': counters['finite'],
        'nonfinite_native_pixels': total_valid - counters['finite'],
        'teacher_wrong_pixels': total_wrong,
        'native_logit_shapes': [list(shape) for shape in sorted(counters['shapes'])],
        'parameters_path': str(parameters_path),
        'parameters_sha256': parameters_sha,
        'parameters_refit': False,
        'a': float(solution['a']), 'b': b,
        'risk_cache': cache,
        'temperature': temperature,
        'effective_temperature': effective,
        'risk_regions': regions,
        'boundary_counts': {
            'u_eq_0p6': counters['u_eq_0p6'],
            'u_eq_0p6_teacher_wrong': counters['u_eq_0p6_wrong'],
            'u_eq_0p8_teacher_wrong': counters['u_eq_0p8_wrong'],
            'u_eq_0p8': counters['u_eq_0p8'],
        },
        'risk_quantile_bins': bins,
        'target_by_risk_decile': bins,
        'risk_evidence': risk_evidence(regions, bins, total_valid, total_wrong),
        'stratified_diagnostics': {
            'class': class_rows,
            'foreground_background': fgbg,
            'boundary_interior': boundary,
            'small_object': small,
        },
        'violations': violations,
        'target_numeric_maxima': {
            'neutral_target_max_abs_error': counters['neutral_max_abs_error'],
            'student_softmax_max_abs_error': counters['student_max_abs_error'],
            'teacher_target_probe_max_abs_difference': (
                counters['teacher_probe_max_abs_difference']
            ),
        },
        'closure_checks': closure,
        'teacher_target_contract': CONTRACT,
        'checks': checks,
        'all_checks_pass': all(checks.values()),
    }


def main():
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    args = parse_args()
    output_path, cache_path, formal = output_paths(args)
    if output_path.exists():
        raise FileExistsError(f'refusing to overwrite O1.2 artifact: {output_path}')
    list_path = resolve_list_path(args)
    cdf_path, gate_path, teacher_path, list_sha, gate = validate_frozen_inputs(
        args, list_path
    )
    scan_context = setup_scan(args, list_path, teacher_path, cdf_path)
    identity = runtime_identity(args, scan_context[0])
    if args.stage == 'solve':
        payload = run_solve(
            args, output_path, cache_path, formal, scan_context
        )
    else:
        if not args.parameters:
            raise ValueError('stage evaluate requires --parameters')
        parameters = load_json(args.parameters)
        payload = run_evaluate(
            args, output_path, cache_path, formal, scan_context, parameters
        )
    cdf = scan_context[4]
    metadata = scan_context[5]
    payload.update(common_provenance(
        args, list_path, list_sha, teacher_path, cdf, metadata
    ))
    ended_at = datetime.now(timezone.utc)
    payload.update({
        'started_at_utc': started_at.isoformat(),
        'ended_at_utc': ended_at.isoformat(),
        'elapsed_seconds': float(time.monotonic() - started_monotonic),
        'runtime_identity': identity,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)
    print(f'wrote={output_path}', flush=True)
    print(f'wrote={cache_path}', flush=True)
    if args.strict and not payload['all_checks_pass']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
