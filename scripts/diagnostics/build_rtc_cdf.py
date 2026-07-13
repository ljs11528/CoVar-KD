#!/usr/bin/env python3
"""Build the frozen training-set reliability CDF used by RTC-KD."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.voc import VOCDataTrainSet
from models.model_zoo import get_segmentation_model
from utils.rtc_temperature import (
    compute_reference_reliability,
    file_sha256,
    reliability_definition_metadata,
    save_frozen_reliability_cdf,
)


SOURCE_PATHS = {
    'build_rtc_cdf': Path(__file__).resolve(),
    'diagnose_rtc_routing': ROOT / 'scripts' / 'diagnostics' / 'diagnose_rtc_routing.py',
    'rtc_temperature': ROOT / 'utils' / 'rtc_temperature.py',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build a frozen RTC reliability CDF on native teacher logits.'
    )
    parser.add_argument(
        '--phase',
        choices=['O1', 'O1.1'],
        default='O1',
    )
    parser.add_argument('--data', default=str(ROOT / 'dataset' / 'VOCAug'))
    parser.add_argument(
        '--list-path', default=str(ROOT / 'dataset' / 'list' / 'voc' / 'train_aug.txt')
    )
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
    parser.add_argument('--max-pixels-per-image', type=int, default=4096)
    parser.add_argument('--num-quantiles', type=int, default=4097)
    parser.add_argument('--assess-temperature', type=float, default=1.0)
    parser.add_argument('--coefficient-a', type=float, default=None)
    parser.add_argument(
        '--reliability-mode',
        choices=['full', 'confidence', 'variance'],
        default='full',
    )
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--no-scale', action='store_true', default=False)
    parser.add_argument('--no-mirror', action='store_true', default=False)
    parser.add_argument(
        '--output',
        default=None,
        help='Output path; phase-specific defaults are used when omitted.',
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


def sample_pixels(values, max_pixels, generator):
    values = values.detach().reshape(-1).float().cpu()
    if max_pixels <= 0 or values.numel() <= max_pixels:
        return values
    indices = torch.randperm(values.numel(), generator=generator)[:max_pixels]
    return values[indices]


def validate_args(args):
    if args.batch_size <= 0:
        raise ValueError('--batch-size must be positive')
    if args.workers < 0:
        raise ValueError('--workers must be non-negative')
    if args.max_images < 0:
        raise ValueError('--max-images must be non-negative')
    if args.max_pixels_per_image < 0:
        raise ValueError('--max-pixels-per-image must be non-negative')
    if args.num_quantiles < 2:
        raise ValueError('--num-quantiles must be at least 2')
    if args.num_classes < 2:
        raise ValueError('--num-classes must be at least 2')
    if args.assess_temperature <= 0:
        raise ValueError('--assess-temperature must be positive')
    if args.coefficient_a is not None and args.coefficient_a < 0:
        raise ValueError('--coefficient-a must be non-negative')
    if args.phase == 'O1.1':
        if args.reliability_mode != 'confidence':
            raise ValueError('O1.1 requires --reliability-mode confidence')
        if args.coefficient_a is None or float(args.coefficient_a) != 0.0:
            raise ValueError('O1.1 requires explicit --coefficient-a 0')


def main():
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == 'npu':
        torch.npu.manual_seed_all(args.seed)
    source_hashes = source_sha256(args.phase)

    list_path = Path(args.list_path).resolve()
    teacher_path = Path(args.teacher_pretrained).resolve()
    dataset = VOCDataTrainSet(
        args.data,
        str(list_path),
        max_iters=None,
        crop_size=tuple(args.crop_size),
        scale=not args.no_scale,
        mirror=not args.no_mirror,
        ignore_label=args.ignore_label,
    )
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
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

    coefficient_a = args.coefficient_a
    if coefficient_a is None:
        coefficient_a = float((args.num_classes - 1) ** 2) / 2.0
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    samples = []
    processed_images = 0
    valid_native_pixels = 0
    finite_valid_pixels = 0
    nonfinite_valid_pixels = 0
    images_with_valid_pixels = 0
    images_with_finite_samples = 0
    images_with_nonfinite_valid_pixels = 0
    finite_sample_candidates = 0
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
            reliability = compute_reference_reliability(
                raw_logits,
                targets != args.ignore_label,
                assess_temperature=args.assess_temperature,
                coefficient_a=coefficient_a,
                reliability_mode=args.reliability_mode,
            )
            native_valid = reliability.valid_mask
            finite_valid = native_valid & reliability.finite_mask
            nonfinite_valid = native_valid & ~reliability.finite_mask
            valid_native_pixels += int(native_valid.sum().item())
            finite_valid_pixels += int(finite_valid.sum().item())
            nonfinite_valid_pixels += int(nonfinite_valid.sum().item())
            images_with_valid_pixels += int(
                native_valid.flatten(1).any(dim=1).sum().item()
            )
            images_with_nonfinite_valid_pixels += int(
                nonfinite_valid.flatten(1).any(dim=1).sum().item()
            )
            for image_index in range(images.shape[0]):
                image_values = reliability.reliability[image_index][
                    finite_valid[image_index]
                ]
                finite_sample_candidates += int(image_values.numel())
                if image_values.numel() > 0:
                    images_with_finite_samples += 1
                    samples.append(
                        sample_pixels(
                            image_values,
                            args.max_pixels_per_image,
                            generator,
                        )
                    )
            processed_images += int(images.shape[0])
            if args.log_every > 0 and processed_images % args.log_every < images.shape[0]:
                sampled_count = sum(item.numel() for item in samples)
                print(
                    f'processed_images={processed_images}/{len(dataset)} '
                    f'valid_native_pixels={valid_native_pixels} '
                    f'nonfinite_valid_pixels={nonfinite_valid_pixels} '
                    f'sampled_pixels={sampled_count}',
                    flush=True,
                )
            if args.max_images > 0 and processed_images >= args.max_images:
                break

    if not samples:
        raise RuntimeError('No finite native-grid reliability samples were collected')
    reliability_samples = torch.cat(samples)
    commit, dirty, dirty_entries = git_provenance()
    full_dataset_scan = args.max_images == 0 and processed_images == len(dataset)
    metadata = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': commit,
        'git_dirty': dirty,
        'git_dirty_entry_count': dirty_entries,
        'source_sha256': source_hashes,
        'argv': list(sys.argv),
        'command_args': dict(vars(args)),
        'phase': args.phase,
        'dataset': 'voc',
        'dataset_class': type(dataset).__name__,
        'split': 'train_aug',
        'data_root': str(Path(args.data).resolve()),
        'train_list_path': str(list_path),
        'train_list_sha256': file_sha256(list_path),
        'teacher_model': args.teacher_model,
        'teacher_backbone': args.teacher_backbone,
        'teacher_path': str(teacher_path),
        'teacher_sha256': file_sha256(teacher_path),
        'num_classes': int(args.num_classes),
        'assess_temperature': float(args.assess_temperature),
        'coefficient_a': float(coefficient_a),
        'reliability_mode': args.reliability_mode,
        **reliability_definition_metadata(
            args.reliability_mode,
            float(coefficient_a),
        ),
        'seed': int(args.seed),
        'processed_images': int(processed_images),
        'dataset_size': int(len(dataset)),
        'max_images': int(args.max_images),
        'full_dataset_scan': bool(full_dataset_scan),
        'batch_size': int(args.batch_size),
        'workers': int(args.workers),
        'max_pixels_per_image': int(args.max_pixels_per_image),
        'num_quantiles': int(args.num_quantiles),
        'crop_size': list(args.crop_size),
        'scale': not args.no_scale,
        'mirror': not args.no_mirror,
        'teacher_output_grid': 'native',
        'native_logit_shapes': [list(shape) for shape in sorted(native_logit_shapes)],
        'valid_mask_resize': 'nearest',
        'valid_native_pixels': int(valid_native_pixels),
        'finite_valid_pixels': int(finite_valid_pixels),
        'nonfinite_valid_pixels': int(nonfinite_valid_pixels),
        'nonfinite_valid_rate': (
            float(nonfinite_valid_pixels / valid_native_pixels)
            if valid_native_pixels > 0 else None
        ),
        'images_with_valid_pixels': int(images_with_valid_pixels),
        'images_with_finite_samples': int(images_with_finite_samples),
        'images_with_nonfinite_valid_pixels': int(images_with_nonfinite_valid_pixels),
        'finite_sample_candidates': int(finite_sample_candidates),
        'sampling_semantics': (
            'uniform without replacement within each image over finite native valid pixels; '
            'per-image cap applies'
        ),
    }
    if args.output:
        output_path = Path(args.output)
    elif args.phase == 'O1.1':
        output_path = (
            ROOT / 'runs' / 'diagnostics' / 'phaseO_o11'
            / 'voc_train_rtc_confidence_cdf.pt'
        )
    else:
        output_path = ROOT / 'runs' / 'diagnostics' / 'phaseO' / 'voc_train_rtc_cdf.pt'
    sidecar_paths = (
        output_path,
        output_path.with_suffix(output_path.suffix + '.summary.json'),
        output_path.with_suffix(output_path.suffix + '.sha256'),
    )
    existing = [str(path) for path in sidecar_paths if path.exists()]
    if existing:
        raise FileExistsError(
            'Refusing to overwrite frozen CDF artifacts: ' + ', '.join(existing)
        )
    checksum = save_frozen_reliability_cdf(
        output_path,
        reliability_samples,
        metadata,
        num_quantiles=args.num_quantiles,
    )
    summary = {
        **metadata,
        'sampled_pixels': int(reliability_samples.numel()),
        'reliability_min': float(reliability_samples.min()),
        'reliability_mean': float(reliability_samples.mean()),
        'reliability_max': float(reliability_samples.max()),
        'num_quantiles': int(args.num_quantiles),
        'cdf_path': str(output_path.resolve()),
        'cdf_sha256': checksum,
    }
    summary_path = output_path.with_suffix(output_path.suffix + '.summary.json')
    checksum_path = output_path.with_suffix(output_path.suffix + '.sha256')
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8'
    )
    checksum_path.write_text(f'{checksum}  {output_path.name}\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
