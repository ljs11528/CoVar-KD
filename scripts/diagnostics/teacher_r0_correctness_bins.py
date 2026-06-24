#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.voc import VOCDataTrainSet, VOCDataValSet
from models.model_zoo import get_segmentation_model


def parse_args():
    parser = argparse.ArgumentParser(description='Teacher r0 vs correctness diagnostic for VOC.')
    parser.add_argument('--data', default=str(ROOT / 'dataset' / 'VOCAug'))
    parser.add_argument('--split', choices=['val', 'train'], default='val')
    parser.add_argument('--list-path', default=None)
    parser.add_argument('--teacher-model', default='deeplabv3')
    parser.add_argument('--teacher-backbone', default='resnet101')
    parser.add_argument('--teacher-pretrained', default=str(ROOT / 'data' / 'winycg' / 'cirkd' / 'teachers' / 'deeplabv3_resnet101_voc_best_model.pth'))
    parser.add_argument('--num-classes', type=int, default=21)
    parser.add_argument('--crop-size', nargs=2, type=int, default=[512, 512])
    parser.add_argument('--ignore-label', type=int, default=-1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-samples', type=int, default=200)
    parser.add_argument('--max-pixels-per-image', type=int, default=4096)
    parser.add_argument('--bins', type=int, default=10)
    parser.add_argument('--binning', choices=['quantile', 'uniform'], default='quantile')
    parser.add_argument('--covar-ref-temp', type=float, default=1.0)
    parser.add_argument('--covar-a', type=float, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output', default=str(ROOT / 'runs' / 'diagnostics' / 'teacher_r0_correctness_bins.csv'))
    return parser.parse_args()


def build_dataset(args):
    if args.list_path is None:
        if args.split == 'val':
            list_path = ROOT / 'dataset' / 'list' / 'voc' / 'val.txt'
        else:
            list_path = ROOT / 'dataset' / 'list' / 'voc' / 'train_aug.txt'
    else:
        list_path = Path(args.list_path)

    if args.split == 'val':
        return VOCDataValSet(args.data, str(list_path), crop_size=tuple(args.crop_size), ignore_label=args.ignore_label)

    return VOCDataTrainSet(
        args.data,
        str(list_path),
        max_iters=None,
        crop_size=tuple(args.crop_size),
        scale=False,
        mirror=False,
        ignore_label=args.ignore_label,
    )


def build_teacher(args, device):
    model = get_segmentation_model(
        model=args.teacher_model,
        backbone=args.teacher_backbone,
        local_rank=0,
        pretrained_base='None',
        pretrained=args.teacher_pretrained,
        aux=True,
        norm_layer=nn.BatchNorm2d,
        num_class=args.num_classes,
    ).to(device)
    model.eval()
    return model


def reliability_terms(logits, ref_temp, a_value, epsilon=1e-8):
    prob = F.softmax(logits / max(ref_temp, epsilon), dim=1)
    c, pred = torch.max(prob, dim=1)
    c = c.clamp(min=epsilon, max=1.0 - epsilon)

    num_classes = prob.shape[1]
    if num_classes <= 1:
        v = torch.zeros_like(c)
    else:
        nonmax_count = float(num_classes - 1)
        sum_nonmax = (1.0 - c).clamp_min(epsilon)
        sq_sum_nonmax = (prob * prob).sum(dim=1) - c * c
        mean_nonmax = sum_nonmax / nonmax_count
        v = sq_sum_nonmax / nonmax_count - mean_nonmax * mean_nonmax
        v = v.clamp_min(0.0)

    if a_value is None:
        a_value = float((max(num_classes, 1) - 1) ** 2) / 2.0

    r = -torch.log(c) + float(a_value) * v / (1.0 - c).clamp_min(epsilon)
    return pred, c, v, r


def sample_valid_pixels(values, max_pixels, generator):
    if max_pixels <= 0 or values[0].numel() <= max_pixels:
        return values
    idx = torch.randperm(values[0].numel(), generator=generator)[:max_pixels]
    return tuple(v[idx] for v in values)


def collect(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    dataset = build_dataset(args)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=(device.type == 'cuda'))
    teacher = build_teacher(args, device)
    rng = torch.Generator(device='cpu')
    rng.manual_seed(args.seed)

    all_r = []
    all_correct = []
    all_conf = []
    all_v = []
    all_names = []
    processed = 0

    with torch.no_grad():
        for image, target, name in loader:
            if args.max_samples > 0 and processed >= args.max_samples:
                break

            image = image.to(device)
            target = target.long().to(device)
            output = teacher(image)
            logits = output[0] if isinstance(output, (list, tuple)) else output
            logits = F.interpolate(logits, size=target.shape[-2:], mode='bilinear', align_corners=True)

            pred, conf, v, r = reliability_terms(logits, args.covar_ref_temp, args.covar_a)
            valid = target != args.ignore_label
            if not valid.any():
                continue

            correct = (pred == target).float()
            r_valid = r[valid].detach().cpu().float()
            correct_valid = correct[valid].detach().cpu().float()
            conf_valid = conf[valid].detach().cpu().float()
            v_valid = v[valid].detach().cpu().float()
            r_valid, correct_valid, conf_valid, v_valid = sample_valid_pixels(
                (r_valid, correct_valid, conf_valid, v_valid),
                args.max_pixels_per_image,
                rng,
            )

            all_r.append(r_valid)
            all_correct.append(correct_valid)
            all_conf.append(conf_valid)
            all_v.append(v_valid)
            all_names.append(str(name))
            processed += 1

    if not all_r:
        raise RuntimeError('No valid pixels collected. Check data/list paths and ignore label.')

    return {
        'r': torch.cat(all_r),
        'correct': torch.cat(all_correct),
        'confidence': torch.cat(all_conf),
        'variance': torch.cat(all_v),
        'processed_images': processed,
        'dataset_size': len(dataset),
    }


def make_edges(r, bins, mode):
    bins = max(int(bins), 1)
    if mode == 'uniform':
        return torch.linspace(r.min(), r.max(), bins + 1)
    q = torch.linspace(0.0, 1.0, bins + 1)
    return torch.quantile(r, q)


def summarize(data_dict, args):
    r = data_dict['r']
    correct = data_dict['correct']
    wrong = 1.0 - correct
    conf = data_dict['confidence']
    var = data_dict['variance']
    edges = make_edges(r, args.bins, args.binning)

    rows = []
    for i in range(args.bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == args.bins - 1:
            mask = (r >= lo) & (r <= hi)
        else:
            mask = (r >= lo) & (r < hi)
        count = int(mask.sum().item())
        if count == 0:
            rows.append({
                'bin': i,
                'r_low': float(lo.item()),
                'r_high': float(hi.item()),
                'count': 0,
                'teacher_acc': 0.0,
                'wrong_rate': 0.0,
                'r_mean': 0.0,
                'confidence_mean': 0.0,
                'variance_mean': 0.0,
            })
            continue
        rows.append({
            'bin': i,
            'r_low': float(lo.item()),
            'r_high': float(hi.item()),
            'count': count,
            'teacher_acc': float(correct[mask].mean().item()),
            'wrong_rate': float(wrong[mask].mean().item()),
            'r_mean': float(r[mask].mean().item()),
            'confidence_mean': float(conf[mask].mean().item()),
            'variance_mean': float(var[mask].mean().item()),
        })

    r_center = r - r.mean()
    wrong_center = wrong - wrong.mean()
    denom = r_center.std(unbiased=False) * wrong_center.std(unbiased=False)
    if denom.item() > 0:
        corr_r_wrong = float((r_center * wrong_center).mean().div(denom).item())
    else:
        corr_r_wrong = 0.0

    summary = {
        'processed_images': data_dict['processed_images'],
        'dataset_size': data_dict['dataset_size'],
        'sampled_pixels': int(r.numel()),
        'teacher_acc': float(correct.mean().item()),
        'wrong_rate': float(wrong.mean().item()),
        'r_mean': float(r.mean().item()),
        'r_min': float(r.min().item()),
        'r_max': float(r.max().item()),
        'confidence_mean': float(conf.mean().item()),
        'variance_mean': float(var.mean().item()),
        'corr_r_wrong': corr_r_wrong,
        'bins': args.bins,
        'binning': args.binning,
        'covar_ref_temp': args.covar_ref_temp,
        'covar_a': args.covar_a,
    }
    return rows, summary


def save_outputs(rows, summary, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as f:
        fieldnames = ['bin', 'r_low', 'r_high', 'count', 'teacher_acc', 'wrong_rate', 'r_mean', 'confidence_mean', 'variance_mean']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output.with_suffix(output.suffix + '.summary.json')
    with summary_path.open('w') as f:
        json.dump(summary, f, indent=2)

    return output, summary_path


def main():
    args = parse_args()
    data_dict = collect(args)
    rows, summary = summarize(data_dict, args)
    output, summary_path = save_outputs(rows, summary, args.output)

    print(f'Wrote bins: {output}')
    print(f'Wrote summary: {summary_path}')
    print(json.dumps(summary, indent=2))
    print('bin,r_low,r_high,count,teacher_acc,wrong_rate,r_mean,confidence_mean,variance_mean')
    for row in rows:
        print(','.join(str(row[k]) for k in ['bin', 'r_low', 'r_high', 'count', 'teacher_acc', 'wrong_rate', 'r_mean', 'confidence_mean', 'variance_mean']))


if __name__ == '__main__':
    main()
