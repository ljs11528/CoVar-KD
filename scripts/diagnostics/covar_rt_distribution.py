#!/usr/bin/env python3
import argparse
import csv
import json
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

from dataset.voc import VOCDataValSet
from models.model_zoo import get_segmentation_model


def parse_args():
    parser = argparse.ArgumentParser(description='CoVar r/T distribution diagnostic on VOC val.')
    parser.add_argument('--data', default=str(ROOT / 'dataset' / 'VOCAug'))
    parser.add_argument('--list-path', default=str(ROOT / 'dataset' / 'list' / 'voc' / 'val.txt'))
    parser.add_argument('--output-dir', default=str(ROOT / 'runs' / 'diagnostics' / 'aaai_h3'))
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-samples', type=int, default=-1)
    parser.add_argument('--max-pixels-per-image', type=int, default=4096)
    parser.add_argument('--bins', type=int, default=30)
    parser.add_argument('--ignore-label', type=int, default=-1)
    parser.add_argument('--num-classes', type=int, default=21)
    parser.add_argument('--teacher-model', default='deeplabv3')
    parser.add_argument('--teacher-backbone', default='resnet101')
    parser.add_argument('--teacher-pretrained', default=str(ROOT / 'data' / 'winycg' / 'cirkd' / 'teachers' / 'deeplabv3_resnet101_voc_best_model.pth'))
    parser.add_argument('--teacher-output-temp', type=float, default=3.0)
    parser.add_argument('--covar-temp-base', type=float, default=1.0)
    parser.add_argument('--covar-temp-min', type=float, default=0.5)
    parser.add_argument('--covar-temp-max', type=float, default=8.0)
    parser.add_argument('--covar-grad-eta', type=float, default=0.6)
    parser.add_argument('--covar-grad-max-iter', type=int, default=8)
    parser.add_argument('--covar-newton-hessian-eps', type=float, default=1e-5)
    parser.add_argument('--covar-newton-max-step', type=float, default=0.25)
    parser.add_argument('--covar-a', type=float, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg.startswith('npu'):
        try:
            import torch_npu  # noqa: F401
            device = torch.device(device_arg)
            try:
                torch.npu.set_device(device)
            except Exception:
                pass
            return device
        except Exception as exc:
            print(f'Warning: NPU requested but unavailable ({exc}); falling back to CPU.')
            return torch.device('cpu')
    if device_arg.startswith('cuda') and not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(device_arg)


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


def sort_logits_for_temperature(logits):
    return torch.sort(logits.permute(0, 2, 3, 1).contiguous(), dim=-1, descending=True).values


def covar_a(num_classes, like, configured_a=None):
    value = float((max(num_classes, 1) - 1) ** 2) / 2.0 if configured_a is None else float(configured_a)
    return torch.full_like(like, value)


@torch.no_grad()
def reliability_terms(sorted_logits, temperature_map, a, epsilon=1e-8):
    temp = temperature_map.clamp_min(epsilon)
    prob = F.softmax(sorted_logits / temp.unsqueeze(-1), dim=-1)
    c = prob[..., 0].clamp(min=epsilon, max=1.0 - epsilon)
    nonmax_prob = prob[..., 1:]
    if nonmax_prob.shape[-1] == 0:
        v = torch.zeros_like(c)
    else:
        mu = nonmax_prob.mean(dim=-1, keepdim=True)
        v = torch.mean((nonmax_prob - mu) ** 2, dim=-1)
    s = (1.0 - c).clamp_min(epsilon)
    r = -torch.log(c) + a * v / s
    return prob, c, v, r


@torch.no_grad()
def r_derivatives(sorted_logits, temperature_map, a, epsilon=1e-8):
    temp = temperature_map.clamp_min(epsilon)
    prob, c, v, r = reliability_terms(sorted_logits, temp, a, epsilon=epsilon)
    nonmax_prob = prob[..., 1:]
    if nonmax_prob.shape[-1] == 0:
        zeros = torch.zeros_like(temp)
        return zeros, zeros, r, c, v

    bar_z = torch.sum(prob * sorted_logits, dim=-1)
    centered_logits = bar_z.unsqueeze(-1) - sorted_logits
    temp_sq = temp ** 2
    temp_cu = temp_sq * temp
    temp_qd = temp_sq * temp_sq
    prob_prime = prob * centered_logits / temp_sq.unsqueeze(-1)
    logit_var = torch.sum(prob * centered_logits ** 2, dim=-1)
    prob_double_prime = prob * (
        (centered_logits ** 2 - logit_var.unsqueeze(-1)) / temp_qd.unsqueeze(-1)
        - 2.0 * centered_logits / temp_cu.unsqueeze(-1)
    )

    s = (1.0 - c).clamp_min(epsilon)
    num_nonmax = nonmax_prob.shape[-1]
    mu = s / num_nonmax
    dc_dt = prob_prime[..., 0]
    d2c_dt2 = prob_double_prime[..., 0]
    nonmax_prime = prob_prime[..., 1:]
    nonmax_double = prob_double_prime[..., 1:]
    mu_prime = -dc_dt / num_nonmax
    mu_double = -d2c_dt2 / num_nonmax

    dv_dt = (2.0 / num_nonmax) * torch.sum(nonmax_prob * nonmax_prime, dim=-1) - 2.0 * mu * mu_prime
    d2v_dt2 = (2.0 / num_nonmax) * torch.sum(nonmax_prime ** 2 + nonmax_prob * nonmax_double, dim=-1) - 2.0 * (mu_prime ** 2 + mu * mu_double)

    coeff_c = -1.0 / c + a * v / (s ** 2)
    dr_dt = coeff_c * dc_dt + (a / s) * dv_dt
    d2r_dt2 = (
        (dc_dt ** 2) / (c ** 2)
        - d2c_dt2 / c
        + a * (
            d2v_dt2 / s
            + v * d2c_dt2 / (s ** 2)
            + 2.0 * dc_dt * dv_dt / (s ** 2)
            + 2.0 * v * (dc_dt ** 2) / (s ** 3)
        )
    )
    return dr_dt, d2r_dt2, r, c, v


@torch.no_grad()
def newton_temperature_map(logits, valid_mask, args):
    sorted_logits = sort_logits_for_temperature(logits)
    base_temp = float(args.covar_temp_base)
    t_min = float(args.covar_temp_min)
    t_max = float(args.covar_temp_max)
    eta = float(args.covar_grad_eta)
    hessian_eps = float(args.covar_newton_hessian_eps)
    max_step = float(args.covar_newton_max_step)
    temperature = torch.full(sorted_logits.shape[:-1], base_temp, device=logits.device, dtype=logits.dtype)
    a = covar_a(sorted_logits.shape[-1], valid_mask.float(), args.covar_a)

    for _ in range(max(int(args.covar_grad_max_iter), 0)):
        dr_dt, d2r_dt2, _, _, _ = r_derivatives(sorted_logits, temperature, a)
        valid_hessian = torch.isfinite(d2r_dt2) & (d2r_dt2.abs() >= hessian_eps) & (d2r_dt2 > 0)
        safe_hessian = torch.where(valid_hessian, d2r_dt2, torch.ones_like(d2r_dt2))
        delta_t = torch.where(valid_hessian, eta * dr_dt / safe_hessian, eta * dr_dt)
        if max_step > 0:
            delta_t = torch.clamp(delta_t, min=-max_step, max=max_step)
        temperature = torch.clamp(temperature - delta_t, min=t_min, max=t_max)

    _, _, r_map, c_map, v_map = r_derivatives(sorted_logits, temperature, a)
    temperature = torch.where(valid_mask, temperature, torch.full_like(temperature, base_temp))
    r_map = torch.where(valid_mask, r_map, torch.zeros_like(r_map))
    c_map = torch.where(valid_mask, c_map, torch.zeros_like(c_map))
    v_map = torch.where(valid_mask, v_map, torch.zeros_like(v_map))
    return temperature, r_map, c_map, v_map


def sample_pixels(values, max_pixels, rng):
    count = values[0].numel()
    if max_pixels <= 0 or count <= max_pixels:
        return values
    idx = torch.randperm(count, generator=rng)[:max_pixels]
    return tuple(v[idx] for v in values)


def quantile_stats(x):
    q = torch.tensor([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0], dtype=torch.float32)
    quantiles = torch.quantile(x.float(), q)
    return {
        'count': int(x.numel()),
        'mean': float(x.float().mean().item()),
        'std': float(x.float().std(unbiased=False).item()),
        'min': float(quantiles[0].item()),
        'p01': float(quantiles[1].item()),
        'p05': float(quantiles[2].item()),
        'p25': float(quantiles[3].item()),
        'p50': float(quantiles[4].item()),
        'p75': float(quantiles[5].item()),
        'p95': float(quantiles[6].item()),
        'p99': float(quantiles[7].item()),
        'max': float(quantiles[8].item()),
    }


def corrcoef(x, y):
    x = x.float()
    y = y.float()
    x_center = x - x.mean()
    y_center = y - y.mean()
    denom = x_center.std(unbiased=False) * y_center.std(unbiased=False)
    if denom.item() <= 0:
        return 0.0
    return float((x_center * y_center).mean().div(denom).item())


def write_histogram(path, values, bins, value_name):
    hist, edges = np.histogram(values, bins=bins)
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['bin', f'{value_name}_low', f'{value_name}_high', 'count', 'fraction'])
        writer.writeheader()
        total = max(int(hist.sum()), 1)
        for i, count in enumerate(hist):
            writer.writerow({
                'bin': i,
                f'{value_name}_low': float(edges[i]),
                f'{value_name}_high': float(edges[i + 1]),
                'count': int(count),
                'fraction': float(count / total),
            })


def save_plot(path, arrays, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.reshape(-1)
    axes[0].hist(arrays['r'], bins=80, color='#b33b2e', alpha=0.9)
    axes[0].set_title('Reliability r')
    axes[0].set_yscale('log')

    axes[1].hist(arrays['temperature'], bins=80, color='#2769a3', alpha=0.9)
    axes[1].set_title('Adaptive temperature T')
    axes[1].set_yscale('log')

    axes[2].hist(arrays['confidence'], bins=80, color='#2f7d4f', alpha=0.9)
    axes[2].set_title('Teacher confidence c')
    axes[2].set_yscale('log')

    sample = min(arrays['r'].shape[0], 50000)
    rng = np.random.default_rng(1234)
    idx = rng.choice(arrays['r'].shape[0], size=sample, replace=False)
    axes[3].hexbin(arrays['r'][idx], arrays['temperature'][idx], gridsize=70, bins='log', cmap='viridis')
    axes[3].set_title(f"r vs T, corr={summary['correlations']['r_temperature']:.3f}")
    axes[3].set_xlabel('r')
    axes[3].set_ylabel('T')

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dataset = VOCDataValSet(args.data, args.list_path, crop_size=(512, 512), ignore_label=args.ignore_label)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False)
    teacher = build_teacher(args, device)
    rng = torch.Generator(device='cpu')
    rng.manual_seed(args.seed)

    all_r = []
    all_t = []
    all_c = []
    all_v = []
    all_r_over_t2 = []
    all_wrong = []
    image_rows = []

    with torch.no_grad():
        for idx, (image, target, meta) in enumerate(loader):
            if args.max_samples > 0 and idx >= args.max_samples:
                break
            image = image.to(device)
            target = target.long().to(device)
            valid = target != args.ignore_label
            if not valid.any():
                continue

            output = teacher(image)
            logits = output[0] if isinstance(output, (list, tuple)) else output
            if args.teacher_output_temp != 1.0:
                logits = logits / args.teacher_output_temp
            logits = F.interpolate(logits, size=target.shape[-2:], mode='bilinear', align_corners=True)

            pred = torch.argmax(logits, dim=1)
            temperature, r_map, c_map, v_map = newton_temperature_map(logits, valid, args)
            r_over_t2 = r_map / temperature.clamp_min(1e-8).pow(2)
            wrong = ((pred != target) & valid).float()

            r_valid = r_map[valid].detach().cpu().float()
            t_valid = temperature[valid].detach().cpu().float()
            c_valid = c_map[valid].detach().cpu().float()
            v_valid = v_map[valid].detach().cpu().float()
            rt_valid = r_over_t2[valid].detach().cpu().float()
            wrong_valid = wrong[valid].detach().cpu().float()

            r_s, t_s, c_s, v_s, rt_s, wrong_s = sample_pixels(
                (r_valid, t_valid, c_valid, v_valid, rt_valid, wrong_valid),
                args.max_pixels_per_image,
                rng,
            )
            all_r.append(r_s)
            all_t.append(t_s)
            all_c.append(c_s)
            all_v.append(v_s)
            all_r_over_t2.append(rt_s)
            all_wrong.append(wrong_s)

            if isinstance(meta, (list, tuple)) and len(meta) == 2:
                name = meta[1][0] if isinstance(meta[1], (list, tuple)) else meta[1]
            else:
                name = str(meta)
            image_rows.append({
                'index': idx,
                'name': str(name),
                'valid_pixels': int(valid.sum().item()),
                'sampled_pixels': int(r_s.numel()),
                'r_mean': float(r_s.mean().item()),
                'r_p95': float(torch.quantile(r_s, 0.95).item()),
                'temperature_mean': float(t_s.mean().item()),
                'temperature_p95': float(torch.quantile(t_s, 0.95).item()),
                'wrong_rate': float(wrong_s.mean().item()),
            })

    if not all_r:
        raise RuntimeError('No valid pixels collected.')

    tensors = {
        'r': torch.cat(all_r),
        'temperature': torch.cat(all_t),
        'confidence': torch.cat(all_c),
        'variance': torch.cat(all_v),
        'r_over_t2': torch.cat(all_r_over_t2),
        'wrong': torch.cat(all_wrong),
    }
    arrays = {key: value.numpy() for key, value in tensors.items()}

    summary = {
        'scope': {
            'dataset': 'VOC val',
            'dataset_size': len(dataset),
            'processed_images': len(image_rows),
            'sampled_pixels': int(tensors['r'].numel()),
            'max_pixels_per_image': args.max_pixels_per_image,
        },
        'config': {
            'teacher_output_temp': args.teacher_output_temp,
            'covar_temp_base': args.covar_temp_base,
            'covar_temp_min': args.covar_temp_min,
            'covar_temp_max': args.covar_temp_max,
            'covar_grad_eta': args.covar_grad_eta,
            'covar_grad_max_iter': args.covar_grad_max_iter,
            'covar_newton_max_step': args.covar_newton_max_step,
            'covar_a': args.covar_a if args.covar_a is not None else float((args.num_classes - 1) ** 2 / 2.0),
        },
        'stats': {key: quantile_stats(value) for key, value in tensors.items() if key != 'wrong'},
        'temperature_fractions': {
            'at_min': float((tensors['temperature'] <= args.covar_temp_min + 1e-6).float().mean().item()),
            'below_0_75': float((tensors['temperature'] < 0.75).float().mean().item()),
            'near_base_0_75_1_25': float(((tensors['temperature'] >= 0.75) & (tensors['temperature'] <= 1.25)).float().mean().item()),
            'above_1_25': float((tensors['temperature'] > 1.25).float().mean().item()),
            'at_max': float((tensors['temperature'] >= args.covar_temp_max - 1e-6).float().mean().item()),
        },
        'correlations': {
            'r_temperature': corrcoef(tensors['r'], tensors['temperature']),
            'r_confidence': corrcoef(tensors['r'], tensors['confidence']),
            'r_variance': corrcoef(tensors['r'], tensors['variance']),
            'r_wrong': corrcoef(tensors['r'], tensors['wrong']),
            'temperature_wrong': corrcoef(tensors['temperature'], tensors['wrong']),
        },
        'teacher_wrong_rate': float(tensors['wrong'].mean().item()),
    }

    summary_path = output_dir / 'h3_rt_distribution_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')

    image_csv = output_dir / 'h3_rt_distribution_image_summary.csv'
    with image_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(image_rows)

    for key in ['r', 'temperature', 'confidence', 'variance', 'r_over_t2']:
        write_histogram(output_dir / f'h3_{key}_histogram.csv', arrays[key], args.bins, key)

    np.savez_compressed(output_dir / 'h3_rt_distribution_samples.npz', **arrays)
    plot_path = output_dir / 'h3_rt_distribution.png'
    save_plot(plot_path, arrays, summary)

    print(f'Wrote summary: {summary_path}')
    print(f'Wrote image summary: {image_csv}')
    print(f'Wrote plot: {plot_path}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
