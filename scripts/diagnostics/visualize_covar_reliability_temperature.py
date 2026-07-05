#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.voc import VOCDataValSet
from models.model_zoo import get_segmentation_model
from utils.visualize import get_color_pallete


MEAN_BGR = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CoVar reliability and Newton temperature maps.')
    parser.add_argument('--data', default=str(ROOT / 'dataset' / 'VOCAug'))
    parser.add_argument('--list-path', default=str(ROOT / 'dataset' / 'list' / 'voc' / 'val.txt'))
    parser.add_argument('--output-dir', default=str(ROOT / 'runs' / 'diagnostics' / 'aaai_h2'))
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-samples', type=int, default=200)
    parser.add_argument('--num-examples', type=int, default=4)
    parser.add_argument('--ignore-label', type=int, default=-1)
    parser.add_argument('--num-classes', type=int, default=21)
    parser.add_argument('--teacher-model', default='deeplabv3')
    parser.add_argument('--teacher-backbone', default='resnet101')
    parser.add_argument('--teacher-pretrained', default=str(ROOT / 'data' / 'winycg' / 'cirkd' / 'teachers' / 'deeplabv3_resnet101_voc_best_model.pth'))
    parser.add_argument('--student-model', default='deeplabv3_mobilenet_ssseg')
    parser.add_argument('--student-backbone', default='mobilenetv3_small')
    parser.add_argument('--student-checkpoint', default=str(ROOT / 'data' / 'winycg' / 'checkpoints' / 'cirkd_checkpoints' / 'voc' / 'covar_npu_phaseD_main_table' / 'phaseD_covar_newton_gamma2_tout1' / 'kd_deeplabv3_mobilenet_ssseg_mobilenetv3_small_voc_best_model.pth'))
    parser.add_argument('--teacher-output-temp', type=float, default=1.0)
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
    if device_arg.startswith('cuda') and not torch.cuda.is_available():
        return torch.device('cpu')
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
            print(f'Warning: NPU device requested but torch_npu is unavailable ({exc}); falling back to CPU.')
            return torch.device('cpu')
    return torch.device(device_arg)


def load_state_dict_compatible(module, state_dict, strict=True):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith('module.') else key
        cleaned[new_key] = value
    module.load_state_dict(cleaned, strict=strict)


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


def build_student(args, device):
    if not args.student_checkpoint:
        return None
    model = get_segmentation_model(
        model=args.student_model,
        backbone=args.student_backbone,
        local_rank=0,
        pretrained_base='None',
        pretrained='None',
        aux=False,
        norm_layer=nn.BatchNorm2d,
        num_class=args.num_classes,
    ).to(device)
    checkpoint = torch.load(args.student_checkpoint, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'student' in checkpoint:
        checkpoint = checkpoint['student']
    load_state_dict_compatible(model, checkpoint, strict=True)
    model.eval()
    return model


def unpack_meta(meta):
    if isinstance(meta, (list, tuple)) and len(meta) == 2:
        img_path, name = meta
        if isinstance(img_path, (list, tuple)):
            img_path = img_path[0]
        if isinstance(name, (list, tuple)):
            name = name[0]
        return str(img_path), str(name)
    return '', str(meta)


def image_tensor_to_rgb(image):
    chw = image.detach().float().cpu().numpy()
    bgr = chw.transpose(1, 2, 0) + MEAN_BGR.reshape(1, 1, 3)
    rgb = bgr[:, :, ::-1]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def label_to_rgb(label):
    label = np.asarray(label).astype(np.int32)
    return np.asarray(get_color_pallete(label.copy(), dataset='voc').convert('RGB'))


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
    dc_dT = prob_prime[..., 0]
    d2c_dT2 = prob_double_prime[..., 0]
    nonmax_prob_prime = prob_prime[..., 1:]
    nonmax_prob_double_prime = prob_double_prime[..., 1:]
    mu_prime = -dc_dT / num_nonmax
    mu_double_prime = -d2c_dT2 / num_nonmax

    dv_dT = (2.0 / num_nonmax) * torch.sum(nonmax_prob * nonmax_prob_prime, dim=-1) - 2.0 * mu * mu_prime
    d2v_dT2 = (2.0 / num_nonmax) * torch.sum(
        nonmax_prob_prime ** 2 + nonmax_prob * nonmax_prob_double_prime,
        dim=-1,
    ) - 2.0 * (mu_prime ** 2 + mu * mu_double_prime)

    coeff_c = -1.0 / c + a * v / (s ** 2)
    dr_dT = coeff_c * dc_dT + (a / s) * dv_dT
    d2r_dT2 = (
        (dc_dT ** 2) / (c ** 2)
        - d2c_dT2 / c
        + a * (
            d2v_dT2 / s
            + v * d2c_dT2 / (s ** 2)
            + 2.0 * dc_dT * dv_dT / (s ** 2)
            + 2.0 * v * (dc_dT ** 2) / (s ** 3)
        )
    )
    return dr_dT, d2r_dT2, r, c, v


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
    return temperature, r_map, c_map, v_map


def to_numpy_map(x):
    return x.detach().float().cpu().squeeze(0).numpy()


def map_stats(x, valid):
    values = x[valid]
    if values.size == 0:
        return {'mean': 0.0, 'p95': 0.0, 'max': 0.0}
    return {
        'mean': float(values.mean()),
        'p95': float(np.quantile(values, 0.95)),
        'max': float(values.max()),
    }


def save_figure(candidate, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        ('Image', candidate['image_rgb'], None),
        ('GT', candidate['gt_rgb'], None),
        ('Teacher', candidate['teacher_rgb'], None),
    ]
    if candidate.get('student_rgb') is not None:
        panels.append(('Student', candidate['student_rgb'], None))
    panels.extend([
        ('Reliability r', candidate['r_map'], 'magma'),
        ('Temperature T', candidate['temperature_map'], 'viridis'),
    ])

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.0))
    axes = axes.reshape(-1)
    for ax, (title, image, cmap) in zip(axes, panels):
        ax.set_title(title)
        ax.axis('off')
        if cmap is None:
            ax.imshow(image)
        elif title == 'Temperature T':
            im = ax.imshow(image, cmap=cmap, vmin=candidate['t_min'], vmax=candidate['t_max'])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        else:
            vmax = max(float(np.quantile(image[candidate['valid_mask']], 0.99)), 1e-6)
            im = ax.imshow(image, cmap=cmap, vmin=0.0, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for ax in axes[len(panels):]:
        ax.axis('off')

    fig.suptitle(
        f"{candidate['name']} | teacher wrong={candidate['wrong_rate'] * 100:.2f}% | "
        f"r_p95={candidate['r_p95']:.3f} | T_mean={candidate['t_mean']:.3f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dataset = VOCDataValSet(args.data, args.list_path, crop_size=(512, 512), ignore_label=args.ignore_label)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False)
    teacher = build_teacher(args, device)
    student = build_student(args, device)

    candidates = []
    with torch.no_grad():
        for idx, (image, target, meta) in enumerate(loader):
            if args.max_samples > 0 and idx >= args.max_samples:
                break
            image = image.to(device)
            target = target.long().to(device)
            valid = target != args.ignore_label
            image_path, name = unpack_meta(meta)

            teacher_outputs = teacher(image)
            teacher_logits = teacher_outputs[0] if isinstance(teacher_outputs, (list, tuple)) else teacher_outputs
            if args.teacher_output_temp != 1.0:
                teacher_logits = teacher_logits / args.teacher_output_temp
            teacher_logits = F.interpolate(teacher_logits, size=target.shape[-2:], mode='bilinear', align_corners=True)
            teacher_pred = torch.argmax(teacher_logits, dim=1)

            student_pred = None
            if student is not None:
                student_outputs = student(image)
                student_logits = student_outputs[0] if isinstance(student_outputs, (list, tuple)) else student_outputs
                student_logits = F.interpolate(student_logits, size=target.shape[-2:], mode='bilinear', align_corners=True)
                student_pred = torch.argmax(student_logits, dim=1)

            temperature, r_map, _, _ = newton_temperature_map(teacher_logits, valid, args)
            wrong = ((teacher_pred != target) & valid).float()
            valid_count = max(float(valid.float().sum().item()), 1.0)
            wrong_rate = float(wrong.sum().item() / valid_count)

            r_np = to_numpy_map(r_map)
            t_np = to_numpy_map(temperature)
            valid_np = to_numpy_map(valid).astype(bool)
            r_stat = map_stats(r_np, valid_np)
            t_stat = map_stats(t_np, valid_np)
            score = wrong_rate * (1.0 + r_stat['p95'])

            candidate = {
                'score': score,
                'index': idx,
                'name': name,
                'image_path': image_path,
                'wrong_rate': wrong_rate,
                'r_mean': r_stat['mean'],
                'r_p95': r_stat['p95'],
                'r_max': r_stat['max'],
                't_mean': t_stat['mean'],
                't_p95': t_stat['p95'],
                't_max_actual': t_stat['max'],
                't_min': float(args.covar_temp_min),
                't_max': float(min(args.covar_temp_max, max(float(t_np[valid_np].max()) if valid_np.any() else args.covar_temp_max, args.covar_temp_min))),
                'valid_mask': valid_np,
                'image_rgb': image_tensor_to_rgb(image[0]),
                'gt_rgb': label_to_rgb(to_numpy_map(target).astype(np.int32)),
                'teacher_rgb': label_to_rgb(to_numpy_map(teacher_pred).astype(np.int32)),
                'student_rgb': label_to_rgb(to_numpy_map(student_pred).astype(np.int32)) if student_pred is not None else None,
                'r_map': r_np,
                'temperature_map': t_np,
            }
            candidates.append(candidate)
            candidates = sorted(candidates, key=lambda item: item['score'], reverse=True)[:args.num_examples]

    summary_rows = []
    for rank, candidate in enumerate(sorted(candidates, key=lambda item: item['score'], reverse=True), start=1):
        figure_path = output_dir / f"h2_rank{rank:02d}_{candidate['name']}.png"
        save_figure(candidate, figure_path)
        row = {
            'rank': rank,
            'index': candidate['index'],
            'name': candidate['name'],
            'image_path': candidate['image_path'],
            'figure': str(figure_path),
            'score': candidate['score'],
            'teacher_wrong_rate': candidate['wrong_rate'],
            'r_mean': candidate['r_mean'],
            'r_p95': candidate['r_p95'],
            'r_max': candidate['r_max'],
            't_mean': candidate['t_mean'],
            't_p95': candidate['t_p95'],
            't_max': candidate['t_max_actual'],
        }
        summary_rows.append(row)

    csv_path = output_dir / 'h2_visual_examples_summary.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = output_dir / 'h2_visual_examples_summary.json'
    json_path.write_text(json.dumps(summary_rows, indent=2) + '\n')

    print(f'Wrote summary CSV: {csv_path}')
    print(f'Wrote summary JSON: {json_path}')
    for row in summary_rows:
        print(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
