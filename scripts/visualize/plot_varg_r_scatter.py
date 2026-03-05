import argparse
import os

import numpy as np


def quantile_sample_pairs(x, y, sample_num=1000):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size == 0 or y.size == 0:
        return x, y
    if x.size != y.size:
        raise ValueError(f"x and y must have same length, got {x.size} and {y.size}")

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    if x_sorted.size > sample_num:
        n = x_sorted.size
        q_idx = (np.arange(sample_num) * n) // sample_num
        x_sorted = x_sorted[q_idx]
        y_sorted = y_sorted[q_idx]
    return x_sorted, y_sorted


def save_varg_r_scatter(var_g, r, output_path, sample_num=1000, title=None):
    try:
        import torch
        if isinstance(var_g, torch.Tensor):
            var_g = var_g.detach().float().cpu().numpy()
        if isinstance(r, torch.Tensor):
            r = r.detach().float().cpu().numpy()
    except Exception:
        pass

    x_sampled, y_sampled = quantile_sample_pairs(var_g, r, sample_num=sample_num)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib is required to save scatter plot: {e}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.scatter(x_sampled, y_sampled, s=9, alpha=0.45)
    plt.xlabel('var(g)')
    plt.ylabel('r (scaled_residual_variance)')
    plt.title(title or 'var(g)-r Scatter')
    plt.grid(alpha=0.25)
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close()
    return int(x_sampled.size)


def _load_pairs(input_path, x_key='var_g', y_key='r'):
    suffix = os.path.splitext(input_path)[1].lower()
    if suffix == '.npz':
        arr = np.load(input_path)
        if x_key not in arr or y_key not in arr:
            raise KeyError(f"npz must contain keys '{x_key}' and '{y_key}'")
        return arr[x_key], arr[y_key]

    if suffix in ['.pt', '.pth']:
        import torch
        try:
            obj = torch.load(input_path, map_location='cpu', weights_only=True)
        except TypeError:
            obj = torch.load(input_path, map_location='cpu')

        if isinstance(obj, dict):
            if x_key in obj and y_key in obj:
                return obj[x_key], obj[y_key]

            if 'pairs' in obj and isinstance(obj['pairs'], dict):
                pairs = obj['pairs']
                if x_key in pairs and y_key in pairs:
                    return pairs[x_key], pairs[y_key]

            top_keys = list(obj.keys())[:15]
            raise KeyError(
                f"Input '{input_path}' does not contain required keys '{x_key}' and '{y_key}'. "
                f"This file looks like a model checkpoint/state_dict (top keys sample: {top_keys}). "
                f"Please provide a .pt/.npz file that stores sampled pairs, e.g. {{'var_g': ..., 'r': ...}}."
            )
        raise TypeError('pt/pth input must be a dict containing var_g and r arrays')

    raise ValueError('Unsupported input format, use .npz or .pt/.pth')


def parse_args():
    parser = argparse.ArgumentParser(description='Plot var(g)-r scatter with quantile sampling')
    parser.add_argument('--input', type=str, required=True,
                        help='input file path (.npz or .pt/.pth) containing var_g and r arrays')
    parser.add_argument('--output', type=str, required=True,
                        help='output image path (.png)')
    parser.add_argument('--samples', type=int, default=1000,
                        help='number of quantile-sampled points to draw')
    parser.add_argument('--x-key', type=str, default='var_g', help='x key in input file')
    parser.add_argument('--y-key', type=str, default='r', help='y key in input file')
    parser.add_argument('--title', type=str, default='var(g)-r Scatter', help='plot title')
    return parser.parse_args()


def main():
    args = parse_args()
    var_g, r = _load_pairs(args.input, x_key=args.x_key, y_key=args.y_key)
    used = save_varg_r_scatter(var_g, r, args.output, sample_num=args.samples, title=args.title)
    print(f"Saved scatter to {args.output} with {used} quantile-sampled points")


if __name__ == '__main__':
    main()
