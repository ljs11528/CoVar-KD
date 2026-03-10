import argparse
import os

import numpy as np


def _to_numpy_1d(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().float().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x).reshape(-1)


def quantile_sample_triplets(x, y1, y2, sample_num=1000):
    x = _to_numpy_1d(x)
    y1 = _to_numpy_1d(y1)
    y2 = _to_numpy_1d(y2)
    if x.size == 0 or y1.size == 0 or y2.size == 0:
        return x, y1, y2
    if x.size != y1.size or x.size != y2.size:
        raise ValueError(
            f"x/y1/y2 must have same length, got {x.size}, {y1.size}, {y2.size}"
        )

    order = np.argsort(x)
    x_sorted = x[order]
    y1_sorted = y1[order]
    y2_sorted = y2[order]

    if x_sorted.size > sample_num:
        n = x_sorted.size
        q_idx = (np.arange(sample_num) * n) // sample_num
        x_sorted = x_sorted[q_idx]
        y1_sorted = y1_sorted[q_idx]
        y2_sorted = y2_sorted[q_idx]
    return x_sorted, y1_sorted, y2_sorted


def save_varg_r_scatter(g2, var_g, r, output_path, sample_num=1000, title=None):
    # x-axis is r; left y-axis is g^2; right y-axis is var(g).
    r_sampled, g2_sampled, varg_sampled = quantile_sample_triplets(r, g2, var_g, sample_num=sample_num)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib is required to save scatter plot: {e}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Quantify overlap: for probability KD, mean_c(g) is often near zero,
    # so E[g^2] and var(g) can be nearly identical.
    abs_diff = np.abs(g2_sampled - varg_sampled)
    mean_abs_diff = float(abs_diff.mean()) if abs_diff.size > 0 else 0.0
    max_abs_diff = float(abs_diff.max()) if abs_diff.size > 0 else 0.0
    if g2_sampled.size > 1:
        corr = float(np.corrcoef(g2_sampled, varg_sampled)[0, 1])
    else:
        corr = float('nan')

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Draw var(g) first, then g^2 with hollow markers so overlap remains visible.
    s2 = ax2.scatter(r_sampled, varg_sampled, s=10, alpha=0.5, c='tab:red', label='var(g)', zorder=2)
    s1 = ax1.scatter(
        r_sampled,
        g2_sampled,
        s=24,
        facecolors='none',
        edgecolors='tab:blue',
        linewidths=0.8,
        label='g^2',
        zorder=3,
    )

    ax1.set_xlabel('r (scaled_residual_variance)')
    ax1.set_ylabel('g^2', color='tab:blue')
    ax2.set_ylabel('var(g)', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(alpha=0.25)
    ax1.set_title(title or 'r-g^2-var(g) Scatter')

    handles = [s1, s2]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc='best')

    ax1.text(
        0.02,
        0.98,
        f"corr(g^2,var(g))={corr:.6f}\nmean|g^2-var|={mean_abs_diff:.3e}\nmax|g^2-var|={max_abs_diff:.3e}",
        transform=ax1.transAxes,
        ha='left',
        va='top',
        fontsize=8,
        bbox={'facecolor': 'white', 'alpha': 0.75, 'edgecolor': 'none'},
    )

    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return {
        'num_points': int(r_sampled.size),
        'corr': corr,
        'mean_abs_diff': mean_abs_diff,
        'max_abs_diff': max_abs_diff,
    }


def _load_pairs(input_path, r_key='r', g2_key='g2', var_key='var_g'):
    suffix = os.path.splitext(input_path)[1].lower()
    if suffix == '.npz':
        arr = np.load(input_path)
        if r_key not in arr or var_key not in arr:
            raise KeyError(f"npz must contain keys '{r_key}' and '{var_key}'")
        if g2_key in arr:
            return arr[r_key], arr[g2_key], arr[var_key]
        if 'g' in arr:
            return arr[r_key], np.asarray(arr['g']) ** 2, arr[var_key]
        raise KeyError(f"npz must contain '{g2_key}' (or legacy 'g')")

    if suffix in ['.pt', '.pth']:
        import torch
        try:
            obj = torch.load(input_path, map_location='cpu', weights_only=True)
        except TypeError:
            obj = torch.load(input_path, map_location='cpu')

        if isinstance(obj, dict):
            if r_key in obj and var_key in obj:
                if g2_key in obj:
                    return obj[r_key], obj[g2_key], obj[var_key]
                if 'g' in obj:
                    return obj[r_key], _to_numpy_1d(obj['g']) ** 2, obj[var_key]

            if 'pairs' in obj and isinstance(obj['pairs'], dict):
                pairs = obj['pairs']
                if r_key in pairs and var_key in pairs:
                    if g2_key in pairs:
                        return pairs[r_key], pairs[g2_key], pairs[var_key]
                    if 'g' in pairs:
                        return pairs[r_key], _to_numpy_1d(pairs['g']) ** 2, pairs[var_key]

            top_keys = list(obj.keys())[:15]
            raise KeyError(
                f"Input '{input_path}' does not contain required keys '{r_key}', '{g2_key}', '{var_key}'. "
                f"This file looks like a model checkpoint/state_dict (top keys sample: {top_keys}). "
                f"Please provide a .pt/.npz file that stores sampled pairs, e.g. "
                f"{{'r': ..., 'g2': ..., 'var_g': ...}}."
            )
        raise TypeError('pt/pth input must be a dict containing r, g2 and var_g arrays')

    raise ValueError('Unsupported input format, use .npz or .pt/.pth')


def parse_args():
    parser = argparse.ArgumentParser(description='Plot r-g^2-var(g) scatter with quantile sampling')
    parser.add_argument('--input', type=str, required=True,
                        help='input file path (.npz or .pt/.pth) containing r, g2 and var_g arrays')
    parser.add_argument('--output', type=str, required=True,
                        help='output image path (.png)')
    parser.add_argument('--samples', type=int, default=1000,
                        help='number of quantile-sampled points to draw')
    parser.add_argument('--r-key', type=str, default='r', help='r key in input file')
    parser.add_argument('--g2-key', type=str, default='g2', help='g^2 key in input file')
    parser.add_argument('--var-key', type=str, default='var_g', help='var(g) key in input file')
    parser.add_argument('--title', type=str, default='r-g^2-var(g) Scatter', help='plot title')
    return parser.parse_args()


def main():
    args = parse_args()
    r, g2, var_g = _load_pairs(args.input, r_key=args.r_key, g2_key=args.g2_key, var_key=args.var_key)
    stats = save_varg_r_scatter(g2, var_g, r, args.output, sample_num=args.samples, title=args.title)
    print(
        f"Saved scatter to {args.output} with {stats['num_points']} quantile-sampled points; "
        f"corr={stats['corr']:.6f}, mean|g^2-var|={stats['mean_abs_diff']:.3e}, "
        f"max|g^2-var|={stats['max_abs_diff']:.3e}"
    )


if __name__ == '__main__':
    main()
