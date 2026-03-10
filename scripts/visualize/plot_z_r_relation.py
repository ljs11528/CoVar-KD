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


def random_sample_triplets(x, y_left, y_right, sample_num=1000, seed=0):
    x = _to_numpy_1d(x)
    y_left = _to_numpy_1d(y_left)
    y_right = _to_numpy_1d(y_right)
    if x.size == 0 or y_left.size == 0 or y_right.size == 0:
        return x, y_left, y_right
    if x.size != y_left.size or x.size != y_right.size:
        raise ValueError(
            f"x, y_left and y_right must have same length, got {x.size}, {y_left.size}, {y_right.size}"
        )

    n = x.size
    if n > sample_num:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample_num, replace=False)
        x = x[idx]
        y_left = y_left[idx]
        y_right = y_right[idx]

    order = np.argsort(x)
    x = x[order]
    y_left = y_left[order]
    y_right = y_right[order]

    return x, y_left, y_right


def _linear_fit(x, y):
    if x.size < 2:
        return None
    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    corr = np.corrcoef(x, y)[0, 1]
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r2': float(r2),
        'corr': float(corr),
    }


def save_z_r_relation_plot(delta_z_norm2, delta_p_norm2, r, output_path, sample_num=1000, title=None):
    # Target relation: x=||delta z||^2, y_left=||delta p||^2, y_right=r
    x, y_left, y_right = random_sample_triplets(
        delta_z_norm2, delta_p_norm2, r, sample_num=sample_num
    )

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib is required to save z-r plot: {e}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fit_left = _linear_fit(x, y_left)
    fit_right = _linear_fit(x, y_right)

    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx()

    left_scatter = ax_left.scatter(
        x, y_left, s=10, alpha=0.35, color='tab:blue', label='||delta p||^2'
    )
    right_scatter = ax_right.scatter(
        x, y_right, s=10, alpha=0.35, color='tab:orange', label='r=||delta p||^2/(1-C_t)'
    )

    xx = np.linspace(float(x.min()), float(x.max()), 100) if x.size > 0 else None
    left_line = None
    right_line = None
    if fit_left is not None and xx is not None:
        yy_left = fit_left['slope'] * xx + fit_left['intercept']
        left_line = ax_left.plot(
            xx,
            yy_left,
            color='tab:blue',
            linewidth=1.5,
            alpha=0.9,
            label=f"fit(||delta p||^2): R^2={fit_left['r2']:.4f}",
        )[0]
    if fit_right is not None and xx is not None:
        yy_right = fit_right['slope'] * xx + fit_right['intercept']
        right_line = ax_right.plot(
            xx,
            yy_right,
            color='tab:red',
            linewidth=1.5,
            alpha=0.9,
            label=f"fit(r): R^2={fit_right['r2']:.4f}",
        )[0]

    ax_left.set_xlabel('||delta z||^2')
    ax_left.set_ylabel('||delta p||^2', color='tab:blue')
    ax_right.set_ylabel('r = ||delta p||^2 / (1 - C_t)', color='tab:orange')
    ax_left.set_title(title or 'Relation: ||delta z||^2 vs ||delta p||^2 and r')
    ax_left.grid(alpha=0.3)

    handles = [left_scatter, right_scatter]
    if left_line is not None:
        handles.append(left_line)
    if right_line is not None:
        handles.append(right_line)
    labels = [h.get_label() for h in handles]
    ax_left.legend(handles, labels, loc='best')

    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close()

    return {
        'num_points': int(x.size),
        'fit_delta_p2': fit_left,
        'fit_r': fit_right,
    }


def _load_pairs(input_path, z_key='delta_z_norm2', dp_key='delta_p_norm2', r_key='r'):
    suffix = os.path.splitext(input_path)[1].lower()

    if suffix == '.npz':
        arr = np.load(input_path)
        if z_key not in arr or dp_key not in arr or r_key not in arr:
            raise KeyError(f"npz must contain keys '{z_key}', '{dp_key}' and '{r_key}'")
        return arr[z_key], arr[dp_key], arr[r_key]

    if suffix in ['.pt', '.pth']:
        import torch
        try:
            obj = torch.load(input_path, map_location='cpu', weights_only=True)
        except TypeError:
            obj = torch.load(input_path, map_location='cpu')

        if isinstance(obj, dict):
            if z_key in obj and r_key in obj:
                if dp_key not in obj:
                    raise KeyError(f"pt/pth input must contain key '{dp_key}'")
                return obj[z_key], obj[dp_key], obj[r_key]

            if 'pairs' in obj and isinstance(obj['pairs'], dict):
                pairs = obj['pairs']
                if z_key in pairs and dp_key in pairs and r_key in pairs:
                    return pairs[z_key], pairs[dp_key], pairs[r_key]

            top_keys = list(obj.keys())[:15]
            raise KeyError(
                f"Input '{input_path}' does not contain required keys '{z_key}', '{dp_key}' and '{r_key}'. "
                f"Top keys sample: {top_keys}."
            )

        raise TypeError('pt/pth input must be a dict containing delta_z_norm2, delta_p_norm2 and r arrays')

    raise ValueError('Unsupported input format, use .npz or .pt/.pth')


def parse_args():
    parser = argparse.ArgumentParser(description='Plot relation between ||delta z||^2, ||delta p||^2 and r')
    parser.add_argument('--input', type=str, required=True,
                        help='input file path (.npz or .pt/.pth) containing delta_z_norm2, delta_p_norm2 and r arrays')
    parser.add_argument('--output', type=str, required=True,
                        help='output image path (.png)')
    parser.add_argument('--samples', type=int, default=1000,
                        help='number of randomly sampled points to draw')
    parser.add_argument('--z-key', type=str, default='delta_z_norm2', help='delta-z-squared key in input file')
    parser.add_argument('--dp-key', type=str, default='delta_p_norm2', help='delta-p-squared key in input file')
    parser.add_argument('--r-key', type=str, default='r', help='r key in input file')
    parser.add_argument('--title', type=str, default='Relation: ||delta z||^2 vs ||delta p||^2 and r',
                        help='plot title')
    return parser.parse_args()


def main():
    args = parse_args()
    z, dp, r = _load_pairs(args.input, z_key=args.z_key, dp_key=args.dp_key, r_key=args.r_key)
    stats = save_z_r_relation_plot(z, dp, r, args.output, sample_num=args.samples, title=args.title)
    fit_dp = stats['fit_delta_p2']
    fit_r = stats['fit_r']
    msg = f"Saved z-dp-r plot to {args.output} with {stats['num_points']} points"
    if fit_dp is not None:
        msg += f"; dp2 corr={fit_dp['corr']:.4f}, R^2={fit_dp['r2']:.4f}"
    if fit_r is not None:
        msg += f"; r corr={fit_r['corr']:.4f}, R^2={fit_r['r2']:.4f}"
    print(msg)


if __name__ == '__main__':
    main()
