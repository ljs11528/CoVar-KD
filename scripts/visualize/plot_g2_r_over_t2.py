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


def random_sample_pairs(x, y, sample_num=1000, seed=None):
    x = _to_numpy_1d(x)
    y = _to_numpy_1d(y)
    if x.size == 0 or y.size == 0:
        return x, y
    if x.size != y.size:
        raise ValueError(f"x and y must have same length, got {x.size} and {y.size}")

    n = x.size
    k = min(sample_num, n)
    rng = np.random.default_rng(seed)
    if k < n:
        idx = rng.choice(n, size=k, replace=False)
        return x[idx], y[idx]
    return x, y


def save_g2_r_over_t2_scatter(g2, r_over_t2, output_path, sample_num=1000, title=None, seed=None):
    x, y = random_sample_pairs(g2, r_over_t2, sample_num=sample_num, seed=seed)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib is required to save scatter plot: {e}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if x.size > 1:
        corr = float(np.corrcoef(x, y)[0, 1])
    else:
        corr = float('nan')

    plt.figure()
    plt.scatter(x, y, s=10, alpha=0.45, c='tab:green')
    plt.xlabel('g^2')
    plt.ylabel('r / T^2')
    plt.title(title or 'g^2 vs r/T^2')
    plt.grid(alpha=0.3)
    plt.text(
        0.02,
        0.98,
        f"corr={corr:.6f}",
        transform=plt.gca().transAxes,
        ha='left',
        va='top',
        fontsize=9,
        bbox={'facecolor': 'white', 'alpha': 0.75, 'edgecolor': 'none'},
    )
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close()

    return {'num_points': int(x.size), 'corr': corr}


def _load_pairs(input_path, g2_key='g2', y_key='r_over_t2'):
    suffix = os.path.splitext(input_path)[1].lower()

    if suffix == '.npz':
        arr = np.load(input_path)
        if g2_key in arr and y_key in arr:
            return arr[g2_key], arr[y_key]
        if g2_key in arr and 'r' in arr and 'temp' in arr:
            return arr[g2_key], np.asarray(arr['r']) / (np.asarray(arr['temp']) ** 2 + 1e-12)
        raise KeyError(f"npz must contain keys '{g2_key}' and '{y_key}' (or 'r' and 'temp')")

    if suffix in ['.pt', '.pth']:
        import torch
        try:
            obj = torch.load(input_path, map_location='cpu', weights_only=True)
        except TypeError:
            obj = torch.load(input_path, map_location='cpu')

        if isinstance(obj, dict):
            if g2_key in obj and y_key in obj:
                return obj[g2_key], obj[y_key]
            if g2_key in obj and 'r' in obj and 'temp' in obj:
                temp = _to_numpy_1d(obj['temp'])
                return obj[g2_key], _to_numpy_1d(obj['r']) / (temp ** 2 + 1e-12)

            if 'pairs' in obj and isinstance(obj['pairs'], dict):
                pairs = obj['pairs']
                if g2_key in pairs and y_key in pairs:
                    return pairs[g2_key], pairs[y_key]
                if g2_key in pairs and 'r' in pairs and 'temp' in pairs:
                    temp = _to_numpy_1d(pairs['temp'])
                    return pairs[g2_key], _to_numpy_1d(pairs['r']) / (temp ** 2 + 1e-12)

            top_keys = list(obj.keys())[:15]
            raise KeyError(
                f"Input '{input_path}' does not contain '{g2_key}' and '{y_key}'. "
                f"Top keys sample: {top_keys}."
            )

        raise TypeError('pt/pth input must be a dict containing pair arrays')

    raise ValueError('Unsupported input format, use .npz or .pt/.pth')


def parse_args():
    parser = argparse.ArgumentParser(description='Plot g^2 vs r/T^2 scatter with random sampling')
    parser.add_argument('--input', type=str, required=True,
                        help='input file path (.npz or .pt/.pth) containing g2 and r_over_t2')
    parser.add_argument('--output', type=str, required=True,
                        help='output image path (.png)')
    parser.add_argument('--samples', type=int, default=1000,
                        help='number of random sampled points to draw')
    parser.add_argument('--g2-key', type=str, default='g2', help='g^2 key in input file')
    parser.add_argument('--y-key', type=str, default='r_over_t2', help='r/T^2 key in input file')
    parser.add_argument('--seed', type=int, default=None, help='random seed for sampling')
    parser.add_argument('--title', type=str, default='g^2 vs r/T^2', help='plot title')
    return parser.parse_args()


def main():
    args = parse_args()
    g2, r_over_t2 = _load_pairs(args.input, g2_key=args.g2_key, y_key=args.y_key)
    stats = save_g2_r_over_t2_scatter(
        g2,
        r_over_t2,
        args.output,
        sample_num=args.samples,
        title=args.title,
        seed=args.seed,
    )
    print(
        f"Saved scatter to {args.output} with {stats['num_points']} random sampled points; "
        f"corr={stats['corr']:.6f}"
    )


if __name__ == '__main__':
    main()
