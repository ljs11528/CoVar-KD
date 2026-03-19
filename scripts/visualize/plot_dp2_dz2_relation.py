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


def _load_pairs(input_path, dz2_key="delta_z_norm2", dp2_key="delta_p_norm2"):
    suffix = os.path.splitext(input_path)[1].lower()

    if suffix == ".npz":
        arr = np.load(input_path)
        if dz2_key not in arr or dp2_key not in arr:
            raise KeyError(f"npz must contain keys '{dz2_key}' and '{dp2_key}'")
        return arr[dz2_key], arr[dp2_key]

    if suffix in [".pt", ".pth"]:
        try:
            import torch
        except Exception as e:
            raise RuntimeError(f"Loading .pt/.pth requires torch: {e}")

        try:
            obj = torch.load(input_path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(input_path, map_location="cpu")

        if isinstance(obj, dict):
            if dz2_key in obj and dp2_key in obj:
                return obj[dz2_key], obj[dp2_key]
            if "pairs" in obj and isinstance(obj["pairs"], dict):
                pairs = obj["pairs"]
                if dz2_key in pairs and dp2_key in pairs:
                    return pairs[dz2_key], pairs[dp2_key]

            top_keys = list(obj.keys())[:15]
            raise KeyError(
                f"Input '{input_path}' missing keys '{dz2_key}' and '{dp2_key}'. "
                f"Top keys sample: {top_keys}"
            )

        raise TypeError("pt/pth input must be a dict containing pair arrays")

    raise ValueError("Unsupported input format, use .npz or .pt/.pth")


def _sample_points(x, y, max_points=0, seed=0):
    if max_points is None or max_points <= 0 or x.size <= max_points:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=max_points, replace=False)
    return x[idx], y[idx]


def _fit_through_origin(x, y):
    denom = float(np.dot(x, x)) + 1e-12
    k = float(np.dot(x, y) / denom)
    y_hat = k * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return k, r2


def save_dp2_dz2_relation_plot(dz2, dp2, output_path, max_points=0, seed=0, title=None):
    x = _to_numpy_1d(dz2)
    y = _to_numpy_1d(dp2)
    if x.size == 0 or y.size == 0:
        raise ValueError("Input arrays are empty")
    if x.size != y.size:
        raise ValueError(f"delta_z^2 and delta_p^2 must have same length, got {x.size} and {y.size}")

    keep = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (y >= 0.0)
    x = x[keep]
    y = y[keep]
    if x.size == 0:
        raise ValueError("No valid non-negative finite pairs left after filtering")

    x, y = _sample_points(x, y, max_points=max_points, seed=seed)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    k, r2_origin = _fit_through_origin(x, y)
    corr = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else float("nan")
    ct_est = 1.0 - k

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib is required: {e}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(7.2, 5.6))
    plt.scatter(x, y, s=8, alpha=0.35, color="tab:blue", label="samples")

    xx = np.linspace(float(x.min()), float(x.max()), 200)
    yy = k * xx
    plt.plot(
        xx,
        yy,
        color="tab:red",
        linewidth=1.8,
        label=f"fit y=(1-C_T)x, 1-C_T={k:.5f}, C_T={ct_est:.5f}",
    )

    plt.xlabel("||delta z||^2")
    plt.ylabel("||delta p||^2")
    plt.title(title or "delta_p^2 vs delta_z^2")
    plt.grid(alpha=0.28)
    plt.legend(loc="best")
    plt.text(
        0.02,
        0.98,
        f"corr={corr:.6f}\nR^2(through-origin)={r2_origin:.6f}",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
    )

    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()

    return {
        "num_points": int(x.size),
        "corr": corr,
        "r2_through_origin": float(r2_origin),
        "one_minus_ct": float(k),
        "ct_est": float(ct_est),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot relation between delta_p^2 and delta_z^2 to verify ||delta p||^2 ~= (1-C_T)||delta z||^2"
    )
    parser.add_argument("--input", type=str, required=True, help="input .pt/.pth/.npz containing delta_z_norm2 and delta_p_norm2")
    parser.add_argument("--output", type=str, required=True, help="output plot path (.png)")
    parser.add_argument("--dz2-key", type=str, default="delta_z_norm2", help="key for delta_z^2")
    parser.add_argument("--dp2-key", type=str, default="delta_p_norm2", help="key for delta_p^2")
    parser.add_argument("--max-points", type=int, default=0, help="max points to draw; <=0 means all")
    parser.add_argument("--seed", type=int, default=0, help="random seed for down-sampling")
    parser.add_argument("--title", type=str, default="delta_p^2 vs delta_z^2", help="plot title")
    return parser.parse_args()


def main():
    args = parse_args()
    dz2, dp2 = _load_pairs(args.input, dz2_key=args.dz2_key, dp2_key=args.dp2_key)
    stats = save_dp2_dz2_relation_plot(
        dz2,
        dp2,
        args.output,
        max_points=args.max_points,
        seed=args.seed,
        title=args.title,
    )
    print(
        f"Saved plot to {args.output} with {stats['num_points']} points; "
        f"corr={stats['corr']:.6f}, R2_origin={stats['r2_through_origin']:.6f}, "
        f"1-C_T={stats['one_minus_ct']:.6f}, C_T={stats['ct_est']:.6f}"
    )


if __name__ == "__main__":
    main()
