#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Plot teacher r0 correctness-bin diagnostics.')
    parser.add_argument('--input', required=True, help='CSV from teacher_r0_correctness_bins.py')
    parser.add_argument('--output', required=True, help='Output PNG path')
    parser.add_argument('--stats-output', default=None, help='Optional JSON stats output path')
    parser.add_argument('--title', default='Teacher reliability score vs error rate')
    return parser.parse_args()


def load_rows(path):
    rows = []
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({
                'bin': int(row['bin']),
                'r_mean': float(row['r_mean']),
                'wrong_rate': float(row['wrong_rate']),
                'teacher_acc': float(row['teacher_acc']),
                'confidence_mean': float(row['confidence_mean']),
                'variance_mean': float(row['variance_mean']),
                'count': int(row['count']),
            })
    if not rows:
        raise RuntimeError(f'No rows loaded from {path}')
    return rows


def compute_stats(rows):
    r = np.asarray([row['r_mean'] for row in rows], dtype=np.float64)
    wrong = np.asarray([row['wrong_rate'] for row in rows], dtype=np.float64)
    acc = np.asarray([row['teacher_acc'] for row in rows], dtype=np.float64)

    corr_r_wrong = float(np.corrcoef(r, wrong)[0, 1]) if len(rows) > 1 else float('nan')
    corr_r_acc = float(np.corrcoef(r, acc)[0, 1]) if len(rows) > 1 else float('nan')
    slope_wrong = float(np.polyfit(r, wrong, deg=1)[0]) if len(rows) > 1 else float('nan')
    return {
        'bins': len(rows),
        'corr_r_wrong': corr_r_wrong,
        'corr_r_acc': corr_r_acc,
        'slope_wrong': slope_wrong,
        'first_wrong_rate': float(wrong[0]),
        'last_wrong_rate': float(wrong[-1]),
        'wrong_rate_ratio_last_first': float(wrong[-1] / max(wrong[0], 1e-12)),
    }


def save_plot(rows, output, title, stats):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f'matplotlib is required: {exc}')

    r = np.asarray([row['r_mean'] for row in rows], dtype=np.float64)
    wrong = np.asarray([row['wrong_rate'] for row in rows], dtype=np.float64)
    acc = np.asarray([row['teacher_acc'] for row in rows], dtype=np.float64)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(6.2, 4.0))
    ax2 = ax1.twinx()

    ax1.plot(r, wrong * 100.0, marker='o', linewidth=2.0, color='tab:red', label='Teacher error rate')
    ax2.plot(r, acc * 100.0, marker='s', linewidth=1.6, color='tab:blue', label='Teacher accuracy')

    ax1.set_xlabel('Mean reliability score r per bin')
    ax1.set_ylabel('Teacher error rate (%)', color='tab:red')
    ax2.set_ylabel('Teacher accuracy (%)', color='tab:blue')
    ax1.grid(alpha=0.3)
    ax1.set_title(title)

    text = f"corr(r,error)={stats['corr_r_wrong']:.3f}\nlast/first error={stats['wrong_rate_ratio_last_first']:.1f}x"
    ax1.text(
        0.56,
        0.18,
        text,
        transform=ax1.transAxes,
        ha='left',
        va='bottom',
        fontsize=9,
        bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'},
    )

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(
        lines,
        [line.get_label() for line in lines],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    rows = load_rows(args.input)
    stats = compute_stats(rows)
    save_plot(rows, args.output, args.title, stats)

    if args.stats_output:
        stats_path = Path(args.stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2) + '\n')

    print(json.dumps(stats, indent=2))
    print(f'Wrote plot: {args.output}')


if __name__ == '__main__':
    main()
