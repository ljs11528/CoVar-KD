#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def as_float(value):
    if value in (None, ''):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_curve(path, variant=None):
    path = Path(path)
    if not path.exists():
        return {}
    rows = {}
    with path.open('r', newline='') as f:
        for row in csv.DictReader(f):
            if variant and row.get('variant') != variant:
                continue
            step = as_float(row.get('step'))
            miou = as_float(row.get('miou'))
            if step is None or miou is None:
                continue
            rows[int(step)] = row
    return rows


def fmt(value):
    if value is None:
        return ''
    return f'{value:.3f}'


def delta(a, b):
    if a is None or b is None:
        return None
    return a - b


def main():
    parser = argparse.ArgumentParser(description='Compare full-run progress against reference validation curves.')
    parser.add_argument('--full', default='runs/covar_calib_conf_full/full_validation_curve_80000.csv')
    parser.add_argument('--full-variant', default='04_calib_gamma1_tmax4')
    parser.add_argument('--reference', default='runs/covar_calib_conf_stage1/existing_root_logs_validation_curve.csv')
    parser.add_argument('--reference-variant', default='train1')
    parser.add_argument('--stage1', default='runs/covar_calib_conf_stage1/stage1_validation_curve_10000.csv')
    parser.add_argument('--stage1-variant', default='04_calib_gamma1_tmax4')
    parser.add_argument('--output', default='')
    args = parser.parse_args()

    full = load_curve(args.full, args.full_variant)
    ref = load_curve(args.reference, args.reference_variant)
    stage1 = load_curve(args.stage1, args.stage1_variant)

    fieldnames = [
        'step', 'full_variant', 'full_miou', 'reference_variant', 'reference_miou',
        'delta_vs_reference', 'stage1_variant', 'stage1_miou', 'delta_vs_stage1',
    ]
    out_rows = []
    for step in sorted(full):
        full_miou = as_float(full[step].get('miou'))
        ref_miou = as_float(ref.get(step, {}).get('miou'))
        stage1_miou = as_float(stage1.get(step, {}).get('miou'))
        out_rows.append({
            'step': step,
            'full_variant': args.full_variant,
            'full_miou': fmt(full_miou),
            'reference_variant': args.reference_variant,
            'reference_miou': fmt(ref_miou),
            'delta_vs_reference': fmt(delta(full_miou, ref_miou)),
            'stage1_variant': args.stage1_variant,
            'stage1_miou': fmt(stage1_miou),
            'delta_vs_stage1': fmt(delta(full_miou, stage1_miou)),
        })

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f'Wrote {len(out_rows)} rows to {output}')

    writer = csv.DictWriter(__import__('sys').stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)
    if out_rows:
        latest = out_rows[-1]
        print(
            'Latest: '
            f"step={latest['step']} full={latest['full_miou']} "
            f"ref={latest['reference_miou']} "
            f"delta_ref={latest['delta_vs_reference']} "
            f"stage1={latest['stage1_miou']} "
            f"delta_stage1={latest['delta_vs_stage1']}"
        )
    else:
        print('No full-run validation rows found yet.')


if __name__ == '__main__':
    main()
