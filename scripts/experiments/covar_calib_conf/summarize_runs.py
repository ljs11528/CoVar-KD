#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


VAL_START_RE = re.compile(r"Start validation, Total sample:\s+(\d+)")
VAL_RE = re.compile(r"Sample:\s+(\d+), Validation pixAcc:\s+([0-9.]+), mIoU:\s+([0-9.]+)")
ITER_RE = re.compile(r"Iters:\s+(\d+)/(\d+)")
T_RE = re.compile(r"T_mean:\s+([0-9.eE+-]+).*?T_min:\s+([0-9.eE+-]+).*?T_max:\s+([0-9.eE+-]+)")
CALIB_RE = re.compile(
    r"CalibT:.*?err\(logc\):\s+([0-9.eE+-]+)(?:/([0-9.eE+-]+))?.*?"
    r"c0/tgt/cf:\s+([0-9.eE+-]+)/([0-9.eE+-]+)/([0-9.eE+-]+).*?"
    r"Tq10/50/90:\s+([0-9.eE+-]+)/([0-9.eE+-]+)/([0-9.eE+-]+).*?"
    r"Tmax:\s+([0-9.eE+-]+)%"
)
NEWT_RE = re.compile(r"NewtonT:.*?\|dr/dT\|:\s+([0-9.eE+-]+).*?clamp:\s+([0-9.eE+-]+)%")


def parse_log(path, variant=None):
    row = {
        'variant': variant or path.parent.name or path.stem,
        'log_path': str(path),
        'best_miou': '',
        'best_step': '',
        'last_miou': '',
        'last_step': '',
        'last_iter': '',
        'max_iterations': '',
        'training_complete': '',
        'last_t_mean': '',
        'last_t_min': '',
        'last_t_max': '',
        'last_calib_err_all': '',
        'last_calib_err_active': '',
        'last_c0': '',
        'last_c_target': '',
        'last_c_final': '',
        'last_t_q10': '',
        'last_t_q50': '',
        'last_t_q90': '',
        'last_tmax_ratio': '',
        'last_newton_abs_dr': '',
        'last_newton_clamp_ratio': '',
    }
    current_step = ''
    best_miou = None
    active_val = None

    def finalize_validation():
        nonlocal active_val, best_miou
        if not active_val or active_val.get('last_miou') is None:
            active_val = None
            return
        miou = active_val['last_miou']
        row['last_miou'] = miou
        row['last_step'] = active_val['step']
        if best_miou is None or miou > best_miou:
            best_miou = miou
            row['best_miou'] = miou
            row['best_step'] = active_val['step']
        active_val = None

    with path.open('r', errors='ignore') as f:
        for line in f:
            m = ITER_RE.search(line)
            if m:
                if active_val and active_val.get('complete'):
                    finalize_validation()
                current_step = m.group(1)
                last_iter = int(m.group(1))
                max_iterations = int(m.group(2))
                row['last_iter'] = last_iter
                row['max_iterations'] = max_iterations
                row['training_complete'] = last_iter >= max_iterations

            m = VAL_START_RE.search(line)
            if m:
                finalize_validation()
                active_val = {
                    'step': current_step,
                    'total': int(m.group(1)),
                    'last_sample': 0,
                    'last_miou': None,
                    'complete': False,
                }

            m = VAL_RE.search(line)
            if m:
                sample_idx = int(m.group(1))
                miou = float(m.group(3))
                if active_val is None:
                    active_val = {
                        'step': current_step,
                        'total': sample_idx,
                        'last_sample': sample_idx,
                        'last_miou': miou,
                        'complete': False,
                    }
                else:
                    active_val['last_sample'] = sample_idx
                    active_val['last_miou'] = miou
                if sample_idx >= int(active_val.get('total') or sample_idx):
                    active_val['complete'] = True

            m = T_RE.search(line)
            if m:
                row['last_t_mean'] = float(m.group(1))
                row['last_t_min'] = float(m.group(2))
                row['last_t_max'] = float(m.group(3))

            m = CALIB_RE.search(line)
            if m:
                row['last_calib_err_all'] = float(m.group(1))
                row['last_calib_err_active'] = float(m.group(2)) if m.group(2) else ''
                row['last_c0'] = float(m.group(3))
                row['last_c_target'] = float(m.group(4))
                row['last_c_final'] = float(m.group(5))
                row['last_t_q10'] = float(m.group(6))
                row['last_t_q50'] = float(m.group(7))
                row['last_t_q90'] = float(m.group(8))
                row['last_tmax_ratio'] = float(m.group(9))

            m = NEWT_RE.search(line)
            if m:
                row['last_newton_abs_dr'] = float(m.group(1))
                row['last_newton_clamp_ratio'] = float(m.group(2))

    if active_val and active_val.get('complete'):
        finalize_validation()

    return row


def find_logs(root):
    root = Path(root)
    logs = sorted(root.glob('*/*_log.txt'))
    logs += sorted(root.glob('*/*.log'))
    return sorted(set(logs))


def main():
    parser = argparse.ArgumentParser(description='Summarize CoVar calibration experiment logs.')
    parser.add_argument('--root', default='data/winycg/checkpoints/cirkd_checkpoints/voc/covar_calib_conf')
    parser.add_argument(
        '--logs',
        nargs='*',
        default=None,
        help='Optional explicit log files to summarize; variants use the log stem.',
    )
    parser.add_argument('--output', default='runs/covar_calib_conf_summary.csv')
    args = parser.parse_args()

    rows = [parse_log(path) for path in find_logs(args.root)]
    if args.logs:
        rows.extend(parse_log(Path(path), variant=Path(path).stem) for path in args.logs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'variant', 'best_miou', 'best_step', 'last_miou', 'last_step',
        'last_iter', 'max_iterations', 'training_complete',
        'last_t_mean', 'last_t_min', 'last_t_max',
        'last_calib_err_all', 'last_calib_err_active',
        'last_c0', 'last_c_target', 'last_c_final',
        'last_t_q10', 'last_t_q50', 'last_t_q90', 'last_tmax_ratio',
        'last_newton_abs_dr', 'last_newton_clamp_ratio', 'log_path',
    ]
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Wrote {len(rows)} rows to {output}')
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
