#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

CALIB_PREFIXES = ('03_', '04_', '05_')
EXPECTED_CALIB_VARIANTS = (
    '03_calib_gamma0_tmax4',
    '04_calib_gamma1_tmax4',
    '05_calib_gamma0_tmax8',
)


def as_float(value):
    if value is None or value == '':
        return None
    try:
        v = float(value)
    except ValueError:
        return None
    if math.isnan(v):
        return None
    return v


def as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None or value == '':
        return None
    return str(value).strip().lower() in ('1', 'true', 'yes', 'y')


def load_rows(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open('r', newline='') as f:
        return list(csv.DictReader(f))


def load_curve_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open('r', newline='') as f:
        return list(csv.DictReader(f))


def row_status(row):
    miou = as_float(row.get('best_miou'))
    if miou is None:
        return 'pending-no-val'
    complete = as_bool(row.get('training_complete'))
    if complete is False:
        return 'running'
    return 'ready'


def warnings_for(row, tmax_warn, active_err_warn):
    warnings = []
    variant = row.get('variant', '')
    if variant.startswith(CALIB_PREFIXES):
        tmax = as_float(row.get('last_tmax_ratio'))
        active_err = as_float(row.get('last_calib_err_active'))
        if tmax is not None and tmax >= tmax_warn:
            warnings.append(f'Tmax saturation {tmax:.1f}% >= {tmax_warn:.1f}%')
        if active_err is not None and active_err >= active_err_warn:
            warnings.append(f'active err {active_err:.2e} >= {active_err_warn:.2e}')
    return warnings


def print_table(rows, tmax_warn, active_err_warn):
    print('variant,status,best_miou,best_step,last_miou,last_step,last_iter,max_iterations,training_complete,last_t_mean,last_tmax_ratio,last_calib_err_active,warnings')
    for row in rows:
        warnings = '; '.join(warnings_for(row, tmax_warn, active_err_warn))
        print(','.join([
            row.get('variant', ''),
            row_status(row),
            row.get('best_miou', ''),
            row.get('best_step', ''),
            row.get('last_miou', ''),
            row.get('last_step', ''),
            row.get('last_iter', ''),
            row.get('max_iterations', ''),
            row.get('training_complete', ''),
            row.get('last_t_mean', ''),
            row.get('last_tmax_ratio', ''),
            row.get('last_calib_err_active', ''),
            warnings,
        ]))


def print_matched_step_delta(curve_rows, baseline_variant):
    by_variant = {}
    for row in curve_rows:
        variant = row.get('variant', '')
        step = as_float(row.get('step'))
        miou = as_float(row.get('miou'))
        baseline_miou = as_float(row.get('baseline_miou'))
        delta = as_float(row.get('delta_vs_baseline'))
        if not variant or step is None or miou is None or baseline_miou is None or delta is None:
            continue
        current = by_variant.get(variant)
        if current is None or step > current['step']:
            by_variant[variant] = {
                'step': int(step),
                'miou': miou,
                'baseline_miou': baseline_miou,
                'delta': delta,
            }

    if not by_variant:
        return

    print('\nMatched-step latest delta vs baseline:')
    for variant in sorted(by_variant):
        row = by_variant[variant]
        if variant == baseline_variant:
            continue
        print(
            f"{variant}: step={row['step']} "
            f"mIoU={row['miou']:.3f} "
            f"baseline={row['baseline_miou']:.3f} "
            f"delta={row['delta']:+.4f}"
        )


def candidate_gate(row, min_candidate_step):
    last_step = as_float(row.get('last_step'))
    if last_step is None or last_step < min_candidate_step:
        return False
    complete = as_bool(row.get('training_complete'))
    return complete is not False


def recommend(rows, tmax_warn, active_err_warn, curve_rows, baseline_variant,
              min_candidate_step, full_max_iterations, full_save_root):
    ready = [r for r in rows if row_status(r) == 'ready']
    if not ready:
        print('\nRecommendation: no completed validation results yet. Wait for active variants, then rerun this analyzer.')
        return

    baseline = next((r for r in ready if r.get('variant') == baseline_variant), None)
    calib = [r for r in ready if r.get('variant', '').startswith(CALIB_PREFIXES)]

    ready_sorted = sorted(ready, key=lambda r: as_float(r.get('best_miou')) or -1.0, reverse=True)
    print('\nRanking by best_mIoU:')
    for i, row in enumerate(ready_sorted, 1):
        print(f"{i}. {row.get('variant')} best_mIoU={row.get('best_miou')} step={row.get('best_step')}")

    if baseline is not None:
        b = as_float(baseline.get('best_miou'))
        print('\nBest-vs-best delta vs baseline:')
        for row in ready_sorted:
            m = as_float(row.get('best_miou'))
            if m is not None and b is not None:
                print(f"{row.get('variant')}: {m - b:+.4f}")

    print_matched_step_delta(curve_rows, baseline_variant)

    usable_calib = []
    for row in calib:
        if not candidate_gate(row, min_candidate_step):
            continue
        if not warnings_for(row, tmax_warn, active_err_warn):
            usable_calib.append(row)
    if not usable_calib:
        usable_calib = [row for row in calib if candidate_gate(row, min_candidate_step)]

    selected = sorted(usable_calib, key=lambda r: as_float(r.get('best_miou')) or -1.0, reverse=True)[:2]

    if selected:
        print('\nNext full-run candidates:')
        for row in selected:
            variant = row.get('variant')
            script = variant_to_script(variant)
            if script:
                print(
                    f"SAVE_ROOT={full_save_root} "
                    f"MAX_ITERATIONS={int(full_max_iterations)} "
                    f"bash scripts/experiments/covar_calib_conf/{script}"
                )
            else:
                print(f"{variant}: no direct script mapping")
    else:
        print('\nNext full-run candidates: none yet')
        print(
            f'Wait for calibrated variants to complete training and reach '
            f'a complete validation step >= {int(min_candidate_step)} before selecting full runs.'
        )

    pending = [r.get('variant') for r in rows if row_status(r) != 'ready']
    if pending:
        print('\nPending variants:', ', '.join(pending))

    seen = {r.get('variant') for r in rows}
    missing_calib = [v for v in EXPECTED_CALIB_VARIANTS if v not in seen]
    if missing_calib:
        print('\nNot started calibrated variants:', ', '.join(missing_calib))


def variant_to_script(variant):
    mapping = {
        '01_baseline_no_covar': '01_baseline_no_covar.sh',
        '02_old_newton': '02_old_newton.sh',
        '03_calib_gamma0_tmax4': '03_calib_gamma0_tmax4.sh',
        '04_calib_gamma1_tmax4': '04_calib_gamma1_tmax4.sh',
        '05_calib_gamma0_tmax8': '05_calib_gamma0_tmax8.sh',
    }
    return mapping.get(variant)


def inferred_candidate_step(stage_max_iterations, val_per_iters):
    if val_per_iters <= 0:
        return stage_max_iterations
    return math.floor(stage_max_iterations / val_per_iters) * val_per_iters


def main():
    parser = argparse.ArgumentParser(description='Analyze CoVar calib_conf Stage 1 summary and recommend next full runs.')
    parser.add_argument('--summary', default='runs/covar_calib_conf_stage1/stage1_summary_10000.csv')
    parser.add_argument('--curve', default='runs/covar_calib_conf_stage1/stage1_validation_curve_10000.csv')
    parser.add_argument('--baseline', default='01_baseline_no_covar')
    parser.add_argument('--stage-max-iterations', type=float, default=10000.0)
    parser.add_argument('--val-per-iters', type=float, default=800.0)
    parser.add_argument('--min-candidate-step', type=float, default=None)
    parser.add_argument('--full-max-iterations', type=float, default=80000.0)
    parser.add_argument(
        '--full-save-root',
        default='data/winycg/checkpoints/cirkd_checkpoints/voc/covar_calib_conf_full',
    )
    parser.add_argument('--tmax-warn', type=float, default=30.0)
    parser.add_argument('--active-err-warn', type=float, default=1e-3)
    args = parser.parse_args()

    rows = load_rows(args.summary)
    if not rows:
        print('No rows in summary yet.')
        return
    curve_rows = load_curve_rows(args.curve)
    min_candidate_step = args.min_candidate_step
    if min_candidate_step is None:
        min_candidate_step = inferred_candidate_step(args.stage_max_iterations, args.val_per_iters)
    print_table(rows, args.tmax_warn, args.active_err_warn)
    recommend(
        rows,
        args.tmax_warn,
        args.active_err_warn,
        curve_rows,
        args.baseline,
        min_candidate_step,
        args.full_max_iterations,
        args.full_save_root,
    )


if __name__ == '__main__':
    main()
