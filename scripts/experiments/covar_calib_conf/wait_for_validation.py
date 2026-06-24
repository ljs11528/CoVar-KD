#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


def read_pid(path):
    try:
        text = Path(path).read_text().strip()
    except FileNotFoundError:
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def pid_is_running(pid):
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def as_float(value):
    if value in (None, ''):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def run_summary(root, output):
    cmd = [
        sys.executable,
        'scripts/experiments/covar_calib_conf/summarize_runs.py',
        '--root',
        str(root),
        '--output',
        str(output),
    ]
    subprocess.run(cmd, check=True)


def load_rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open('r', newline='') as f:
        return list(csv.DictReader(f))


def find_best_row(rows, variant):
    candidates = [r for r in rows if not variant or r.get('variant') == variant]
    if not candidates:
        return None
    return max(candidates, key=lambda r: as_float(r.get('last_iter')) or -1.0)


def main():
    parser = argparse.ArgumentParser(description='Wait until a CoVar full-run reaches a validation step.')
    parser.add_argument('--root', default='data/winycg/checkpoints/cirkd_checkpoints/voc/covar_calib_conf_full')
    parser.add_argument('--summary', default='runs/covar_calib_conf_full/full_summary_80000.csv')
    parser.add_argument('--pid-file', default='runs/covar_calib_conf_full/full_candidates_80000.pid')
    parser.add_argument('--variant', default='04_calib_gamma1_tmax4')
    parser.add_argument('--target-step', type=float, default=800.0)
    parser.add_argument('--poll-seconds', type=float, default=60.0)
    parser.add_argument('--timeout-seconds', type=float, default=1800.0)
    args = parser.parse_args()

    started = time.monotonic()
    last_status = None
    while True:
        run_summary(args.root, args.summary)
        rows = load_rows(args.summary)
        row = find_best_row(rows, args.variant)
        pid = read_pid(args.pid_file)
        running = pid_is_running(pid)
        elapsed = time.monotonic() - started

        if row:
            last_iter = row.get('last_iter') or ''
            last_step = row.get('last_step') or ''
            last_miou = row.get('last_miou') or ''
            best_miou = row.get('best_miou') or ''
            status = (
                f"variant={row.get('variant')} last_iter={last_iter} "
                f"last_step={last_step} last_miou={last_miou} "
                f"best_miou={best_miou} pid={pid} running={running}"
            )
            if status != last_status:
                print(status, flush=True)
                last_status = status
            step = as_float(row.get('last_step'))
            if step is not None and step >= args.target_step:
                print(f"Reached target validation step {int(args.target_step)}.", flush=True)
                return 0
        else:
            status = f'No matching row yet variant={args.variant} pid={pid} running={running}'
            if status != last_status:
                print(status, flush=True)
                last_status = status

        if running is False:
            print('Training queue is not running before target validation step was reached.', flush=True)
            return 2
        if elapsed >= args.timeout_seconds:
            print(f'Timed out after {elapsed:.0f}s before target validation step was reached.', flush=True)
            return 3
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
