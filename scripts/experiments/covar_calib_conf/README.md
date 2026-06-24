# CoVar Calibration-Confidence Experiments

This directory contains runnable entry points for the reliability-calibrated teacher temperature experiments on VOC/CIRKDv2.

## Environment

Use the project venv:

```bash
cd /workspace/covar+cirkd/CIRKD-main
source .venv/bin/activate
```

All scripts default to single-GPU execution with `GPU_ID=0`. Override paths or runtime knobs through environment variables:

```bash
GPU_ID=0 MAX_ITERATIONS=20000 bash scripts/experiments/covar_calib_conf/03_calib_gamma0_tmax4.sh
```

## Stage 0: sanity checks

```bash
bash scripts/experiments/covar_calib_conf/smoke_calib_gamma0_tmax4.sh
```

Expected early log signals for `calib_conf`:

- `T_mean >= 1.0`
- `c0/tgt/cf` has `cf` close to `tgt`
- `err(logc)` is small
- `Tmax` is not saturated for most pixels

## Stage 1: short trend runs

Run each variant for 10k-20k iterations first:

```bash
MAX_ITERATIONS=20000 bash scripts/experiments/covar_calib_conf/run_stage1_short.sh
```

Variants:

1. `01_baseline_no_covar`: CIRKD baseline with CoVar disabled.
2. `02_old_newton`: previous `min r(T)` Newton method.
3. `03_calib_gamma0_tmax4`: calibration confidence, `gamma=0`, `Tmax=4`.
4. `04_calib_gamma1_tmax4`: calibration confidence, `gamma=1`, `Tmax=4`.
5. `05_calib_gamma0_tmax8`: calibration confidence, `gamma=0`, `Tmax=8`.

## Full runs

After short-run trend selection, run the best calibrated variants for full 80k iterations in a separate save root so Stage 1 evidence is preserved:

```bash
MAX_ITERATIONS=80000 bash scripts/experiments/covar_calib_conf/launch_full_candidates.sh
```

The current full-run queue runs `04_calib_gamma1_tmax4` first, then `03_calib_gamma0_tmax4`, and saves under `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_calib_conf_full` by default. Monitor it with:

```bash
MAX_ITERATIONS=80000 bash scripts/experiments/covar_calib_conf/monitor_full_candidates.sh
```

Wait for a specific full-run validation point without stopping the queue:

```bash
.venv/bin/python scripts/experiments/covar_calib_conf/wait_for_validation.py \
  --target-step 800 \
  --timeout-seconds 1200 \
  --poll-seconds 60
```

Export the current full-run validation curve and compare it against existing 80k Newton logs plus the Stage 1 curve:

```bash
.venv/bin/python scripts/experiments/covar_calib_conf/compare_validation_curves.py \
  --root data/winycg/checkpoints/cirkd_checkpoints/voc/covar_calib_conf_full \
  --baseline 04_calib_gamma1_tmax4 \
  --output runs/covar_calib_conf_full/full_validation_curve_80000.csv

.venv/bin/python scripts/experiments/covar_calib_conf/compare_validation_curves.py \
  --root /tmp/no-such-covar-root \
  --logs train.log train1.log \
  --baseline train1 \
  --output runs/covar_calib_conf_stage1/existing_root_logs_validation_curve.csv

.venv/bin/python scripts/experiments/covar_calib_conf/analyze_full_progress.py \
  --output runs/covar_calib_conf_full/full_progress_vs_refs.csv
```

## Diagnostic before over-tuning

Run the teacher reliability diagnostic to check whether `r0` actually correlates with teacher errors:

```bash
.venv/bin/python scripts/diagnostics/teacher_r0_correctness_bins.py \
  --data dataset/VOCAug/ \
  --teacher-pretrained data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth \
  --max-samples 200 \
  --max-pixels-per-image 4096 \
  --output runs/diagnostics/teacher_r0_correctness_bins.csv
```

If high-`r0` bins do not show lower teacher accuracy, prioritize improving the reliability definition before tuning more temperature parameters.

## Summarize completed runs

After one or more variants finish validation, collect their best mIoU and latest temperature diagnostics:

```bash
.venv/bin/python scripts/experiments/covar_calib_conf/summarize_runs.py \
  --root data/winycg/checkpoints/cirkd_checkpoints/voc/covar_calib_conf \
  --output runs/covar_calib_conf_summary.csv
```

Explicit root-level logs such as `train.log` and `train1.log` can be summarized with `--logs`:

```bash
.venv/bin/python scripts/experiments/covar_calib_conf/summarize_runs.py \
  --root /tmp/no-such-covar-root \
  --logs train.log train1.log \
  --output runs/covar_calib_conf_stage1/existing_root_logs_summary.csv
```

## Background Stage 1 Queue

Launch the five short-run variants sequentially in the background. The default is `MAX_ITERATIONS=10000` for the first trend check:

```bash
MAX_ITERATIONS=10000 bash scripts/experiments/covar_calib_conf/launch_stage1_short.sh
```

Monitor progress:

```bash
MAX_ITERATIONS=10000 bash scripts/experiments/covar_calib_conf/monitor_stage1_short.sh
```

Stop the queue if needed:

```bash
MAX_ITERATIONS=10000 bash scripts/experiments/covar_calib_conf/stop_stage1_short.sh
```

## Analyze Stage 1 Results

After the monitor has produced a summary CSV, rank variants and print recommended full-run commands:

```bash
.venv/bin/python scripts/experiments/covar_calib_conf/analyze_stage1_results.py \
  --summary runs/covar_calib_conf_stage1/stage1_summary_10000.csv \
  --curve runs/covar_calib_conf_stage1/stage1_validation_curve_10000.csv
```

If a calibrated run has high `Tmax` saturation or poor active calibration error, the analyzer warns before recommending it.

For matched-step validation curves, use:

```bash
.venv/bin/python scripts/experiments/covar_calib_conf/compare_validation_curves.py \
  --root data/winycg/checkpoints/cirkd_checkpoints/voc/covar_calib_conf \
  --output runs/covar_calib_conf_stage1/stage1_validation_curve_10000.csv
```

