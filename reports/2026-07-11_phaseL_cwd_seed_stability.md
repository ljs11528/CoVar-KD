# Phase L CWD Seed Stability Report

- Generated: 2026-07-11T17:44:55+08:00
- Seeds: `1234, 2025, 3407`
- Budget: `80000` iterations
- CWD recipe: task CE + KD + adversarial KD + CWD feature/logit

## Per-Seed Results

| Seed | CWD best | Best iter | CWD final | CoVar/CIRKD best | CoVar/CIRKD final | CWD-CoVar best | CWD-CoVar final | Complete |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1234 | 0.6640 | 79200 | 0.6610 | 0.6539 | 0.6490 | +0.0101 | +0.0120 | True |
| 2025 | 0.6600 | 73600 | 0.6580 | 0.6411 | 0.6407 | +0.0189 | +0.0173 | True |
| 3407 | 0.6650 | 75200 | 0.6640 | 0.6448 | 0.6448 | +0.0202 | +0.0192 | True |

## Aggregate

- CWD best mIoU: `0.6630 +/- 0.0022`.
- CWD final mIoU: `0.6610 +/- 0.0024`.
- Compare method-level means cautiously: CWD and CoVar use different base recipes.

## Log Paths

- CWD seed 1234: `/home/ma-user/work/ljs/runs/logs/kd_baselines_npu/phaseK_voc_80k/cwd_80k/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- CWD seed 2025: `/home/ma-user/work/ljs/runs/logs/kd_baselines_npu/phaseL_cwd_seed_stability/cwd_80k_seed2025/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- CWD seed 3407: `/home/ma-user/work/ljs/runs/logs/kd_baselines_npu/phaseL_cwd_seed_stability/cwd_80k_seed3407/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
