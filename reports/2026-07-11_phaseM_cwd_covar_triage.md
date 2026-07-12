# Phase M CWD + CoVar 20k Triage Report

- Generated: 2026-07-11T20:16:55+08:00
- Seed: `1234`
- Budget: `20000` iterations
- Controlled change: only the logit KD temperature mechanism differs between fixed and CoVar rows.

## Results

| Variant | Best mIoU | Best iter | Final mIoU | Validations | Runtime | Complete |
|---|---:|---:|---:|---:|---|---|
| CWD, Tout=1 historical | 0.6390 | 18400 | 0.6360 | 25 | 2:26:24.568313 (0.4392s / it) | True |
| CWD, Tout=3 fixed | 0.6430 | 19200 | 0.6420 | 25 | 2:24:59.836589 (0.4350s / it) | True |
| CWD, Tout=3 Newton CoVar | 0.6460 | 20000 | 0.6460 | 25 | 2:31:14.501236 (0.4537s / it) | True |

## Controlled Delta

- Best mIoU delta, CoVar - fixed: `+0.0030`.
- Final mIoU delta, CoVar - fixed: `+0.0040`.

## Log Paths

- Historical CWD: `/home/ma-user/work/ljs/runs/logs/kd_baselines_npu/phaseJ_voc_20k/cwd_20k/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- Fixed Tout=3: `/home/ma-user/work/ljs/runs/logs/kd_baselines_npu/phaseM_cwd_covar/triage_20k/cwd_tout3_fixed_20k_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- Newton CoVar: `/home/ma-user/work/ljs/runs/logs/kd_baselines_npu/phaseM_cwd_covar/triage_20k/cwd_covar_newton_tout3_20k_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
