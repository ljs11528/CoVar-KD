# Phase C NPU Triage Report

- Generated: 2026-06-30T08:39:58
- Save root: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseC`
- Max iterations per variant: `20000`
- Variants: `phaseC_lc_newton_gamma0, phaseC_lc_newton_gamma2_repro, phaseC_lc_newton_gamma1, phaseC_lc_no_covar_tout3`

## Summary

| Variant | Last iter | Validations | Best mIoU | Final mIoU | Last T mean/min/max |
|---|---:|---:|---:|---:|---|
| `phaseC_lc_newton_gamma0` | 20000 | 24 | 0.6087 | 0.6087 | 0.5772/0.5000/2.0873 |
| `phaseC_lc_newton_gamma2_repro` | 20000 | 25 | 0.6353 | 0.6353 | 0.5578/0.5000/2.0924 |
| `phaseC_lc_newton_gamma1` | 20000 | 25 | 0.6282 | 0.6282 | 0.5578/0.5000/2.0924 |
| `phaseC_lc_no_covar_tout3` | 20000 | 25 | 0.6331 | 0.6331 | n/a |

## Details

### phaseC_lc_newton_gamma0

- Log: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseC/phaseC_lc_newton_gamma0/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- Training complete: `True`
- Total time: `3:05:50.039875 (0.5807s / it)`
- Last Newton diagnostic: `NewtonT: 47.3ms(avg 13.4, inner 5.9) || |dr/dT|: 3.50e-01 || conv<1e-02: 8.6% || |T-T0|: 0.491 || clamp: 91.2% || Cost Time: 3:04:53 || Estimated Time: 0:00:00`

### phaseC_lc_newton_gamma2_repro

- Log: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseC/phaseC_lc_newton_gamma2_repro/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- Training complete: `True`
- Total time: `3:11:38.915560 (0.5749s / it)`
- Last Newton diagnostic: `NewtonT: 47.6ms(avg 13.9, inner 6.0) || |dr/dT|: 3.33e-01 || conv<1e-02: 7.1% || |T-T0|: 0.491 || clamp: 93.1% || Cost Time: 3:10:41 || Estimated Time: 0:00:00`

### phaseC_lc_newton_gamma1

- Log: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseC/phaseC_lc_newton_gamma1/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- Training complete: `True`
- Total time: `3:11:32.286106 (0.5746s / it)`
- Last Newton diagnostic: `NewtonT: 46.9ms(avg 13.9, inner 5.9) || |dr/dT|: 3.33e-01 || conv<1e-02: 7.1% || |T-T0|: 0.491 || clamp: 93.1% || Cost Time: 3:10:36 || Estimated Time: 0:00:00`

### phaseC_lc_no_covar_tout3

- Log: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseC/phaseC_lc_no_covar_tout3/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- Training complete: `True`
- Total time: `2:55:30.081801 (0.5265s / it)`
- Last Newton diagnostic: `n/a`

