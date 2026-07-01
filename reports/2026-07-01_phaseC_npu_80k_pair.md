# Phase C NPU Triage Report

- Generated: 2026-07-01T16:18:24
- Save root: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseC_80k`
- Max iterations per variant: `80000`
- Variants: `phaseC_lc_newton_gamma2_repro, phaseC_lc_no_covar_tout3`

## Summary

| Variant | Last iter | Validations | Best mIoU | Final mIoU | Last T mean/min/max |
|---|---:|---:|---:|---:|---|
| `phaseC_lc_newton_gamma2_repro` | 80000 | 100 | 0.6539 | 0.6490 | 0.5911/0.5000/2.0135 |
| `phaseC_lc_no_covar_tout3` | 80000 | 100 | 0.6475 | 0.6383 | n/a |

## Details

### phaseC_lc_newton_gamma2_repro

- Log: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseC_80k/phaseC_lc_newton_gamma2_repro/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- Training complete: `True`
- Total time: `13:11:18.364429 (0.5935s / it)`
- Last Newton diagnostic: `NewtonT: 49.6ms(avg 14.3, inner 6.2) || |dr/dT|: 3.97e-01 || conv<1e-02: 13.2% || |T-T0|: 0.474 || clamp: 87.8% || Cost Time: 13:10:20 || Estimated Time: 0:00:00`

### phaseC_lc_no_covar_tout3

- Log: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseC_80k/phaseC_lc_no_covar_tout3/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- Training complete: `True`
- Total time: `11:57:26.683598 (0.5381s / it)`
- Last Newton diagnostic: `n/a`

