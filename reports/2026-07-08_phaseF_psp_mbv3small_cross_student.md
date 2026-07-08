# Phase F F1 PSPNet-MobileNetV3-Small Cross-Student Report

- Generated: 2026-07-08T20:16:36+08:00
- Teacher: `DeepLabV3-ResNet101`
- Student: `PSPNet-MobileNetV3-Small`
- Dataset: VOC
- Setting: `Tout=3.0`, seed `1234`, `80000` iterations
- Queue log: `runs/covar_npu_phaseF_psp_mbv3small/phaseF_psp_mbv3small_tout3_80000.log`
- Save root: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseF_psp_mbv3small`

## Results

| Variant | CoVar | Best mIoU | Final mIoU | Final pixAcc | Validations | Total time |
|---|---|---:|---:|---:|---:|---|
| `phaseF_psp_mbv3small_no_covar_tout3_seed1234` | off | `0.6366` | `0.6324` | `0.9074` | `100` | `11:50:59` |
| `phaseF_psp_mbv3small_covar_newton_gamma2_tout3_seed1234` | on | `0.6387` | `0.6359` | `0.9089` | `100` | `12:58:49` |

## Delta

- Best mIoU delta: `+0.0021`
- Final mIoU delta: `+0.0035`
- CoVar overhead: about `+1:07:50` wall time, from `0.5332s/it` to `0.5841s/it`.

## Interpretation

F1 is positive for the cross-student setting. CoVar improves both the best checkpoint and the final 80k validation over the PSPNet-MobileNetV3-Small no-CoVar baseline. The margin is smaller than the main DeepLabV3-MobileNetV3-Small final-seed result, but it supports that CoVar is not tied to the original student head.

This is still a single-seed cross-student result, so it should be reported as generalization evidence rather than a seed-stability claim.

## Artifacts

- No-CoVar log: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseF_psp_mbv3small/phaseF_psp_mbv3small_no_covar_tout3_seed1234/psp_mobile_resnet101_mobilenetv3_small_log.txt`
- No-CoVar best checkpoint: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseF_psp_mbv3small/phaseF_psp_mbv3small_no_covar_tout3_seed1234/kd_psp_mobile_mobilenetv3_small_voc_miou-0.6366.pth`
- CoVar log: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseF_psp_mbv3small/phaseF_psp_mbv3small_covar_newton_gamma2_tout3_seed1234/psp_mobile_resnet101_mobilenetv3_small_log.txt`
- CoVar best checkpoint: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseF_psp_mbv3small/phaseF_psp_mbv3small_covar_newton_gamma2_tout3_seed1234/kd_psp_mobile_mobilenetv3_small_voc_miou-0.6387.pth`

