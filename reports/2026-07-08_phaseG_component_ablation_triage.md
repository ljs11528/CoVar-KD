# Phase G Component Ablation Triage Report

- Generated: 2026-07-08T22:42:31
- Save root: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseG_component_ablation`
- Max iterations: `20000`
- Seed: `1234`
- Operational status: Phase G was resumed at 2026-07-08 22:40 CST after the previous queue stopped without a Python traceback. Current queue PID is `23660`; current NPU worker PIDs are `23865/23866`.
- Resume state: no-CoVar loaded `training_state_latest.pth` at iteration `12000`, preserving best mIoU `0.5790`; logs are appended to preserve pre-resume validation history.

## Summary

| Label | Variant | Last iter | Validations | Best mIoU | Final mIoU | Delta vs off | Delta vs full | Final pixAcc | T mean/min/max | Complete |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| off | `phaseG_triage_no_covar_tout3_seed1234` | 12140 | 15 | 0.5790 | 0.5580 | 0.0000 | pending | 0.8834 | n/a | False |
| confidence | `phaseG_triage_covar_confidence_only_tout3_seed1234` | pending | 0 | pending | pending | pending | pending | pending | n/a | False |
| variance | `phaseG_triage_covar_variance_only_tout3_seed1234` | pending | 0 | pending | pending | pending | pending | pending | n/a | False |
| full | `phaseG_triage_covar_full_tout3_seed1234` | pending | 0 | pending | pending | pending | pending | pending | n/a | False |

## Details

### off

- Variant: `phaseG_triage_no_covar_tout3_seed1234`
- Log: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseG_component_ablation/phaseG_triage_no_covar_tout3_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- Last reliability mode: `n/a`
- Total time: `pending`
- Last Newton diagnostic: `n/a`

### confidence

- Variant: `phaseG_triage_covar_confidence_only_tout3_seed1234`
- Log: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseG_component_ablation/phaseG_triage_covar_confidence_only_tout3_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- Last reliability mode: `n/a`
- Total time: `pending`
- Last Newton diagnostic: `n/a`

### variance

- Variant: `phaseG_triage_covar_variance_only_tout3_seed1234`
- Log: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseG_component_ablation/phaseG_triage_covar_variance_only_tout3_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- Last reliability mode: `n/a`
- Total time: `pending`
- Last Newton diagnostic: `n/a`

### full

- Variant: `phaseG_triage_covar_full_tout3_seed1234`
- Log: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseG_component_ablation/phaseG_triage_covar_full_tout3_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- Last reliability mode: `n/a`
- Total time: `pending`
- Last Newton diagnostic: `n/a`
