# Phase G Component Ablation Triage Report

- Generated: 2026-07-09T09:43:58
- Save root: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseG_component_ablation`
- Max iterations: `20000`
- Seed: `1234`

## Summary

| Label | Variant | Last iter | Validations | Best mIoU | Final mIoU | Delta vs off | Delta vs full | Final pixAcc | T mean/min/max | Complete |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| off | `phaseG_triage_no_covar_tout3_seed1234` | 20000 | 25 | 0.6267 | 0.6267 | 0.0000 | -0.0035 | 0.9035 | n/a | True |
| confidence | `phaseG_triage_covar_confidence_only_tout3_seed1234` | 20000 | 25 | 0.6285 | 0.6285 | 0.0018 | -0.0017 | 0.9055 | 0.5000/0.5000/0.5000 | True |
| variance | `phaseG_triage_covar_variance_only_tout3_seed1234` | 20000 | 25 | 0.6242 | 0.6242 | -0.0025 | -0.0059 | 0.9039 | 1.2668/0.5000/3.0000 | True |
| full | `phaseG_triage_covar_full_tout3_seed1234` | 20000 | 25 | 0.6301 | 0.6301 | 0.0035 | 0.0000 | 0.9048 | 0.5578/0.5000/2.0924 | True |

## Details

### off

- Variant: `phaseG_triage_no_covar_tout3_seed1234`
- Log: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseG_component_ablation/phaseG_triage_no_covar_tout3_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- Last reliability mode: `n/a`
- Total time: `1:11:46.771801 (0.5383s / it)`
- Last Newton diagnostic: `n/a`

### confidence

- Variant: `phaseG_triage_covar_confidence_only_tout3_seed1234`
- Log: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseG_component_ablation/phaseG_triage_covar_confidence_only_tout3_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- Last reliability mode: `confidence`
- Total time: `3:15:46.225335 (0.5873s / it)`
- Last Newton diagnostic: `NewtonT[confidence]: 47.6ms(avg 13.6, inner 6.0) || |dr/dT|: 1.91e-01 || conv<1e-02: 0.8% || |T-T0|: 0.500 || clamp: 100.0% || Cost Time: 3:14:48 || Estimated Time: 0:00:00`

### variance

- Variant: `phaseG_triage_covar_variance_only_tout3_seed1234`
- Log: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseG_component_ablation/phaseG_triage_covar_variance_only_tout3_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- Last reliability mode: `variance`
- Total time: `3:16:06.295588 (0.5883s / it)`
- Last Newton diagnostic: `NewtonT[variance]: 47.6ms(avg 13.8, inner 6.0) || |dr/dT|: 5.58e-02 || conv<1e-02: 3.2% || |T-T0|: 0.591 || clamp: 12.7% || Cost Time: 3:15:08 || Estimated Time: 0:00:00`

### full

- Variant: `phaseG_triage_covar_full_tout3_seed1234`
- Log: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseG_component_ablation/phaseG_triage_covar_full_tout3_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- Last reliability mode: `full`
- Total time: `3:17:34.611052 (0.5927s / it)`
- Last Newton diagnostic: `NewtonT[full]: 50.2ms(avg 14.8, inner 6.3) || |dr/dT|: 3.33e-01 || conv<1e-02: 7.1% || |T-T0|: 0.491 || clamp: 93.1% || Cost Time: 3:16:35 || Estimated Time: 0:00:00`

