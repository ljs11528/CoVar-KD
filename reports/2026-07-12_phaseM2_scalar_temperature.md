# Phase M2 Scalar-Temperature Controls

- Finalized: `2026-07-13T07:37:41+08:00` (Asia/Shanghai)
- Dataset/model: Pascal VOC, DeepLabV3-ResNet101 teacher, DeepLabV3-MobileNetV3-Small student.
- Seed/budget: `1234`, `20000` iterations.
- Shared recipe: CWD with teacher-output temperature `Tout=3`; only the logit KD temperature mechanism differs.
- Parser: `scripts/experiments/kd_baselines_npu/summarize_phaseM2_scalar_temperature.py`.
- Metrics are rendered to 6 decimal places. Historical Phase M logs contain only 3-decimal mIoU values, so trailing zeros do not imply added precision.

## Results

| Variant | Best mIoU | Best iter | Final mIoU | Last-10 val mean | Validations | Runtime | Complete |
|---|---:|---:|---:|---:|---:|---|---|
| CWD, Tout=3, scalar KD T=0.5 | 0.648235 | 20000 | 0.648235 | 0.626266 | 25 | 2:25:26.069913 (0.4363s / it) | true |
| CWD, Tout=3, scalar KD T=0.6 | **0.653381** | 20000 | **0.653381** | 0.624513 | 25 | 2:25:00.807508 (0.4350s / it) | true |
| CWD, Tout=3, scalar KD T=1.0 | 0.643000 | 19200 | 0.642000 | **0.626800** | 25 | 2:24:59.836589 (0.4350s / it) | true |
| CWD, Tout=3, Newton CoVar | 0.646000 | 20000 | 0.646000 | 0.622100 | 25 | 2:31:14.501236 (0.4537s / it) | true |

All four runs contain `Iters: 20000/20000`, a parseable final validation, 25 validation points, and `Total training time:`. Both new scalar runs also produced final/best model checkpoints and latest/best full training states.

## Controlled deltas

| Comparison | Final delta | Best delta | Last-10 mean delta |
|---|---:|---:|---:|
| `T=0.5 - T=1.0` | `+0.006235` | `+0.005235` | `-0.000534` |
| `T=0.6 - T=1.0` | `+0.011381` | `+0.010381` | `-0.002287` |
| `CoVar - T=0.5` | `-0.002235` | `-0.002235` | `-0.004166` |
| `CoVar - T=0.6` | `-0.007381` | `-0.007381` | `-0.002413` |

The strongest matched scalar control by the preregistered primary metric is `T=0.6`. Therefore `Delta_final = CoVar - S* = -0.007381`.

## Preregistered decision

The result hits Phase M2 rule 3 (`Delta_final <= -0.002`): **scalar temperature is stronger**. CoVar must not be automatically promoted to the planned 80k multi-seed CWD experiment. The prior CoVar improvement over scalar `T=1.0` (`+0.004000` final) is not evidence that spatial temperature allocation is responsible, because both matched low-temperature scalar controls outperform CoVar on final mIoU.

This is a single-seed, 20k triage result, so it is a resource-allocation and causal-warning result rather than a statistical generalization claim. The next experiment is Phase N: a controlled 80k, seed-1234 comparison of scalar `T=0.6` versus scalar `T=1.0`, followed by multi-seed promotion only if the long-budget gain remains competitive.

## Log paths

- Scalar `T=0.5`: `runs/logs/kd_baselines_npu/phaseM2_scalar_temperature/triage_20k/cwd_tout3_kdtemp0p5_20k_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- Scalar `T=0.6`: `runs/logs/kd_baselines_npu/phaseM2_scalar_temperature/triage_20k/cwd_tout3_kdtemp0p6_20k_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- Scalar `T=1.0`: `runs/logs/kd_baselines_npu/phaseM_cwd_covar/triage_20k/cwd_tout3_fixed_20k_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- Newton CoVar: `runs/logs/kd_baselines_npu/phaseM_cwd_covar/triage_20k/cwd_covar_newton_tout3_20k_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
