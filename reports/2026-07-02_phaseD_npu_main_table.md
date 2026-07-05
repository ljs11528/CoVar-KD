# Phase D NPU Main Table Report

- Generated: 2026-07-03T00:14:14
- Max iterations: `80000`
- Phase D root: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseD_main_table`
- Phase C 80k root: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseC_80k`

## Tout x CoVar Main Table

| Teacher output temp | CoVar | Variant | Last iter | Validations | Best mIoU | Final mIoU | Total time | Last T mean/min/max |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1.0 | off | `phaseD_cirkd_no_covar_tout1` | 80000 | 100 | 0.6426 | 0.6416 | `11:55:26.456171 (0.5366s / it)` | n/a |
| 1.0 | on | `phaseD_covar_newton_gamma2_tout1` | 80000 | 17 | 0.6453 | 0.6440 | `2:10:02.950271 (0.5737s / it)` | 0.5959/0.5000/3.0000 |
| 3.0 | off | `phaseC_lc_no_covar_tout3` | 80000 | 100 | 0.6475 | 0.6383 | `11:57:26.683598 (0.5381s / it)` | n/a |
| 3.0 | on | `phaseC_lc_newton_gamma2_repro` | 80000 | 100 | 0.6539 | 0.6490 | `13:11:18.364429 (0.5935s / it)` | 0.5911/0.5000/2.0135 |

## Interpretation Guide

- `Tout=1.0, CoVar=off` is the same-code original CIRKD control.
- `Tout=1.0, CoVar=on` tests whether CoVar helps without teacher softening.
- `Tout=3.0, CoVar=off` isolates teacher softening.
- `Tout=3.0, CoVar=on` is the current main method.
- The key claim is strongest if CoVar improves over no-CoVar at both `Tout=1.0` and `Tout=3.0`.

## Log Paths

- `phaseD_cirkd_no_covar_tout1`: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseD_main_table/phaseD_cirkd_no_covar_tout1/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- `phaseD_covar_newton_gamma2_tout1`: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseD_main_table/phaseD_covar_newton_gamma2_tout1/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- `phaseC_lc_no_covar_tout3`: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseC_80k/phaseC_lc_no_covar_tout3/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- `phaseC_lc_newton_gamma2_repro`: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseC_80k/phaseC_lc_newton_gamma2_repro/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
