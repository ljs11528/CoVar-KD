# Phase E Tout=1.0 Seed Stability Report

- Generated: 2026-07-03T09:06:17
- Max iterations: `80000`
- Phase E root: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseE_seed_stability`
- Existing seed-1234 root: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseD_main_table`
- Seeds: `1234, 2025, 3407`

## Per-Seed Results

| Seed | CoVar | Variant | Last iter | Validations | Best mIoU | Final mIoU | Complete | Total time |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1234 | off | `phaseD_cirkd_no_covar_tout1` | 80000 | 100 | 0.6426 | 0.6416 | True | `11:55:26.456171 (0.5366s / it)` |
| 1234 | on | `phaseD_covar_newton_gamma2_tout1` | 80000 | 17 | 0.6453 | 0.6440 | True | `2:10:02.950271 (0.5737s / it)` |
| 2025 | off | `phaseE_seed2025_cirkd_no_covar_tout1` | pending | 0 | pending | pending | False | `pending` |
| 2025 | on | `phaseE_seed2025_covar_newton_gamma2_tout1` | pending | 0 | pending | pending | False | `pending` |
| 3407 | off | `phaseE_seed3407_cirkd_no_covar_tout1` | pending | 0 | pending | pending | False | `pending` |
| 3407 | on | `phaseE_seed3407_covar_newton_gamma2_tout1` | pending | 0 | pending | pending | False | `pending` |

## Aggregate

- `best_miou`: off `0.6426 +/- 0.0000`; on `0.6453 +/- 0.0000`; delta `0.0028`.
- `final_miou`: off `0.6416 +/- 0.0000`; on `0.6440 +/- 0.0000`; delta `0.0024`.

## Log Paths

- `phaseD_cirkd_no_covar_tout1`: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseD_main_table/phaseD_cirkd_no_covar_tout1/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- `phaseD_covar_newton_gamma2_tout1`: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseD_main_table/phaseD_covar_newton_gamma2_tout1/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- `phaseE_seed2025_cirkd_no_covar_tout1`: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseE_seed_stability/phaseE_seed2025_cirkd_no_covar_tout1/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- `phaseE_seed2025_covar_newton_gamma2_tout1`: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseE_seed_stability/phaseE_seed2025_covar_newton_gamma2_tout1/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- `phaseE_seed3407_cirkd_no_covar_tout1`: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseE_seed_stability/phaseE_seed3407_cirkd_no_covar_tout1/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- `phaseE_seed3407_covar_newton_gamma2_tout1`: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseE_seed_stability/phaseE_seed3407_covar_newton_gamma2_tout1/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
