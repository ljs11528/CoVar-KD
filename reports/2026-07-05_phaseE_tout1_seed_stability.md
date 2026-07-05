# Phase E Tout=1.0 Seed Stability Report

- Generated: 2026-07-05T09:41:39
- Max iterations: `80000`
- Phase E root: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseE_seed_stability`
- Existing seed-1234 root: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseD_main_table`
- Seeds: `1234, 2025, 3407`

## Per-Seed Results

| Seed | CoVar | Variant | Last iter | Validations | Best mIoU | Final mIoU | Complete | Total time |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1234 | off | `phaseD_cirkd_no_covar_tout1` | 80000 | 100 | 0.6426 | 0.6416 | True | `11:55:26.456171 (0.5366s / it)` |
| 1234 | on | `phaseD_covar_newton_gamma2_tout1` | 80000 | 17 | 0.6453 | 0.6440 | True | `2:10:02.950271 (0.5737s / it)` |
| 2025 | off | `phaseE_seed2025_cirkd_no_covar_tout1` | 80000 | 100 | 0.6468 | 0.6459 | True | `11:35:37.771897 (0.5217s / it)` |
| 2025 | on | `phaseE_seed2025_covar_newton_gamma2_tout1` | 80000 | 100 | 0.6458 | 0.6449 | True | `12:40:35.872716 (0.5704s / it)` |
| 3407 | off | `phaseE_seed3407_cirkd_no_covar_tout1` | 80000 | 100 | 0.6382 | 0.6382 | True | `11:35:36.684417 (0.5217s / it)` |
| 3407 | on | `phaseE_seed3407_covar_newton_gamma2_tout1` | 80000 | 100 | 0.6421 | 0.6412 | True | `12:41:09.457481 (0.5709s / it)` |

## Aggregate

- `best_miou`: off `0.6425 +/- 0.0043`; on `0.6444 +/- 0.0020`; delta `0.0019`.
- `final_miou`: off `0.6419 +/- 0.0039`; on `0.6434 +/- 0.0019`; delta `0.0015`.

## Log Paths

- `phaseD_cirkd_no_covar_tout1`: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseD_main_table/phaseD_cirkd_no_covar_tout1/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- `phaseD_covar_newton_gamma2_tout1`: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseD_main_table/phaseD_covar_newton_gamma2_tout1/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- `phaseE_seed2025_cirkd_no_covar_tout1`: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseE_seed_stability/phaseE_seed2025_cirkd_no_covar_tout1/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- `phaseE_seed2025_covar_newton_gamma2_tout1`: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseE_seed_stability/phaseE_seed2025_covar_newton_gamma2_tout1/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- `phaseE_seed3407_cirkd_no_covar_tout1`: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseE_seed_stability/phaseE_seed3407_cirkd_no_covar_tout1/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- `phaseE_seed3407_covar_newton_gamma2_tout1`: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseE_seed_stability/phaseE_seed3407_covar_newton_gamma2_tout1/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
