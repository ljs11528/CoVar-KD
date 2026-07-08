# Phase E Tout=3.0 Seed Stability Report

- Generated: 2026-07-07T12:04:45
- Max iterations: `80000`
- Phase E root: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseE_tout3_seed_stability`
- Existing seed-1234 root: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseC_80k`
- Seeds: `1234, 2025, 3407`

## Per-Seed Results

| Seed | CoVar | Variant | Last iter | Validations | Best mIoU | Final mIoU | Complete | Total time |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1234 | off | `phaseC_lc_no_covar_tout3` | 80000 | 100 | 0.6475 | 0.6383 | True | `11:57:26.683598 (0.5381s / it)` |
| 1234 | on | `phaseC_lc_newton_gamma2_repro` | 80000 | 100 | 0.6539 | 0.6490 | True | `13:11:18.364429 (0.5935s / it)` |
| 2025 | off | `phaseE_seed2025_cirkd_no_covar_tout3` | 80000 | 1 | 0.6370 | 0.6370 | True | `0:07:10.345880 (0.5379s / it)` |
| 2025 | on | `phaseE_seed2025_covar_newton_gamma2_tout3` | 80000 | 100 | 0.6411 | 0.6407 | True | `12:43:58.974591 (0.5730s / it)` |
| 3407 | off | `phaseE_seed3407_cirkd_no_covar_tout3` | 80000 | 100 | 0.6387 | 0.6346 | True | `11:51:02.164684 (0.5333s / it)` |
| 3407 | on | `phaseE_seed3407_covar_newton_gamma2_tout3` | 80000 | 100 | 0.6448 | 0.6448 | True | `12:51:56.789902 (0.5790s / it)` |

## Aggregate

- `best_miou`: off `0.6411 +/- 0.0056`; on `0.6466 +/- 0.0066`; delta `0.0055`.
- `final_miou`: off `0.6366 +/- 0.0019`; on `0.6448 +/- 0.0042`; delta `0.0082`.

## Log Paths

- `phaseC_lc_no_covar_tout3`: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseC_80k/phaseC_lc_no_covar_tout3/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- `phaseC_lc_newton_gamma2_repro`: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseC_80k/phaseC_lc_newton_gamma2_repro/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- `phaseE_seed2025_cirkd_no_covar_tout3`: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseE_tout3_seed_stability/phaseE_seed2025_cirkd_no_covar_tout3/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- `phaseE_seed2025_covar_newton_gamma2_tout3`: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseE_tout3_seed_stability/phaseE_seed2025_covar_newton_gamma2_tout3/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- `phaseE_seed3407_cirkd_no_covar_tout3`: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseE_tout3_seed_stability/phaseE_seed3407_cirkd_no_covar_tout3/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- `phaseE_seed3407_covar_newton_gamma2_tout3`: `/home/ma-user/work/ljs/data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseE_tout3_seed_stability/phaseE_seed3407_covar_newton_gamma2_tout3/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
