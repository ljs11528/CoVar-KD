# AAAI Gap Analysis and Experiment Completion Plan

Date: 2026-07-09

## Short verdict

Current CoVar-KD evidence is promising but not yet sufficient for a strong AAAI main-track submission. The method has a clear positive signal, seed stability, component evidence, cross-student evidence, and mechanism diagnostics. The main remaining risk is experimental competitiveness: the current package still lacks direct comparison with recent semantic-segmentation KD methods and lacks a second real dataset run.

## Current evidence

- Main VOC 80k table: best/final mIoU improves from 0.6475/0.6383 to 0.6539/0.6490 at Tout=3.0, delta +0.0064 best and +0.0107 final.
- Seed stability: Tout=3.0 has positive paired deltas for all tested seeds; mean best delta +0.0055 and final delta +0.0082.
- Cross-student: PSPNet-MobileNetV3-Small improves 0.6366/0.6324 to 0.6387/0.6359, delta +0.0021 best and +0.0035 final.
- Component triage: confidence-only +0.0018, variance-only -0.0025, full CoVar +0.0035 at 20k, supporting the combined reliability term.
- Mechanism: reliability correlates with teacher error; bin-level corr(r,error)=0.9599 and pixel corr(r,wrong)=0.3984. H3 shows r/T coupling is behaving as intended.

## Recent related work to account for

- BPKD, boundary/body-region distillation, WACV 2024: https://arxiv.org/abs/2306.08075
- AttnFD, CBAM-refined feature distillation, arXiv 2024/2025 revision: https://arxiv.org/abs/2403.05451
- RDD, pixel-level relative difficulty distillation, 2024: https://arxiv.org/abs/2407.03719
- LAD, label-assisted teacher, ECCV 2024: https://arxiv.org/abs/2407.13254
- FAKD, feature augmentation KD, WACV 2024: https://openaccess.thecvf.com/content/WACV2024/papers/Yuan_FAKD_Feature_Augmented_Knowledge_Distillation_for_Semantic_Segmentation_WACV_2024_paper.pdf
- Raw/Angular feature distillation, WACV 2024: https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Rethinking_Knowledge_Distillation_With_Raw_Features_for_Semantic_Segmentation_WACV_2024_paper.pdf
- I2CKD, intra/inter-class prototype KD, Neurocomputing 2025: https://www.sciencedirect.com/science/article/pii/S0925231225014638

## Needed for AAAI readiness

1. Must add baseline comparisons on the same VOC teacher/student setup.
   - Completed: KD-only, CWD, SKD, IFVD, FitNet, AT, and DSD 20k triage.
   - Promoted to 80k: CWD and SKD by triage rank, IFVD for segmentation-specific coverage, plus KD-only as the common control.
   - Phase K runs the four promoted/control experiments on two independent NPUs and will compare them against CoVar/CIRKD 80k.

2. Should add one recent method or close proxy.
   - Best first target: RDD-style pixel difficulty weighting, because it is conceptually closest to CoVar reliability and can likely be implemented inside the existing logit KD path.
   - Backup: boundary-aware/BPKD-style masks, but this requires boundary-label generation and is more engineering-heavy.
   - LAD is less suitable as an immediate baseline because it changes teacher training.

3. Must add cross-dataset evidence if data and teacher become available.
   - Local check: only VOC data and VOC DeepLabV3-ResNet101 teacher are present.
   - Cityscapes/CamVid/COCO list files exist, but actual datasets and matching teachers are not available locally.

4. Should strengthen generalization.
   - Add 2 more seeds for PSPNet-MobileNetV3-Small if compute allows.
   - Add overhead/runtime table for CoVar temperature computation.
   - Consider 80k component ablation only if reviewers are likely to challenge the reliability design.

## Experiment execution

Phase J VOC 20k baseline triage completed at 2026-07-10 04:14 CST. All seven variants produced 25 validations and final/best checkpoints with no traceback or OOM.

| Variant | Best mIoU | Final mIoU | Delta vs KD-only best | Runtime |
|---|---:|---:|---:|---:|
| `cwd_20k` | 0.639 | 0.636 | +0.032 | 2:26:24 |
| `skd_20k` | 0.620 | 0.620 | +0.013 | 2:25:35 |
| `fitnet_20k` | 0.616 | 0.616 | +0.009 | 2:16:27 |
| `ifvd_20k` | 0.615 | 0.615 | +0.008 | 2:42:22 |
| `dsd_20k` | 0.614 | 0.614 | +0.007 | 2:15:43 |
| `at_20k` | 0.610 | 0.610 | +0.003 | 2:15:21 |
| `kdonly_20k` | 0.607 | 0.606 | 0.000 | 2:15:20 |

Phase K VOC 80k promotion runs started at 2026-07-10 08:02 CST, batch size 16 and seed 1234.

- NPU0 queue, PID `2275879`: `cwd_80k` then `ifvd_80k`.
- NPU1 queue, PID `2275881`: `skd_80k` then `kdonly_80k`.
- Launcher: `scripts/experiments/kd_baselines_npu/launch_phaseK_voc_80k_baselines.sh`.
- Monitor: `scripts/experiments/kd_baselines_npu/monitor_phaseK_voc_80k_baselines.sh`.
- Save root: `data/winycg/checkpoints/kd_baselines_npu/phaseK_voc_80k`.
- At iteration 120, both active runs were healthy with their method-specific losses nonzero. The first pair should finish around 2026-07-10 17:40-18:00 CST; the full four-run queue should finish around 2026-07-11 04:30-05:30 CST.

## Promotion rule

- At each 800-iteration validation, track best mIoU and stability of the validation curve.
- Compare CWD/SKD/IFVD/KD-only at 80k against CoVar best/final mIoU, not only the 20k ordering.
- Do not make the final AAAI competitiveness claim until Phase K completes.
