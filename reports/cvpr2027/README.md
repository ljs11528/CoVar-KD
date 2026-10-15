# Controlled dense-prediction distillation experiments

Snapshot: 2026-10-15. The current research question is when a local distillation
diagnostic predicts the opposite ordering from the final segmentation result.

## Existing evidence

[The Chinese analysis report](posthoc/analysis_report_zh.md) describes 30 completed
MobileNetV3-Large student runs: two datasets, three seeds, and CE plus a complete
temperature-placement by loss-coefficient factorial. The endpoint is final 80k
validation mIoU, not a best checkpoint.

| Condition | VOC mIoU, mean ± sample SD | Cityscapes mIoU, mean ± sample SD |
| --- | ---: | ---: |
| CE | 69.016 ± 0.223 | 71.375 ± 0.168 |
| Teacher c=1 | 68.924 ± 0.503 | 71.988 ± 0.320 |
| Teacher c=4 | 65.809 ± 3.462 | 73.185 ± 0.506 |
| Shared c=1 | 68.552 ± 0.343 | 72.374 ± 0.346 |
| Shared c=4 | 67.213 ± 1.071 | 73.113 ± 0.346 |

Teacher uses T_teacher=2, T_student=1. Shared uses both temperatures equal to 2.
The explicit coefficient c is not multiplied by another T². All original seeds,
including the low VOC Teacher c=4 result, are retained.

The dataset protocols also differ in teacher weights, learning rate, batch size
and crop. The cross-setting reversal is not an isolated causal dataset effect.
Existing single-step probes use CE checkpoints, batch 2 and eval-mode BN; these
do not reproduce full KD training. Pixel-averaged CE and class-averaged mIoU
weight outcomes differently, but the existing results do not establish that
this distinction causes the reversal.

## Reproduction material

- [Original cohort records](baselines/) retain P11 and P12 protocols/results.
- [Posthoc registration](posthoc/registration.json), [final audit](posthoc/final_audit.json)
  and [statistics](posthoc/analysis_summary.json) record the evidence boundary.
- [Frozen CUDA runtimes](../../repro/rtx3090/) preserve the executed source.
  The VOC and Cityscapes snapshots differ in dataset handling and metric code.
- [Posthoc scripts](../../repro/posthoc_20261015/) are historical source snapshots.
  Their absolute paths document the original execution environment.
- [Source manifest](source_manifest.json) records content hashes and archive provenance.

Datasets, checkpoints, prediction tensors, SSH configuration and credentials
are excluded. Current-tree credential removal does not remove credentials from
earlier Git history; previously committed keys require replacement.

## Approved next stage

See [the research protocol](utility_protocol_20261015.md). The first stage is
restored-state diagnostics on existing checkpoints. Full selector training is
conditional on that stage; the approved upper bound is 12 fresh 80k trajectories.
Planned experiments are not reported as completed results.
