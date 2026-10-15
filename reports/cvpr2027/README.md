# Controlled dense-prediction distillation experiments

Snapshot: 2026-10-15. The current research question is when local distillation
diagnostics can guide subsequent dense-prediction training.

The Maker cohort completed all 324 formal CE-start one-step branches. Its
registered replication gate failed on both datasets, and the workflow stopped
before KD-start diagnostics, 128-step rollouts or selector training. The
[complete analysis](diagnostics_324_20261015/analysis_zh.md) audits the inherited
records, training-state effects, metric rankings and gradient controls. The
[revised follow-up plan](followup_protocol_20261015.md) prioritizes causal controls
and same-checkpoint continuation tests before allocating full training. That
plan's P0/P1 queue is now running on Maker, with P1 conditional on P0 numerical
calibration. This is an execution update, not evidence that a new selector
works. All future model experiments must run on Maker; the old server is a
read-only checkpoint source.

The [P15 execution contract](p15_execution_20261015.md) freezes the concrete
P0/P1 implementation and memory settings. The
[verified P15 launch receipt](engineering/p15_launch_20261015.json) records
47 passing CPU checks, full-batch GPU preflights for both datasets, four-card
scoring agreement and then-live formal P0 workers on all four cards. P0 has
72 primary one-step replays and 16 separate cross-card scores. If its registered
numerical-resolution gate passes, the detached queue proceeds to the 36 fixed
P1 cases (912 scored endpoints), starting with the four discovery cases.
P2 and full training remain outside this launch. The scientific follow-up
document is preserved as the original prelaunch plan; its historical
`PLAN_NOT_LAUNCHED` status is superseded for P0/P1 by this launch receipt.

Existing keepalive programs now occupy memory on all four GPUs and remain
running. Full student batches are preserved by storing backward activations
on CPU; the model still computes in FP32 on CUDA. Each worker caps its
PyTorch allocation at 6144 MiB, and independent tasks use all four cards when
the registered admission checks pass. Results and registration are copied
to the large persistent volume as SHA-verified snapshots. The receipt is
a timestamped launch snapshot; live progress and final P0/P1 outcomes belong
to the execution output, not to this snapshot.

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

## Original conditional stage and execution provenance

The [original research protocol](utility_protocol_20261015.md) is retained for
audit. Its first stage used restored-state diagnostics on existing checkpoints.
Its gate has failed, so its selector-training grid remains suspended. The
revised plan also caps its conditional final comparison at 12 fresh 80k
trajectories and lists short diagnostic/continuation costs separately.

The [Maker execution amendment](maker_migration_20261015.md) moves the new
diagnostics to a separate RTX 4090 cohort. The user initially allocated idle GPUs
2 and 3, then authorized all four cards while preserving the keepalive programs
on 0 and 1. Old P14 compute is stopped,
with artifacts retained. Jobs use fixed original inputs and independent
seed/checkpoint shards, without changing batch size or the scientific gate.

[The initial verified Maker launch receipt](engineering/maker_launch_20261015.json)
records 24 passing CPU checks, both passing GPU smoke contracts, and then-live
CE-start jobs on physical GPUs 2 and 3. The [migration contract audit](engineering/maker_contract_20261015.json)
confirms identical registered seeds, update streams, monitor bytes and origin
checkpoint hashes. This is a timestamped execution snapshot, not a final result
or an assertion that all checkpoint transfers or 80k training runs have finished.

The [four-GPU receipt](engineering/maker_four_gpu_launch_20261015.json) records
the subsequent allocation, 31 passing CPU checks and per-card resource preflights.
Its [continuation audit](engineering/maker_four_gpu_contract_20261015.json) binds
the unchanged scientific contract and inherited observations. Completed r2 cases
retain their original registration hashes and are counted once. Missing verified
checkpoints remain an input dependency even when GPU lanes are available.

The P14 implementation is in [utility_study](../../utility_study/). Its historical
invocation below documents reproduction, not authorization to restart the
stopped queue or use the old server. A new P15 execution requires its own
frozen registration and output directory.

```bash
python -m pytest tests/test_utility_study.py -q
python -u -m utility_study.workflow --root /path/to/CoVar-KD-3090 --out /path/to/new/P14_Utility --gpus 0,2
```

The workflow preserves all existing model checkpoints, checks frozen runtime
hashes against the published source snapshot, and stops before any selector
full training. Its primary gate and intermediate branch records are explicit
JSON artifacts in the output directory. Do not start a second controller for
the same output. Shared-card execution requires explicit authorization and the
registered keepalive identity and free-memory checks.
