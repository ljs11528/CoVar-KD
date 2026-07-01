# Phase D NPU Main Table Plan

- Created: 2026-07-01
- Purpose: complete the `teacher_output_temp x CoVar` main table for AAAI submission.
- Hardware: 2 x Ascend 910 via `torch.distributed.run` + HCCL.
- Max iterations: 80000 per missing variant.

## Existing 80k Cells

| Teacher output temp | CoVar | Variant | Best mIoU | Final mIoU |
|---:|---|---|---:|---:|
| 3.0 | off | `phaseC_lc_no_covar_tout3` | 0.6475 | 0.6383 |
| 3.0 | on | `phaseC_lc_newton_gamma2_repro` | 0.6539 | 0.6490 |

## Missing Cells to Run

| Teacher output temp | CoVar | Variant | Key args |
|---:|---|---|---|
| 1.0 | off | `phaseD_cirkd_no_covar_tout1` | `--teacher-output-temp 1.0 --no-covar` |
| 1.0 | on | `phaseD_covar_newton_gamma2_tout1` | `--teacher-output-temp 1.0 --covar-temp-mode newton --covar-kd-temp-power 2.0` |

## Decision Logic

- If CoVar improves over no-CoVar at both `Tout=1.0` and `Tout=3.0`, the main claim is strong: CoVar provides teacher-reliability gains independent of teacher softening.
- If CoVar only improves at `Tout=3.0`, the paper should frame CoVar as complementary to softened-teacher distillation.
- If `Tout=1.0` no-CoVar is stronger than current CoVar, the baseline section must explicitly discuss this and avoid overclaiming.

## Commands

```bash
bash scripts/experiments/covar_npu/launch_phaseD_main_table.sh
bash scripts/experiments/covar_npu/monitor_phaseD_main_table.sh
```

Final report will be generated as:

```text
reports/YYYY-MM-DD_phaseD_npu_main_table.md
```
