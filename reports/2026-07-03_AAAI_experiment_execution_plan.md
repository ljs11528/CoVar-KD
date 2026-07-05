# AAAI CoVar-KD Experiment Execution Plan

- Created: 2026-07-03
- Goal: strengthen the CoVar-KD submission package for AAAI with mechanism evidence, statistical stability, model generalization, ablations, and optional cross-dataset validation.
- Current main result source: `reports/2026-07-02_phaseD_npu_main_table.md`
- Sync rule: update this file after every launched run, finished run, failed run, and result summary. Keep command/log/report paths explicit.

## Current Claim

CoVar is a teacher-reliability-aware pixel-wise adaptive temperature distillation method. It uses teacher confidence and residual variance to estimate pixel reliability, sharpens reliable teacher signals, and smooths unreliable teacher signals. The strongest current evidence is that CoVar improves over no-CoVar at both original teacher logits and softened teacher logits.

Decision after Phase E `Tout=1.0`: fix the main method as `Tout=3.0 + CoVar Newton + gamma=2`. Do not change the reliability definition, Newton update, or loss unless a later experiment exposes a concrete failure. Treat `Tout=1.0` as supplemental stability evidence and prioritize main-setting stability, ablations, and generalization over further parameter tuning.

## Completed Main Table

| Teacher output temp | CoVar | Variant | Best mIoU | Final mIoU | Status |
|---:|---|---|---:|---:|---|
| 1.0 | off | `phaseD_cirkd_no_covar_tout1` | 0.6426 | 0.6416 | complete |
| 1.0 | on | `phaseD_covar_newton_gamma2_tout1` | 0.6453 | 0.6440 | complete |
| 3.0 | off | `phaseC_lc_no_covar_tout3` | 0.6475 | 0.6383 | complete |
| 3.0 | on | `phaseC_lc_newton_gamma2_repro` | 0.6539 | 0.6490 | complete |

Interpretation: CoVar improves no-CoVar by +0.0027 best mIoU at `Tout=1.0` and +0.0064 best mIoU at `Tout=3.0`. The `Tout=3.0 + CoVar` cell is the current main method.

## Execution Priority

### Phase H: Mechanism Analysis

Purpose: prove that the reliability score is meaningful and that the adaptive temperature behaves as intended.

| ID | Task | Output | Status |
|---|---|---|---|
| H1 | `r` vs teacher correctness bins on VOC val | CSV + summary JSON + plotted curve | complete |
| H2 | reliability map and temperature map visual examples | figure grid for paper | complete |
| H3 | `r`/`T` distribution for best CoVar run | histogram or curve from logs/checkpoints | pending |

Run first because it is fast and directly strengthens the method story.

### Phase E: Seed Stability

Purpose: convert single-run improvements into mean +/- std evidence.

| ID | Variant | Seeds | Status |
|---|---|---|---|
| E1 | `Tout=1.0, CoVar off` | 3 seeds | complete: seeds 1234/2025/3407 complete |
| E2 | `Tout=1.0, CoVar on` | 3 seeds | complete: seeds 1234/2025/3407 complete; report: `reports/2026-07-05_phaseE_tout1_seed_stability.md` |
| E3 | `Tout=3.0, CoVar off` | 3 seeds | running: seed 1234 complete via Phase C; seed 2025 active at `46000/80000` as of 2026-07-05 16:50 CST; seed 3407 queued |
| E4 | `Tout=3.0, CoVar on` | 3 seeds | queued: seed 1234 complete via Phase C; seeds 2025/3407 queued |

Next seed-stability priority is E3/E4, because `Tout=3.0 + CoVar` is the fixed main method.

Phase E current aggregate: best mIoU off `0.6425 +/- 0.0043`, on `0.6444 +/- 0.0020`, delta `+0.0019`; final mIoU off `0.6419 +/- 0.0039`, on `0.6434 +/- 0.0019`, delta `+0.0015`.

### Phase F: Cross-Student Generalization

Purpose: show the method is not a MobileNetV3-Small-only effect.

| ID | Teacher | Student | Setting | Status |
|---|---|---|---|---|
| F1 | DeepLabV3-ResNet101 | PSPNet-MobileNetV3-Small | `Tout=3.0`, off/on | pending |
| F2 | DeepLabV3-ResNet101 | DeepLabV3-ResNet18 | `Tout=3.0`, off/on | pending |

Run at least one student pair before submission.

### Phase G: Component Ablation

Purpose: prove each reliability component contributes.

| ID | Reliability / temperature setup | Status |
|---|---|---|
| G1 | confidence only: `-log(c)` | pending |
| G2 | residual variance only: `a*v/(1-c)` | pending |
| G3 | full CoVar: `-log(c)+a*v/(1-c)` | covered by main method; summarize |
| G4 | fixed-temperature baseline | pending or cite existing no-CoVar/fixed KD |
| G5 | gamma ablation: `gamma=0/1/2` | partially covered; summarize existing Phase C/B logs |

Use 20k triage before 80k if queue pressure is high.

### Phase I: Cross-Dataset Validation

Purpose: strengthen external validity.

Preferred dataset: Cityscapes if teacher/checkpoints/data are ready. Otherwise run COCO-Stuff only if setup is already stable.

Minimum design:

| ID | Dataset | Variants | Status |
|---|---|---|---|
| I1 | Cityscapes | no-CoVar vs CoVar, `Tout=3.0` | pending |
| I2 | Cityscapes | full `Tout=1/3 x off/on` table | optional |

## Immediate Next Actions

1. Launch and monitor Phase E `Tout=3.0` seed-stability queue for seeds 2025/3407 off/on.
2. Summarize `Tout=3.0` mean +/- std after all four new runs finish.
3. Then run one cross-student generalization pair before deeper ablations.

## Phase H Results

### H1: VOC Val Teacher Reliability Bins

- Command log: `runs/diagnostics/aaai_h1/teacher_r0_correctness_bins_val_full.log`
- CSV: `runs/diagnostics/aaai_h1/teacher_r0_correctness_bins_val_full.csv`
- Summary JSON: `runs/diagnostics/aaai_h1/teacher_r0_correctness_bins_val_full.csv.summary.json`
- Plot: `runs/diagnostics/aaai_h1/teacher_r0_correctness_bins_val_full.png`
- Plot stats: `runs/diagnostics/aaai_h1/teacher_r0_correctness_bins_val_full.plot_stats.json`
- Scope: full VOC val, `1449` images, `5,935,104` sampled valid pixels.
- Pixel-level `corr(r, wrong)`: `0.3984`.
- Bin-level `corr(r, error)`: `0.9599`.
- Lowest-r bin error rate: `0.0104%`.
- Highest-r bin error rate: `29.5577%`.
- Interpretation: high reliability-score pixels are much more likely to be teacher mistakes, which directly supports the CoVar premise that unreliable teacher regions should be temperature-smoothed.

### H2: Reliability and Temperature Visual Examples

- Script: `scripts/diagnostics/visualize_covar_reliability_temperature.py`
- Log: `runs/diagnostics/aaai_h2/h2_visual_examples.log`
- Summary CSV: `runs/diagnostics/aaai_h2/h2_visual_examples_summary.csv`
- Summary JSON: `runs/diagnostics/aaai_h2/h2_visual_examples_summary.json`
- Figures:
  - `runs/diagnostics/aaai_h2/h2_rank01_2007_008260.png`
  - `runs/diagnostics/aaai_h2/h2_rank02_2007_008964.png`
  - `runs/diagnostics/aaai_h2/h2_rank03_2007_005149.png`
  - `runs/diagnostics/aaai_h2/h2_rank04_2007_004112.png`
  - `runs/diagnostics/aaai_h2/h2_rank05_2007_002426.png`
  - `runs/diagnostics/aaai_h2/h2_rank06_2007_003134.png`
- Scan scope: first `300` VOC-val images, automatically ranked by teacher wrong rate and high-r severity.
- Best paper candidates:
  - `h2_rank03_2007_005149.png`: teacher wrong `29.17%`, `r_p95=3.378`, `T_mean=0.704`, high-r/high-T regions align with teacher confusion over the dogs.
  - `h2_rank01_2007_008260.png`: teacher wrong `79.86%`, useful as a strong failure case where CoVar highlights a large unreliable teacher region.
  - `h2_rank02_2007_008964.png`: teacher wrong `49.03%`, `r_p95=2.443`, useful as another severe teacher-error example.

## Sync Log

| Time | Event |
|---|---|
| 2026-07-03 | Created AAAI execution plan from the agreed priority order: H -> E -> F -> G -> I. |
| 2026-07-03 | Launched H1 full VOC-val teacher reliability diagnostic on NPU 0. PID: `872947`; log: `runs/diagnostics/aaai_h1/teacher_r0_correctness_bins_val_full.log`; CSV target: `runs/diagnostics/aaai_h1/teacher_r0_correctness_bins_val_full.csv`. |
| 2026-07-03 | Completed H1 and generated plot. Main result: full-val bin-level `corr(r,error)=0.9599`, highest-r bin error rate `29.56%` vs lowest-r bin `0.0104%`. |
| 2026-07-03 | Added H2 visualization script and launched formal H2 scan on NPU 0. PID: `882857`; log: `runs/diagnostics/aaai_h2/h2_visual_examples.log`; output dir: `runs/diagnostics/aaai_h2/`. |
| 2026-07-03 | Completed H2. Generated six reliability/temperature visual examples and selected `h2_rank03_2007_005149.png` as the best paper-style mechanism figure candidate. |
| 2026-07-03 | Added Phase E seed-stability queue scripts and launched `Tout=1.0` seed queue. PID: `888441`; queue log: `runs/covar_npu_phaseE_seed_stability/phaseE_tout1_seed_stability_80000.log`; first active run: `phaseE_seed2025_cirkd_no_covar_tout1`. |
| 2026-07-04 09:05 CST | Phase E progress check: `phaseE_seed2025_cirkd_no_covar_tout1` completed at 2026-07-03 20:42 CST with best mIoU `0.6468` and final mIoU `0.6459` (`11:35:37`). Active run is `phaseE_seed2025_covar_newton_gamma2_tout1` at `78040/80000`, best-so-far mIoU `0.6458`, log ETA about `0:18:37`; seed 3407 off/on remain queued. |
| 2026-07-04 20:00 CST | Phase E progress check: `phaseE_seed2025_covar_newton_gamma2_tout1` completed at 2026-07-04 09:23 CST with best mIoU `0.6458` and final mIoU `0.6449` (`12:40:35`). Active run is `phaseE_seed3407_cirkd_no_covar_tout1` at `73160/80000`, best-so-far mIoU `0.6363`, latest validation mIoU `0.6248`, log ETA about `0:59:26`; final queued run is `phaseE_seed3407_covar_newton_gamma2_tout1`. |
| 2026-07-04 22:18 CST | Phase E progress check: `phaseE_seed3407_cirkd_no_covar_tout1` completed at 2026-07-04 20:59 CST with best/final mIoU `0.6382` (`11:35:36`). Active run is the final Phase E job, `phaseE_seed3407_covar_newton_gamma2_tout1`, at `8160/80000`, latest validation mIoU `0.4978`, log ETA about `11:24:44`. |
| 2026-07-05 09:21 CST | Phase E progress check: final Phase E job `phaseE_seed3407_covar_newton_gamma2_tout1` is active at `77880/80000` (`97.4%`), best-so-far/latest full validation mIoU `0.6421` at 2026-07-05 09:18 CST, log ETA about `0:20:09`; estimated training finish around 2026-07-05 09:41 CST, with final validation/reporting likely by 09:50 CST. This run has now surpassed the same-seed no-CoVar result `0.6382`. |
| 2026-07-05 09:54 CST | Phase E `Tout=1.0` seed-stability queue completed at 2026-07-05 09:41 CST; NPU released. Final seed3407 CoVar result: best mIoU `0.6421`, final mIoU `0.6412`, total time `12:41:09`. Aggregate report written to `reports/2026-07-05_phaseE_tout1_seed_stability.md`: best delta `+0.0019`, final delta `+0.0015`. |
| 2026-07-05 10:11 CST | Added Phase E `Tout=3.0` seed-stability queue scripts and launched seeds 2025/3407 off/on. PID: `645413`; queue log: `runs/covar_npu_phaseE_tout3_seed_stability/phaseE_tout3_seed_stability_80000.log`; save root: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseE_tout3_seed_stability`; first active run: `phaseE_seed2025_cirkd_no_covar_tout3`. Seed 1234 off/on are reused from Phase C (`phaseC_lc_no_covar_tout3`, `phaseC_lc_newton_gamma2_repro`). |
| 2026-07-05 10:13 CST | Phase E `Tout=3.0` launch health check passed: `phaseE_seed2025_cirkd_no_covar_tout3` is active at `180/80000`, NPU processes `645668/645669` are running, and training logs are writing normally. Early per-run ETA is still stabilizing; historical no-CoVar runtime is about 11.5-12.0h and CoVar runtime about 12.7-13.2h. |
| 2026-07-05 16:50 CST | Phase E `Tout=3.0` progress check: queue PID `645413` is still running. Active run `phaseE_seed2025_cirkd_no_covar_tout3` reached `46000/80000` (`57.5%`), latest validation mIoU `0.6033` at 2026-07-05 16:47 CST, best-so-far mIoU `0.6085` at 2026-07-05 16:33 CST, and log ETA is about `4:55:02`. Estimated current-run finish is around 2026-07-05 21:45-22:00 CST; remaining queued runs are seed2025 CoVar, seed3407 no-CoVar, and seed3407 CoVar. |
