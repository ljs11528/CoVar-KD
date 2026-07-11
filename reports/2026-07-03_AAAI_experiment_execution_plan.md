# AAAI CoVar-KD Experiment Execution Plan

- Created: 2026-07-03
- Goal: strengthen the CoVar-KD submission package for AAAI with mechanism evidence, statistical stability, model generalization, ablations, and optional cross-dataset validation.
- Current main result sources: `reports/2026-07-02_phaseD_npu_main_table.md`, `reports/2026-07-07_phaseE_tout3_seed_stability.md`, `reports/2026-07-07_phaseH_h3_rt_distribution.md`, `reports/2026-07-08_phaseF_psp_mbv3small_cross_student.md`
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
| H3 | `r`/`T` distribution for best CoVar run | histogram or curve from logs/checkpoints | complete |

Run first because it is fast and directly strengthens the method story.

### Phase E: Seed Stability

Purpose: convert single-run improvements into mean +/- std evidence.

| ID | Variant | Seeds | Status |
|---|---|---|---|
| E1 | `Tout=1.0, CoVar off` | 3 seeds | complete: seeds 1234/2025/3407 complete |
| E2 | `Tout=1.0, CoVar on` | 3 seeds | complete: seeds 1234/2025/3407 complete; report: `reports/2026-07-05_phaseE_tout1_seed_stability.md` |
| E3 | `Tout=3.0, CoVar off` | 3 seeds | complete: seeds 1234/2025/3407 complete; report: `reports/2026-07-07_phaseE_tout3_seed_stability.md` |
| E4 | `Tout=3.0, CoVar on` | 3 seeds | complete: seeds 1234/2025/3407 complete; report: `reports/2026-07-07_phaseE_tout3_seed_stability.md` |

Phase E seed stability is now complete for both teacher-output temperatures.

- `Tout=1.0`: best mIoU off `0.6425 +/- 0.0043`, on `0.6444 +/- 0.0020`, delta `+0.0019`; final mIoU off `0.6419 +/- 0.0039`, on `0.6434 +/- 0.0019`, delta `+0.0015`.
- `Tout=3.0`: best mIoU off `0.6411 +/- 0.0056`, on `0.6466 +/- 0.0066`, delta `+0.0055`; final mIoU off `0.6366 +/- 0.0019`, on `0.6448 +/- 0.0042`, delta `+0.0082`.
- `Tout=3.0` paired seed deltas are positive for all three seeds: best `+0.0064/+0.0041/+0.0061`; final `+0.0107/+0.0037/+0.0102` for seeds 1234/2025/3407.
- Caveat: `phaseE_seed2025_cirkd_no_covar_tout3` was resumed after disconnect, so the generated report's validation count and total time for that row reflect only the resumed segment; the final and best mIoU values remain valid for aggregate comparison.

### Phase F: Cross-Student Generalization

Purpose: show the method is not a MobileNetV3-Small-only effect.

| ID | Teacher | Student | Setting | Status |
|---|---|---|---|---|
| F1 | DeepLabV3-ResNet101 | PSPNet-MobileNetV3-Small | `Tout=3.0`, off/on | complete; report: `reports/2026-07-08_phaseF_psp_mbv3small_cross_student.md` |
| F2 | DeepLabV3-ResNet101 | DeepLabV3-ResNet18 | `Tout=3.0`, off/on | deferred until local ResNet18 ImageNet pretrained weight/config is ready |

Run at least one student pair before submission.

Phase F F1 tracking:

- Scripts:
  - `scripts/experiments/covar_npu/run_phaseF_psp_mbv3small_tout3.sh`
  - `scripts/experiments/covar_npu/launch_phaseF_psp_mbv3small_tout3.sh`
  - `scripts/experiments/covar_npu/monitor_phaseF_psp_mbv3small_tout3.sh`
- Queue PID: `3309200`
- Queue log: `runs/covar_npu_phaseF_psp_mbv3small/phaseF_psp_mbv3small_tout3_80000.log`
- PID file: `runs/covar_npu_phaseF_psp_mbv3small/phaseF_psp_mbv3small_tout3_80000.pid`
- Save root: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseF_psp_mbv3small`
- Variants:
  - off: `phaseF_psp_mbv3small_no_covar_tout3_seed1234` complete, best mIoU `0.6366`, final mIoU `0.6324`, total time `11:50:59`.
  - on: `phaseF_psp_mbv3small_covar_newton_gamma2_tout3_seed1234` complete, best mIoU `0.6387`, final mIoU `0.6359`, total time `12:58:49`.
- Smoke test: `MAX_ITERATIONS=20`, off/on, `SKIP_VAL=1`, passed at 2026-07-07 19:16 CST. Logs: `runs/covar_npu_phaseF_psp_mbv3small_smoke/smoke_20.log`.
- First validation health check: off variant reached `820/80000` at 2026-07-07 19:24 CST; first 800-iter validation produced pixAcc `80.246`, mIoU `28.353`; training continued normally.
- No-CoVar final validation: pixAcc `90.736`, mIoU `63.244` at 2026-07-08 07:07 CST; top checkpoint mIoU `0.6366` from 2026-07-08 06:32 CST.
- CoVar first validation health check: reached `820/80000` at 2026-07-08 07:16 CST; first 800-iter validation produced pixAcc `77.022`, mIoU `23.545`; training continued normally. Early 800-iter mIoU is a health signal only, not an outcome comparison.
- CoVar final validation: pixAcc `90.893`, mIoU `63.594` at 2026-07-08 20:07 CST; top checkpoint mIoU `0.6387` from 2026-07-08 19:43 CST.
- Phase F F1 delta: best mIoU `+0.0021`, final mIoU `+0.0035`. This is a positive single-seed cross-student generalization result.
- Runtime: no-CoVar took `11:50:59`, CoVar took `12:58:49`, full queue took about `24:51:04`.

### Phase G: Component Ablation

Purpose: prove each reliability component contributes.

| ID | Reliability / temperature setup | Status |
|---|---|---|
| G1 | confidence only: `-log(c)` | queued in Phase G 20k triage; smoke passed |
| G2 | residual variance only: `a*v/(1-c)` | queued in Phase G 20k triage; smoke passed |
| G3 | full CoVar: `-log(c)+a*v/(1-c)` | queued in Phase G 20k triage |
| G4 | fixed-temperature baseline | running as no-CoVar `Tout=3.0` 20k baseline in Phase G triage |
| G5 | gamma ablation: `gamma=0/1/2` | partially covered; summarize existing Phase C/B logs |

Use 20k triage before 80k if queue pressure is high.

Phase G tracking:

- Code support: `train_cirkdv2.py` now exposes `--covar-reliability-mode {full,confidence,variance}`; default `full` preserves existing behavior.
- Scripts:
  - `scripts/experiments/covar_npu/run_phaseG_component_ablation_triage.sh`
  - `scripts/experiments/covar_npu/launch_phaseG_component_ablation_triage.sh`
  - `scripts/experiments/covar_npu/monitor_phaseG_component_ablation_triage.sh`
  - `scripts/experiments/covar_npu/summarize_phaseG_component_ablation.py`
- Smoke test: `MAX_ITERATIONS=20`, `PHASEG_VARIANTS="confidence variance"`, `SKIP_VAL=1`, passed at 2026-07-08 20:27 CST. Smoke save root: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseG_component_ablation_smoke`.
- Formal queue: initial PID `765102` stopped unexpectedly after the no-CoVar run reached `12200/20000` with no Python traceback. Resumed at 2026-07-08 22:40 CST by appending to the same queue log. Current queue PID: `23660`; current worker processes: `23865/23866`; queue log `runs/covar_npu_phaseG_component_ablation/phaseG_component_ablation_20000.log`; PID file `runs/covar_npu_phaseG_component_ablation/phaseG_component_ablation_20000.pid`; save root `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseG_component_ablation`.
- Formal variant order: `off`, `confidence`, `variance`, `full`.
- Health check: active no-CoVar baseline reached `80/20000` at 2026-07-08 20:28 CST and continued normally.
- Latest progress: resume health check passed at 2026-07-08 22:42 CST. The no-CoVar baseline loaded `training_state_latest.pth` at iteration `12000`, preserving best mIoU `0.5790`, and has written new resumed iterations through `12140/20000`. The refreshed report is `reports/2026-07-08_phaseG_component_ablation_triage.md`.
- Runtime estimate: after the resume delay, expect the no-CoVar baseline to finish around 2026-07-08 23:50-2026-07-09 00:10 CST if uninterrupted; expected full four-variant queue completion is around 2026-07-09 09:00-10:30 CST.

### Phase I: Cross-Dataset Validation

Purpose: strengthen external validity.

Preferred dataset: Cityscapes if teacher/checkpoints/data are ready. Otherwise run COCO-Stuff only if setup is already stable.

Minimum design:

| ID | Dataset | Variants | Status |
|---|---|---|---|
| I1 | Cityscapes | no-CoVar vs CoVar, `Tout=3.0` | pending |
| I2 | Cityscapes | full `Tout=1/3 x off/on` table | optional |

## Immediate Next Actions

1. Use the completed `Tout=3.0` seed-stability aggregate as the main statistical evidence.
2. Use completed H1/H2/H3 diagnostics as the mechanism evidence package.
3. Use completed Phase F F1 as cross-student generalization evidence.
4. Monitor Phase G component ablation triage and summarize the four 20k variants when complete.
5. Resume Phase F F2 only after the ResNet18 student pretrained/config is available locally.

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

### H3: VOC Val r/T Distribution

- Script: `scripts/diagnostics/covar_rt_distribution.py`
- Report: `reports/2026-07-07_phaseH_h3_rt_distribution.md`
- Log: `runs/diagnostics/aaai_h3/h3_rt_distribution.log`
- Summary JSON: `runs/diagnostics/aaai_h3/h3_rt_distribution_summary.json`
- Figure: `runs/diagnostics/aaai_h3/h3_rt_distribution.png`
- Scope: full VOC val, `1449` images, `5,935,104` sampled valid pixels.
- Config: `Tout=3.0`, Newton CoVar temperature, `T in [0.5, 8.0]`, `eta=0.6`, `max_iter=8`, `max_step=0.25`, `a=200`.
- Teacher wrong rate: `5.0490%`.
- `r` distribution: mean `0.2258`, median `0.0125`, p95 `1.6600`, p99 `1.9452`.
- `T` distribution: mean `0.5815`, median `0.5000`, p95 `1.3226`, p99 `1.6919`.
- Temperature fractions: `90.35%` at `T_min=0.5`, `90.41%` below `0.75`, `5.96%` above `1.25`, `0.00%` at `T_max=8.0`.
- Correlations: `corr(r,T)=0.9159`, `corr(r,teacher_wrong)=0.4130`, `corr(T,teacher_wrong)=0.3688`.
- Interpretation: most high-confidence reliable pixels are sharpened, while the high-`r` tail is smoothed; the distribution supports the claim that CoVar performs pixel-wise reliability-aware temperature adaptation rather than a global temperature shift.

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
| 2026-07-05 20:12 CST | Phase E `Tout=3.0` progress check: queue PID `645413` remains healthy. Active run `phaseE_seed2025_cirkd_no_covar_tout3` reached `69160/80000` (`86.45%`), latest/best-so-far full validation mIoU is `0.6346` at 2026-07-05 20:09 CST, and log ETA is about `1:34:08`. Estimated current-run finish is around 2026-07-05 21:45-22:00 CST; no-CoVar seed2025 is approaching the seed1234 no-CoVar final mIoU `0.6383` but has not yet matched seed1234 best mIoU `0.6475`. |
| 2026-07-05 20:37 CST | Phase E `Tout=3.0` progress check: active run `phaseE_seed2025_cirkd_no_covar_tout3` reached `72020/80000` (`90.03%`). The 72k validation completed with pixAcc `90.890` and mIoU `0.6361`, refreshing best-so-far from `0.6346` to `0.6361`; log ETA after validation is about `1:09:20`. Estimated current-run finish remains around 2026-07-05 21:45-21:55 CST. |
| 2026-07-05 20:55 CST | Phase E `Tout=3.0` progress check: active run `phaseE_seed2025_cirkd_no_covar_tout3` reached `74100/80000` (`92.63%`), with log ETA `0:51:13`. Latest validation at 2026-07-05 20:50 CST produced mIoU `0.6347`, below the current best `0.6361`, so best checkpoint remains the 72k validation. Estimated current-run finish is around 2026-07-05 21:45-21:50 CST. |
| 2026-07-05 22:22 CST | After disconnect, Phase E `Tout=3.0` queue PID `645413` was no longer running. NPU was idle. Active run `phaseE_seed2025_cirkd_no_covar_tout3` had last logged `79400/80000` at 2026-07-05 21:41 CST; `training_state_latest.pth` was at iteration `79200` with best mIoU `0.6364`. |
| 2026-07-05 22:28 CST | Relaunched Phase E `Tout=3.0` queue by appending to the existing queue log instead of overwriting it. New PID: `26458`; log: `runs/covar_npu_phaseE_tout3_seed_stability/phaseE_tout3_seed_stability_80000.log`. The run resumed from `training_state_latest.pth` at iteration `79200`, preserving best mIoU `0.6364`. |
| 2026-07-05 22:30 CST | Resume health check passed: NPU processes `26698/26699` are active, queue PID `26458` is running, and `phaseE_seed2025_cirkd_no_covar_tout3` has written new iterations through `79400/80000` with ETA about `0:05:03`. |
| 2026-07-05 22:38 CST | `phaseE_seed2025_cirkd_no_covar_tout3` completed after resume. Final validation pixAcc `90.951`, mIoU `0.6370`; top checkpoint `kd_deeplabv3_mobilenet_ssseg_mobilenetv3_small_voc_miou-0.6370.pth`. Queue automatically advanced to `phaseE_seed2025_covar_newton_gamma2_tout3`, active at `220/80000` with NPU processes `38420/38421`; early ETA about `12.3h`. |
| 2026-07-06 09:04 CST | Phase E `Tout=3.0` progress check: queue PID `26458` remains healthy on NPU processes `38420/38421`. Active run `phaseE_seed2025_covar_newton_gamma2_tout3` reached `65780/80000` (`82.2%`), latest validation mIoU `0.6166` at 2026-07-06 09:02 CST, best-so-far mIoU `0.6389` at 2026-07-06 08:24 CST, log ETA about `2:15:38`. This best-so-far is already above same-seed no-CoVar `0.6370`. Estimated current-run finish is around 2026-07-06 11:20-11:30 CST; remaining queued runs are seed3407 no-CoVar and seed3407 CoVar. Expected full Phase E `Tout=3.0` queue completion is around 2026-07-07 late morning to midday CST if uninterrupted. |
| 2026-07-07 18:23 CST | Phase E `Tout=3.0` seed-stability queue completed earlier at 2026-07-07 12:04 CST; PID `26458` is no longer running and NPU is idle. Report: `reports/2026-07-07_phaseE_tout3_seed_stability.md`. Aggregate: best mIoU off `0.6411 +/- 0.0056`, on `0.6466 +/- 0.0066`, delta `+0.0055`; final mIoU off `0.6366 +/- 0.0019`, on `0.6448 +/- 0.0042`, delta `+0.0082`. Per-seed final deltas are seed1234 `+0.0107`, seed2025 `+0.0037`, seed3407 `+0.0102`. |
| 2026-07-07 19:02 CST | Completed H3 full VOC-val `r`/`T` distribution diagnostic. Outputs: `runs/diagnostics/aaai_h3/`; report: `reports/2026-07-07_phaseH_h3_rt_distribution.md`. Main result: `corr(r,T)=0.9159`, `corr(r,teacher_wrong)=0.4130`, `corr(T,teacher_wrong)=0.3688`; `90.35%` of pixels are at `T_min=0.5`, `5.96%` are above `1.25`, and none hit `T_max=8.0`. |
| 2026-07-07 19:16 CST | Added Phase F F1 PSPNet-MobileNetV3-Small scripts and passed a 20-iteration off/on smoke test with `SKIP_VAL=1`. Smoke log: `runs/covar_npu_phaseF_psp_mbv3small_smoke/smoke_20.log`; smoke save root: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseF_psp_mbv3small_smoke`. |
| 2026-07-07 19:17 CST | Launched Phase F F1 formal 80k off/on queue. PID: `3309200`; log: `runs/covar_npu_phaseF_psp_mbv3small/phaseF_psp_mbv3small_tout3_80000.log`; save root: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseF_psp_mbv3small`. Active run: `phaseF_psp_mbv3small_no_covar_tout3_seed1234`; health check reached `40/80000` with NPU worker processes `3309389/3309390`. Expected full off/on wall time: about `24-26h`. |
| 2026-07-07 19:24 CST | Phase F F1 first validation health check passed. Active no-CoVar run reached `820/80000`; first 800-iter validation pixAcc `80.246`, mIoU `28.353`; queue PID `3309200` and worker processes `3309389/3309390` remain healthy. Current no-CoVar ETA is about `12.25h`; full off/on completion estimate remains around 2026-07-08 20:00-22:00 CST. |
| 2026-07-08 07:08 CST | Phase F F1 no-CoVar completed and queue advanced automatically to CoVar. No-CoVar result: best mIoU `0.6366`, final mIoU `0.6324`, final validation pixAcc `90.736`, total time `11:50:59`. Top checkpoint: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseF_psp_mbv3small/phaseF_psp_mbv3small_no_covar_tout3_seed1234/kd_psp_mobile_mobilenetv3_small_voc_miou-0.6366.pth`. |
| 2026-07-08 07:16 CST | Phase F F1 CoVar health check: active run `phaseF_psp_mbv3small_covar_newton_gamma2_tout3_seed1234` reached `820/80000`; first 800-iter validation pixAcc `77.022`, mIoU `23.545`; queue PID `3309200` and worker processes `4075085/4075088` remain healthy. Current ETA after first validation is about `12.9h`, estimating F1 completion around 2026-07-08 20:15-20:45 CST. |
| 2026-07-08 19:42 CST | Phase F F1 CoVar late-stage progress check: queue PID `3309200` remains healthy with worker processes `4075085/4075088`. Active run reached `77560/80000` (`96.95%`), best-so-far/latest validation mIoU `0.6358` at 2026-07-08 19:36 CST, latest ETA `0:23:43`. It is now close to the no-CoVar best `0.6366` and above the no-CoVar final `0.6324`; final comparison should wait for the 80k validation. Expected completion is around 2026-07-08 20:10-20:25 CST. |
| 2026-07-08 20:16 CST | Phase F F1 PSPNet-MobileNetV3-Small off/on queue completed and NPU is idle. Report written: `reports/2026-07-08_phaseF_psp_mbv3small_cross_student.md`. Result: no-CoVar best/final `0.6366/0.6324`; CoVar best/final `0.6387/0.6359`; deltas best `+0.0021`, final `+0.0035`. Full queue wall time was about `24:51:04`. |
| 2026-07-08 20:22 CST | Checked Phase F F2 readiness before starting the next queue. Local search did not find a ResNet18 ImageNet pretrained `.pth` under `data/` or `data/winycg/`; existing ResNet18 scripts reference `resnet18-imagenet.pth`, so F2 is deferred until the pretrained/config is present. |
| 2026-07-08 20:29 CST | Added Phase G component-ablation support and scripts. Smoke test passed for confidence-only and variance-only modes (`MAX_ITERATIONS=20`, `SKIP_VAL=1`), then launched formal Phase G 20k queue with variants `off confidence variance full`. Queue PID: `765102`; worker PIDs: `765301/765302`; log: `runs/covar_npu_phaseG_component_ablation/phaseG_component_ablation_20000.log`; save root: `data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseG_component_ablation`. Active no-CoVar run reached `80/20000` at 2026-07-08 20:28 CST. Expected full queue wall time: about `12-13h`, completion around 2026-07-09 08:45-09:30 CST if uninterrupted. |
| 2026-07-08 21:26 CST | Phase G progress check: queue PID `765102` and worker processes `765301/765302` remain healthy. Active run is still `phaseG_triage_no_covar_tout3_seed1234`, reached `6600/20000` (`33.0%`) with `8` completed validations. Current best/latest mIoU is `0.5209` and latest pixAcc is `0.8784`; next validation is at `7200`. No-CoVar ETA is about `2.0h`, estimated completion around 2026-07-08 23:25 CST. Full queue completion estimate remains around 2026-07-09 08:50-09:30 CST. Updated in-progress report: `reports/2026-07-08_phaseG_component_ablation_triage.md`. |
| 2026-07-08 21:45 CST | Phase G progress check: queue PID `765102` and worker processes `765301/765302` remain healthy. Active run `phaseG_triage_no_covar_tout3_seed1234` reached `8760/20000` (`43.8%`) with `10` completed validations. Current best mIoU is `0.5366` at 7200 iter; latest 8000-iter validation mIoU is `0.5009`, pixAcc `0.8691`. No-CoVar ETA is about `1.65h`, still estimating completion around 2026-07-08 23:25 CST. Full queue completion estimate is around 2026-07-09 08:40-09:30 CST. Updated in-progress report: `reports/2026-07-08_phaseG_component_ablation_triage.md`. |
| 2026-07-08 22:09 CST | Phase G progress check: queue PID `765102` and worker processes `765301/765302` remain healthy. Active run `phaseG_triage_no_covar_tout3_seed1234` reached `11440/20000` (`57.2%`) with `14` completed validations. Current best/latest mIoU is `0.5790` at 11200 iter, latest pixAcc `0.8918`. No-CoVar ETA is about `1.27h`, still estimating completion around 2026-07-08 23:25 CST. Full queue completion estimate is around 2026-07-09 08:30-09:20 CST. Updated in-progress report: `reports/2026-07-08_phaseG_component_ablation_triage.md`. |
| 2026-07-08 22:16 CST | Phase G queue stopped unexpectedly after the active no-CoVar run last logged `12200/20000`; no Python traceback appeared in the queue or variant log. The latest durable `training_state_latest.pth` checkpoint was at iteration `12000`, preserving best mIoU `0.5790`. |
| 2026-07-08 22:40 CST | Updated `train_cirkdv2.py` so auto-resume runs append to existing variant logs instead of overwriting them, preserving validation history for summarizers. Relaunched Phase G by appending to `runs/covar_npu_phaseG_component_ablation/phaseG_component_ablation_20000.log`. New queue PID: `23660`. |
| 2026-07-08 22:42 CST | Phase G resume health check passed. NPU worker processes `23865/23866` are active; the run resumed from `training_state_latest.pth` at iteration `12000`, best mIoU `0.5790`, and wrote new logs through `12140/20000`. Refreshed report: `reports/2026-07-08_phaseG_component_ablation_triage.md`. |
