# CoVar-KD Paper Experiment Record

- Updated: 2026-07-13
- Repository: `/home/ma-user/work/ljs`
- Hardware/runtime verified 2026-07-12: 2 x Ascend 910, CANN 8.5.0, Python 3.11.10, PyTorch 2.8.0+cpu with torch_npu 2.8.0.post2
- Dataset: Pascal VOC / VOCAug, 21 classes, VOC val with 1,449 images
- Main teacher: DeepLabV3-ResNet101
- Main student: DeepLabV3-MobileNetV3-Small
- Cross-student: PSPNet-MobileNetV3-Small
- Default training budget: 80,000 iterations, global batch 16, crop 512 x 512
- Primary metrics: best validation mIoU and final-iteration validation mIoU

## 1. Scope and comparison rules

The paper currently has two distinct evidence tracks. They must not be conflated.

1. **Controlled CoVar evidence:** CoVar on/off comparisons use the same CIRKD code path, teacher/student, seed, schedule, and loss recipe. These rows support causal claims about adaptive temperature.
2. **Method-level baselines:** KD-only, CWD, SKD, and IFVD use the repository's official `train_kd.py` recipes. In particular, the official CWD recipe is task CE + logit KD + adversarial KD + CWD feature/logit losses. These rows support absolute competitiveness claims, not isolated-loss attribution.

All main claims should use 80k results. The 20k runs are triage or ablation evidence unless explicitly labeled otherwise.

## 2. Main CoVar result: teacher softening x adaptive temperature

Source: `reports/2026-07-02_phaseD_npu_main_table.md`.

| Teacher output temp | CoVar | Best mIoU | Final mIoU | Delta best | Delta final | Runtime |
|---:|---|---:|---:|---:|---:|---:|
| 1.0 | off | 0.6426 | 0.6416 | - | - | 11:55:26 |
| 1.0 | on | 0.6453 | 0.6440 | +0.0027 | +0.0024 | partial row: 2:10:02 |
| 3.0 | off | 0.6475 | 0.6383 | - | - | 11:57:26 |
| 3.0 | on | **0.6539** | **0.6490** | **+0.0064** | **+0.0107** | 13:11:18 |

Paper use:

- The `Tout=3.0` pair is the primary controlled result.
- CoVar improves both best and final mIoU, with a larger final gain that suggests reduced late-training degradation.
- The `Tout=1.0` result is supportive but weaker; its seed-1234 on-row contains only the resumed 17-validation segment and must not be used for runtime or curve-shape claims.

## 3. Seed stability

### 3.1 Primary setting: Tout=3.0

Source: `reports/2026-07-07_phaseE_tout3_seed_stability.md`.

| Seed | Off best/final | CoVar best/final | Delta best | Delta final |
|---:|---:|---:|---:|---:|
| 1234 | 0.6475 / 0.6383 | 0.6539 / 0.6490 | +0.0064 | +0.0107 |
| 2025 | 0.6370 / 0.6370 | 0.6411 / 0.6407 | +0.0041 | +0.0037 |
| 3407 | 0.6387 / 0.6346 | 0.6448 / 0.6448 | +0.0061 | +0.0102 |
| Mean +/- SD | 0.6411 +/- 0.0056 / 0.6366 +/- 0.0019 | **0.6466 +/- 0.0066 / 0.6448 +/- 0.0042** | **+0.0055** | **+0.0082** |

All three paired deltas are positive. This is the strongest statistical result currently available for CoVar.

Caveat: seed-2025 off was resumed near completion, so its report shows one validation and a partial runtime. Its recovered best/final values are valid; its validation count and runtime are not comparable.

### 3.2 Strong-teacher setting: Tout=1.0

Source: `reports/2026-07-05_phaseE_tout1_seed_stability.md`.

| Metric | Off mean +/- SD | CoVar mean +/- SD | Delta |
|---|---:|---:|---:|
| Best mIoU | 0.6425 +/- 0.0043 | 0.6444 +/- 0.0020 | +0.0019 |
| Final mIoU | 0.6419 +/- 0.0039 | 0.6434 +/- 0.0019 | +0.0015 |

Interpretation: CoVar is most useful when the softened teacher exposes a meaningful uncertainty structure. The gain under `Tout=1.0` is modest and one seed is negative, so this table belongs in the ablation/supplement rather than the headline result.

## 4. Cross-student generalization

Source: `reports/2026-07-08_phaseF_psp_mbv3small_cross_student.md`.

| Student | CoVar | Best mIoU | Final mIoU | Final pixAcc | Runtime |
|---|---|---:|---:|---:|---:|
| PSPNet-MobileNetV3-Small | off | 0.6366 | 0.6324 | 0.9074 | 11:50:59 |
| PSPNet-MobileNetV3-Small | on | **0.6387** | **0.6359** | **0.9089** | 12:58:49 |

Delta: +0.0021 best and +0.0035 final. This supports architecture transfer, but it is a single-seed result and must not be described as a stability experiment.

## 5. Method and component ablations

### 5.1 Temperature-power triage

Source: `reports/2026-06-30_phaseC_npu_triage.md`. All rows use 20k iterations and `Tout=3.0`.

| Variant | Best/final mIoU | Interpretation |
|---|---:|---|
| Newton, gamma=0 | 0.6087 / 0.6087 | Under-scales the pixel-wise KD term |
| Newton, gamma=1 | 0.6282 / 0.6282 | Intermediate |
| Newton, gamma=2 | **0.6353 / 0.6353** | Selected setting |
| No CoVar | 0.6331 / 0.6331 | Fixed-temperature control |

This supports gamma=2 for the Newton formulation. It should be reported as a short-budget hyperparameter ablation, not mixed into the 80k main table.

### 5.2 Reliability components

Source: `reports/2026-07-09_phaseG_component_ablation_triage.md`. All rows use 20k iterations, seed 1234, and `Tout=3.0`.

| Reliability input | Best/final mIoU | Delta vs off |
|---|---:|---:|
| Off | 0.6267 / 0.6267 | - |
| Confidence only | 0.6285 / 0.6285 | +0.0018 |
| Variance only | 0.6242 / 0.6242 | -0.0025 |
| Confidence + variance | **0.6301 / 0.6301** | **+0.0035** |

Interpretation: variance is not independently useful under this parameterization, but it complements confidence in the full reliability score. The claim should be "the joint score is better than either isolated component," not "both components independently improve accuracy."

## 6. External KD baseline comparison

### 6.1 Final 80k table

Source logs:

- `data/winycg/checkpoints/kd_baselines_npu/phaseK_voc_80k/phaseK_npu0_cwd_ifvd.nohup.log`
- `data/winycg/checkpoints/kd_baselines_npu/phaseK_voc_80k/phaseK_npu1_skd_kdonly.nohup.log`

All rows use DeepLabV3-ResNet101 -> DeepLabV3-MobileNetV3-Small, seed 1234, global batch 16, and 80k iterations.

| Method recipe | Best mIoU | Best iter | Final mIoU | Runtime |
|---|---:|---:|---:|---:|
| KD-only | 0.625 | 79.2k | 0.618 | 9:02:17 |
| IFVD | 0.630 | 79.2k | 0.626 | 10:49:02 |
| SKD | 0.634 | 76.8k | 0.631 | 9:40:36 |
| CoVar on CIRKD, Tout=3.0 | 0.6539 | - | 0.6490 | 13:11:18 |
| CWD official recipe | **0.664** | 79.2k | **0.661** | 9:42:12 |

CWD exceeds the current CoVar result by +0.0101 best and +0.0120 final. Its last-ten-validation mean is 0.6576 with SD 0.0041, so the lead is not only a single early checkpoint.

This changes the paper positioning:

- Do not claim state-of-the-art or strongest absolute VOC performance for the current CIRKD+CoVar model.
- The controlled CoVar gain remains valid.
- The next decisive experiment is Newton CoVar on the official CWD recipe.

### 6.2 Full 20k screening table

| Method | Best/final mIoU |
|---|---:|
| KD-only | 0.607 / 0.606 |
| CWD | **0.639 / 0.636** |
| SKD | 0.620 / 0.620 |
| IFVD | 0.615 / 0.615 |
| FitNet | 0.616 / 0.616 |
| AT | 0.610 / 0.610 |
| DSD | 0.614 / 0.614 |

Use this only to document the promotion process or in supplementary material.

## 7. Mechanism evidence

### 7.1 H1: reliability predicts teacher error

Artifacts: `runs/diagnostics/aaai_h1/`.

- Full VOC val: 1,449 images and 5,935,104 sampled valid pixels.
- Pixel-level `corr(r, teacher_wrong) = 0.3984`.
- Quantile-bin `corr(r, error_rate) = 0.9599`.
- Lowest-r bin teacher error: 0.0104%.
- Highest-r bin teacher error: 29.5577%, a 2,852x ratio.

This directly supports the premise that the confidence-variance score identifies unreliable teacher supervision.

### 7.2 H2: qualitative alignment

Artifacts: `runs/diagnostics/aaai_h2/`.

- Six automatically selected VOC examples are available.
- Preferred paper figure: `h2_rank03_2007_005149.png`.
- In that example, teacher wrong rate is 29.17%, `r_p95=3.378`, and `T_mean=0.704`; high-r/high-T regions align with teacher confusion over the dogs.
- `h2_rank01_2007_008260.png` is a useful severe failure case with 79.86% teacher error.

### 7.3 H3: reliability-to-temperature behavior

Source: `reports/2026-07-07_phaseH_h3_rt_distribution.md`; artifacts: `runs/diagnostics/aaai_h3/`.

- `corr(r,T)=0.9159`.
- `corr(r,teacher_wrong)=0.4130`.
- `corr(T,teacher_wrong)=0.3688`.
- Temperature mean/median/p95/p99: 0.5815 / 0.5000 / 1.3226 / 1.6919.
- 90.35% of pixels are at `T_min=0.5`; 5.96% are above 1.25; 0% hit `T_max=8.0`.

The solver mainly sharpens reliable pixels and smooths a small unreliable tail. The high lower-bound occupancy must be acknowledged; it motivates either a bound sensitivity study or a concise explanation that most VOC teacher pixels are already correct.

## 8. Efficiency evidence

| Setting | Baseline sec/iter | CoVar sec/iter | Relative overhead |
|---|---:|---:|---:|
| Main DeepLab student, Tout=3.0 | 0.5381 | 0.5935 | +10.3% |
| PSPNet cross-student | 0.5332 | 0.5841 | +9.5% |

The paper can report approximately 10% training-time overhead. Inference is unchanged because CoVar is used only during distillation training.

## 9. Latest experiments and required next evidence

### 9.1 Phase L: CWD seed stability

Status: completed on 2026-07-11. Source: `reports/2026-07-11_phaseL_cwd_seed_stability.md`.

| Seed | Best mIoU | Best iter | Final mIoU | Status |
|---:|---:|---:|---:|---|
| 1234 | 0.6640 | 79200 | 0.6610 | complete |
| 2025 | 0.6600 | 73600 | 0.6580 | complete |
| 3407 | 0.6650 | 75200 | 0.6640 | complete |

The completed three-seed aggregate is `0.6630 +/- 0.0022` best and `0.6610 +/- 0.0024` final using the current summary script's population SD. For paper reporting with sample SD (`ddof=1`), the corresponding values are approximately `0.6630 +/- 0.0026` best and `0.6610 +/- 0.0030` final. Method-level comparisons against the existing three-seed CoVar/CIRKD table remain non-causal because the base recipes differ.

### 9.2 CWD + CoVar

Status: smoke and 20k triage completed on 2026-07-11. Source: `reports/2026-07-11_phaseM_cwd_covar_triage.md`.

| Variant | Teacher output temp | Logit KD temperature | Best mIoU | Best iter | Final mIoU | Sec/iter |
|---|---:|---|---:|---:|---:|---:|
| Historical CWD | 1.0 | scalar `T=1.0` | 0.6390 | 18400 | 0.6360 | 0.4392 |
| CWD fixed | 3.0 | scalar `T=1.0` | 0.6430 | 19200 | 0.6420 | 0.4350 |
| CWD + Newton CoVar | 3.0 | pixel-wise `T(x)` | **0.6460** | 20000 | **0.6460** | 0.4537 |

- Against the controlled `Tout=3.0`, scalar-`T=1.0` row, CoVar improves best mIoU by `+0.0030` and final mIoU by `+0.0040`.
- Training time rises from `0.4350` to `0.4537` sec/iter, a `+4.3%` overhead in this CWD integration.
- Only the logit KD temperature mechanism changes in the controlled pair; the official CWD task, KD, adversarial, feature-CWD, and logit-CWD recipe remains fixed.
- This is positive preliminary portability evidence, but it does not yet isolate spatial adaptation from the global low-temperature sharpening effect.

### 9.3 Phase M2: matched scalar-temperature controls

Status: completed on 2026-07-12. Sources: [final result](2026-07-12_phaseM2_scalar_temperature.md) and [preregistration/run record](2026-07-12_phaseM2_scalar_temperature_plan.md).

| Variant | Best mIoU | Final mIoU | Last-10 mean | Runtime |
|---|---:|---:|---:|---:|
| Scalar `T=0.5` | 0.648235 | 0.648235 | 0.626266 | 2:25:26 |
| Scalar `T=0.6` | **0.653381** | **0.653381** | 0.624513 | 2:25:01 |
| Scalar `T=1.0` | 0.643000 | 0.642000 | **0.626800** | 2:25:00 |
| Newton CoVar | 0.646000 | 0.646000 | 0.622100 | 2:31:15 |

- `T=0.6` is the strongest matched scalar by the preregistered primary metric. Its final mIoU is `+0.011381` over `T=1.0` and `+0.007381` over CoVar.
- `CoVar - T=0.6 = -0.007381` final mIoU, which hits the preregistered `<= -0.002` rule: **scalar temperature is stronger**. The automatic 80k multi-seed promotion of CoVar is stopped.
- Both matched low-temperature controls outperform CoVar on final mIoU. Therefore the Phase M CoVar gain over `T=1.0` cannot be attributed to spatial adaptation without further evidence; global low-temperature sharpening is the more plausible explanation under this CWD setting.
- The last-10 mean ranks `T=1.0` first, so the 20k endpoints and late-window average are not perfectly aligned. This is a single-seed triage result and must not be reported as statistical superiority.

### 9.4 Phase N: 80k scalar-temperature confirmation

Status: running since `2026-07-13T08:10:52+08:00`; paired smoke completed first. Source: [Phase N plan and live run record](2026-07-13_phaseN_scalar_temperature_80k_plan.md).

- Compare scalar `T=1.0` on NPU 0 against scalar `T=0.6` on NPU 1 under the same CWD recipe, `Tout=3.0`, seed `1234`, and 80k budget.
- Both runs passed the strict 20-iteration smoke, reached the first 800-iteration validation, produced all model/training-state artifacts, and continued without traceback/NaN/OOM. The first validation is a health check, not a result claim.
- Primary metric: final mIoU. Secondary metrics: best mIoU/best iteration and last-10-validation mean.
- This pair tests whether the Phase M2 low-temperature endpoint gain persists at the paper's full training budget. It does not test spatial adaptation.
- Multi-seed promotion is conditional on the preregistered final-mIoU delta and late-window behavior; no multi-seed conclusion is authorized before the seed-1234 pair completes.

## 10. Paper-ready claims and prohibited overclaims

Supported:

1. The confidence-variance reliability score strongly tracks teacher error.
2. Newton CoVar maps high unreliability to higher pixel temperatures as intended.
3. On the same CIRKD base, CoVar improves all three `Tout=3.0` seeds and a second student head.
4. The full confidence+variance score outperforms either isolated component in 20k ablation.
5. CoVar adds about 10% training overhead and no inference-time module.

Not yet supported:

1. CoVar is the strongest method on VOC: CWD is currently higher.
2. Variance alone improves performance: the isolated variance row is negative.
3. Cross-dataset generalization: only VOC data and a VOC teacher are locally available.
4. Cross-student statistical stability: the PSPNet result is one seed.
5. CWD+CoVar superiority over matched scalar temperatures: Phase M2 instead favors scalar `T=0.6` by `+0.007381` final mIoU at 20k.
6. An 80k or multi-seed benefit from scalar `T=0.6`: Phase N is designed to test the first of these claims.

## 11. Recommended paper placement

- Main table: 80k method comparison, with CWD seed statistics when complete.
- Main ablation: `Tout=3.0` CoVar off/on, three-seed aggregate, and reliability components.
- Main mechanism figure: H1 reliability/error curve plus H2 rank-3 qualitative map.
- Supplement: `Tout=1.0` seeds, gamma triage, all 20k baseline rows, H3 full distributions, and additional H2 examples.
- Limitations: single dataset, CWD currently stronger in absolute mIoU, high `T_min` occupancy, and the Phase M2 evidence that matched scalar low temperature can outperform CoVar.
