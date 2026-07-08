# Phase H3 r/T Distribution Diagnostic Report

- Generated: 2026-07-07T19:05:00+08:00
- Script: `scripts/diagnostics/covar_rt_distribution.py`
- Output directory: `runs/diagnostics/aaai_h3`
- Summary JSON: `runs/diagnostics/aaai_h3/h3_rt_distribution_summary.json`
- Figure: `runs/diagnostics/aaai_h3/h3_rt_distribution.png`
- Log: `runs/diagnostics/aaai_h3/h3_rt_distribution.log`

## Run Scope

- Dataset: VOC val
- Images: `1449 / 1449`
- Sampled valid pixels: `5,935,104`
- Max pixels per image: `4096`
- Teacher output temperature: `3.0`
- CoVar temperature mode: Newton
- Temperature bounds: `[0.5, 8.0]`
- Newton settings: `eta=0.6`, `max_iter=8`, `max_step=0.25`, `hessian_eps=1e-5`
- CoVar coefficient: `a=200.0`

## Key Results

| Quantity | Value |
|---|---:|
| Teacher wrong rate | `5.0490%` |
| `r` mean / median / p95 / p99 | `0.2258 / 0.0125 / 1.6600 / 1.9452` |
| `T` mean / median / p95 / p99 | `0.5815 / 0.5000 / 1.3226 / 1.6919` |
| Fraction at `T_min=0.5` | `90.35%` |
| Fraction `T < 0.75` | `90.41%` |
| Fraction `0.75 <= T <= 1.25` | `3.62%` |
| Fraction `T > 1.25` | `5.96%` |
| Fraction at `T_max=8.0` | `0.00%` |
| `corr(r, T)` | `0.9159` |
| `corr(r, confidence)` | `-0.9817` |
| `corr(r, variance)` | `0.9734` |
| `corr(r, teacher_wrong)` | `0.4130` |
| `corr(T, teacher_wrong)` | `0.3688` |

## Interpretation

The distribution matches the intended mechanism. Most pixels are high-confidence reliable teacher pixels and are sharpened to the lower temperature bound, while the high-`r` tail receives larger temperatures. The strong positive `corr(r, T)=0.9159` confirms that the Newton solver maps higher unreliability to stronger smoothing. The positive `corr(r, teacher_wrong)=0.4130` and `corr(T, teacher_wrong)=0.3688` align H3 with H1: pixels assigned higher unreliability and higher temperature are more likely to be teacher errors.

This supports the paper claim that CoVar is not just adding noise or a global temperature shift. It is producing a pixel-wise temperature distribution concentrated on sharpening reliable teacher regions and smoothing the small but important unreliable tail.

## Artifacts

- Per-image summary: `runs/diagnostics/aaai_h3/h3_rt_distribution_image_summary.csv`
- Raw sampled arrays: `runs/diagnostics/aaai_h3/h3_rt_distribution_samples.npz`
- Histograms:
  - `runs/diagnostics/aaai_h3/h3_r_histogram.csv`
  - `runs/diagnostics/aaai_h3/h3_temperature_histogram.csv`
  - `runs/diagnostics/aaai_h3/h3_confidence_histogram.csv`
  - `runs/diagnostics/aaai_h3/h3_variance_histogram.csv`
  - `runs/diagnostics/aaai_h3/h3_r_over_t2_histogram.csv`

