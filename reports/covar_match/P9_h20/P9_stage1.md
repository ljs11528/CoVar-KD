# P9: second teacher-student pair fixed-temperature response

## Outcome

Identifiability result: result_A_no_unique_reproducible_grid_winner.
Final grid sample-mean winner: T=2.0 at 69.305603 mIoU.
Delta=0.2 pp near-optimal grid set: 1.5, 2.0.
Conditional stage-2 status: not_required (coarse_grid_does_not_identify_an_unresolved_candidate_peak_inside_0p5_to_1p5).

## Locked protocol

- Teacher: DeepLabV3-ResNet101; student: DeepLabV3-MobileNetV3-Large.
- Student capacity and execution protocol both differ from P7.
- Execution: Single CUDA rank. The student head uses BatchNorm instead of SyncBatchNorm; backbone batch statistics use 16 rather than 8 samples per rank. Sampling and validation reduction differ from P7. Same-numbered seeds do not establish a controlled P7 comparison.
- Teacher-only temperature, student T=1, no T-squared compensation, CE + KD.
- Seeds: 1234, 2025, 3407; fresh 80k poly schedule.
- Validation: 20k, 40k, 60k, 80k; primary endpoint: final 80k mIoU.

## Raw trajectories

| seed | T | 20k | 40k | 60k | 80k/final | best observed |
|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 0.25 | 59.622890 | 61.924523 | 67.492074 | 69.480413 | 69.480413 |
| 1234 | 0.5 | 59.929723 | 60.579044 | 67.794269 | 68.759316 | 68.759316 |
| 1234 | 1.0 | 56.869495 | 62.359279 | 65.753555 | 67.800862 | 67.800862 |
| 1234 | 1.5 | 59.353191 | 61.694688 | 67.097867 | 69.165152 | 69.165152 |
| 1234 | 2.0 | 60.939342 | 65.211493 | 66.531247 | 69.335973 | 69.335973 |
| 2025 | 0.25 | 58.267230 | 63.677323 | 65.159625 | 68.664455 | 68.664455 |
| 2025 | 0.5 | 61.789423 | 60.922652 | 65.821910 | 69.214028 | 69.214028 |
| 2025 | 1.0 | 59.115952 | 55.225861 | 66.291106 | 68.012673 | 68.012673 |
| 2025 | 1.5 | 58.979142 | 60.132855 | 65.769517 | 68.912899 | 68.912899 |
| 2025 | 2.0 | 61.017811 | 62.209457 | 66.855758 | 68.994683 | 68.994683 |
| 3407 | 0.25 | 62.363464 | 62.869537 | 67.041171 | 68.462300 | 68.462300 |
| 3407 | 0.5 | 60.983914 | 61.737239 | 67.244428 | 68.115371 | 68.115371 |
| 3407 | 1.0 | 61.890501 | 63.001603 | 67.096633 | 69.029480 | 69.029480 |
| 3407 | 1.5 | 62.498707 | 63.843375 | 67.157447 | 69.320095 | 69.320095 |
| 3407 | 2.0 | 62.531990 | 63.585877 | 67.727947 | 69.586152 | 69.586152 |

## Final response estimates

| T | mean final mIoU | sample SD | values by seed |
|---:|---:|---:|---|
| 0.25 | 68.869056 | 0.539013 | 69.480413, 68.664455, 68.462300 |
| 0.5 | 68.696238 | 0.552038 | 68.759316, 69.214028, 68.115371 |
| 1.0 | 68.281005 | 0.656793 | 67.800862, 68.012673, 69.029480 |
| 1.5 | 69.132715 | 0.205527 | 69.165152, 68.912899, 69.320095 |
| 2.0 | 69.305603 | 0.296902 | 69.335973, 68.994683, 69.586152 |

## Per-seed winners

| seed | winner T |
|---:|---:|
| 1234 | 0.25 |
| 2025 | 0.5 |
| 3407 | 2.0 |

## CoVar and cross-pair comparison

| T | mean r_c | mean r_v | mean r |
|---:|---:|---:|---:|
| 0.25 | 0.005927386 | 0.045944399 | 0.051871785 |
| 0.5 | 0.012145204 | 0.091226898 | 0.103372101 |
| 1.0 | 0.026652498 | 0.179955250 | 0.206607748 |
| 1.5 | 0.050836446 | 0.243055924 | 0.293892370 |
| 2.0 | 0.102252647 | 0.251133329 | 0.353385976 |

- Pair 1 mean winner: T=1.5; pair 2 mean winner: T=2.0.
- Exact overlap of near-optimal CoVar grid points: 1.5.
- Different winners with overlapping CoVar coordinates: True.

With a fixed teacher, CoVar coordinates at each temperature are identical by construction. Their overlap is descriptive and does not validate a temperature-transfer rule.

Capacity and execution protocol are confounded in comparisons against P7.

## Evidence boundary

The conclusion is limited to this dense-prediction teacher-student pair and locked training protocol. With three seeds, sample means and sample SDs are emphasized rather than p-values.

## H20 cohort

All runs use the runtime in [protocol.json](protocol.json). Earlier H100 runs are not pooled. The nominally best temperature and delta sets are descriptive on this finite grid with three seeds; they are not a proof of a population optimum or of non-identifiability.

Per-seed delta-near-optimal sets: `{"1234": ["0.25", "2.0"], "2025": ["0.5"], "3407": ["2.0"]}`.

Classification KD interactions were studied by [Frank and Davis (2026)](https://arxiv.org/abs/2603.02430). Here the measurements concern dense prediction, temperature response and teacher CoVar coordinates.
