# P2 same checkpoint continuation execution

Date: 2026-10-16. The user authorizes P2 after the completed P1 analysis supports
its mechanism prerequisite, using all four Maker RTX 4090 cards where the
existing keepalive programs leave capacity. P3, P4 and 80k training are not
launched by this execution. P14's failed gate remains part of the record.

Engineering revision: before any formal P2 probe score, the initial queue was
preserved and superseded. Its four 16-step benchmarks took about16--17 seconds
per update. Replace hundreds of per-parameter CUDA finite reductions with one
read-only concatenated finite check, retaining exactly the same Boolean
predicate and unmodified gradients. The new registration binds all four
original16-step states and requires max absolute state error at most1e-4 before
formal probes. All mathematical settings, streams and workload remain fixed.
Revision r2 measured the concatenated check without materially improving step
time. Before formal probes, revision r3 additionally enables pinned CPU saved
activation storage and nonblocking host-to-device restoration, without dtype
conversion or any change to CUDA arithmetic. It must pass the same four
16-step comparisons against the original unpinned implementation, using the
same predeclared1e-4 maximum absolute state tolerance. CPU checks also compare
the offload options for state, gradients and RNG. Each revision has its own
registration and output; engineering observations are not scientific probes.

P1 completed 912 scored endpoints, 576 optimizer updates and 72 buffer-only
forwards. At fixed dropout and original buffers, changing BN forward statistics
reversed c4-c1 pixel-CE ordering in 15/18 VOC and 14/18 Cityscapes paired flows.
The seed-mean BN intervention had the same direction in all three training
seeds of each dataset. Buffer transplantation alone caused no resolved ordering
flips. The full counts include the four predesignated discovery cases; excluding
them, fixed-dropout-off confirmation comparisons reversed13/16 on each dataset,
with the same seed-mean intervention direction in all three seeds. Reducing the
original global learning rate to 0.1 reversed all six VOC
eval comparisons in the seed1234 control. These observations support testing
state and horizon matched diagnostic validity. They do not establish that a
local signal predicts subsequent validation performance.

## Inputs and streams

Use the 12 original 20k states: two datasets, training seeds 1234/2025/3407,
and CE or Shared c1 trajectories. Verify all checkpoint bytes against P14/P15
provenance and validate their saved metadata. Restore their true SGD momentum,
weight decay and criterion states. Continue the original 80k poly schedule;
the first new update uses global iteration20001 and the last fork update22000.
The frozen teacher, original full student batch/crop, FP32 and TF32-off
arithmetic match P15. Teacher eval inference uses chunks of two before the
student, and must preserve CPU/CUDA RNG. Saved student activations use CPU
storage without dtype conversion. There is no mixed precision, student batch
reduction, gradient accumulation, gradient clipping or momentum reset.

Pool B is exactly P15's frozen 256-image train monitoring pool. All update
draws exclude both A and B. Three probe stream seeds are 2026101601/2/3; the
actual 2k forks use separate seed2026101604. Streams are derived by dataset,
training seed, stage20000 and stream seed, with per-step without-replacement
batch draws and replacement between steps. Source condition and candidate
coefficient do not enter sampling or augmentation/dropout RNG derivation.
Thus CE/KD sources and c0/c1/c4 share the intended input stream. Each stream's
indices and per-step augmentation/dropout seeds are hashed before GPU work.
Streams and source conditions remain nested repeated observations, not new
training seeds.

## Workload and decisions

There are 108 train-mode rollouts to128 steps and 36 exact first-flow replays.
Save model endpoints at1/16/128 for432 monitor scores. Add108 matched full-batch
eval-mode one-step branches and12 B-pool source baselines. Total probe cost is
18,540 optimizer updates,540 endpoint scores and12 baselines. Train mode sets
BN batch statistics and dropout on independently of the model's parent flag;
eval mode sets BN running statistics and dropout off. Rollout scoring occurs
in separate processes on saved float32 states and never advances training RNG.

After all probe scores, freeze four choices for each source state before any
official validation score is computed. The first three rules use the three-flow
mean of eval pixel CE at1 step, train pixel CE at1 step, or train soft mIoU at1
step. Metric ties retain the source coefficient (CE c0 or Shared c1), otherwise
prefer the smaller coefficient. Eval ties use the corresponding P0 band;
train ties use max(P0 band,5 times largest candidate first-flow replay span).

The fourth rule uses train hard mIoU at128 steps. A candidate may replace the
source coefficient only if it exceeds that coefficient's score on all three
flows by more than max(P0 hard-mIoU band,5 times the largest candidate128-step
replay span). Choose the eligible candidate with highest flow mean; ties
prefer smaller c. With no eligible replacement, retain the source coefficient.
The numerical bands are reported for all horizons and four metrics.

Freeze the complete decisions, every probe-score hash and registration hash,
then copy the choices sequentially to persistent storage with full SHA readback.
Only afterward launch36 actual2000-step c0/c1/c4 forks,72000 optimizer updates.
At22k compute full official validation hard mIoU on VOC1449 and Cityscapes500
images using the frozen runtime's native-resolution loader, bilinear logit
resizing with align_corners=True, valid-mask confusion pooling and class mean.
No best checkpoint selection or evaluation-driven coefficient changes occur.

Report each rule's selection regret in percentage points against the best
coefficient of that same source state, used only as a posthoc evaluation upper
bound. Average CE/KD source outcomes within each training seed. The existing
P4 resource rule requires at least0.10pp lower mean regret than train pixel CE
on both datasets and improvement in at least two of three training seeds per
dataset. This is a practical allocation criterion, not a significance test.
The workflow reports it and stops; it cannot launch P3/P4 or80k trajectories.

## Engineering and recovery

All model work, CPU model checks and GPU benchmarks run on Maker. Before
formal probes, execute16 full-batch updates for CE/KD seed1234 on both datasets
across physical GPUs0/1/2/3. Restore and verify complete source state. Then
score four complete B baselines and verify CE primary scores against P15.
Publish actual steady-step and native-score timing estimates after this preflight.

The resource policy inherits the explicitly preserved live keepalive identity
rules, requiring8192MiB free on two samples two seconds apart and a6144MiB
per-worker allocator cap. Reserve16GiB host memory per active update and a32GiB
available floor. Each card runs an independent task. Only this controller's
own children may be terminated on its failure; no other processes are signalled.

Forks save complete local restart states every250 updates, bound to their job
and registration hashes. Persist1000/2000-step full states using sequential
content-addressed writes and SHA readback. All source checkpoints remain intact.
Published numerical failures are retained and stop the queue. Documented
resource failures retry the same task; a fork resumes only an authenticated
committed restart. Three memory failures of a task stop for engineering audit.
Scientific JSON, registration, stream schedules, source bundle and checkpoint
archive receipts are periodically stored as SHA-verified persistent snapshots.
