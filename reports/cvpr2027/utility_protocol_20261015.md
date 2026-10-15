# Utility diagnostics and coefficient selection: execution protocol

Approved research direction: mechanism study plus a simple intervention, targeting
CVPR 2027. Registration date: 2026-10-15. No results for this new stage exist at
registration. Existing validation results motivated the hypotheses and are
exploratory evidence, not a new independent test.

## Stage A: distinguish alternative explanations

Reuse seeds 1234, 2025 and 3407, both VOC and Cityscapes, and existing 20k/40k/60k
CE and KD checkpoints. Preserve the corresponding dataset, teacher, optimizer,
80k learning-rate schedule, precision, valid mask and class mapping.

The primary comparison is Shared c=4 minus Shared c=1; Teacher-only is a secondary
temperature-placement control. Include c=0 as the CE reference. No extra T².
Compare legacy eval/batch2, matched-batch eval, and matched-batch training modes.
Training-mode BN/dropout and saved SGD momentum are restored before each branch.
Teacher BN is always frozen. Diagnostic TF32 is disabled in all compared branches.

Record pixel-averaged CE, class-averaged CE, soft mIoU and hard mIoU from the same
post-update predictions, with before-update values and exact counts. Compare one
step with 128 steps. Use deterministic disjoint update and monitor data. Add a
Shared KD parameter-gradient norm matched to Teacher c=1 control for scale.
All candidate branches share update images, augmentation and randomness.

Restore model parameters, all buffers, optimizer state, learning rates, training
flags and Python/NumPy/CPU/CUDA RNG after virtual updates. Never overwrite source
checkpoints. Stages and batches are repeated measurements, not new training seeds.
Report all seed-level values and sample SD; do not infer significance from pixels.

Engineering smoke precedes full diagnostics: analytic KL gradients including
ignore masks, metric sufficient-statistic reductions, finite gradients, BN
updates, optimizer/momentum restoration, RNG restoration, deterministic replay,
and verification that input-file hashes are unchanged.

The primary replication gate requires the Shared c4-c1 next-batch CE difference
to remain positive on average in matched training mode at at least two of the
three checkpoints in each dataset. Individual seed directions are always shown.
This is a resource-allocation rule, not a significance test. If this gate fails,
stop selector training and report the protocol dependence; do not attribute the
historical reversal to metric choice. CE and KD starting trajectories are
reported separately. Any additional mode-specific result remains secondary.

## Stage B: freeze the intervention

For both conditions, train CE+c*masked_KL with Shared teacher/student temperature
2, coefficient candidates {0,1,4}, and no extra T². Use c=1 through iteration
20000. At 20000, 40000 and 60000, restore the current state for each candidate,
run 128 training-mode virtual SGD steps with the original schedule, score, then
discard every virtual state. Apply the selected c to the unadvanced main state
for the next 20k segment. Main data-loader and RNG streams must remain unchanged
by lookahead. Separate deterministic update streams are used for lookahead.

The metric condition maximizes pooled soft mIoU; the ablation minimizes pooled
pixel-averaged CE. This scoring metric is the sole intended difference. Numerical
ties use relative tolerance 1e-6*max(1,abs(best_score)); retain the previous
coefficient when tied, otherwise use the smallest tied coefficient.

Soft IoU for class k is I_k/U_k, where I_k=sum_i p_ik*1[y_i=k] and
U_k=sum_i p_ik+sum_i 1[y_i=k]-I_k. Aggregate sufficient statistics over the
entire monitoring pool before class averaging, excluding ignore pixels. Every
class is present in the registered pool. This is a selection score, not an added
training loss. Hard mIoU is measured independently from int64 confusion counts.

For each dataset, create a fixed 256-image train monitoring pool with sampling
seed 20261015: independently shuffle the image IDs containing each GT class,
take eight per class, union, and randomly fill to 256 from remaining train IDs.
All classes must have at least eight eligible images. Use canonical validation
preprocessing on these training images, without stochastic augmentation.
Exclude monitor images from virtual-update batches only; they remain in main
training. Monitoring is not an independent held-out test. Validation labels must
never select coefficients. Freeze this pool and its hashes before running.

## Stage C: full-training allocation and endpoints

At most 12 fresh 80k trajectories: 2 scoring conditions * 2 datasets * 3 seeds.
First execute seed1234 for both conditions/datasets (four trajectories). Advance
to the remaining eight only when soft-mIoU selection improves Cityscapes final
mIoU by at least 0.2 pp against CE selection and loses no more than 0.2 pp on VOC.
Do not eliminate a formal trajectory by its early validation ranking.

The final target is Cityscapes improvement on at least two seeds, VOC mean loss
no greater than 0.2 pp, and proximity to the existing strong fixed references:
VOC CE and Cityscapes c=4. These are practical criteria, not equivalence or
significance claims. Retain all attempted formal runs and failures.

Primary endpoint: 80k final validation mIoU. Retain 20k/40k/60k/80k full states,
all seed-level scores, per-class results, coefficient choices, wall time, GPU
memory and lookahead overhead. Keep datasets/protocols and hardware cohorts
explicit. Use at most two otherwise-idle GPUs; no sharing occupied GPUs or
terminating unrelated processes. Existing raw evidence remains immutable.

Recheck the diagnostic gate and actual server resources before launch. A source
sync or passing unit test is not an experimental completion claim.
