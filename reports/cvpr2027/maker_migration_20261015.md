# Maker execution amendment, 2026-10-15

The user requested rerunning the required supplementary experiments on the
Maker server with four RTX 4090 GPUs. Inspection found unrelated jobs on GPUs
0 and 1. The user then explicitly allocated **only GPUs 2 and 3**, preserving
the existing jobs. This amendment changes execution resources, not the primary
hypothesis, contrast, monitoring pool, update streams or scientific stopping gate.

The old P14 revision 3 controller and its remaining diagnostic work were stopped
after checking exact command lines and output paths. Its checkpoints and partial
observations remain preserved. No other old-server processes were stopped.

Stage A reuses the 54 immutable existing CE/Shared-c1/Shared-c4 states at
20k/40k/60k. Each copied state must match its original P14 SHA256 and original
iteration, optimizer and training-argument metadata. Original checkpoint args
are retained as provenance; data, teacher, runtime and checkpoint execution
paths are separately remapped. The fixed monitoring images must match their
previous hashes. Every training split entry must resolve in the new fast cache.

Hashes of all 54 states are fixed from the origin registration before replay.
Copies may still be pending when the new registration is written. Availability
and verified inputs at registration are explicit. A transfer becomes visible
only after stream/readback hash verification and atomic rename. The runner
rechecks each state hash and original metadata before using it. Deferred copies
do not change the expected bytes or allow a different checkpoint to enter.

New observations are a separate `maker_rtx4090_20261015` hardware cohort. They
are not pooled with old 3090 measurements. The new registration records software,
driver and hardware versions. Match PyTorch 2.0.1+cu118, torchvision 0.15.2+cu118,
NumPy 1.24.4, OpenCV 4.8.0.74 and Pillow 9.5.0 where available; record remaining
environment differences. FP32, disabled TF32 and unchanged single-card batch
sizes remain mandatory. Hardware-dependent numerical identity is not assumed.

Training/validation data for this study are cached on the local NVMe because
the root overlay has only about 21 GiB free. The complete user datasets remain
in `/root/maker/makerdata/liujinshi/covar-kd`. The fast working directory is
`/tmp/liujinshi-covar-kd-20261015`, linked from
`/root/maker/liujinshi/covar-kd`. Persistent small reports are copied to the user
object-storage directory. NVMe is working storage, not the only archival copy.

Run one independent seed job per allocated card, with an immutable GPU allowlist
and idle-card checks. Each worker has distinct progress and summary files.
Formal jobs partition both datasets by seed and checkpoint; longer Cityscapes
jobs enter the queue first among available inputs, and each card immediately
takes the next ready case after finishing. Pending copies do not block ready
cases; unverified partial files are never used. The queue fails explicitly if
required copies have not arrived within 24 hours of the phase starting.
This preserves each example/augmentation stream and the original BN batch.
Do not use GPUs 0 or 1, increase batch size or combine seeds with DDP.

Execution remains: CPU verification, GPU smoke, CE-start one-step diagnostics,
then the registered primary gate. The same positive Shared-c4 minus Shared-c1
pixel-CE seed mean is required at at least two of three stages in **each**
dataset. If it fails, selector full training stops and the result is reported.
If it passes, KD-start and 128-step diagnostics follow before deciding whether
the registered selector experiments are scientifically supported. Source sync,
data transfer and smoke results are not full-training results.
