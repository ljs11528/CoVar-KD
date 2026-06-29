# CoVar NPU Phase C experiments

These scripts run VOC CIRKD+CoVar experiments on a two-card Ascend 910 server via
`torch.distributed.run` + HCCL.

Default queue:

1. `phaseC_lc_newton_gamma0`: low-confidence teacher, Newton temperature, `gamma=0`
2. `phaseC_lc_newton_gamma2_repro`: reproduce current best low-confidence Newton, `gamma=2`
3. `phaseC_lc_newton_gamma1`: intermediate `gamma=1`
4. `phaseC_lc_no_covar_tout3`: softened-teacher KD control without CoVar

Launch:

```bash
bash scripts/experiments/covar_npu/launch_phaseC_triage.sh
```

Monitor:

```bash
bash scripts/experiments/covar_npu/monitor_phaseC_triage.sh
```

Useful overrides:

```bash
MAX_ITERATIONS=80000 MASTER_PORT=29630 bash scripts/experiments/covar_npu/launch_phaseC_triage.sh
WORKERS=4 BATCH_SIZE=16 bash scripts/experiments/covar_npu/phaseC_lc_newton_gamma0.sh
```

Checkpoints are written under
`data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseC/`.
Each run auto-resumes from `training_state_latest.pth`.
