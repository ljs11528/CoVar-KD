# CLAUDE.md — CIRKD+CoVar Knowledge Distillation for Semantic Segmentation

## 1. Project Overview

**CoVar+CIRKD** extends the Cross-Image Relational Knowledge Distillation framework with **per-pixel adaptive temperature maps** based on teacher reliability. A student model (DeepLabV3-MobileNetV3-Small) is distilled from a teacher (DeepLabV3-ResNet101) on Pascal VOC.

**Core formula**: Teacher reliability per pixel:
```
r = -log(c) + a·v/(1-c)
```
where `c` = max softmax confidence, `v` = residual variance over non-max classes, `a = (K-1)²/2`.

High `r` → teacher uncertain → raise temperature T > 1 (smooth/weaken KD signal).
Low `r` → teacher reliable → lower temperature T < 1 (sharpen/strengthen KD signal).

The temperature map `T(pixel)` is used in: `KL(student_logits/T, teacher_logits/T) * T^gamma`.

## 2. Environment

| Component | Version |
|-----------|---------|
| Python | 3.10.12 |
| PyTorch | 2.0.0+cu118 |
| CUDA | 12.4 |
| OS | Ubuntu 22.04 |
| GPU | NVIDIA A100 80GB PCIe |

```bash
# Clone and setup
git clone git@github.com:ljs11528/CoVar-KD.git
cd CoVar-KD/CIRKD-main
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: `mmcv==2.0.1`, `mmengine==0.8.1`, `timm==0.6.13`, `opencv-python==4.7.0.72`, `scikit-learn==1.6.1`.

## 3. Required Data Files

| File | Path | Size | Purpose |
|------|------|------|---------|
| VOC Aug dataset | `dataset/VOCAug/` | ~2GB | Training data |
| Teacher checkpoint | `data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth` | ~240MB | VOC-finetuned teacher |
| Student backbone | `data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth` | ~11MB | ImageNet backbone |
| Experiment checkpoints | `data/winycg/checkpoints/cirkd_checkpoints/voc/` | varies | All experiment outputs |

**Migrating to new server**: Copy `dataset/VOCAug/` and `data/winycg/` directories. Update `ROOT_DIR` in shell scripts to point to the new base path.

## 4. Architecture Map

### Key Python Files

| File | Lines | Role |
|------|-------|------|
| `train_cirkdv2.py` | 2204 | **Main training script** — all CoVar logic, training loop, argument parsing |
| `PCOS.py` | 114 | Teacher reliability metric computation (`r`, `c`, `v`) |
| `models/deeplabv3.py` | 185 | DeepLabV3 model (teacher) with ASPP |
| `models/deeplabv3_mobilenetv3.py` | — | DeepLabV3+MobileNetV3-Small (student) |
| `models/segbase.py` | 56 | Abstract base model with backbone loading |
| `models/model_zoo.py` | 29 | Model factory: `get_segmentation_model()` |
| `models/base_models/resnetv1b.py` | ~310 | ResNet backbones (18/34/50/101/152) |
| `losses/kd.py` | — | `CriterionKD`: KL-divergence loss |
| `losses/cirkd_memory.py` | — | Pixel/region memory-bank contrastive KD |
| `losses/cirkd_mini_batch.py` | — | Cross-image mini-batch contrastive KD |
| `losses/cirkd_channel.py` | — | Channel-wise contrastive KD |

### Data Flow (training step)

```
images → teacher (frozen) → t_logits
                           ↓
                  PCOS.py: c, v, r
                           ↓
                  get_covar_metadata() → temperature_map
                           ↓
        images → student → s_logits
                           ↓
        KD loss: KL(s_logits/T, t_logits/T) * T^gamma
        Task loss: CE(s_logits, labels)
        CIRKD losses: contrastive pixel/region/channel
                           ↓
        total_loss = task + kd + contrastive losses + fitnet
```

### Temperature Modes (`--covar-temp-mode`)

| Mode | Description | Key params |
|------|-------------|------------|
| `sqrt` | Simple heuristic: T = T0 + α·(√r - mean(√r)) | `covar-temp-alpha` |
| `grad` | Gradient descent: minimize r(T) | `covar-grad-eta`, `covar-grad-max-iter` |
| `newton` | Newton's method: solve dr/dT=0 | above + `covar-newton-hessian-eps`, `covar-newton-max-step` |
| `calib_conf` | Calibrate max confidence to target | `covar-calib-centered`, anchor params |

## 5. CoVar Centered Calibration (`calib_conf` mode)

### Mechanism

Instead of solving `dr/dT=0` (newton), centered calibration sets a target confidence:
```
c_target = c0 * exp(-α * (r0 - r_anchor))
```

- r_anchor = quantile(r0_valid, p) — the "center" of reliability
- Pixels with r0 < r_anchor → c_target > c0 → sharpen (T < ref_temp)
- Pixels with r0 > r_anchor → c_target < c0 → smooth (T > ref_temp)
- Then solve: max_softmax(z/T) = c_target via Newton+bracket method

### Anchor Modes (`--covar-calib-anchor-mode`)

| Mode | Formula | Behavior |
|------|---------|----------|
| `median` (default) | r_anchor = median(r0) | Always 50% sharpen / 50% smooth |
| `adaptive_mc` | p = mean(teacher_mc)^power, r_anchor = quantile(r0, p) | Auto-adapts to teacher quality |
| `fixed` | r_anchor = quantile(r0, anchor_quantile) | Explicit control |

### Full CoVar Parameter Reference

```
--use-covar / --no-covar          Master switch (default: enabled)
--covar-temp-mode {sqrt,grad,newton,calib_conf}
--covar-temp-min FLOAT            Minimum T (default: 0.5)
--covar-temp-max FLOAT            Maximum T (default: 8.0)
--covar-ref-temp FLOAT            Reference T for calibration (default: 1.0)
--covar-temp-base FLOAT           Initial T guess (default: 1.0)
--covar-kd-temp-power FLOAT       T^gamma scaling (default: 2.0)
--covar-grad-eta FLOAT            Step size / damping (default: 0.6)
--covar-grad-max-iter INT         Max solver iterations (default: 8)
--covar-newton-hessian-eps FLOAT  Min |d2r/dT2| for Newton (default: 1e-5)
--covar-newton-max-step FLOAT     Max Newton step (default: 0.25)
--covar-grad-converge-thresh FLOAT Convergence threshold (default: 0.01)
--covar-grad-detail-interval INT  Diagnostic print interval (default: 100)
--covar-a FLOAT                   CoVar constant (default: auto (K-1)^2/2)
--covar-calib-centered            Enable bidirectional calibration
--covar-calib-anchor-mode {median,adaptive_mc,fixed}
--covar-calib-anchor-power FLOAT  Exponent for adaptive_mc (default: 1.0)
--covar-calib-anchor-quantile FLOAT Explicit quantile [0-1] for fixed mode
--covar-calib-anchor-alpha FLOAT  Alpha for exp transition (default: 1.0)
--teacher-output-temp FLOAT       Scale teacher logits (default: 1.0)
--enable-visualizations           Enable diagnostic plotting
```

## 6. Experiment Structure

### Shell Script Pattern

All experiments use a common pattern:

```bash
# common_voc_cirkdv2.sh defines:
#   - ROOT_DIR, PYTHON, GPU_ID, DATA_DIR, SAVE_ROOT
#   - COMMON_ARGS[] with training hyperparams
#   - run_variant() { ... } helper function

# Each experiment script:
source common_voc_cirkdv2.sh
run_variant "variant_name" \
  --covar-temp-mode calib_conf \
  --covar-calib-centered \
  ...
```

### Common Config Scripts

| Script | Teacher | Key difference |
|--------|---------|----------------|
| `common_voc_cirkdv2.sh` | VOC-finetuned (mc≈0.97) | Standard strong teacher |
| `common_voc_lowconf_teacher.sh` | T_out=3.0 (mc≈0.73) | Teacher logit temperature scaling |
| `common_voc_weak_teacher.sh` | Random init | No pretrained teacher (broken—don't use) |

### Running Experiments

```bash
# Single experiment
bash scripts/experiments/covar_calib_conf/06_centered_gamma1_tmin0.5.sh

# Background launch
nohup bash scripts/experiments/covar_calib_conf/phaseB_b3_tmin0.3_alpha1.sh \
  > runs/experiment.log 2>&1 &

# Sequential auto-chain
nohup bash -c '
  bash scripts/.../exp1.sh
  bash scripts/.../exp2.sh
' > runs/chain.log 2>&1 &
```

**Typical runtime**: ~30 hours for 80k iterations on single A100.

### Monitoring

```bash
# GPU status
nvidia-smi

# Training progress (iteration, ETA)
tail -f <save_dir>/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt

# Validation milestones
grep "Sample: 1449, Validation" <save_dir>/..._log.txt | tail -5

# T distribution and anchor detail (every 100 iters in calib_conf mode)
grep "CalibConfCenteredTemp Detail" <save_dir>/..._log.txt | tail -3
```

## 7. Complete Experimental Results

All experiments: DeepLabV3-ResNet101 → DeepLabV3-MobileNetV3-Small, VOC dataset, 80k iterations, batch_size=16, crop_size=512².

### Phase 1: Baseline Comparisons (Original Teacher, mc≈0.97)

| # | Experiment | Method | Key Config | Best mIoU | Final |
|---|-----------|--------|------------|-----------|-------|
| P0 | `06_centered_gamma1_tmin0.5` | centered | γ=1.0, t_min=0.5, ref=1.0 | 0.615 | 0.612 |
| P1 | `07_centered_gamma0_tmin0.5` | centered | γ=0.0, t_min=0.5, ref=1.0 | 0.632 | 0.632 |
| P2 | `08_old_newton_rerun` | newton | t_min=0.5 | **0.639** | 0.639 |

**Findings**: γ=1.0 temperature compensation is harmful; γ=0.0 recovers +0.017. Newton wins by +0.007.

### Phase 2: Low-Confidence Teacher (T_out=3.0, mc≈0.73)

| # | Experiment | Method | Key Config | Best mIoU | Final |
|---|-----------|--------|------------|-----------|-------|
| P1-LC | `lowconf_teacher_p1_centered_gamma0` | centered | γ=0.0, t_min=0.5, T_out=3.0 | 0.636 | 0.632 |
| P2-LC | `lowconf_teacher_p2_newton` | newton | t_min=0.5, T_out=3.0 | **0.646** | 0.638 |

**Findings**: Both methods improve with softer teacher. Newton advantage widens (+0.010 vs +0.007). T_out=3.0 + newton = best overall (0.646).

### Phase B: Adaptive Anchor Centered Calibration

| # | Experiment | anchor | t_min | α | ref | Best mIoU |
|---|-----------|--------|-------|----|------|-----------|
| B1 | `phaseB_b1_adaptive_anchor_orig_teacher` | adaptive_mc | 0.5 | 1.0 | 1.0 | 0.626 |
| B2 | `phaseB_b2_adaptive_anchor_lowconf_teacher` | adaptive_mc | 0.5 | 1.0 | 1.0 | 0.630 |
| B3 | `phaseB_b3_tmin0.3_alpha1` | adaptive_mc | 0.3 | 1.0 | 1.0 | 0.626 |
| B4 | `phaseB_b4_tmin0.3_alpha2` | adaptive_mc | 0.3 | 2.0 | 1.0 | 0.619 |
| B5 | `phaseB_b5_reftemp0.6_tmin0.3` | adaptive_mc | **0.3** | 1.0 | **0.6** | **0.636** |

**Findings**: 
- Anchor mechanism tracks teacher mc correctly (anchor_q=0.98 for mc=0.97, 0.73 for mc=0.73).
- Lowering t_min alone (B3) or raising α (B4) doesn't improve — identity zone is the bottleneck.
- **Lowering ref_temp to 0.6 (B5) is the key breakthrough** — T_mean drops from 0.74 to 0.58 matching newton (0.60). B5 (0.636) is statistically tied with newton (0.639).

### Final Leaderboard

| Rank | Experiment | mIoU | Method | Scenario |
|:----:|-----------|------|--------|----------|
| 🥇 | P2-LC | **0.646** | newton | T_out=3.0 |
| 🥈 | P2 | **0.639** | newton | original |
| 🥉 | B5 | **0.636** | centered adaptive ref=0.6 | original |
| 🥉 | P1-LC | **0.636** | centered γ=0 | T_out=3.0 |
| 5 | P1 | 0.632 | centered γ=0 | original |
| 6 | B2 | 0.630 | adaptive anchor | T_out=3.0 |
| 7 | B1/B3 | 0.626 | adaptive / t_min↓ | original |
| 8 | B4 | 0.619 | adaptive α=2 | original |
| 9 | P0 | 0.615 | centered γ=1 | original |

### Key Conclusions

1. **Newton wins but margin is small**: 0.639 vs B5's 0.636 (Δ=-0.003, statistically tied in last 5 validations)
2. **γ=1.0 is harmful** (P0: 0.615) — don't use KL·T scaling; set `--covar-kd-temp-power 0.0`
3. **ref_temp is the strongest lever** for centered calibration: 1.0→0.6 improves +0.010 (B1→B5)
4. **Teacher output temperature** (T_out=3.0) acts as label smoothing — improves both methods
5. **Adaptive anchor correctly tracks teacher mc** but doesn't translate to mIoU gains by itself
6. **Centered calibration's exp formula produces too many "identity" pixels** (r0≈r_anchor → T=ref_temp); newton's direct optimization avoids this

## 8. Recommended Configurations

### Best overall (if teacher quality unknown)
```bash
# Newton + teacher output temp 3.0 (P2-LC: 0.646)
--covar-temp-mode newton --teacher-output-temp 3.0
```

### Best centered calibration
```bash
# Adaptive anchor + ref_temp=0.6 + t_min=0.3 (B5: 0.636)
--covar-temp-mode calib_conf --covar-calib-centered \
--covar-calib-anchor-mode adaptive_mc --covar-calib-anchor-power 1.0 \
--covar-calib-anchor-alpha 1.0 --covar-ref-temp 0.6 --covar-temp-base 0.6 \
--covar-temp-min 0.3 --covar-temp-max 4.0 --covar-kd-temp-power 0.0
```

### Simple baseline (no adaptive)
```bash
# Centered γ=0.0 (P1: 0.632)
--covar-temp-mode calib_conf --covar-calib-centered \
--covar-kd-temp-power 0.0
```

## 9. Migration Checklist

### Files to copy to new server:
```
□ dataset/VOCAug/                          # VOC Augmented dataset (~2GB)
□ data/winycg/cirkd/teachers/              # Teacher checkpoint
□ data/winycg/imagenet_pretrained/          # Student backbone
□ Entire git repo: github.com/ljs11528/CoVar-KD
```

### Setup steps on new server:
```bash
1. git clone git@github.com:ljs11528/CoVar-KD.git
2. cd CoVar-KD/CIRKD-main
3. python3 -m venv .venv && source .venv/bin/activate
4. pip install -r requirements.txt
5. # Copy dataset and pretrained weights to correct paths
6. # Update ROOT_DIR in shell scripts or set environment variable
7. # Verify: python -c "import torch; print(torch.cuda.is_available())"
```

### Quick smoke test:
```bash
bash scripts/experiments/covar_calib_conf/smoke_calib_gamma0_tmax4.sh
# Should complete in ~5 minutes with no errors
```

## 10. Continuing with Claude on New Server

When you resume with Claude on the new server, point Claude to this CLAUDE.md. Key context to provide:

- "This is a semantic segmentation KD project. The main training script is `train_cirkdv2.py`."
- "We've run 10 CoVar calibration experiments. Full results are in the CLAUDE.md leaderboard."
- "The best centered calibration config is B5 (ref_temp=0.6). The best overall is P2-LC (newton + T_out=3.0)."
- "Key insight: centered calibration can't match newton because its smooth exp transition creates an identity zone where pixels get no treatment."
- "Future direction: fused approach — newton for low-confidence pixels, centered for high-confidence ones."

### Quick reference for common tasks:
```bash
# Launch a new experiment
cd CoVar-KD/CIRKD-main
source .venv/bin/activate
nohup bash scripts/experiments/covar_calib_conf/<script>.sh > runs/<name>.log 2>&1 &

# Monitor progress
tail -f <save_dir>/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt | grep "Iters:"

# Check GPU
nvidia-smi

# Extract validation curve
grep "Sample: 1449, Validation" <save_dir>/..._log.txt | awk -F'mIoU: ' '{print $2}'
```
