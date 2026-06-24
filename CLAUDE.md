# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoVar+CIRKD — **Co**nfidence-**Var**iance adaptive **C**ross-**I**mage **R**elational **K**nowledge **D**istillation for semantic segmentation. This repository extends the CIRKD framework with a per-pixel teacher reliability metric (CoVar) that generates dynamic temperature maps for adaptive knowledge distillation.

The core insight: at each pixel, teacher reliability is `r = -log(c) + a * v / (1-c)` where `c` = max softmax confidence and `v` = residual variance over non-max classes. High `r` means the teacher is uncertain — the pixel gets a higher distillation temperature, down-weighting the KD loss there.

## Environment

- Python 3.9, PyTorch 2.0+cu118, CUDA 12.4, Ubuntu 22.04
- Key deps: `mmcv`, `mmengine`, `timm`, `opencv-python`, `scikit-learn`
- Install: `pip install -r requirements.txt`

## Architecture Map

### Models (`models/`)
- `segbase.py` — Abstract base segmentation model with backbone loading
- `deeplabv3.py` / `deeplabv3_mobile.py` / `deeplabv3_mobilenetv3.py` — DeepLabV3 variants
- `pspnet.py` / `psp_mobile.py` — PSPNet variants
- `upernet.py` — UPerNet (lite versions for KD)
- `segformer.py` — SegFormer (MiT-B0 through B5) — standalone, not SegBaseModel subclass
- `model_zoo.py` — `get_segmentation_model()` factory; dispatches by model name string
- `base_models/` — Backbone implementations: ResNet-v1b, MobileNetV2/V3, MobileViT, HRNet

### Losses (`losses/`)
- `kd.py` — `CriterionKD`: standard KL-divergence distillation loss
- `cirkd_memory.py` — `StudentSegContrast`: memory-bank pixel/region contrastive KD with momentum queues
- `cirkd_mini_batch.py` — `CriterionMiniBatchCrossImagePair`: cross-image pairwise similarity KD within a mini-batch
- `cirkd_channel.py` — `StudentSegChannelContrast`: channel-wise contrastive KD (mini-batch + memory-bank)
- Other losses: `fitnet.py`, `at.py`, `cwd.py`, `skd.py`, `ifvd.py`, `dsd.py`, `adv.py`, `gcn.py`, `task.py`

### CoVar Core (`PCOS.py`)
- `get_max_confidence_and_residual_variance_components()` — Computes per-pixel `c`, `v`, and scaled `r = a * v / (1-c)` from teacher softmax probabilities
- `batch_class_stats()` — 2-class clustering of `(c, v)` features per image to find high-confidence regions
- Helper functions for SVD-based class assignment and center computation

### Main Training Script (`train_cirkdv2.py`)
The `Trainer` class orchestrates the full training loop:
1. **Model setup**: Freezes teacher, trains student + projection heads + memory banks
2. **CoVar temperature generation** (`get_covar_metadata()`):
   - Computes `r` from teacher logits via `PCOS.py`
   - Three temperature modes controlled by `--covar-temp-mode`:
     - `sqrt`: simple heuristic `T = T0 + alpha * (sqrt(r) - mean(sqrt(r)))`
     - `grad` / `newton`: iterative gradient descent or Newton's method on `r(T)` to push `r` toward 0
     - `calib_conf`: calibrates confidence to match a target derived from `r0` using bisection-Newton
3. **Loss assembly**: task_loss + covar_kd_loss + minibatch_pixel_contrast + memory_pixel_contrast + memory_region_contrast + fitnet + minibatch_channel_contrast + memory_channel_contrast + channel_mse
4. **Validation**: Runs every `--val-per-iters` steps, tracks best/top-K mIoU checkpoints
5. **Visualization** (opt-in with `--enable-visualizations`): z-r diagnostic plots, var(g)-r scatter, g²-vs-r/T² scatter

### Other Training Entry Points
- `train_baseline.py` — Train a student without distillation
- `train_kd.py` — Standard logit-level KD (no CIRKD losses)
- `train_cirkdv2_segformer.py` — CIRKD v2 for SegFormer models (different forward signature)
- `semi_supervised_train.py` — Semi-supervised extension with CutMix consistency
- Corresponding `*_segformer.py` variants for baseline and KD scripts

### Evaluation (`eval.py`)
- Multi-scale and flip evaluation on validation sets
- Saves predictions as color-mapped PNGs with `--save-pred`
- `test.py` / `test_voc_fixedset.py` for Cityscapes test server and VOC fixed-set evaluation

### Datasets (`dataset/`)
- Cityscapes, Pascal VOC Aug, ADE20K, CamVid, COCO-Stuff-164K
- Each dataset module provides `TrainSet` and `ValSet` classes; Cityscapes uses a combined `CSTrainValSet`

### Utilities (`utils/`)
- `distributed.py` — Samplers, batch samplers, synchronize helpers for multi-GPU
- `logger.py` — Logging setup
- `score.py` — `SegmentationMetric` (pixel accuracy, mIoU)
- `flops.py` — FLOPs/params calculation
- `visualize.py` — Color palette generation
- `sagan.py` — Self-attention GAN utilities

### Scripts (`scripts/`)
Shell scripts organized by purpose:
- `train_baseline/<dataset>/<model>.sh` — Baseline training runs
- `train_cirkdv2/<dataset>/<model>.sh` — CIRKD v2 training runs
- `evaluation/<dataset>/<model>.sh` — Evaluation runs
- `train_kd/` — KD baseline training
- `experiments/covar_calib_conf/` — CoVar calibration experiments (launch, monitor, stop, analyze)
- `visualize/` — Plotting scripts for diagnostics
- `diagnostics/` — Teacher correctness binning

## Common Commands

### Training (CIRKD v2 + CoVar)
```bash
# Multi-GPU (example: DeepLabV3-ResNet18 student, ResNet101 teacher, Cityscapes)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12397 \
train_cirkdv2.py \
  --teacher-model deeplabv3 --student-model deeplabv3 \
  --teacher-backbone resnet101 --student-backbone resnet18 \
  --dataset citys --data /path/to/cityscapes/ \
  --batch-size 8 --lr 0.01 --max-iterations 80000 \
  --lambda-kd 1. --lambda-fitnet 1. \
  --lambda-minibatch-pixel 1. --lambda-memory-pixel 0.1 --lambda-memory-region 0.1 \
  --lambda-minibatch-channel 1. --lambda-memory-channel 0.1 --lambda-channel-kd 100. \
  --teacher-pretrained /path/to/teacher.pth \
  --student-pretrained-base /path/to/backbone.pth \
  --save-dir /path/to/checkpoints/ --save-dir-name <experiment_name> \
  --use-covar --covar-temp-mode newton
```

### Evaluation
```bash
python eval.py \
  --model deeplabv3 --backbone resnet18 --dataset citys \
  --pretrained /path/to/checkpoint.pth \
  --data /path/to/cityscapes/ --scales 1.0
```

### Baseline Training (no KD)
```bash
python -m torch.distributed.launch --nproc_per_node=8 train_baseline.py \
  --model deeplabv3 --backbone resnet18 --dataset citys --data /path/to/cityscapes/ \
  --batch-size 16 --lr 0.02 --max-iterations 80000
```

## Key Design Notes

- **Teacher is always frozen** — `t_model.eval()` and all params have `requires_grad=False`
- **Model outputs differ by type**: CNN models (DeepLabV3, PSPNet) return `[logits, aux_logits?, final_features]`; SegFormer returns `[logits, fused_decoder_features]`
- **Distributed training** uses `torch.distributed.launch`; memory-bank losses use `all_gather` to share teacher features across GPUs
- **Temperature adaptation** only affects the KD loss term — the task loss (cross-entropy) is computed normally
- **CoVar constant `a`** defaults to `(K-1)²/2` where K = number of classes; can be overridden with `--covar-a`
- The `crop_size` argument takes `[height, width]` — e.g., `[512, 1024]` for Cityscapes, `[520, 520]` for ADE20K
- Scripts in `scripts/` reference absolute paths (`/data/winycg/...`) — update these for your environment
