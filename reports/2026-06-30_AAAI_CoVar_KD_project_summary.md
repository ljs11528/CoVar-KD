# CoVar-KD AAAI 投稿前项目总结

- 生成时间：2026-06-30
- 项目路径：`/home/ma-user/work/ljs`
- 当前硬件：2 x Ascend 910 NPU，CANN 8.5.0，PyTorch 2.6.0 NPU 环境
- 当前状态：Phase N 已停止；O1.1 confidence-only 诊断与联合门禁已完成；O1.2 预注册已冻结，尚未实现或启动学生训练

> 2026-07-13 主线同步：Phase N 已停止；O1.1 confidence-only 风险联合门禁已通过，但旧温度映射的中位数约为 0.5、调和均值约为 0.6。当前冻结主线转为“高风险优先、预算约束、教师目标单侧校准”的 O1.2；总览见 [Phase O 主记录](2026-07-13_phaseO_rtc_method_reconstruction_plan.md)，唯一规范见 [O1.2 预注册](2026-07-13_phaseO_rtc_o12_budgeted_routing_plan.md)。尚未启动 O1.2 学生训练。

## 1. 一句话总结

本项目研究语义分割知识蒸馏中的教师不确定性问题，提出 CoVar-KD：基于教师预测的置信度与非主类别残差方差，为每个像素构造自适应温度图，在可靠像素上增强蒸馏信号，在不可靠像素上削弱或平滑蒸馏信号，从而提升轻量学生模型的分割性能。

## 2. 背景与问题

语义分割蒸馏通常使用强教师模型指导轻量学生模型，例如 DeepLabV3-ResNet101 蒸馏到 DeepLabV3-MobileNetV3-Small。传统 KD 通常使用全局固定温度，默认所有像素的教师输出都同等可信。但语义分割是密集预测任务，不同像素的可靠性差异很大：目标边界、遮挡区域、小物体、类别混淆区域的教师预测往往不稳定。

如果对所有像素使用相同蒸馏强度，可靠像素和不可靠像素会被混在一起处理。不可靠教师信号可能向学生注入噪声，可靠教师信号又可能没有被充分利用。因此，本项目的核心问题是：如何在像素级别估计教师可靠性，并把这种可靠性转化为蒸馏温度或蒸馏权重。

## 3. 动机

教师模型输出中已经包含了可靠性线索。最大 softmax 置信度可以反映教师对预测类别的确信程度，但单独使用最大置信度不够完整，因为非主类别的概率分布形状也很重要：如果非主类别分布很分散，说明该像素存在更强类别不确定性。

因此，本项目用两个量刻画教师可靠性：

- `c`：教师最大 softmax 置信度，越大表示越可靠。
- `v`：非主类别残差概率的方差，反映非主类别分布结构。

当前 CoVar 可靠性指标定义为：

```text
r = -log(c) + a * v / (1 - c)
a = (K - 1)^2 / 2
```

其中 `K` 是类别数。`r` 越大表示教师越不可靠；`r` 越小表示教师越可靠。

## 4. 核心 idea

CoVar-KD 的核心是把教师可靠性 `r` 映射成每个像素的温度 `T(pixel)`：

- 可靠像素：降低温度，锐化教师分布，增强 KD 监督。
- 不可靠像素：提高温度，平滑教师分布，降低错误硬监督的伤害。
- 蒸馏损失：`KL(student_logits / T, teacher_logits / T) * T^gamma`。

项目中已经实现并比较了几类温度求解方式：

| 方法 | 含义 | 当前结论 |
|---|---|---|
| `sqrt` | 简单启发式映射 | 作为早期探索，不是当前主线 |
| `grad` | 通过梯度下降调整温度 | 可用，但收敛和稳定性不如 Newton |
| `newton` | 用 Newton 方式求解可靠性目标 | 当前最强主线方法 |
| `calib_conf` | 把教师置信度校准到目标置信度 | 可解释性强，但性能略弱于 Newton |

## 5. 方法结构

### 5.1 训练框架

- 教师：DeepLabV3-ResNet101。
- 学生：DeepLabV3-MobileNetV3-Small。
- 数据集：Pascal VOC / VOCAug。
- 基础框架：CIRKD，包括 task CE、logit KD、fitnet、mini-batch pixel/channel contrast、memory pixel/region/channel contrast。
- 本项目改动：在原有 CIRKD 的 logit KD 路径中引入 CoVar 像素级温度图。

训练步骤：

```text
image -> teacher logits -> softmax -> c, v, r
                          -> CoVar temperature map T
image -> student logits
student/teacher logits + T -> pixel-wise KD
CIRKD contrastive losses + task CE + fitnet -> total loss
```

### 5.2 NPU 迁移和工程状态

当前 `train_cirkdv2.py` 已支持：

- `--device-type npu`。
- Ascend NPU DDP/HCCL 双卡训练。
- 自动断点续训：`training_state_latest.pth`。
- 保存完整训练状态：student、memory contrast、fitnet、channel contrast、GCN、optimizer、iteration、best mIoU、top-k checkpoint 信息、温度历史和参数。
- 保留评估兼容 checkpoint：`kd_*.pth`。

这对当前服务器很关键，因为服务器可能意外重启，80k 长实验必须可以自动恢复。

## 6. 已完成实验方案

### 6.1 旧 GPU/A100 阶段 80k 实验

主要用于验证方法方向、比较 centered calibration 与 Newton、分析 teacher softening 的作用。

| 阶段 | 目的 | 实验重点 |
|---|---|---|
| Phase 1 | 原始强教师下比较温度策略 | centered gamma、Newton |
| Phase 2 | 低置信 teacher 场景 | `teacher_output_temp=3.0` |
| Phase B | centered calibration 改进 | adaptive anchor、ref temp、t_min、alpha |

### 6.2 当前 NPU 阶段 20k triage

为了在 Ascend 910 上快速复核关键变量，已完成四个 20k 实验：

| Variant | Last iter | Best mIoU | Final mIoU | 结论 |
|---|---:|---:|---:|---|
| `phaseC_lc_newton_gamma0` | 20000 | 0.6087 | 0.6087 | 明显较差，不继续 |
| `phaseC_lc_newton_gamma2_repro` | 20000 | 0.6353 | 0.6353 | 20k 最优 |
| `phaseC_lc_newton_gamma1` | 20000 | 0.6282 | 0.6282 | 中间态，弱于 gamma2 |
| `phaseC_lc_no_covar_tout3` | 20000 | 0.6331 | 0.6331 | 与 gamma2 接近，是最关键 control |

20k triage 的最重要发现是：在 low-confidence teacher + Newton 场景下，`gamma=2` 反而最好；之前“gamma 补偿有害”的结论主要来自 centered calibration，并不能直接推广到 Newton。

## 7. 已有实验结果总结

### 7.1 旧 80k 结果

| Rank | Experiment | Method | Scenario | Best mIoU |
|---:|---|---|---|---:|
| 1 | `lowconf_teacher_p2_newton` | Newton | `teacher_output_temp=3.0` | 0.646 |
| 2 | `08_old_newton_rerun` | Newton | 原始强教师 | 0.639 |
| 3 | `phaseB_b5_reftemp0.6_tmin0.3` | centered adaptive | 原始强教师 | 0.636 |
| 4 | `lowconf_teacher_p1_centered_gamma0` | centered | `teacher_output_temp=3.0` | 0.636 |
| 5 | `07_centered_gamma0_tmin0.5` | centered | 原始强教师 | 0.632 |
| 6 | `phaseB_b2_adaptive_anchor_lowconf_teacher` | centered adaptive | `teacher_output_temp=3.0` | 0.630 |
| 7 | `phaseB_b1/b3` | centered adaptive | 原始强教师 | 0.626 |
| 8 | `phaseB_b4_tmin0.3_alpha2` | centered adaptive | 原始强教师 | 0.619 |
| 9 | `06_centered_gamma1_tmin0.5` | centered | 原始强教师 | 0.615 |

主要结论：

- Newton 是目前最强 CoVar 温度策略。
- `teacher_output_temp=3.0` 能显著改善低置信 teacher 场景。
- centered calibration 的可解释性较好，但容易出现较大 identity zone，性能略弱。
- centered calibration 中 `ref_temp=0.6` 是关键参数，比单纯调 `t_min` 或 `alpha` 更有效。

### 7.2 当前 NPU 20k 结果

NPU 20k 结果改变了下一步判断重点：`phaseC_lc_newton_gamma2_repro` 只比 `phaseC_lc_no_covar_tout3` 高约 0.22 mIoU 点。这个差距太小，不能直接说明 CoVar 是主要贡献。

因此，当前最关键科学问题变成：

```text
teacher_output_temp=3.0 带来的 teacher softening 是主贡献，
还是 CoVar Newton 温度图在 80k 长训练后会拉开明显差距？
```

## 8. 正在进行的关键 80k 实验

2026-06-30 15:08:29 已启动 80k 双实验队列：

1. `phaseC_lc_newton_gamma2_repro`
2. `phaseC_lc_no_covar_tout3`

保存路径：

```text
data/winycg/checkpoints/cirkd_checkpoints/voc/covar_npu_phaseC_80k/
```

队列日志：

```text
runs/covar_npu_phaseC_80k/phaseC_80k_pair_80000.log
```

截至 2026-06-30 16:52 左右，第一项 `gamma2_repro` 已运行到约 `10720/80000`，第一次完整验证 mIoU 为 `52.666`，日志估计第一项剩余约 11 小时 10 分钟。两张 Ascend 910 均有训练进程，HBM 占用约 26-27GB。

预估完成时间：

- `gamma2_repro`：约 2026-07-01 04:00 前后完成。
- `no_covar_tout3`：顺序接着运行，约再需要 11.5-12 小时。
- 整个 80k pair：预计 2026-07-01 15:30 到 17:00 完成。

## 9. 目前可以写进摘要的保守论点

在摘要提交前，建议采用稳健表述，不把尚未完成的 80k pair 结果写死。

可主打的论点：

- 固定温度 KD 忽略密集预测中像素级教师可靠性差异。
- CoVar-KD 用教师置信度和非主类别残差方差构造像素级可靠性估计。
- 基于可靠性的自适应温度图可以动态调节教师分布的锐化或平滑。
- 在 VOC 语义分割蒸馏中，CoVar-KD 在多组设置下优于或接近强 baseline，当前最佳旧结果达到 0.646 mIoU。
- 最新 NPU 实验正在进一步隔离 CoVar 与 teacher softening 的贡献。

不建议在摘要里过早写死的点：

- 不要直接声称 CoVar 是 `teacher_output_temp=3.0` 场景下的唯一主贡献。
- 不要声称 gamma scaling 总是有害，因为 NPU 20k 结果显示 Newton 场景下 `gamma=2` 更好。
- 不要过度强调 centered calibration，除非论文定位为“可解释校准方法”；从性能看 Newton 目前更强。

## 10. 投稿定位建议

### 推荐论文主线

建议把论文主线定位为：

```text
Reliability-aware adaptive temperature distillation for dense prediction
```

核心贡献可以写成三点：

1. 提出一种由教师预测置信度和非主类别残差方差组成的像素级可靠性指标，用于刻画 dense prediction 中教师监督噪声。
2. 提出 CoVar adaptive temperature，将可靠性映射为像素级蒸馏温度，在可靠像素增强教师信号，在不可靠像素降低错误监督影响。
3. 在 CIRKD 语义分割蒸馏框架上系统验证 CoVar-KD，并通过 temperature mode、gamma、teacher softening、centered calibration 等消融分析揭示不同机制的作用。

### 论文标题候选

- CoVar-KD: Confidence-Variance Adaptive Temperature Distillation for Semantic Segmentation
- Reliability-Aware Pixel-Level Temperature Distillation for Semantic Segmentation
- Confidence-Variance Guided Knowledge Distillation for Dense Prediction

## 11. 实验风险与当前短板

当前最大风险是：`teacher_output_temp=3.0` 的 no-covar control 已经很强，20k 时只比 CoVar Newton 低约 0.22 个 mIoU 点。如果 80k 后差距仍然很小，论文不能把主要增益完全归因于 CoVar，而应转向更诚实的结论：

```text
Low-confidence teacher softening is the dominant contributor;
CoVar provides additional but modest gains.
```

另一个短板是 baseline 表格还不够完整。AAAI 正文最好补齐：

- 原始 CIRKD baseline。
- 固定 KD temperature baseline。
- `teacher_output_temp=3.0` no-covar baseline。
- CoVar Newton。
- centered calibration。
- gamma 消融。
- 不同 teacher confidence 场景。
- 至少 2-3 个 seed 或者报告最后若干验证点均值/方差。

## 12. 下一步实验计划

### 12.1 最高优先级

当前正在跑的 80k pair 必须完成：

| 优先级 | 实验 | 目的 | 预期解释 |
|---:|---|---|---|
| 1 | `phaseC_lc_newton_gamma2_repro` 80k | 主方法 | 判断 CoVar Newton 长训练上限 |
| 2 | `phaseC_lc_no_covar_tout3` 80k | 最强 control | 隔离 teacher softening 的贡献 |

判断标准：

- 如果 `gamma2_repro` 明显超过 no-covar，例如超过 0.5-1.0 mIoU 点，可以强调 CoVar 的真实增益。
- 如果差距仍约 0.2 mIoU 点，则应把主结论调整为 teacher softening 是主贡献，CoVar 是小幅补充增益。

### 12.2 摘要后、正文前应补

| 实验 | 目的 | 建议 |
|---|---|---|
| 原始 CIRKD NPU 80k baseline | 提供硬件一致主 baseline | 必补 |
| 固定 temperature KD | 证明 adaptive T 的必要性 | 必补 |
| `teacher_output_temp=3.0` + fixed KD T | 分离 teacher softening 与 KD 温度 | 建议补 |
| CoVar Newton 原始 teacher 80k | 与旧 GPU 结果对齐 | 建议补 |
| 最佳配置多 seed | 证明稳定性 | 正文或 rebuttal 前补 |
| 可视化 T map 与 error map | 增强可解释性 | 建议补 |

### 12.3 可能的增强实验

- 在 Cityscapes 或 ADE20K 上补一个小规模迁移实验，提升 AAAI 说服力。
- 对边界像素、小目标、低置信像素分组统计 mIoU 或 loss 改善。
- 画出 `r`、`T`、teacher error、student improvement 的相关性。
- 比较 entropy-based reliability 与 CoVar reliability，证明 `v` 不只是冗余项。

## 13. 预期效果

### 乐观情况

80k 中 CoVar Newton 明显超过 no-covar control，最终形成清晰故事：

```text
Teacher softening improves KD, but CoVar reliability-aware temperature provides additional gains by selectively modulating pixel-level supervision.
```

这种情况下，论文可以主打 CoVar 的方法贡献和性能增益。

### 中性情况

80k 中 CoVar 只比 no-covar 高 0.1-0.3 mIoU。论文仍可投稿，但主张需要更谨慎：

```text
CoVar is a reliability-aware diagnostic and adaptive KD mechanism.
It gives modest improvement over a strong softened-teacher baseline and provides interpretable pixel-level control.
```

这种情况下，论文需要靠更充分的分析、可视化、跨设置稳定性来支撑贡献。

### 不利情况

80k 中 no-covar 持平或超过 CoVar。则需要调整论文方向：

- 把 `teacher_output_temp` 和 reliability analysis 作为发现。
- 重新设计 CoVar 权重，让温度图不只是改变 logits 温度，也参与 loss weighting。
- 尝试更稳定的 temperature mapping 或分段策略，避免大量像素 clamp 到 `T_min=0.5`。

## 14. 摘要草稿要点

中文摘要要点：

```text
现有语义分割知识蒸馏方法通常采用全局固定温度，忽略了密集预测任务中不同像素的教师可靠性差异。本文提出 CoVar-KD，一种基于置信度与非主类别残差方差的像素级自适应温度蒸馏方法。我们首先构造教师可靠性指标，用于估计每个像素的蒸馏噪声风险；然后将该指标映射为温度图，在可靠像素上锐化教师分布，在不可靠像素上平滑教师分布。该方法可无缝集成到 CIRKD 等语义分割蒸馏框架中。Pascal VOC 上的实验和消融表明，可靠性感知温度调节能够提升轻量学生模型性能，并揭示 teacher softening 与像素级自适应蒸馏之间的互补关系。
```

英文摘要要点：

```text
Knowledge distillation for semantic segmentation commonly applies a global temperature to all pixels, despite the fact that teacher reliability varies substantially across dense predictions. We propose CoVar-KD, a confidence-variance guided adaptive temperature distillation method. CoVar-KD estimates pixel-level teacher reliability from the maximum confidence and the residual variance over non-dominant classes, and converts this reliability into a temperature map that sharpens reliable teacher predictions while smoothing uncertain ones. The method can be integrated into relational distillation frameworks such as CIRKD. Experiments on Pascal VOC with DeepLabV3-ResNet101 as teacher and DeepLabV3-MobileNetV3-Small as student show that reliability-aware temperature modulation improves distillation and provides interpretable control over pixel-level supervision.
```

## 15. 当前操作记录

- 已完成 NPU 训练代码迁移和断点续训。
- 已完成 NPU 20k triage，并生成 `reports/2026-06-30_phaseC_npu_triage.md`。
- 已启动 80k pair：`phaseC_lc_newton_gamma2_repro` 与 `phaseC_lc_no_covar_tout3`。
- GitHub 同步曾因 HTTPS 凭据缺失、SSH 22 端口超时失败；当前本地代码和结果仍在服务器上。

## 16. 结论

项目已有一个相对完整的技术故事：像素级教师可靠性估计、可靠性感知温度调节、与 CIRKD 的集成、以及围绕 Newton、centered calibration、gamma 和 teacher softening 的系统消融。AAAI 摘要阶段可以先采用保守但有力的表述，强调固定温度 KD 的缺陷和 CoVar-KD 的可靠性感知机制。

真正决定论文主结论强弱的是当前 80k pair。如果 CoVar Newton 拉开 no-covar control，论文可以主打 CoVar 的性能贡献；如果差距仍小，论文需要转向“teacher softening 是主贡献，CoVar 提供可解释的微弱增益和分析框架”这一更稳健的定位。
