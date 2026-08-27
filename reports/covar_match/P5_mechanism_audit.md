# P5：P4a 局部优势传递机制审计

## 结论先行

P5.1 预设的参数空间断裂没有出现。18 个冻结 checkpoint/minibatch probe 的平均结果为：

\[
A_z^{\mathrm{P4a}}-A_z^{\mathrm{fixed}}=+0.023018,
\qquad
A_\theta^{\mathrm{P4a}}-A_\theta^{\mathrm{fixed}}=+0.031731.
\]

因此，当前证据不支持“P4a 的 logit-space 优势在网络 Jacobian 投影后立即消失”。相反，P4a 在三个阶段的平均参数梯度余弦均不低于固定 \(T=1.5\)。

最早出现不稳定传递的是参数空间一阶余弦到有限步虚拟更新的接口。使用唯一预设步长 \(\eta=0.02\) 时，P4a 的 same-batch uplift 均值为 \(-0.007305\)，但中位数为 \(+0.001905\)，18 个 probe 中有 61.11% 为正。少数较大的退化 batch 改变了均值方向。独立 batch 的 uplift 均值为 \(+0.001574\)，中位数为 \(-0.001082\)，胜率恰为 50%；固定与 P4a 更新后的独立 batch CE gain 均值分别为 \(-0.0089950\) 和 \(-0.0074206\)，两者平均都没有泛化为正收益。

本轮最精确的结论是：

\[
\boxed{
\text{task-aligned 局部优势平均能够到达参数空间，}
\quad
\text{但不能稳定转化为有限步或跨 batch utility。}
}
\]

该结论把断点定位到有限更新与跨 batch 传递接口，但不能仅凭本审计唯一归因为曲率、梯度尺度或 batch specificity。

## 审计协议

- 学生状态复用 P2/P3A 的 P1 \(T=1\) 轨迹 checkpoint：early 4k、middle 12k、late 20k。这样审计的是产生 P3A 局部证据的同一学生轨迹，不补训 P4a 阶段模型。
- 用 seed 1234 在 VOC val 上独立固定一个 30-image pool，并取排序后的前 24 张图像。每个阶段使用完全相同的 6 对 minibatch，batch size 为 2；同 batch 与对应独立 batch 没有图像重叠。该集合不是 P3A 的 30-image 子集。
- 不同尺寸图像只在 minibatch 内向右/下补零，标签补 ignore label；CE/KD 只统计有效标签。
- 模型处于 eval mode，BN running statistics 不更新。没有 optimizer、训练 step、checkpoint 写入或新训练轨迹。
- \(A_z\) 保持 P4a 的原生 logits 网格、8×8 region、逐像素归一化 KD 方向定义。
- \(A_\theta\) 中的 CE 与 train_kd.py 的真实主头目标代数等价：先将 logits 双线性上采样到标签分辨率，再以 log-softmax、target gather 和 valid-pixel mean 实现确定性 CE reduction。KD 使用原生 logits 网格上的 mean-valid \(\mathrm{KL}(p_t\|p_s)\)。
- 虚拟更新通过参数张量副本和 torch.func.functional_call 完成：
  \[
  \theta'=\theta-0.02\,g_\theta^{\mathrm{KD}}.
  \]
  它不包含 momentum、weight decay 或 CE 梯度，因此是用户指定公式的隔离审计，不等同于完整 SGD iteration。
- 运行环境：PyTorch 2.11.0+cu128，NVIDIA GeForce RTX 4090。

## P5.1 参数空间梯度对齐

| stage | pairs | \(A_z\) fixed | \(A_z\) P4a | \(\Delta A_z\) | \(A_\theta\) fixed | \(A_\theta\) P4a | \(\Delta A_\theta\) | joint break |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| early | 6 | 0.155281 | 0.183329 | +0.028048 | 0.950663 | 0.952287 | +0.001624 | 33.33% |
| middle | 6 | 0.095995 | 0.119316 | +0.023321 | 0.858715 | 0.903784 | +0.045069 | 16.67% |
| late | 6 | 0.066501 | 0.084186 | +0.017685 | 0.855893 | 0.904394 | +0.048501 | 16.67% |
| overall | 18 | 0.105925 | 0.128944 | +0.023018 | 0.888424 | 0.920155 | +0.031731 | 22.22% |

joint break 表示同一 minibatch 满足 \(A_z^{\mathrm{P4a}}>A_z^{\mathrm{fixed}}\)，但 \(A_\theta^{\mathrm{P4a}}\le A_\theta^{\mathrm{fixed}}\) 的比例。P4a 对 \(A_z\) 的非负提升是 hard argmax 对自身目标的代数结果；参数余弦并非如此。整体及三个阶段的平均参数余弦均提升，故不能确认“logit 到 parameter”是主要断点；不过 22.22% 的逐 batch joint break 说明传递也不是逐样本必然成立。

## P5.2 同 batch 与独立 batch 虚拟更新

### 当前 minibatch

| stage | fixed mean gain | P4a mean gain | uplift mean | uplift median | P(P4a>fixed) |
|---|---:|---:|---:|---:|---:|
| early | 0.23999630 | 0.24008893 | +0.00009263 | +0.00190539 | 83.33% |
| middle | 0.10022526 | 0.09151386 | -0.00871140 | +0.00178272 | 50.00% |
| late | 0.06747472 | 0.05417803 | -0.01329669 | +0.00198905 | 50.00% |
| overall | 0.13589876 | 0.12859361 | -0.00730515 | +0.00190539 | 61.11% |

### 独立 minibatch

| stage | fixed mean gain | P4a mean gain | uplift mean | uplift median | P(P4a>fixed) |
|---|---:|---:|---:|---:|---:|
| early | -0.03819186 | -0.03981907 | -0.00162721 | -0.00185880 | 50.00% |
| middle | 0.01370537 | 0.01456155 | +0.00085617 | -0.00108193 | 50.00% |
| late | -0.00249845 | 0.00299569 | +0.00549414 | +0.00018372 | 50.00% |
| overall | -0.00899498 | -0.00742061 | +0.00157437 | -0.00108193 | 50.00% |

正 gain 表示虚拟 KD 更新降低了 CE。same-batch uplift 的总体范围为 \([-0.06517,0.00790]\)，独立 batch uplift 的范围为 \([-0.01299,0.03142]\)。均值、中位数和胜率没有给出一致排序，因此不能写成“P4a 普遍改善 same batch”或“P4a 普遍伤害 next batch”。

更保守的机制解释是：参数梯度夹角只描述无穷小方向，不约束给定步长下的曲率项；P4a 的 KD 梯度范数和 target sharpness 又与固定温度不同。典型 batch 的中位数模式符合“即时改善、独立 batch 不改善”，但少数有限步 outlier 足以反转均值。这说明局部贪心信号没有形成稳定的训练 utility。

## Teacher target entropy 与 KD 梯度范数

| stage | entropy fixed | entropy P4a | \(\|g_\theta^{KD}\|\) fixed | \(\|g_\theta^{KD}\|\) P4a |
|---|---:|---:|---:|---:|
| early | 0.166306 | 0.106389 | 5.984505 | 6.220933 |
| middle | 0.166306 | 0.105889 | 3.245795 | 3.466992 |
| late | 0.166306 | 0.101275 | 3.367642 | 3.689253 |
| overall | 0.166306 | 0.104518 | 4.199314 | 4.459059 |

P4a target entropy 比固定 \(T=1.5\) 低约 37%，参数 KD 梯度范数平均高约 6.2%。这与训练中越来越多区域选择 \(T=0.5\) 一致，也为有限步 outlier 提供了一个可检验解释，但本轮没有做步长扫描或曲率分解，因此不能把它写成已确认因果。

训练日志没有记录 target entropy 或 KD 参数梯度范数。上表是同一批冻结 checkpoint probe，不是从三个点插值得到的训练曲线。

## P5.3 现有训练过程对照

| phase | mean CE fixed/P4a | mean KD fixed/P4a | P4a 温度分布 | phase mean \(\Delta A\) | cumulative \(\sum\Delta A\) | cumulative mean \(\Delta A\) |
|---|---:|---:|---|---:|---:|---:|
| early 1–4k | 0.720266/0.751453 | 0.626516/0.695214 | 0.5:47.71%, 0.75:24.75%, 1:11.01%, 1.25:4.16%, 1.5:5.02%, 2:7.36% | 0.00625786 | 19,716.537 | 0.00625786 |
| middle 4–12k | 0.433481/0.443297 | 0.358347/0.400608 | 0.5:55.41%, 0.75:22.36%, 1:7.77%, 1.25:3.84%, 1.5:4.86%, 2:5.77% | 0.00774350 | 68,529.186 | 0.00724841 |
| late 12–20k | 0.285675/0.288761 | 0.220062/0.250791 | 0.5:61.68%, 0.75:18.92%, 1:5.85%, 1.25:3.80%, 1.5:4.63%, 2:5.13% | 0.00899008 | 125,210.373 | 0.00794521 |

两条正式训练日志各包含 1,000 个 CE/KD 点，P4a 还包含 1,000 个 selector 统计点。累计局部 \(\Delta A\) 从 19,716.537 单调增加到 125,210.373，累计 region-weighted mean 也从 0.006258 增至 0.007945；与此同时，P4a 的阶段平均 CE 与 KD loss 均高于固定 \(T=1.5\)。

两组实验都只在 20k validation 一次，因此不存在可提取的中间 mIoU 曲线。唯一终点为：

| method | final mIoU | final pixAcc |
|---|---:|---:|
| fixed \(T=1.5\) | 60.861409% | 90.041566% |
| P4a | 60.517776% | 89.898586% |

P4a 的 final mIoU 低 0.343633 pp。现有日志严格支持以下两件事同时成立：

\[
\text{累计局部 proxy 持续占优},
\qquad
\text{最终 mIoU 不占优}.
\]

## 决策与论文边界

P5 排除了一个简单解释：P4a 并非在平均意义上因为网络 Jacobian 而丢失全部参数梯度对齐。失败更晚地表现为有限步收益对 minibatch 高度敏感，并且没有稳定的跨 batch 排序。该结果与 P4a 的长期负结果一致，但没有证明某一个二阶项或优化器机制是唯一原因。

因此本轮不引出新方法，也不继续调温度、region、margin gate 或步长。论文中可以将机制结论写为：

> Region-wise task alignment improves the native logit-space objective and, on average, the CE–KD parameter-gradient cosine. However, this advantage is not stable under a finite parameter update or across disjoint minibatches, and it does not translate into final mIoU.

MetaOptimize 显式以未来损失的折扣和定义长期 regret，说明即时一步目标与长时程训练目标的区分具有相关元优化语境；该工作只支持问题定位，不证明本实验的具体断点。[MetaOptimize: A Framework for Optimizing Step Sizes and Other Meta-parameters, ICML 2025](https://proceedings.mlr.press/v267/sharifnassab25a.html)。

## 执行门禁与产物

- 18/18 个正式 probe 有限；三个 checkpoint 合约、固定样本去重、P4a hard-argmax 非负性和既有日志门禁全部通过。
- P4a/P5 相关单元测试 12/12 通过。
- 原始结果：runs/covar_match/P5_mechanism_audit/raw_results.json。
- 结构化报告：reports/covar_match/P5_mechanism_audit.json。
- 本报告：reports/covar_match/P5_mechanism_audit.md。
- 本轮没有训练新策略、覆盖 P4a 结果或写入 checkpoint。
