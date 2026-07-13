# CoVar-KD 当前方法、公式、实现与实验审查

- 审查快照：2026-07-13（Asia/Shanghai）
- 仓库：`/home/ma-user/work/ljs`
- 文档目的：把当前有效方法、实际代码、冻结实验配方、已完成结果、负结果与未解决问题放到同一份记录中，便于整体检查研究逻辑。
- 事实口径：方法以当前代码和冻结 shell 为准；实验以 `reports/` 中的阶段报告及已核验日志为准；README 中的上游发布结果不计为本项目本地完成实验。
- 状态标记：`已完成` 表示训练预算、最终验证和日志完整性已经核验；`进行中` 和 `计划中` 不进入结果结论。

> 执行方向更新（2026-07-13）：第 0 至 12 节继续保留 Newton/历史实验审查快照；当前主线已转为 O1.2。O1.2-A 独立实现、全量 train/val 机制诊断和联合门禁已完成并通过，温度中位数约为 0.982/0.979、调和均值约为 0.988/0.988。另行授权的 O1.2-B `neutral` 与 `unreliable_only` 20-step fresh 及各自终点零步恢复审计也均通过，但只构成学生训练链路 smoke；B 不并入 A 的机制 gate，且没有 validation、mIoU、性能或泛化结论。当前结论与停止线见第 13 节、[O1.2 预注册与结果](2026-07-13_phaseO_rtc_o12_budgeted_routing_plan.md)、[O1.2-B smoke 报告](2026-07-13_phaseO_rtc_o12b_smoke_report.md)及 [Phase O 主记录](2026-07-13_phaseO_rtc_method_reconstruction_plan.md)。

## 0. 先给结论

当前研究链条是清楚的：教师像素输出经过置信度—残差方差评分，评分再通过投影 Newton 更新转成像素温度，最后只在训练期修改 logit KD。机制诊断也显示该评分能识别教师错误，温度图确实随不可靠度变化。

已有证据中最扎实的部分是：在**同一 CIRKD 配方**下，`Tout=3.0` 的 CoVar on/off 三个 seed 都得到正增益，平均 best/final mIoU 增益为 `+0.0055/+0.0082`；换成 PSPNet 学生头后，单 seed 仍有 `+0.0021/+0.0035`。

但当前还不能把这些增益严格归因于“像素级空间自适应”本身，原因有三项：

1. CoVar 温度中 `90.35%` 的像素落在 `T_min=0.5`，整体行为以大面积低温锐化为主。
2. CWD 上的匹配标量对照中，固定 `T=0.6` 在 20k final mIoU 上比 Newton CoVar 高 `0.007381`，说明全局低温锐化至少是一个很强的替代解释。
3. 更严重的是，当前 CIRKD 的 CoVar-on KD 只平均 valid pixels，而 CoVar-off 的普通 KD 会平均包括 padding/ignore 区域在内的全部 logits 位置。因此现有 on/off 对照同时改变了“温度机制”和“KD 有效像素集合”，并非完全单变量。

所以，当前最稳妥的论文定位应是：**提出一种教师可靠性驱动的像素温度机制，并证明它在既有 CIRKD 配方内稳定改善结果；空间自适应相对匹配标量温度的独立收益仍待严格验证。** 现在不应声称 VOC 最强、方差项单独有效、跨数据集泛化已经成立，或 CWD+CoVar 优于合适的标量温度。

## 1. 研究问题与证据线

### 1.1 研究问题

语义分割是密集预测任务。教师在物体边界、遮挡、小目标和类别混淆区域可能出错，而传统 KD 通常对所有像素采用同一个温度与强度。当前工作希望回答：

> 能否仅利用教师自身输出，在像素级估计教师不可靠度，并用自适应温度强化可靠像素、平滑不可靠像素，从而减少错误知识传递？

### 1.2 两条证据线必须分开

1. **受控 CoVar 证据线：CoVar + CIRKD。** 入口为 `train_cirkdv2.py`。同一教师、学生、seed、schedule 和 CIRKD 损失下比较 CoVar on/off，原则上用于判断温度机制的增量效果；但第 8.1 节指出当前实现还存在 valid-mask 混杂。
2. **方法级强基线与可移植性线：CWD/KD/SKD/IFVD 等。** 入口为 `train_kd.py`。这些方法使用各自完整配方，只能比较最终方法性能，不能把差异归因于某一个损失。CoVar 接入 CWD 后，只有标准 logit-KD 分支被替换，其他 CWD 分支保持不变。

所有主结论应以 80k 为准；20k 结果主要用于超参数筛选、组件消融和资源晋级。

## 2. 任务、模型与记号

主实验使用 Pascal VOC/VOCAug：训练集 10,582 张，验证集 1,449 张，类别数 `K=21`。

- 教师：DeepLabV3-ResNet101，加载 VOC 分割整网权重，训练时冻结并保持 `eval()`。
- 主学生：DeepLabV3-MobileNetV3-Small，MobileNetV3-Small 使用 ImageNet 权重初始化。
- 跨学生：PSPNet-MobileNetV3-Small。
- 教师输出主 logits、aux logits 和 ASPP 后特征；主学生输出主 logits 和 ASPP 后特征。

记第 `i` 个像素、第 `k` 类的教师和学生原始 logits 为 `z^t_{ik}` 与 `z^s_{ik}`，有效像素掩码为 `M_i in {0,1}`，ignore label 为 `-1`。

当前 `teacher_output_temp` 只额外软化教师的 logit-KD 目标：

$$
\tilde z^t_{ik}=\frac{z^t_{ik}}{\tau_{out}}.
$$

主结果取 `tau_out=3.0`，强教师对照取 `tau_out=1.0`。它不是师生共同使用的标准 KD 温度；后续像素温度还会再次作用于教师和学生。

## 3. CoVar 可靠性指标

### 3.1 温度下的教师分布

对像素温度候选 `T_i>0`，定义：

$$
p^t_{ik}(T_i)=
\frac{\exp(\tilde z^t_{ik}/T_i)}
{\sum_j\exp(\tilde z^t_{ij}/T_i)}.
$$

由于温度为正，logits 的类别排序不变。令最大概率类别为 `k_i^*`，则：

$$
c_i(T_i)=p^t_{i k_i^*}(T_i),
\qquad
\mu_i(T_i)=\frac{1-c_i(T_i)}{K-1}.
$$

非主类别残差概率的总体方差为：

$$
v_i(T_i)=\frac{1}{K-1}
\sum_{k\ne k_i^*}
\left(p^t_{ik}(T_i)-\mu_i(T_i)\right)^2.
$$

### 3.2 不可靠度

当前完整评分为：

$$
r_i(T_i)=
-\log c_i(T_i)
+a\frac{v_i(T_i)}{1-c_i(T_i)},
\qquad
a=\frac{(K-1)^2}{2}.
$$

VOC 中 `K=21`，所以 `a=200`。`r` 越大代表教师越不可靠。代码还支持：

$$
r_{conf}=-\log c,
\qquad
r_{var}=a\frac{v}{1-c},
\qquad
r_{full}=r_{conf}+r_{var}.
$$

系数 `a` 可写成更直观的两两差异形式。令非主类共有 `n=K-1` 个，则：

$$
a v
=\frac{1}{2}\sum_{j<k,\,j,k\ne k_i^*}
(p^t_{ij}-p^t_{ik})^2.
$$

因此第二项刻画的是非主类别残差质量内部的不均匀程度，并由总残差质量 `1-c` 归一化。论文若使用 “CoVar” 名称，需要明确它表示 confidence–variance 组合，而不是协方差矩阵。

核心独立实现位于 `utils/covar_temperature.py:33-64`；CIRKD 训练器内还有一份实现，原始统计组件位于 `PCOS.py`。

## 4. 像素温度的投影 Newton 更新

### 4.1 概率导数

记温度下的教师 logit 均值与方差为：

$$
\bar z_i=\sum_k p^t_{ik}\tilde z^t_{ik},
\qquad
\operatorname{Var}_{p_i}(\tilde z_i^t)=
\sum_kp^t_{ik}(\bar z_i-\tilde z^t_{ik})^2.
$$

代码使用：

$$
\frac{\partial p^t_{ik}}{\partial T_i}
=p^t_{ik}\frac{\bar z_i-\tilde z^t_{ik}}{T_i^2},
$$

$$
\frac{\partial^2 p^t_{ik}}{\partial T_i^2}
=p^t_{ik}
\left[
\frac{(\bar z_i-\tilde z^t_{ik})^2-\operatorname{Var}_{p_i}(\tilde z_i^t)}{T_i^4}
-\frac{2(\bar z_i-\tilde z^t_{ik})}{T_i^3}
\right].
$$

令 `s=1-c`，则完整评分的一、二阶导数为：

$$
r'
=-\frac{c'}{c}
+a\left(\frac{v'}{s}+\frac{vc'}{s^2}\right),
$$

$$
r''
=\frac{(c')^2}{c^2}-\frac{c''}{c}
+a\left[
\frac{v''}{s}
+\frac{vc''}{s^2}
+\frac{2c'v'}{s^2}
+\frac{2v(c')^2}{s^3}
\right].
$$

其中非主类均值 `mu=s/(K-1)`，代码按下式计算方差导数：

$$
v'=\frac{2}{K-1}\sum_{k\ne k^*}p_kp'_k-2\mu\mu',
$$

$$
v''=\frac{2}{K-1}\sum_{k\ne k^*}
\left((p'_k)^2+p_kp''_k\right)
-2\left((\mu')^2+\mu\mu''\right).
$$

### 4.2 更新规则

初始化 `T_i^(0)=T_0`。第 `m` 次更新：

$$
\Delta_i^{(m)}=
\begin{cases}
\eta\,r'_i/r''_i,
&r''_i>0,\ |r''_i|\ge\epsilon_h,\ r''_i\text{ 有限},\\
\eta\,r'_i,
&\text{否则退化为梯度步}.
\end{cases}
$$

$$
\Delta_i^{(m)}\leftarrow
\operatorname{clip}(\Delta_i^{(m)},-\Delta_{max},\Delta_{max}),
$$

$$
T_i^{(m+1)}=
\operatorname{clip}
\left(T_i^{(m)}-\Delta_i^{(m)},T_{min},T_{max}\right).
$$

当前主配置为：

| 参数 | 值 |
|---|---:|
| `T_0` | 1.0 |
| `T_min`, `T_max` | 0.5, 8.0 |
| `eta` | 0.6 |
| 更新次数 | 8 |
| `epsilon_h` | `1e-5` |
| `Delta_max` | 0.25 |
| reliability mode | `full` |

温度图在 `no_grad` 下计算，不对教师或温度求解过程反向传播。无效像素最终被置回 `T_0`。

严格说，这不是“Newton 已求得最优解”，而是**固定 8 步、带梯度回退和上下界投影的阻尼 Newton 更新**。`covar_grad_converge_thresh` 只参与日志诊断，不会提前停止。20k 主配置最后一次诊断只有约 `7.1%` 像素满足 `|dr/dT|<0.01`，同时约 `93.1%` 像素触及某个边界，因此论文中不宜使用“已收敛求解”表述。

### 4.3 已实现但不是当前主线的策略

- `sqrt`：`T_i=clip[T_0+alpha(sqrt(r_i)-mean_valid(sqrt(r))),T_min,T_max]`。
- `grad`：只使用 `T^(m+1)=clip(T^m-eta r'(T^m))`。
- `calib_conf`：先由参考不可靠度构造目标置信度，再用带区间保护的 Newton/二分混合求 `log c_i(T_i)=log c_i^*`；centered 版本允许双向锐化和平滑。

这些分支属于历史探索或备用实现，不应与当前 Newton 主结果混写。

## 5. 像素级 KD 与总训练目标

### 5.1 CoVar logit KD

学生与教师的最终蒸馏分布分别是：

$$
p^s_i=\operatorname{softmax}\left(\frac{z^s_i}{T_i}\right),
\qquad
p^t_i=\operatorname{softmax}\left(\frac{z^t_i}{\tau_{out}T_i}\right).
$$

因此教师的有效温度是 `tau_out*T_i`，学生的有效温度是 `T_i`。像素 KD 为：

$$
\mathcal L_{CoVar-KD}
=\frac{1}{\sum_iM_i}
\sum_i M_iT_i^\gamma
D_{KL}(p^t_i\Vert p^s_i).
$$

当前 `gamma=2`。这是标准标量 KD 中 `T^2` 梯度补偿的像素级推广；gamma 消融见第 7.4 节。当所有像素有效且 `T_i` 为同一常数时，它与仓库普通标量温度 KD 的定义一致。

### 5.2 CoVar + CIRKD 完整目标

主方法不是单独的 logit KD，而是在 CIRKD 完整配方中替换 logit-KD 分支：

$$
\begin{aligned}
\mathcal L_{total}={}&
\mathcal L_{CE}
+\lambda_{KD}\mathcal L_{CoVar-KD}
+\lambda_{fit}\mathcal L_{FitNet}\\
&+\lambda_{mb-p}\mathcal L_{mini-pixel}
+\lambda_{mem-p}\mathcal L_{memory-pixel}
+\lambda_{mem-r}\mathcal L_{memory-region}\\
&+\lambda_{mb-c}\mathcal L_{mini-channel}
+\lambda_{mem-c}\mathcal L_{memory-channel}
+\lambda_{ch}\mathcal L_{channel-MSE}.
\end{aligned}
$$

实际权重：

| 项 | 权重 |
|---|---:|
| task CE | 1 |
| logit KD / CoVar-KD | 1 |
| FitNet | 10 |
| mini-batch pixel relation | 1 |
| memory pixel / region | 0.1 / 0.1 |
| mini-batch channel relation | 1 |
| memory channel | 0.1 |
| channel MSE | 100 |

关系蒸馏的做法是：

- mini-batch pixel：对归一化特征构造 batch 内跨图像像素相似度，教师相似度分布监督学生；
- memory pixel/region：按类别维护教师像素队列和区域原型队列，师生对同一个教师 memory bank 的相似度分布做 KL；
- channel：4x4 平均池化并对齐通道后，计算 batch 内通道关系 KL、教师通道 memory KL 和归一化特征 MSE；
- FitNet：学生特征经过 GCN/1x1 对齐后与教师特征做均方误差。

### 5.3 CWD 中的 CoVar 接入

CWD 学生目标为：

$$
\mathcal L_G=
\mathcal L_{CE}
+1.0\mathcal L_{KD}
+0.001\mathcal L_{adv-G}
+50\mathcal L_{CWD-feat}
+3\mathcal L_{CWD-logit},
$$

判别器单独优化：

$$
\mathcal L_D=0.1\mathcal L_{adv-D}.
$$

CWD 在空间维做 channel-wise softmax，内部固定温度为 4。接入 CoVar 时，只有 `1.0*L_KD` 的标量温度被像素温度替换；CWD-feature、CWD-logit 和 adversarial 分支仍使用原始教师输出。`teacher_output_temp=3` 也只作用于标准 logit-KD 目标。

这使 CWD 内的 CoVar/fixed 比较相对干净，但也意味着 CoVar 只改变完整 CWD 配方中的一个分支。

## 6. 实际训练与评估做法

### 6.1 主配置

| 项目 | 当前冻结值 |
|---|---|
| 数据 | Pascal VOC/VOCAug，21 类 |
| 主 teacher/student | DeepLabV3-R101 / DeepLabV3-MobileNetV3-Small |
| crop | 512x512 |
| 全局 batch | 16；双卡时每卡 8 |
| workers | 8 |
| optimizer | SGD |
| 初始学习率 | 0.02 |
| momentum / weight decay | 0.9 / `1e-4` |
| 学习率 | `lr_t=lr_0*(1-t/Tmax)^0.9` |
| 主预算 | 80,000 iterations |
| 日志 / 保存 / 验证 | 每 20 / 800 / 800 iterations |
| 主 seeds | 1234、2025、3407 |
| 主指标 | best validation mIoU、iteration-80000 final mIoU |
| 次指标 | late-window mean、pixAcc、运行时间、完整性 |

训练数据使用随机尺度、随机裁剪和水平翻转；较小图像/标签会 padding，其中标签 padding 为 ignore label。这一点与第 8.1 节的 KD mask 问题直接相关。

### 6.2 每个 iteration

1. 读取增强后的图像和标签。
2. 冻结教师在 `no_grad` 下前向；logit-KD 分支可先除以 `tau_out`。
3. 学生前向得到 logits 和特征。
4. 将 `target != ignore_label` 的 mask 最近邻缩放到 logits 分辨率。
5. 从教师 logits 计算 `c,v,r`，固定执行 8 步投影 Newton 得到 `T(x)`。
6. 计算 task CE、CoVar-KD，以及 CIRKD 或 CWD 的其他损失。
7. 使用 poly 学习率反向更新学生和蒸馏投影模块；教师不更新。
8. 每 800 iterations 验证并保存 checkpoint/training state。

### 6.3 推理

推理只加载学生模型并执行一次学生前向、插值和 argmax。教师、可靠性统计、Newton、memory bank 和 CoVar 温度均不参与，因此没有推理期额外模块、参数或计算量。

## 7. 已完成实验记录

### 7.1 80k 主受控表：teacher softening x CoVar

| `Tout` | CoVar | Best mIoU | Final mIoU | CoVar delta best/final | 备注 |
|---:|---|---:|---:|---:|---|
| 1.0 | off | 0.6426 | 0.6416 | - | 100 次验证 |
| 1.0 | on | 0.6453 | 0.6440 | +0.0027 / +0.0024 | 恢复日志仅保留 17 次验证片段；指标可用，runtime/曲线不可比 |
| 3.0 | off | 0.6475 | 0.6383 | - | 100 次验证 |
| 3.0 | on | **0.6539** | **0.6490** | **+0.0064 / +0.0107** | 当前主方法，100 次验证 |

`Tout=3.0` 的 CoVar 对 final 的改善大于 best，表面上说明后期退化较小；但在修复第 8.1 节的 mask 混杂前，这仍不是严格的单变量温度因果证据。

### 7.2 80k 三种子稳定性

`Tout=3.0`：

| Seed | Off best/final | CoVar best/final | Delta best | Delta final |
|---:|---:|---:|---:|---:|
| 1234 | 0.6475 / 0.6383 | 0.6539 / 0.6490 | +0.0064 | +0.0107 |
| 2025 | 0.6370 / 0.6370 | 0.6411 / 0.6407 | +0.0041 | +0.0037 |
| 3407 | 0.6387 / 0.6346 | 0.6448 / 0.6448 | +0.0061 | +0.0102 |
| Mean +/- report SD | 0.6411+/-0.0056 / 0.6366+/-0.0019 | **0.6466+/-0.0066 / 0.6448+/-0.0042** | **+0.0055** | **+0.0082** |

三个 paired delta 都为正，这是当前最强的稳定性证据。seed 2025 的 off 行从接近结束处恢复，报告只保留最后一次验证，因此该行的 recovered best/final 可用，验证次数和 runtime 不可比。

`Tout=1.0` 三种子聚合：

| 指标 | Off | CoVar | Delta |
|---|---:|---:|---:|
| Best mIoU | 0.6425+/-0.0043 | 0.6444+/-0.0020 | +0.0019 |
| Final mIoU | 0.6419+/-0.0039 | 0.6434+/-0.0019 | +0.0015 |

这里有一个 seed 为负，增益明显弱于 `Tout=3.0`，应放在补充或消融，不宜作为 headline。

注意：现有阶段汇总脚本使用 population SD。论文若使用 sample SD (`ddof=1`)，必须统一重算并明确统计口径；三种子也不足以支撑夸张的显著性表述。

### 7.3 跨学生 80k，单 seed

| Student | CoVar | Best | Final | Final pixAcc |
|---|---|---:|---:|---:|
| PSPNet-MobileNetV3-Small | off | 0.6366 | 0.6324 | 0.9074 |
| PSPNet-MobileNetV3-Small | on | **0.6387** | **0.6359** | **0.9089** |

delta 为 `+0.0021` best、`+0.0035` final。它说明方法不只绑定 DeepLabV3 学生头，但仍是同一 MobileNetV3-Small backbone、单 seed，不能称为跨架构统计稳定性。

### 7.4 20k gamma 消融

| 变体 | Best/final mIoU | 结论 |
|---|---:|---|
| Newton, gamma=0 | 0.6087 / 0.6087 | 像素 KD 缩放不足 |
| Newton, gamma=1 | 0.6282 / 0.6282 | 中间水平 |
| Newton, gamma=2 | **0.6353 / 0.6353** | 当前选择 |
| No CoVar | 0.6331 / 0.6331 | 固定温度对照 |

这是短预算超参数选择，不能直接等价为 80k 结论。历史 centered-calibration 中出现过 gamma=0 优于 gamma=1，那是另一种温度生成方式，不能与当前 Newton gamma 结论混用。

### 7.5 20k 可靠性组件消融

| 输入 | Best/final | 相对 off |
|---|---:|---:|
| Off | 0.6267 | - |
| Confidence only | 0.6285 | +0.0018 |
| Variance only | 0.6242 | -0.0025 |
| Confidence + variance | **0.6301** | **+0.0035** |

正确结论是“联合评分优于两个单分量”；不能写“置信度和方差各自都能提高精度”。方差项只在与置信度联合时表现出互补性。

### 7.6 机制诊断

H1，完整 VOC val：1,449 张图、5,935,104 个采样有效像素。

- `corr(r, teacher_wrong)=0.3984`；
- 按 `r` 分箱后，`corr(bin_r, error_rate)=0.9599`；
- 最低/最高 `r` bin 的教师错误率为 `0.0104% / 29.5577%`。

这支持评分能识别教师错误，但属于相关性而非性能因果证据。

H2 扫描前 300 张 VOC-val 图并自动排序，生成 6 个可视化样例。推荐样例 `h2_rank03_2007_005149.png` 中 teacher wrong 为 `29.17%`、`r_p95=3.378`、`T_mean=0.704`。论文应同时说明扫描范围和排序规则，避免把精选严重失败案例写成普通样例。

H3，完整 VOC val、`Tout=3.0`：

| 统计 | 值 |
|---|---:|
| teacher wrong rate | 5.0490% |
| `r` mean/median/p95/p99 | 0.2258 / 0.0125 / 1.6600 / 1.9452 |
| `T` mean/median/p95/p99 | 0.5815 / 0.5000 / 1.3226 / 1.6919 |
| `T=T_min=0.5` | 90.35% |
| `T>1.25` | 5.96% |
| `T=T_max=8.0` | 0% |
| `corr(r,T)` | 0.9159 |
| `corr(r,teacher_wrong)` | 0.4130 |
| `corr(T,teacher_wrong)` | 0.3688 |

温度图确实呈现“多数像素锐化、少量不可靠尾部平滑”。但当前 H3 的 `r` 是在最终 `T` 处重新计算的 `r(T)`，与 `T` 在同一求解方程中耦合。更有说服力的诊断应报告参考温度 `T_0` 下的 `r_0` 与最终 `T` 的关系：`corr(r_0,T)`、按 `r_0` 分箱的平均 `T` 和错误率。

### 7.7 外部 KD 基线与 CWD

80k、seed 1234 的方法级比较：

| 完整方法配方 | Best | Final |
|---|---:|---:|
| KD-only | 0.625 | 0.618 |
| IFVD | 0.630 | 0.626 |
| SKD | 0.634 | 0.631 |
| CoVar on CIRKD, `Tout=3.0` | 0.6539 | 0.6490 |
| CWD official recipe | **0.664** | **0.661** |

CWD 比当前 CoVar/CIRKD 高 `+0.0101` best、`+0.0120` final。因此当前方法不能声称 VOC 绝对最强。这里的 CWD 与 CoVar/CIRKD 底座损失不同，也不能把差值解释为 CWD 单一组件优于 CoVar 温度。

CWD 80k 三种子：

| Seed | Best | Final |
|---:|---:|---:|
| 1234 | 0.664 | 0.661 |
| 2025 | 0.660 | 0.658 |
| 3407 | 0.665 | 0.664 |

阶段报告的聚合为 `0.6630+/-0.0022` best、`0.6610+/-0.0024` final（population SD；sample SD 约为 `0.0026/0.0030`）。

### 7.8 CWD + CoVar 与 matched scalar 反证

Phase M，20k、seed 1234：

| 变体 | Best | Final |
|---|---:|---:|
| CWD, `Tout=1`, scalar `T=1` | 0.639 | 0.636 |
| CWD, `Tout=3`, scalar `T=1` | 0.643 | 0.642 |
| CWD, `Tout=3`, Newton CoVar | **0.646** | **0.646** |

CoVar 相对 scalar `T=1` 表面提高 `+0.003/+0.004`。

Phase M2 在相同配方下增加匹配标量温度：

| logit KD 温度 | Best | Final | Last-10 mean |
|---|---:|---:|---:|
| scalar `T=0.5` | 0.648235 | 0.648235 | 0.626266 |
| scalar `T=0.6` | **0.653381** | **0.653381** | 0.624513 |
| scalar `T=1.0` | 0.643000 | 0.642000 | **0.626800** |
| Newton CoVar | 0.646000 | 0.646000 | 0.622100 |

关键差值：

- `T=0.6 - T=1.0` final：`+0.011381`；
- `CoVar - T=0.6` final：`-0.007381`；
- last-10 mean 反而由 `T=1.0` 最高。

这是一项重要负结果：20k 单 seed 下，CoVar 对 `T=1` 的改善不能证明空间分配有效，因为简单全局低温更强。它还提示单个 final endpoint 与末段稳定性可能冲突，必须继续报告 final、best 和 late-window mean，而不能只挑最有利指标。

### 7.9 训练开销

| Setting | Fixed sec/iter | CoVar sec/iter | 相对开销 |
|---|---:|---:|---:|
| DeepLabV3 主学生，CIRKD | 0.5381 | 0.5935 | +10.3% |
| PSPNet 跨学生 | 0.5332 | 0.5841 | +9.5% |
| CWD 20k 接入 | 0.4350 | 0.4537 | +4.3% |

开销仅在训练期；推理不变。CIRKD 训练器还每步执行了未进入损失的 `split_quality()`，移除后应重新测量纯 CoVar 开销。

## 8. 方法与实现审查：当前真正需要修正的点

### 8.1 P0：CoVar on/off 的有效像素集合不一致

这是目前最重要的代码级因果混杂。

- CoVar-on 调用像素温度 KD：乘 valid mask，并除以 valid pixel 数。
- CoVar-off 调用普通 `CriterionKD`：把 logits 展平后以 `batchmean` 计算，包含全部空间位置，没有排除 padding/ignore pixels。
- VOC 随机缩放与 512x512 crop 会对较小样本进行 padding，因此两条路径看到的 KD 像素集合确实可能不同。

这意味着现有 on/off 不只改变 `T_i`，还改变了是否在 ignore/padding 处施加 KD。修复建议：所有 fixed/CoVar 变体统一调用同一个 masked KD 函数；fixed control 使用常数温度图 `T_i=tau`，并使用完全相同的 valid mask、归一化和 teacher-output softening。修复后至少重跑：

1. 20k smoke/triage，确认旧差异量级；
2. CIRKD `Tout=3`、CoVar vs scalar `T=1` 的 80k seed 1234；
3. 若结果方向保持，再补 seeds 2025/3407。

在这组复核完成前，三种子正增益仍是重要经验结果，但不应表述为无混杂的温度机制因果效应。

### 8.2 P0：空间自适应尚未与全局锐化隔离

H3 中 `T` 的中位数为 0.5、均值为 0.5815，90.35% 像素触及下界；Phase M2 又显示 scalar `T=0.6` 强于 CoVar。当前证据更直接地支持“较低 logit KD 温度有用”，而不是“按像素分配温度不可替代”。

严格对照应在**同一 CIRKD base**中同时比较：

1. constant `T=1.0`；
2. constant `T=0.5`、`0.6`、以及与 CoVar 平均值匹配的 `T=0.58`；
3. 原始 CoVar map；
4. 每图或每 batch 随机打乱空间位置、但保持温度直方图不变的 shuffled map；
5. 可选的反向排序 map。

constant 对照检验“低温均值”，shuffle 对照检验“正确空间位置”，二者结合才能支撑空间自适应的因果主张。Phase N 目前只做 CWD 中 `T=0.6` vs `1.0`，即使完成，也不能替代 CIRKD 内的这组隔离实验。

### 8.3 P0：Newton 目标存在自引用，需要更准确的理论叙述

当前更新最小化的是由同一个温度改变后的 `r(T)`，而不是把参考温度下的固定 `r_0` 映射到目标 `T`。降低温度本身就会提高最大置信度，因此求解器天然容易把大量像素推向 `T_min`。

算法定义层面并不保证“初始 `r_0` 越大，最终 `T` 必然越大”；目前只有经验相关性。建议：

- 将方法准确描述为“对教师分布形状做投影更新”，不要写成闭式单调映射；
- 补 `r_0 -> T` 单调性诊断；
- 考虑直接设计显式单调映射 `T=f(r_0)`，或固定目标可靠度/目标置信度的校准式求解，以减少自引用；
- 对 `T_min`、更新次数和 `a` 做敏感性实验。

### 8.4 P1：H3 当前诊断有耦合相关性

H3 报告的 `corr(r,T)=0.9159` 使用最终温度处重新计算的 `r(T)`，二者由同一方程共同产生。它证明输出量相关，却不能单独证明输入不可靠度驱动了温度。应补：

- `corr(r_0,T_final)`；
- `r_0` 分位 bin 下的 `T_final` 均值/分位数；
- `r_0`、`T_final` 对 teacher error 的条件关系；
- 若可能，控制 confidence 后检验 variance term 的额外信息。

### 8.5 P1：下界饱和与 Newton “收敛”表述

大部分像素在 8 步后落到下界，梯度收敛比例很低。当前更像投影截断动力学，而不是数值求解收敛。至少需要：

- `T_min in {0.3,0.5,0.7,0.8}` 的敏感性；
- 4/8/12/16 步更新对结果和温度分布的影响；
- 报告 valid-Hessian 比例、gradient fallback 比例、边界占比和 `|r'|`；
- 若性能主要由下界决定，简化为显式映射可能更稳定、更快。

### 8.6 P1：方差项的主张与系数需要收紧

variance-only 的 20k 结果为负，而 full 最好。这支持互补，不支持方差独立有效。`a=200` 虽可写成非主类两两离散度，但仍需要：

- `a` 的理论来源或尺度不变性说明；
- `a in {0,50,100,200,400}` 或归一化版本的消融；
- confidence 相近像素中，variance 是否继续预测 teacher error；
- 避免把相关性写成校准性或概率不确定性的严格估计。

### 8.7 P1：证据覆盖仍窄

- 只有 VOC 本地完整证据，尚无跨数据集结果；
- PSPNet 跨学生只跑一个 seed，且仍使用 MobileNetV3-Small backbone；
- DeepLabV3-ResNet18 跨学生因权重/配置未就绪而 deferred；
- 尚未完成更近期方法基线；
- CWD 的绝对性能显著高于当前 CoVar/CIRKD。

论文应将“泛化”限定为“跨一个学生头的初步迁移证据”，并把跨数据集与更多 backbone 列为必要补充。

### 8.8 P1：重复实现与配置漂移风险

可靠性与 Newton 至少分散在：

- `train_cirkdv2.py` 内部实现；
- `utils/covar_temperature.py` 的复用实现；
- `scripts/diagnostics/covar_rt_distribution.py` 的诊断实现；
- `PCOS.py` 的可靠性统计实现。

当前单元测试主要覆盖 `utils/covar_temperature.py`，不能自动保证 CIRKD 内部实现和 H3 诊断长期等价。建议统一到一个模块，并增加同一 logits/mask/config 下的逐元素等价测试、有限差分导数测试、constant-temperature 退化测试、全 ignore mask 测试和 NPU dtype 测试。

此外，`train_cirkdv2.py` 默认 CoVar 开启、默认预算 40k，而 `train_kd.py` 默认 CoVar 关闭；论文复现必须引用冻结 shell，且所有变体都显式传 `--use-covar` 或 `--no-covar`，不能依赖 argparse 默认值。

### 8.9 P1：断点与环境记录

- CIRKD training state 保存学生、蒸馏模块、optimizer、iteration 等，但不保存 Python/NumPy/Torch/NPU RNG；恢复并非 bit-exact。
- `train_kd.py` 已保存各 rank RNG，但 DataLoader worker/prefetch 游标仍不保存，断点恢复也不是 bit-exact。
- 两个历史主表行只保留恢复后的部分验证片段，指标可用，但 runtime、验证数和曲线不可用于公平比较。
- 最新报告记录的运行栈为 Python 3.11.10、PyTorch 2.8.0+cpu、torch_npu 2.8.0.post2、CANN 8.5.0；公共 shell 默认解释器路径名称仍是 `PyTorch-2.6.0`，README 又是 CUDA 12.4 配置。应从每次实际 Namespace/环境快照生成唯一环境清单，不以目录名推断版本。

### 8.10 P2：无效计算与命名清理

`get_covar_metadata()` 每步计算 `split_quality()` 并返回 `mask_high`，但主训练只用占位符接收，没有进入任何损失。它会增加 SVD/聚类式计算和训练开销。应移除、仅在诊断模式开启，或明确纳入方法。

## 9. 尚未完成，不能写成结果

### 9.1 Phase N：CWD 标量温度 80k 确认

状态：`已停止且结果不完整`。用户于 2026-07-13 约 10:36:45（Asia/Shanghai）暂停两路训练以重构方法；两路到达 20,000/80,000 时正处于验证中，均没有完整的 20k final 测量，因此不能形成性能结论，也不得自动恢复。

Phase N 回答的是“CWD 中 20k 的全局低温 endpoint 增益能否延续到 80k”，不回答“CoVar 空间分配是否优于 matched scalar”。

### 9.2 其他未完成项

- Cityscapes 或其他数据集的 cross-dataset validation；
- DeepLabV3-ResNet18 跨学生；
- 近期 KD 方法基线；
- weak-teacher 正式实验：已有一条 50/80k 日志出现 NaN，不能计为实验结果；
- 部分历史 full/calibration 计划找不到完整日志，不能凭脚本存在推定完成；
- smoke/debug 只证明代码可运行，不计性能实验。

## 10. Newton 审查时的下一步建议（历史，已被 O1.2 预注册取代）

1. **先修 masked KD 公平性。** 让所有 fixed/CoVar 路径使用同一 mask 和归一化，做单元测试和 20k 复核。
2. **在 CIRKD 内做 matched scalar + shuffled-map 对照。** 这是当前空间自适应主张的决定性实验。
3. **补正确的 `r_0 -> T` 诊断。** 同时记录 clamp/fallback/收敛统计。
4. **根据前三步决定论文主线。** 若 CoVar 仍优于 scalar/shuffle，保留空间自适应主张；若只与 scalar 持平，改写为可靠性启发的温度锐化/正则化分析；若更差，保留机制诊断与负结果，重做映射。
5. **再扩展数据集和学生。** 至少一个不同数据集、一个不同 backbone，并为关键比较使用 paired seeds。
6. **统一实现与环境。** 消除四份公式实现，固化运行 manifest、依赖和 checkpoint checksum。

## 11. Newton 历史证据允许与禁止的论文表述

### 可以说

- confidence–variance 评分与 VOC 教师错误显著相关；
- 当前 Newton 更新产生了随不可靠度变化的像素温度图；
- 在现有 CIRKD 实现和配方内，`Tout=3` 的三个 seed 都观察到 CoVar-on 正增益；
- 联合评分在 20k 消融中优于 confidence-only 和 variance-only；
- PSPNet 学生头的单 seed 结果为正；
- CoVar 约增加 10% CIRKD 训练时间，不增加推理模块。

### 现在不能说

- 已严格证明空间自适应优于合适的全局温度；
- 已严格证明当前 CIRKD on/off 差异只由温度造成；
- variance 单独能提升分割精度；
- 当前方法是 VOC 最强或 SOTA；
- 已证明跨数据集泛化或跨学生统计稳定性；
- Newton 已收敛到最优温度；
- Phase N 或其他未完整结束的实验已经支持某个结论。

## 12. 关键代码与证据索引

| 内容 | 文件 |
|---|---|
| CIRKD 主训练/内置 CoVar | `train_cirkdv2.py` |
| 复用版 Newton 与 masked KD | `utils/covar_temperature.py` |
| 原始 confidence/variance 统计 | `PCOS.py` |
| 通用 KD/CWD 与 CoVar 接入 | `train_kd.py` |
| O1.2 预算映射核心 | `utils/rtc_o12_calibration.py` |
| O1.2 正式 diagnose/checker | `scripts/diagnostics/diagnose_rtc_o12_budget.py`、`scripts/diagnostics/check_rtc_o12_gate.py` |
| O1.2 正式产物 | `runs/diagnostics/phaseO_o12/` |
| 普通未 masked KD | `losses/kd.py` |
| CWD | `losses/cwd.py` |
| CIRKD memory/mini-batch/channel | `losses/cirkd_memory.py`、`losses/cirkd_mini_batch.py`、`losses/cirkd_channel.py` |
| CIRKD 冻结 NPU 配方 | `scripts/experiments/covar_npu/common_voc_cirkdv2_npu.sh` |
| 当前 Newton 主变体 | `scripts/experiments/covar_npu/phaseC_lc_newton_gamma2_repro.sh` |
| CWD 冻结配方 | `scripts/experiments/kd_baselines_npu/common_voc_kd_npu.sh`、`run_phaseM_cwd_variant.sh` |
| 最新实验总记录 | `reports/2026-07-11_AAAI_paper_experiment_record.md` |
| 主表/三种子/跨学生/组件 | `reports/2026-07-02_phaseD_npu_main_table.md`、`2026-07-07_phaseE_tout3_seed_stability.md`、`2026-07-08_phaseF_psp_mbv3small_cross_student.md`、`2026-07-09_phaseG_component_ablation_triage.md` |
| 机制诊断 | `reports/2026-07-07_phaseH_h3_rt_distribution.md`、`runs/diagnostics/aaai_h1/`、`aaai_h2/`、`aaai_h3/` |
| CWD/标量温度反证 | `reports/2026-07-11_phaseL_cwd_seed_stability.md`、`2026-07-11_phaseM_cwd_covar_triage.md`、`2026-07-12_phaseM2_scalar_temperature.md` |
| Phase N 已停止的不完整记录 | `reports/2026-07-13_phaseN_scalar_temperature_80k_plan.md` |

## 13. Phase O1.2 当前方法与正式机制结果

### 13.1 当前方法

O1.2 不再使用 Newton 或 confidence+variance 风险。对冻结教师原始参考分布计算 c=max softmax(z_t)，采用 confidence-only 风险 r=-log(c)，再用冻结训练集 CDF 得到相对分位 u=F_train(r)。qR=0.6、qU=0.8 将像素分为轻锐化侧、中性区和高风险平滑侧：

~~~text
gR(u) = ((qR-u)/qR)^1,       u<qR
gR(u) = 0,                   u>=qR

gU(u) = ((u-qU)/(1-qU))^2,   u>qU
gU(u) = 0,                   u<=qU

log T(u) = -a*gR(u) + b*gU(u)
~~~

正式 train 预算求得 a*=0.1053605157、b*=0.3476499170。空间温度只生成 detached teacher target；学生 softmax 温度固定为 1，不乘 T 的幂。教师已有 Tout=3，因此实际教师 softmax 分母是 3*T；这里的“锐化”是相对 Tout=3 的中性目标锐化，不是相对 raw teacher 使用绝对低温。

### 13.2 正式结果

| 指标 | Train | Val |
|---|---:|---:|
| mean(T) | 0.995000 | 0.996092 |
| harmonic(T) | 0.988069 | 0.988161 |
| median(T) | 0.982425 | 0.979061 |
| top-risk decile mean(T) | 1.228580 | 1.233375 |
| T>1.25 覆盖率 | 3.955% | 4.643% |
| 高风险错误率 | 14.656% | 24.385% |
| 高风险错误 recall | 98.697% | 82.628% |
| 高风险错误富集 | 4.9748x | 3.9183x |

独立联合门禁给出 joint_gate_pass=true。所有方向、范围、中性区、置信度/熵、教师 argmax、学生固定温度、teacher-target-only、来源和有限值检查通过；train 求解与 train 复算缓存字节一致。可靠侧目标置信度上升且熵下降，高风险侧目标置信度下降且熵上升，中性区与学生分布的最大数值变化均为 0。

### 13.3 审查结论与限制

O1.2-A 已排除“新映射仍事实上等价于全局 T=0.6”这一主要数值隐患，但仍应保留以下限制：

- 约 60% 像素处于轻锐化侧，均值接近 1 不等于大多数像素没有变化；
- 高风险区 train/val 仍约有 85.34%/75.61% 教师预测正确，平滑可能同时削弱有用监督；
- 正温度缩放不改变教师 argmax，只降低高风险目标的 top-class 集中度，不能修正错误类别；
- mean(T) 与 harmonic(T) 预算不等价于保持平均 KL、梯度或监督强度；
- 当前只有 O1.2-B 的 20-step 学生链路 smoke，没有学生 validation/mIoU、matched scalar、within-image shuffle、跨数据集、跨教师或多 seed 证据；
- VOC-val 参与了方法设计，只能作为探索性机制验证，不能称为独立确认。

O1.2-A 与 O1.2-B 的证据边界必须分开：A 的联合门禁只证明教师目标机制，B 只证明两条学生训练链路可运行。获批的 B smoke 已完成，当前重新停在人工审查线；不得自动启动后续 smoke、20k、C2、C3、80k 或恢复 Phase N。

详细证据：

- [Phase O 主记录](2026-07-13_phaseO_rtc_method_reconstruction_plan.md)
- [O1.2 预注册与结果](2026-07-13_phaseO_rtc_o12_budgeted_routing_plan.md)
- [O1.2 执行记录](2026-07-13_phaseO_rtc_o12_execution_record.md)
- [O1.2 正式机制诊断](2026-07-13_phaseO_rtc_o12_diagnostic_report.md)
- [O1.2-B neutral 与 unreliable_only 20-step smoke 报告](2026-07-13_phaseO_rtc_o12b_smoke_report.md)

### 13.4 O1.2-B 学生链路 smoke

`neutral` 与 `unreliable_only` 的 fresh 运行各完成 20 个 optimizer step，独立 fail-closed 验收均为 `pass=true`。两路均产生 iteration-20 完整训练状态；随后各自的 `resume_audit` 从该终点成功加载，严格执行 0 个 optimizer step，并通过终点状态一致性验收。

本阶段只检查冻结配方下的学生训练、有限 loss、非零有限 KD 学生-logit 梯度、checkpoint 保存、样本顺序契约和终点恢复。fresh 使用 `--skip-val`，没有 validation 或 mIoU，因此不能比较 `neutral` 与 `unreliable_only` 的效果，不能推出高风险平滑有效、空间位置有因果贡献、方法有统计优势或能跨数据集泛化。后续任何 smoke、20k、C2、C3 或 80k 都需要再次明确授权，不能由本次通过自动接棒。

---

本页是审查快照，不替代原始日志。后续任何结论更新都应先核验：配置是否完全一致、运行是否达到冻结预算、final 验证是否存在、是否发生恢复/配置漂移，以及比较是否使用了相同的 mask、归一化和统计口径。
