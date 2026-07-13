# Phase O：可靠性目标置信度蒸馏方法重构与实验预注册

- 创建时间：2026-07-13（Asia/Shanghai）
- 工作名称：RTC-KD（Reliability-Targeted Confidence Distillation，可靠性目标置信度蒸馏）
- 当前状态：方法与实验计划 v1；尚未实现，尚未启动训练
- 上一阶段：Phase N 已按用户要求停止，不自动恢复
- 目的：把“可靠像素强锐化、不可靠像素平滑”写成方向受约束、可独立验证、可与标量温度公平比较的目标置信度方法。

## 0. 重构结论

新方法不直接对温度下的不可靠度进行最小化，而采用下面的因果链：

1. 在固定参考分布上计算教师初始不可靠度 r0；
2. 用冻结的训练集可靠性分布判断像素属于相对可靠侧还是不可靠侧；
3. 可靠侧设定更高的目标置信度，不可靠侧设定更低的目标置信度；
4. 目标置信度由可达温度端点构造，方向和力度均有明确含义；
5. 用无导数的单调二分反求像素温度；
6. 用统一 valid mask 的像素 KD 训练学生；
7. 分别通过 reliable-only 和 unreliable-only 验证两条分支，再用 matched scalar 和 shuffled map 判断空间分配是否真的有效。

第一版主配置使用：

| 项目 | 预注册值 |
|---|---:|
| 可靠性评估温度 Tassess | 1.0，使用原始教师 logits |
| 路由分位点 q | 0.80 |
| 路由过渡宽度 w | 0.05 |
| 强锐化端点 TR | 0.5 |
| 中性端点 T0 | 1.0 |
| 平滑端点 TU | 2.0 |
| 可靠分支力度 alphaR | 1.0 |
| 不可靠分支力度 alphaU | 1.0 |
| 二分次数 | 16 |
| KD 温度幂 gamma | 0 |
| 第一开发底座 | CWD，Tout=3.0，VOC，20k |
| 第二验证底座 | CIRKD，VOC |

这些是首轮冻结值，不得根据运行中的 mIoU 临时修改。若数值诊断显示不可达、NaN 或严重饱和，先登记偏差，再按预注册的安全回退值 alpha=0.5 重跑受影响分支。

## 1. 为什么停止当前方向

### 1.1 当前 Newton 的主要问题

当前 Newton 方法优化的是随温度变化的 r(T)。降低温度会直接提高最大置信度并降低 r(T)，因此求解器天然倾向于大面积锐化。H3 显示：

- 温度中位数为 0.5；
- 90.35% 像素落在 Tmin=0.5；
- 只有 5.96% 像素高于 1.25。

这说明结果中存在很强的全局低温解释，而不是只有“不可靠区域被针对性平滑”。

### 1.2 匹配标量温度已经形成强反证

CWD、Tout=3、20k、seed 1234 下：

| 方法 | Final mIoU |
|---|---:|
| scalar T=1.0 | 0.642000 |
| Newton CoVar | 0.646000 |
| scalar T=0.5 | 0.648235 |
| scalar T=0.6 | 0.653381 |

Newton CoVar 比 scalar T=0.6 低 0.007381。因此，只与 T=1 比较不能证明像素自适应有效。

### 1.3 旧 calib_conf 的经验教训

旧 centered calibration 使用：

$$
c_i^*
=
c_i^0\exp\left[-\alpha(r_i^0-r_{\mathrm{anchor}})\right].
$$

主要问题是：

- anchor 附近目标变化趋近于零，天然形成 identity zone；
- adaptive_mc 使用 batch 内平均置信度决定分位点，分流比例会被 Tout 改变，但教师 argmax 正误并没有随正温缩放改变；
- batch/rank 内分位数会随 batch 内容抖动；
- B5 把参考温度降到 0.6 后 best mIoU 达到 0.636，但最终 Tmean=0.5833、q50=q90=0.6、identity=78.58%、smooth active 仅 0.79%，更像全局 T约0.6，而不是双向机制；
- 旧 centered gamma0 比 gamma1 高 0.017 best，说明 T 的损失缩放会明显改变机制。

新方法保留“先定目标置信度，再反求温度”这一核心原则，但重新定义可靠性路由、目标置信度和求解方式，不沿用 batch adaptive anchor、移动参考温度或旧 centered 指数目标。

## 2. 固定参考可靠性

### 2.1 参考教师分布

教师原始 logits 为 z_i^t。可靠性评估始终使用固定参考温度：

$$
\pi_{ik}^0
=
\operatorname{softmax}\left(\frac{z_{ik}^t}{T_{\mathrm{assess}}}\right),
\qquad
T_{\mathrm{assess}}=1.
$$

这里不使用 teacher_output_temp。这样，后续 KD 是否采用 Tout=1 或 Tout=3，不会改变可靠/不可靠路由。

定义：

$$
c_i^{\mathrm{assess}}=\max_k\pi_{ik}^0,
$$

$$
v_i^0
=
\frac{1}{K-1}
\sum_{k\ne k_i^*}
\left(
\pi_{ik}^0-\frac{1-c_i^{\mathrm{assess}}}{K-1}
\right)^2,
$$

$$
r_i^0
=
-\log c_i^{\mathrm{assess}}
+
a\frac{v_i^0}{1-c_i^{\mathrm{assess}}+\epsilon},
\qquad
a=\frac{(K-1)^2}{2}.
$$

VOC 中 K=21，故 a=200。r0 全程 detach，不因目标温度或学生状态改变。这里的 c_assess 属于 raw-teacher 可靠性分布；第 4 节的 ell0 属于实际 KD logits 在 T0=1 下的中性目标，Tout=3 时二者不是同一个分布。

### 2.2 现有 H1 对路由的支持边界

完整 VOC val 的十等分结果中：

- bottom 80% 的最高单 bin 教师错误率为 5.24%；
- 第 9 个 bin 错误率为 9.72%；
- 最高 bin 错误率为 29.56%；
- top 20% 覆盖约 77.79% 的全部教师错误；
- top 20% 内教师错误精度约 19.64%。

所以 q=0.80 是一个有机制依据的首轮分流点：高 r 的 top 20% 集中了大部分教师错误。不过，最高 r 的 10% 中仍有约 70.44% 像素预测正确，因此“不可靠”只能理解为高风险，不等于教师一定错误。

上述 q 已参考 VOC-val 诊断。为避免继续使用验证标签调参，后续不再按学生 mIoU 修改 q；正式跨数据集实验必须从该数据集训练集的教师输出重新建立 CDF，并报告阈值迁移规则。

## 3. 冻结训练集 CDF 与连续双向路由

### 3.1 全局 CDF

先用冻结教师扫描训练集有效像素，建立 r0 的经验累积分布：

$$
u_i=\widehat F_{\mathrm{train}}(r_i^0)\in[0,1].
$$

CDF 必须：

- 在学生训练前一次性生成并冻结；
- 保存训练列表 checksum、教师权重 checksum、Tassess、a、采样方法和分位点表；
- 不使用每 batch 分位数；
- 不使用每个 DDP rank 各自的局部分位数；
- 在验证集上只查询冻结 CDF，不重新拟合。

### 3.2 连续路由

定义：

$$
d_i
=
\tanh\left(\frac{q-u_i}{w}\right),
\qquad
q=0.80,\quad w=0.05,
$$

$$
g_i^R=\max(d_i,0),
\qquad
g_i^U=\max(-d_i,0).
$$

性质：

- u<q 时 gR>0，只允许锐化；
- u>q 时 gU>0，只允许平滑；
- 远离边界时 gate 接近 1，执行大力度操作；
- q 附近只有窄过渡区，不会出现旧 centered calibration 的大面积非预期 identity zone；
- gR 和 gU 互斥，便于独立消融。

在报告中仍可把 gR>0 称为可靠侧、gU>0 称为不可靠侧，但必须同时报告 gate 强度分布，不能把连续风险路由写成无误的二分类器。

## 4. 由可达端点定义目标置信度

### 4.1 实际 KD 教师 logits

进入 logit KD 的教师 logits 为：

$$
\widetilde z_i^t
=
\frac{z_i^t}{\tau_{\mathrm{out}}}.
$$

第一开发底座沿用 CWD 的 Tout=3.0，以直接面对 scalar T=0.6 的强替代解释。可靠性路由仍由未经过 Tout 软化的原始 logits 决定。

对任意像素温度 T：

$$
c_i(T)
=
\max_k
\operatorname{softmax}
\left(
\frac{\widetilde z_i^t}{T}
\right)_k.
$$

### 4.2 稳定的 top-vs-rest log-odds

为避免 c 接近 1 时的数值饱和，目标使用与最大置信度一一对应的 top-vs-rest log-odds：

$$
\ell_i(T)
=
\log\frac{c_i(T)}{1-c_i(T)}
=
\frac{\widetilde z_{i,k_i^*}^t}{T}
-
\log
\sum_{k\ne k_i^*}
\exp\left(\frac{\widetilde z_{ik}^t}{T}\right).
$$

计算三个端点：

$$
\ell_i^R=\ell_i(T_R),\quad T_R=0.5,
$$

$$
\ell_i^0=\ell_i(T_0),\quad T_0=1.0,
$$

$$
\ell_i^U=\ell_i(T_U),\quad T_U=2.0.
$$

### 4.3 双向目标

目标 log-odds 定义为：

$$
\ell_i^*
=
\ell_i^0
+
\alpha_R g_i^R(\ell_i^R-\ell_i^0)
+
\alpha_U g_i^U(\ell_i^U-\ell_i^0),
$$

第一版取：

$$
\alpha_R=\alpha_U=1.
$$

目标置信度为：

$$
c_i^*=\sigma(\ell_i^*).
$$

该定义保证：

- 可靠侧：ell0 <= ell* <= ellR，因此 c*>=c0，必定锐化；
- 不可靠侧：ellU <= ell* <= ell0，因此 c*<=c0，必定平滑；
- gR 接近 1 时，目标接近 TR=0.5 所能达到的强锐化；
- gU 接近 1 时，目标接近 TU=2.0 所能达到的平滑；
- alpha 表示向对应可达端点移动的比例，含义明确；
- 目标天然位于可达区间，无需把大量不可达目标粗暴夹到边界；
- 不移动全局参考温度，避免把方法退化为整体 T约0.6。

候选 exp(-beta*r0) 只保留为目标构造消融，不作为主公式。其可靠分支不能保证 beta<1 时一定锐化，容易重新形成 identity zone。

## 5. 单调二分反求像素温度

对于无最大 logit 并列的像素：

$$
\frac{\partial\ell_i(T)}{\partial T}
=
\frac{
\mathbb E_{\mathrm{nonmax},T}[\widetilde z_i^t]
-
\widetilde z_{i,k_i^*}^t
}{T^2}
<0.
$$

所以 ell(T) 关于 T 单调递减，每个可达目标对应唯一温度。

求解区间：

- 可靠侧：T_i in [TR,T0]；
- 不可靠侧：T_i in [T0,TU]；
- gate 为 0：T_i=T0。

使用固定 16 步纯二分，不使用导数更新：

1. 初始化对应方向的左右端点；
2. 计算中点温度与 ell(Tmid)；
3. 根据单调性缩小区间；
4. 16 步后取区间中点；
5. 无效像素、最大 logit 并列、NaN/Inf 回退 T0=1。

必须记录：

- abs(ell(T)-ell*) 的 mean/p95/max；
- reliable/unreliable gate 覆盖率和均值；
- T 的 mean、harmonic mean、q10/q50/q90/p95；
- T=TR、T=T0、T=TU 的比例；
- 有效教师温度 Tout*T 的分布；
- 回退、NaN、Inf 和 top-logit tie 数量。

方法核心是“目标置信度校准”；二分只是稳定的单调反求工具，不作为论文创新点。

## 6. KD 损失与 gamma=0

教师和学生分布为：

$$
p_i^t
=
\operatorname{softmax}
\left(
\frac{z_i^t}{\tau_{\mathrm{out}}T_i}
\right),
$$

$$
p_i^s
=
\operatorname{softmax}
\left(
\frac{z_i^s}{T_i}
\right).
$$

第一版主损失：

$$
\mathcal L_{\mathrm{RTC}}
=
\frac{1}{\sum_iM_i}
\sum_iM_i
D_{\mathrm{KL}}(p_i^t\Vert p_i^s),
$$

即 gamma=0，不乘 T 的幂。

一般形式的学生 logit 梯度尺度近似为：

$$
\frac{\partial\mathcal L_i}{\partial z_i^s}
\propto
T_i^{\gamma-1}(p_i^s-p_i^t).
$$

因此：

- gamma=0：低温可靠侧自然增强，高温不可靠侧自然衰减；
- gamma=1：近似抵消一阶温度尺度；
- gamma=2：低温分支被压低，高温分支被补偿，与当前“可靠增强、不可靠削弱”的语义不完全一致。

gamma=0 是新机制的首轮冻结设置，不代表它对所有温度方法普遍最佳。Full RTC 通过核心门槛后，再做 gamma in {0,1,2} 的独立消融。

## 7. 公平性前置修复

任何新实验启动前，必须先解决现有 valid-mask 混杂：

- scalar T=1；
- scalar T=0.6；
- reliable-only；
- unreliable-only；
- full RTC；
- shuffled/reverse；
- 历史 Newton 对照；

全部调用同一个 masked pixel KD：

$$
\frac{1}{\sum_iM_i}
\sum_iM_iT_i^\gamma
D_{\mathrm{KL}}(p_i^t\Vert p_i^s).
$$

fixed scalar 只是在同一函数中令 T_i 恒等于标量。所有变体必须共享：

- valid mask；
- padding/ignore 处理；
- loss reduction；
- teacher-output softening；
- gamma；
- 数据顺序与初始化；
- seed、schedule、验证频率和训练预算。

现有普通 CriterionKD 未排除 padding/ignore 位置，不能继续作为严格的 fixed control。

严格的空间机制比较统一使用 gamma=0。scalar T=0.6 gamma=2 和旧 Newton gamma=2 只作为明确标注的损失缩放/历史方法对照，不参与空间分配的单因素归因。

## 8. 分支消融的定义

核心 2x2 因子实验：

| 变体 | 可靠侧 | 不可靠侧 | 目的 |
|---|---|---|---|
| Neutral | T=1 | T=1 | masked 公平基线 |
| Reliable-only | 目标锐化 | T=1 | 验证可靠像素强锐化 |
| Unreliable-only | T=1 | 目标平滑 | 验证不可靠像素抑噪 |
| Full RTC | 目标锐化 | 目标平滑 | 验证两分支联合效果 |

边际量：

$$
\Delta_R^{\mathrm{direct}}
=
\mathrm{ReliableOnly}-\mathrm{Neutral},
$$

$$
\Delta_U^{\mathrm{direct}}
=
\mathrm{UnreliableOnly}-\mathrm{Neutral},
$$

$$
\Delta_R^{\mathrm{full}}
=
\mathrm{Full}-\mathrm{UnreliableOnly},
$$

$$
\Delta_U^{\mathrm{full}}
=
\mathrm{Full}-\mathrm{ReliableOnly}.
$$

由于训练存在交互，Full 的两个边际量不能假设严格可加。

机制对照：

| 变体 | 回答的问题 |
|---|---|
| scalar T=0.5/0.6/1.0，gamma=0 | 是否只是全局低温 |
| scalar T=0.6，gamma=2 | 与 Phase M2 历史强对照对齐 |
| matched arithmetic-mean T | 是否只是平均温度 |
| matched inverse-mean T | 是否只是 gamma=0 下的平均梯度尺度 |
| per-image shuffled map | 保持直方图后，正确空间位置是否重要 |
| reverse routing | 方向反转是否按机制显著变差 |
| exp(-beta*r0) target | 目标构造方式是否重要 |
| GT oracle | 可靠性判断的机制上界，仅作诊断 |

shuffle 只在每张图的 valid pixels 内进行，保持温度直方图、有效像素数和损失尺度，不得把 invalid/padding 温度混入。

## 9. 分组指标

全局指标继续报告 best/final mIoU、last-window mean、pixAcc 和 runtime，但不能只看全局 mIoU。

冻结可靠侧 R 与不可靠侧 U 后，分别报告：

- 覆盖率和 gate 强度；
- 教师 pixel accuracy、错误精度、错误召回和 AUPRC；
- 学生 pixel accuracy、NLL；
- restricted macro-IoU，并明确缺失类别处理；
- P(student correct | teacher correct, R)；
- P(student correct | teacher wrong, U)；
- 错误模仿率：

$$
P(\widehat y_s=\widehat y_t\ne y\mid U);
$$

- R/U 的平均 KD loss 与学生 logit 梯度范数；
- c0、c*、T、目标残差分布；
- 边界像素与非边界像素的分层结果，作为次要机制分析。

可靠分支成立的最低条件：

- Reliable-only 在 R 上提高 student accuracy 或降低 NLL；
- 不得让 U 上 student accuracy 下降超过预注册容忍度；
- 全局 final/late-window 不出现明显反向。

不可靠分支成立的最低条件：

- Unreliable-only 提高 P(student correct | teacher wrong,U)；
- 降低错误模仿率；
- 全局 final/late-window 不出现明显反向。

GT oracle 中，教师正确像素锐化、教师错误像素平滑。若 oracle 也不能优于强标量温度，说明问题可能在“锐化/平滑”假设本身，而不只是可靠性评分。oracle 不得进入主结果表。

## 10. 实现计划

### 10.1 单一实现源

建议新增独立模块 utils/rtc_temperature.py，避免继续复制公式。模块职责：

1. compute_reference_reliability；
2. load_or_query_frozen_cdf；
3. compute_reliability_gates；
4. compute_top_vs_rest_log_odds；
5. build_target_log_odds；
6. invert_target_by_bisection；
7. masked_temperature_kd_loss；
8. collect_rtc_diagnostics。

train_cirkdv2.py 和 train_kd.py 只能调用该模块，不再各自实现一份公式。H1/H3/可视化脚本也复用同一实现。

### 10.2 建议 CLI

建议使用新的模式名 rtc，不复用旧 calib_conf 的语义：

- --covar-temp-mode rtc
- --rtc-cdf-path
- --rtc-route-quantile 0.80
- --rtc-route-width 0.05
- --rtc-temp-reliable 0.5
- --rtc-temp-neutral 1.0
- --rtc-temp-unreliable 2.0
- --rtc-alpha-reliable 1.0
- --rtc-alpha-unreliable 1.0
- --rtc-enable-reliable
- --rtc-enable-unreliable
- --rtc-bisection-iters 16
- --covar-kd-temp-power 0.0

所有实验 shell 必须显式给出 on/off 和参数，不能依赖 argparse 默认值。

### 10.3 单元测试门槛

实现完成后必须通过：

- c(T) 与 ell(T) 单调性；
- reliable 分支 T<=1，不可靠分支 T>=1；
- gate 互斥；
- 目标位于端点区间；
- 二分残差满足容差；
- alpha=0 退化为 T=1；
- reliable-only/unreliable-only 精确关闭另一分支；
- constant T 与 masked scalar KD 数值等价；
- 全 ignore mask 返回有限零损失；
- shuffled map 保持 valid 温度多重集合；
- DDP rank 使用同一冻结 CDF；
- float32/NPU 下无 NaN/Inf；
- 旧 Newton/RTC 日志标签不会混淆。

## 11. 分阶段实验计划

### Phase O0：停止与快照

状态：已完成。

- Phase N 控制器和两路 worker 已停止；
- 两张 NPU 无运行进程；
- 两路训练达到 20000/80000，但 20k 验证中断；
- 不自动恢复 Phase N；
- 保留日志与 checkpoint，仅作历史记录。

### Phase O1：教师路由诊断，不训练学生

目标：冻结训练集 CDF，并确认 q=0.80 的路由具有足够错误富集。

输出：

- train/val 的 r0 CDF 和十等分表；
- q=0.80 下的 R/U 覆盖率；
- teacher error precision、recall、AUPRC；
- confidence-only、variance-only、full r0 的路由质量；
- raw teacher 与 Tout=3 KD logits 下路由一致性；
- 固定 CDF 文件与 checksum。

晋级条件：

- U 组 teacher error precision 至少为全局错误率的 2 倍；
- U 组覆盖至少 10%；
- U 组覆盖至少 60% 的教师错误；
- train 与 val 的方向一致；
- full r0 不得明显弱于 confidence-only，否则先重新审查方差项。

未满足则停止性能训练，先重构可靠性评分。

### Phase O2：代码正确性和 20-iteration smoke

矩阵：

- Neutral；
- Reliable-only；
- Unreliable-only；
- Full RTC；
- scalar T=0.6 gamma0；
- shuffled map。

要求：

- 每个变体 20/20 完整退出；
- Namespace、mask、gamma、Tout 与预注册一致；
- 无 NaN/Inf；
- 温度方向断言通过；
- 目标残差和回退率达标；
- checkpoint/training state 完整。

### Phase O3：CWD 20k 核心 2x2

公共配置：

- VOC；
- DeepLabV3-R101 到 DeepLabV3-MobileNetV3-Small；
- CWD 官方配方；
- Tout=3.0；
- seed 1234；
- 20k；
- gamma=0；
- q=0.80，w=0.05；
- TR/T0/TU=0.5/1.0/2.0；
- alphaR=alphaU=1；
- 同一 masked KD。

变体：

1. Neutral；
2. Reliable-only；
3. Unreliable-only；
4. Full RTC；
5. scalar T=0.6 gamma0；
6. scalar T=0.6 gamma2。

主指标：final mIoU。次指标：best、last-10 mean 和全部分组机制指标。

20k 是筛选和因果诊断，不作为统计优越性结论。

分支晋级规则：

- Reliable-only 必须在 R 组机制指标上优于 Neutral，且相对 Neutral 的全局 final 降幅不得超过 0.002；
- Unreliable-only 必须提高 U 组 teacher-wrong 条件下学生正确率或降低错误模仿率，且相对 Neutral 的全局 final 降幅不得超过 0.002；
- Full 必须比 Neutral 的 final 至少高 0.002，且 last-10 delta 不为负；
- 若只有一个分支成立，下一阶段只保留成立分支，不强行维持双向叙事；
- 若两分支都不成立，停止性能扩展。

### Phase O4：空间因果对照 20k

只有 Full 通过 O3 才运行：

1. Full RTC；
2. strongest scalar in {0.5,0.6,1.0}，同 gamma；
3. matched arithmetic-mean scalar；
4. matched inverse-mean scalar；
5. per-image shuffled map；
6. reverse routing；
7. exp(-beta*r0) target。

主张空间分配成立的最低门槛：

- Full - strongest scalar final >= 0.002；
- Full - shuffled final >= 0.002；
- 两项 last-10 delta 均不为负；
- Full 的 R/U 机制指标方向正确；
- reverse routing 明显弱于 Full。

若 Full 只与 scalar 持平，结论降级为“可靠性校准可达到合适全局温度的性能”，不能主张空间自适应独立增益。

### Phase O5：CIRKD 20k 迁移

在修复 masked KD 后，重复：

- Neutral；
- Reliable-only；
- Unreliable-only；
- Full；
- strongest scalar；
- shuffled map。

不得直接复用旧 on/off 结果，因为旧 fixed KD 的 mask/reduction 不一致。

### Phase O6：80k seed 1234 主确认

只有 CWD 或 CIRKD 至少一个底座通过 O4 才运行。每个晋级底座包括：

- Neutral；
- Reliable-only；
- Unreliable-only；
- Full；
- strongest scalar/matched scalar。

主指标：

- iteration-80000 final mIoU；
- final paired delta；
- last-10 mean；
- 分组机制指标。

80k 晋级规则：

- Full - strongest scalar final >= 0.002；
- Full - strongest scalar last-10 >= 0；
- Full - Neutral final >= 0.002；
- 运行完整、配置一致、无恢复混杂。

### Phase O7：三种子

保留 seed 1234，新增 2025、3407。核心只跑 Full vs strongest scalar 的 paired comparison，必要时保留 Neutral。

报告：

- seed-wise paired delta；
- mean；
- sample SD，ddof=1；
- 置信区间；
- final、best、last-10；
- 不隐藏负 seed。

只有三种子 paired delta 整体支持，才进入论文主张。

### Phase O8：泛化

优先顺序：

1. PSPNet-MobileNetV3-Small；
2. 不同 backbone 的学生；
3. 至少一个不同数据集；
4. CWD 与 CIRKD 两个底座；
5. Tout=1 与 Tout=3 的解耦检查。

跨数据集不得重新查看 val 结果后修改 q、w、端点或 alpha。若必须重新建立训练集 CDF，只允许使用教师输出和冻结规则。

## 12. 失败解释与止损规则

### 12.1 高置信错误

可靠侧仍可能包含高置信错误，强锐化会放大它们。必须报告高置信错误在 R 组的比例，并按 teacher-correct/teacher-wrong 分层。

### 12.2 高风险不等于错误

U 组 top 20% 中多数像素仍可能正确。平滑过强会浪费有用监督，因此 TU=2 先于更激进的 4 或 8；只有 U 分支机制指标支持，才考虑 TU 消融。

### 12.3 只控制最大置信度

目标只校准 top-vs-rest confidence，不能保证非主类别内部结构最优。variance 项目前用于路由，不直接规定每个非主类目标。

### 12.4 双重温度

Tout 与 T 同时作用于教师。所有日志必须报告有效教师温度 Tout*T，论文不能只写像素 T。

### 12.5 分位路由的跨域风险

CDF 路由固定相对比例，不能保证不同教师和数据集上具有同一错误概率。跨域必须重新报告路由 precision/recall，而不能只复用 q=0.80 的语义。

### 12.6 禁止性结论

在 O4/O7 完成前不得声称：

- RTC 优于合适的全局标量温度；
- 正确的空间位置已经被证明有用；
- 可靠与不可靠两条分支都独立有效；
- r0 可以准确判断教师对错；
- 方法跨数据集稳定。

## 13. 结果回填模板

### 13.1 全局性能

| Variant | Budget | Seed | Best | Final | Last-10 | Runtime | Complete |
|---|---:|---:|---:|---:|---:|---|---|
| Neutral | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Reliable-only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Unreliable-only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Full RTC | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Strongest scalar | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Shuffled | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

### 13.2 分组机制

| Variant | R coverage | U coverage | Acc R | Acc U | Student correct given teacher wrong U | Error imitation U | T mean R/U | Target error |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Neutral | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Reliable-only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Unreliable-only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Full RTC | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 14. 当前执行决定

- Phase N 保持 stopped，不自动恢复；
- 暂不启动任何新训练；
- 下一步先实现共享 masked KD、RTC 单一模块和教师训练集 CDF 诊断；
- 实现完成后先跑单元测试与 20-iteration smoke；
- 只有文档中的 O1/O2 门槛通过，才启动 20k 核心矩阵；
- 所有参数变化、失败和中断必须先写入本页，再启动新运行。
