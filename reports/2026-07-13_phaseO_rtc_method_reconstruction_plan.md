# Phase O：RTC-KD 方法重构与实验预注册

- 创建日期：2026-07-13
- 方法名称：可靠性目标置信度蒸馏（Reliability-Targeted Confidence Distillation，RTC-KD）
- 当前状态：方案草案 v2，尚未实现，尚未启动新训练
- 上一阶段：Phase N 已停止，不自动恢复
- 文档格式：为兼容不同 Markdown 渲染器，公式统一写成纯文本代码块

## 0. 一页结论

RTC-KD 不再直接最小化随温度变化的不可靠度，而采用以下流程：

1. 在固定参考分布上计算教师初始不可靠度 `r0`。
2. 使用冻结教师在训练集上的 `r0` 排名，得到相对低风险侧和相对高风险侧。
3. 低风险侧提高目标置信度，执行强锐化。
4. 高风险侧降低目标置信度，执行平滑。
5. 用单调二分反求达到目标置信度所需的像素温度。
6. 用统一有效像素掩码的 KD 损失训练学生。
7. 分别验证可靠侧锐化和不可靠侧平滑，再与标量温度和打乱温度图比较。

首轮预注册配置如下：

| 项目 | 符号/参数 | 初值 |
|---|---|---:|
| 可靠性评估温度 | `T_assess` | 1.0 |
| 高风险起始分位点 | `q` | 0.80 |
| 路由过渡宽度 | `w` | 0.05 |
| 强锐化温度端点 | `T_R` | 0.5 |
| 中性温度 | `T_0` | 1.0 |
| 平滑温度端点 | `T_U` | 2.0 |
| 可靠侧目标力度 | `alpha_R` | 1.0 |
| 不可靠侧目标力度 | `alpha_U` | 1.0 |
| 二分次数 | `n_bisect` | 16 |
| KD 温度幂 | `gamma` | 0 |
| 第一开发底座 | — | CWD，`T_out=3.0`，VOC，20k |
| 第二验证底座 | — | CIRKD，VOC |

这些参数在首轮实验中冻结。不得根据运行中的 mIoU 临时修改。

## 1. 为什么需要重构

### 1.1 当前 Newton 方法的解释问题

当前 Newton 方法优化的是随温度变化的 `r(T)`。降低温度会直接提高最大置信度，也往往会降低 `r(T)`，因此求解器容易把大量像素推向低温。

H3 的实际分布为：

- 温度中位数：0.5；
- 落在 `T_min=0.5` 的像素：90.35%；
- 温度高于 1.25 的像素：5.96%。

这说明当前结果中存在很强的“整体低温锐化”解释。

### 1.2 匹配标量温度是强替代解释

CWD、`T_out=3.0`、20k、seed 1234 的结果为：

| 方法 | Final mIoU |
|---|---:|
| 标量 `T=1.0` | 0.642000 |
| Newton CoVar | 0.646000 |
| 标量 `T=0.5` | 0.648235 |
| 标量 `T=0.6` | 0.653381 |

标量 `T=0.6` 比 Newton CoVar 高 0.007381。因此，只证明自适应方法优于 `T=1.0`，不能证明空间分配本身有效。

### 1.3 旧 calib_conf 的主要教训

旧 centered calibration 的目标可简写为：

```text
c_target = c0 * exp[-alpha * (r0 - r_anchor)]
```

历史实验暴露出四个问题：

1. `r0` 接近 anchor 时，目标变化很小，容易形成大 identity zone。
2. `adaptive_mc` 根据 batch 平均置信度决定分流比例，比例会受 `T_out` 影响。
3. 每个 batch、每个 DDP rank 单独计算分位数，会引入批次抖动。
4. B5 的提升主要来自把全局参考温度移到 0.6，而不是高风险像素被充分平滑。

B5 的最终统计为：

- `T_mean=0.5833`；
- `T_q50=T_q90=0.6`；
- identity 比例 78.58%；
- smooth-active 比例仅 0.79%。

因此，新方案保留“先设目标置信度，再反求温度”的原则，但不沿用旧 anchor、移动参考温度或旧指数目标。

## 2. 符号表

| 符号 | 含义 |
|---|---|
| `z_t[i,k]` | 像素 `i`、类别 `k` 的教师原始 logit |
| `z_s[i,k]` | 学生 logit |
| `K` | 类别数；VOC 中为 21 |
| `c_assess[i]` | 原始教师参考分布的最大置信度 |
| `v0[i]` | 非主类别残差概率方差 |
| `r0[i]` | 固定参考分布上的教师不可靠度 |
| `u[i]` | `r0` 在冻结训练集 CDF 中的分位位置 |
| `gate_R[i]` | 低风险侧锐化强度 |
| `gate_U[i]` | 高风险侧平滑强度 |
| `T_R / T_0 / T_U` | 锐化、中性、平滑温度端点 |
| `T_out` | 仅作用于教师 KD logits 的全局输出温度 |
| `T[i]` | 反求得到的像素温度 |
| `M[i]` | 非 ignore 有效像素掩码 |

本文中的“可靠侧”和“不可靠侧”均是相对风险简称，不代表教师一定正确或一定错误。

## 3. 固定参考可靠性

### 3.1 参考教师分布

可靠性始终从原始教师 logits 计算，不使用 `T_out`：

```text
p_assess[i,k] = softmax(z_t[i,k] / T_assess)
T_assess       = 1.0
c_assess[i]    = max_k p_assess[i,k]
```

非主类别均值和方差：

```text
mu0[i] = (1 - c_assess[i]) / (K - 1)

v0[i]  = mean over k != top_class of:
         (p_assess[i,k] - mu0[i])^2
```

不可靠度：

```text
a     = (K - 1)^2 / 2
r0[i] = -log(c_assess[i])
        + a * v0[i] / (1 - c_assess[i] + eps)
```

VOC 中 `K=21`，因此 `a=200`。

`r0` 在计算后立即 detach；它不随目标温度、学生状态或训练 iteration 改变。

### 3.2 为什么与 T_out 解耦

`T_out` 只改变教师 KD 分布的软硬程度，不改变正温度下的 argmax 类别。如果用 `T_out` 之后的置信度重新判断可靠性，同一个像素可能仅因全局 softening 就被分到不同风险侧。

因此：

- 风险判断使用原始教师 logits；
- 目标置信度使用实际进入 KD 的教师 logits；
- 两条路径在代码和日志中必须分别命名。

## 4. 冻结训练集 CDF 与风险路由

### 4.1 训练集 CDF

先用冻结教师扫描训练集有效像素，建立 `r0` 的经验累积分布：

```text
u[i] = F_train(r0[i])
```

其中 `u[i]` 位于 0 到 1 之间，数值越大表示该像素在训练集里处于越高风险的尾部。

CDF 文件必须在学生训练前生成并冻结，并记录：

- 训练列表 checksum；
- 教师权重 checksum；
- `T_assess` 和 `a`；
- 像素采样方法；
- CDF 分位点表；
- 生成脚本版本。

禁止以下做法：

- 每个 batch 重算分位数；
- 每个 DDP rank 使用不同 CDF；
- 在验证集上重新拟合 CDF；
- 根据运行中的 mIoU 修改 CDF。

### 4.2 连续双向路由

路由公式：

```text
d[i]      = tanh((q - u[i]) / w)
gate_R[i] = max(d[i], 0)
gate_U[i] = max(-d[i], 0)

q = 0.80
w = 0.05
```

解释：

- `u[i] < q`：低风险侧，只允许锐化；
- `u[i] > q`：高风险侧，只允许平滑；
- 离 `q` 越远，操作力度越大；
- `q` 附近形成窄过渡区；
- `gate_R` 和 `gate_U` 互斥。

### 4.3 q=0.80 的现有依据

VOC-val 的十等分结果显示：

- bottom 80% 中最高单 bin 教师错误率为 5.24%；
- 第 9 个 bin 错误率为 9.72%；
- 最高 bin 错误率为 29.56%；
- top 20% 覆盖约 77.79% 的全部教师错误；
- top 20% 内教师错误精度约 19.64%。

这说明 top 20% 是错误富集区，但不是“教师错误集合”。最高风险的 10% 中仍约有 70.44% 像素预测正确。

### 4.4 泛化性与审稿边界

冻结训练集 CDF 有数值尺度稳定的优点，但它定义的是相对风险，不是绝对错误概率。

换数据集后，以下因素都会改变 `r0` 的含义：

- 教师整体强弱和校准程度；
- 类别数与类别比例；
- 背景、边界和小目标比例；
- 数据集难度与域偏移。

主协议定义为：

1. 在每个新数据集上，只用该数据集训练图像的教师输出重新建立 CDF；
2. `q`、`w`、温度端点和目标力度保持不变；
3. 不使用目标数据集验证标签调参；
4. 重新报告高风险侧的错误 precision、recall 和 AUPRC。

这种做法属于“固定训练流程下的无标签统计适配”，不是直接复用 VOC 统计量的 zero-shot 泛化。

论文中应使用“相对低风险侧/相对高风险侧”，避免声称方法准确判断教师对错。

## 5. 目标置信度构造

### 5.1 实际进入 KD 的教师分布

实际 KD 教师 logits：

```text
z_kd[i,k] = z_t[i,k] / T_out
```

第一开发设置沿用 CWD 的 `T_out=3.0`，因为标量 `T=0.6` 的强替代解释出现在这一设置中。

在像素温度 `T` 下：

```text
c_kd(i, T) = max_k softmax(z_kd[i,k] / T)
```

### 5.2 使用 top-vs-rest log-odds

为避免最大置信度接近 1 时的数值饱和，使用 top-vs-rest log-odds：

```text
log_odds(i, T)
    = log(c_kd(i,T) / (1 - c_kd(i,T)))
    = z_kd[i,top] / T
      - logsumexp(z_kd[i,non_top] / T)
```

三个温度端点：

```text
L_R[i] = log_odds(i, T_R),   T_R = 0.5
L_0[i] = log_odds(i, T_0),   T_0 = 1.0
L_U[i] = log_odds(i, T_U),   T_U = 2.0
```

### 5.3 双向目标

目标 log-odds：

```text
L_target[i]
    = L_0[i]
      + alpha_R * gate_R[i] * (L_R[i] - L_0[i])
      + alpha_U * gate_U[i] * (L_U[i] - L_0[i])

alpha_R = 1.0
alpha_U = 1.0
```

目标置信度：

```text
c_target[i] = sigmoid(L_target[i])
```

方向保证：

- 低风险侧的目标位于 `L_0` 与 `L_R` 之间，只会锐化；
- 高风险侧的目标位于 `L_U` 与 `L_0` 之间，只会平滑；
- gate 接近 1 时，目标接近对应温度端点；
- `alpha_R` 和 `alpha_U` 表示向端点移动的比例；
- 不移动全局参考温度，不把方法退化成整体 `T≈0.6`。

候选目标 `exp(-beta*r0)` 仅作为消融。它在可靠侧不能保证一定锐化，不作为主公式。

## 6. 单调二分反求温度

### 6.1 单调性

在最大 logit 无并列时，`log_odds(i,T)` 随 `T` 单调递减。因此每个可达目标对应唯一温度。

### 6.2 求解区间

| 路由 | 温度区间 |
|---|---|
| 低风险侧 | `[T_R, T_0]` |
| 高风险侧 | `[T_0, T_U]` |
| gate 为 0 | `T=T_0` |

### 6.3 求解步骤

每个像素固定执行 16 步二分：

1. 根据路由方向初始化左右端点；
2. 计算中点温度；
3. 计算中点的 log-odds；
4. 根据单调性更新区间；
5. 16 步后取区间中点。

以下情况回退到 `T_0=1`：

- 无效或 ignore 像素；
- 最大 logit 并列；
- 输入出现 NaN/Inf；
- 目标或二分结果非有限值。

必须记录：

- 目标残差的 mean、p95、max；
- 两侧 gate 覆盖率和平均强度；
- 温度 mean、harmonic mean、q10、q50、q90、p95；
- 落在 `T_R`、`T_0`、`T_U` 的比例；
- 有效教师温度 `T_out * T`；
- 回退、并列、NaN、Inf 数量。

二分只是稳定求解工具，不作为方法创新点。

## 7. RTC-KD 损失

教师和学生分布：

```text
p_teacher[i] = softmax(z_t[i] / (T_out * T[i]))
p_student[i] = softmax(z_s[i] / T[i])
```

主损失：

```text
L_RTC
    = sum_i M[i] * KL(p_teacher[i] || p_student[i])
      / sum_i M[i]
```

第一版使用 `gamma=0`，即不乘 `T^gamma`。

一般情况下，学生 logit 梯度尺度近似为：

```text
gradient_scale ∝ T^(gamma - 1)
```

因此：

| gamma | 近似效果 |
|---:|---|
| 0 | 低温增强、高温衰减，符合当前机制目标 |
| 1 | 近似抵消一阶温度尺度 |
| 2 | 压低低温分支、补偿高温分支 |

`gamma=0` 是首轮机制设置，不代表对所有 KD 方法普遍最优。Full RTC 通过核心门槛后，再消融 `gamma∈{0,1,2}`。

## 8. 公平性前置条件

所有新实验必须使用同一个 masked pixel KD 实现。

统一损失形式：

```text
L
  = sum_i M[i] * T[i]^gamma
    * KL(p_teacher[i] || p_student[i])
    / sum_i M[i]
```

固定标量温度只是在同一函数中令所有有效像素的 `T[i]` 相同。

严格比较必须共享：

- valid mask；
- padding/ignore 处理；
- loss reduction；
- `T_out`；
- `gamma`；
- 数据顺序；
- 模型初始化；
- seed 和训练预算；
- 保存与验证频率。

现有普通 `CriterionKD` 未排除 padding/ignore 位置，不能继续作为严格 fixed control。

空间机制主比较统一使用 `gamma=0`。标量 `T=0.6, gamma=2` 和旧 Newton `gamma=2` 只作为历史/损失缩放对照。

## 9. 核心分支消融

### 9.1 2×2 因子设计

| 变体 | 低风险侧 | 高风险侧 | 回答的问题 |
|---|---|---|---|
| 中性基线（`neutral`） | `T=1` | `T=1` | 统一 masked 基线 |
| 仅锐化（`reliable_only`） | 目标锐化 | `T=1` | 低风险侧锐化是否有效 |
| 仅平滑（`unreliable_only`） | `T=1` | 目标平滑 | 高风险侧抑噪是否有效 |
| 完整 RTC（`full`） | 目标锐化 | 目标平滑 | 两条分支联合效果 |

直接边际：

```text
delta_R_direct = reliable_only   - neutral
delta_U_direct = unreliable_only - neutral
```

Full 条件下的边际：

```text
delta_R_full = full - unreliable_only
delta_U_full = full - reliable_only
```

两条分支可能存在训练交互，不能把两个边际解释为严格可加。

### 9.2 空间因果对照

| 对照 | 目的 |
|---|---|
| 标量 `T=0.5/0.6/1.0, gamma=0` | 检验是否只是全局低温 |
| 标量 `T=0.6, gamma=2` | 对齐 Phase M2 历史强对照 |
| arithmetic-mean matched scalar | 匹配温度算术均值 |
| inverse-mean matched scalar | 匹配 `gamma=0` 下的平均梯度尺度 |
| per-image shuffled map | 保留直方图，破坏空间对应 |
| reverse routing | 反转方向，做机制 sanity check |
| `exp(-beta*r0)` 目标 | 目标构造方式消融 |
| GT oracle | 教师正误已知时的机制上界 |

shuffle 只能在每张图的 valid pixels 内进行，必须保持温度多重集合、有效像素数和损失尺度不变。

## 10. 评价指标

### 10.1 全局指标

- best validation mIoU；
- final mIoU；
- last-10 validation mean；
- pixel accuracy；
- runtime 和 sec/iteration；
- 运行完整性。

### 10.2 风险分组指标

低风险侧和高风险侧分别报告：

- 覆盖率与 gate 强度；
- 教师 pixel accuracy；
- 教师错误 precision、recall、AUPRC；
- 学生 pixel accuracy、NLL；
- restricted macro-IoU，并说明缺失类别处理；
- 教师正确且位于低风险侧时的学生正确率；
- 教师错误且位于高风险侧时的学生正确率；
- 错误模仿率；
- 平均 KD loss 和学生 logit 梯度范数；
- `c_assess`、`c_target`、`T` 和目标残差分布。

错误模仿率定义：

```text
P(student_prediction == teacher_prediction != ground_truth
  | high_risk_side)
```

### 10.3 分支成立条件

仅锐化分支至少满足：

- 低风险侧学生 accuracy 提高或 NLL 降低；
- 高风险侧不能出现超过预注册容忍度的明显退化；
- 全局 final mIoU 相对中性基线降幅不超过 0.002。

仅平滑分支至少满足：

- 教师错误且位于高风险侧时，学生正确率提高；或
- 高风险侧错误模仿率下降；
- 全局 final mIoU 相对中性基线降幅不超过 0.002。

Full RTC 至少满足：

- 相对中性基线 final mIoU 提高 0.002；
- last-10 delta 不为负；
- 两侧机制指标方向正确。

如果只有一个分支成立，下一阶段只保留成立分支，不强行维持双向叙事。

## 11. 实现计划

### 11.1 单一实现源

建议新增 `utils/rtc_temperature.py`，统一提供：

1. `compute_reference_reliability`；
2. `load_or_query_frozen_cdf`；
3. `compute_reliability_gates`；
4. `compute_top_vs_rest_log_odds`；
5. `build_target_log_odds`；
6. `invert_target_by_bisection`；
7. `masked_temperature_kd_loss`；
8. `collect_rtc_diagnostics`。

`train_cirkdv2.py`、`train_kd.py` 和诊断脚本只调用该模块，不再复制公式。

### 11.2 建议命令行参数

```text
--covar-temp-mode rtc
--rtc-cdf-path <path>
--rtc-route-quantile 0.80
--rtc-route-width 0.05
--rtc-temp-reliable 0.5
--rtc-temp-neutral 1.0
--rtc-temp-unreliable 2.0
--rtc-alpha-reliable 1.0
--rtc-alpha-unreliable 1.0
--rtc-enable-reliable
--rtc-enable-unreliable
--rtc-bisection-iters 16
--covar-kd-temp-power 0.0
```

所有实验脚本必须显式传入分支开关与参数，不能依赖 argparse 默认值。

### 11.3 单元测试门槛

实现后必须验证：

- `log_odds(T)` 单调递减；
- 低风险分支温度不高于 1；
- 高风险分支温度不低于 1；
- 两个 gate 互斥；
- 目标位于可达端点之间；
- 二分残差满足容忍度；
- `alpha=0` 精确退化为 `T=1`；
- 单分支模式能完全关闭另一分支；
- constant temperature 与 masked scalar KD 数值等价；
- 全 ignore mask 返回有限零损失；
- shuffled map 保持有效温度多重集合；
- 所有 DDP rank 使用同一 CDF；
- float32/NPU 下无 NaN/Inf。

## 12. 分阶段实验计划

### O0：停止与快照

状态：已完成。

- Phase N 控制器和两个 worker 已停止；
- 两张 NPU 无训练进程；
- 两路达到 20000/80000，但 20k 验证被中断；
- 不自动恢复 Phase N；
- 日志和 checkpoint 仅作为历史记录保留。

### O1：教师路由诊断

不训练学生。输出：

- train/val 的 `r0` CDF 和十等分表；
- `q=0.80` 下两侧覆盖率；
- teacher-error precision、recall、AUPRC；
- confidence-only、variance-only、full score 的排序质量；
- raw teacher 与 `T_out=3` KD 分布的路由一致性；
- 冻结 CDF 文件及 checksum。

晋级条件：

- 高风险侧错误精度至少是全局错误率的 2 倍；
- 高风险侧覆盖至少 10% 像素；
- 高风险侧覆盖至少 60% 的教师错误；
- train 与 val 的风险方向一致；
- full score 不得明显弱于 confidence-only。

未满足则停止性能训练，先修正可靠性评分。

### O2：单元测试与 20-iteration smoke

变体：

- `neutral`；
- `reliable_only`；
- `unreliable_only`；
- `full`；
- 标量 `T=0.6, gamma=0`；
- `shuffled`。

要求：

- 每个变体完整达到 20/20；
- Namespace 与预注册一致；
- 无 NaN/Inf；
- 温度方向断言通过；
- 目标残差与回退率正常；
- checkpoint/training state 完整。

### O3：CWD 20k 核心实验

公共配置：

| 项目 | 值 |
|---|---|
| Dataset | VOC |
| Teacher | DeepLabV3-ResNet101 |
| Student | DeepLabV3-MobileNetV3-Small |
| Base recipe | CWD |
| `T_out` | 3.0 |
| Seed | 1234 |
| Budget | 20k |
| `gamma` | 0 |
| `q, w` | 0.80, 0.05 |
| `T_R, T_0, T_U` | 0.5, 1.0, 2.0 |
| `alpha_R, alpha_U` | 1.0, 1.0 |

运行：

1. `neutral`；
2. `reliable_only`；
3. `unreliable_only`；
4. `full`；
5. 标量 `T=0.6, gamma=0`；
6. 标量 `T=0.6, gamma=2`。

20k 仅用于筛选和机制诊断，不作为统计优越性结论。

### O4：空间因果对照 20k

仅当 `full` 通过 O3 时运行：

1. Full RTC；
2. 同 `gamma` 下最强标量温度；
3. arithmetic-mean matched scalar；
4. inverse-mean matched scalar；
5. per-image shuffled map；
6. reverse routing；
7. `exp(-beta*r0)` 目标。

空间分配成立的最低门槛：

- Full 相对最强标量 final mIoU 至少提高 0.002；
- Full 相对 shuffled final mIoU 至少提高 0.002；
- 两项 last-10 delta 均不为负；
- 两侧机制指标方向正确；
- reverse routing 明显弱于 Full。

若 Full 只与标量温度持平，只能声称“可靠性校准达到合适标量温度的性能”，不能声称空间自适应有独立贡献。

### O5：CIRKD 20k 迁移

修复 masked KD 后，重复：

- `neutral`；
- `reliable_only`；
- `unreliable_only`；
- `full`；
- 最强标量温度；
- `shuffled`。

不得直接复用旧 CoVar on/off 结果，因为旧 fixed KD 的 mask 和 reduction 不一致。

### O6：80k seed 1234

只有至少一个底座通过 O4 才运行：

- `neutral`；
- `reliable_only`；
- `unreliable_only`；
- `full`；
- strongest/matched scalar。

80k 晋级条件：

- Full 相对最强标量 final mIoU 至少提高 0.002；
- 对应 last-10 delta 不为负；
- Full 相对中性基线 final mIoU 至少提高 0.002；
- 全部运行完整且无恢复混杂。

### O7：三种子

保留 seed 1234，增加 2025 和 3407。

核心只运行 Full 与最强标量温度的 paired comparison。报告：

- 每个 seed 的 paired delta；
- mean；
- sample SD，`ddof=1`；
- 置信区间；
- final、best、last-10；
- 所有负 seed。

### O8：泛化验证

顺序：

1. PSPNet-MobileNetV3-Small；
2. 不同 backbone；
3. 至少一个不同数据集；
4. CWD 与 CIRKD 两个底座；
5. `T_out=1` 与 `T_out=3` 的解耦检查。

跨数据集只允许重新建立无标签训练集 CDF；`q`、`w`、温度端点和目标力度保持固定。

同时增加两种检查：

- 直接复用 VOC CDF；
- 使用目标数据集训练输出重新建立 CDF。

如果只有重新建立 CDF 才有效，应将方法定位为 dataset-adaptive training method，而不是 zero-shot 方法。

## 13. 主要风险与止损规则

### 13.1 高置信错误

低风险侧仍可能包含高置信错误。强锐化会放大这些错误，必须按 teacher-correct/teacher-wrong 分层报告。

### 13.2 高风险不等于教师错误

高风险 top 20% 中多数像素仍可能正确。首轮使用 `T_U=2`，不直接采用 4 或 8。

### 13.3 只控制 top-vs-rest 置信度

RTC 只校准最大类别相对其余类别的置信度，不保证非主类别内部结构最优。

### 13.4 双重温度

教师有效温度是 `T_out * T[i]`。所有日志和论文表格必须同时报告 `T_out`、`T[i]` 和有效教师温度。

### 13.5 CDF 跨域风险

CDF 固定相对比例，不能保证不同教师和数据集上的绝对错误率一致。跨域必须重新报告风险富集质量。

### 13.6 禁止性结论

在 O4 和 O7 完成前，不得声称：

- RTC 优于合适的全局标量温度；
- 正确空间位置已经被证明有效；
- 两条分支均独立有效；
- `r0` 能准确判断教师对错；
- 方法具有跨数据集稳定性。

## 14. 结果回填模板

### 14.1 全局性能

| Variant | Budget | Seed | Best | Final | Last-10 | Runtime | Complete |
|---|---:|---:|---:|---:|---:|---|---|
| neutral | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| reliable_only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| unreliable_only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| full | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| strongest_scalar | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| shuffled | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

### 14.2 分组机制

| Variant | Low-risk coverage | High-risk coverage | Acc low | Acc high | Student correct given teacher wrong | Error imitation | T mean low/high | Target error |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| neutral | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| reliable_only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| unreliable_only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| full | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 15. 当前执行决定

- Phase N 保持 stopped；
- 不自动恢复旧训练；
- 暂不启动新训练；
- 先实现共享 masked KD、RTC 单一模块和训练集 CDF 诊断；
- 实现后先通过单元测试和 20-iteration smoke；
- 只有 O1/O2 门槛通过，才启动 20k 核心矩阵；
- 所有参数变化、失败和中断必须先写入本文档。
