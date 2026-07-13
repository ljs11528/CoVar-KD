# Phase O：RTC-KD 方法重构与实验预注册

- 创建日期：2026-07-13
- 方法名称：可靠性目标置信度蒸馏（Reliability-Targeted Confidence Distillation，RTC-KD）
- 当前状态：O1/O1.1 历史诊断完成；O1.2-A 独立实现和全量机制联合门禁通过；O1.2-B 的 `neutral` 与 `unreliable_only` 20-step fresh 及终点零步恢复审计均通过；后续实验未启动
- 上一阶段：Phase N 已停止，不自动恢复
- 文档格式：为兼容不同 Markdown 渲染器，公式统一写成纯文本代码块
- 阅读规则：第 1 至 18 节保留 O1/O1.1 的历史方法与正式结果；O1.2 以第 19 节及其独立预注册文档为唯一规范

## 0. 一页结论

### 当前 O1.2 主线

Phase O 已完成两次风险定义诊断：

1. O1 保留 confidence+variance 历史定义并正式失败；
2. O1.1 改用 confidence-only 风险，正式联合门禁通过；
3. O1.1 同时暴露出旧单阈值温度映射造成大面积强锐化，不能直接进入学生实验。

当前唯一主线是 O1.2：

1. 精确复用 O1.1 confidence-only CDF，不重新选择风险分数；
2. bottom 60% 只做逐渐减弱的轻微锐化；
3. middle 20% 保持严格中性；
4. top 20% 随风险连续增强平滑；
5. 将温度算术均值约束在 0.995，并要求调和均值不低于 0.98；
6. 空间温度只校准教师目标，学生端温度固定为 1；
7. 首个学生比较只运行 neutral 与 unreliable_only；
8. matched scalar 与 within-image shuffle 完成前，不声称空间风险位置具有独立贡献。

O1.2 的完整公式、来源契约、门禁和停止规则见：

- [O1.2 高风险优先预算路由预注册](2026-07-13_phaseO_rtc_o12_budgeted_routing_plan.md)

O1.2-A 已完成全量机制诊断并通过联合门禁。经单独授权后，O1.2-B 仅运行了 `neutral` 与 `unreliable_only` 的 20-step fresh 学生链路 smoke，两路及各自的终点零步恢复审计均通过。A 的机制联合门禁与 B 的学生链路 smoke 是两条独立证据链：B 不并入、也不改写 A 的 gate 结论。B 使用 `--skip-val`，没有 validation、mIoU、性能或泛化结论；当前重新停在人工审查线，不自动启动后续 smoke、20k、C2、C3 或 80k。

### O1/O1.1 历史首轮定义（仅记录）

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

## 16. 2026-07-13 正式执行回填

### 16.1 执行边界

- O0、O0b、正式 CDF 构建和正式 O1 联合诊断均已完成；
- 没有恢复 Phase N，没有启动 O2，也没有运行任何 20k/80k 学生训练；
- 当前没有 Phase O 训练或诊断进程；
- O1 未通过预注册门槛，因此 O2/O3 继续锁定。

### 16.2 正式冻结 CDF

正式 CDF 完整扫描 VOC train_aug 的 10,582 张图像。原生教师网格有效像素数为
32,298,651，有限像素数同为 32,298,651，非有限像素数为 0。

| 项目 | 正式值 |
|---|---|
| Artifact | `runs/diagnostics/phaseO/voc_train_rtc_cdf.pt` |
| CDF SHA256 | `40b4454fc919422899e512fed4ece4a428d12544e7ce828a89dc731d12529e39` |
| 图像进度 | 10,582 / 10,582 |
| CDF 样本数 | 32,298,651 |
| `r0` 最小值 | 2.8937583440580283e-09 |
| `r0` 均值 | 0.20497506856918335 |
| `r0` 最大值 | 5.4413299560546875 |
| 构建脚本 SHA256 | `7c9f6d5de606ce0a2e2d4de0af9b728b372e0d2ef6c0121fcd79af3753d8e7c2` |
| RTC 核心 SHA256 | `4986b545536bb9d0417b60f1b57855a4ca1bdbe982364e09224db53a68628224` |

### 16.3 正式 O1 结果

train 和 val 均为完整扫描，且使用同一份冻结训练集 CDF。标签只用于离线验证
路由质量，不进入 CDF、温度生成或训练时路由。

| 指标 | train_aug | val |
|---|---:|---:|
| 完整图像数 | 10,582 / 10,582 | 1,449 / 1,449 |
| 原生有效像素 | 32,246,990 | 3,878,674 |
| 教师错误率 | 0.0294596 | 0.0622352 |
| 高风险覆盖率 | 0.1984741 | 0.2093762 |
| 高风险侧错误 precision | 0.1466368 | 0.2439632 |
| 教师错误 recall | 0.9879156 | 0.8207589 |
| 相对全局错误率富集倍数 | 4.9776 | 3.9200 |
| 低风险侧错误率 | 0.0004442 | 0.0141092 |
| Full AP | 0.3855771 | 0.3698922 |
| Confidence-only AP | 0.3943625 | 0.3856800 |
| Variance-only AP | 0.3840141 | 0.3676450 |
| Full − Confidence-only AP | -0.0087854 | -0.0157878 |
| Full AUC | 0.9634619 | 0.9017147 |
| 温度方向违规数 | 0 | 0 |
| fallback / tie / nonfinite | 0 / 0 / 0 | 0 / 0 / 0 |
| 目标残差 p95 | 7.58171e-05 | 7.53403e-05 |

风险十分位与错误率具有明显单调关系。train 最高风险十分位错误率为 0.2611881，
val 为 0.3514240；最低风险十分位分别为 0 和 0.0001780。这说明当前无标签风险
排序确实能区分相对可靠侧和不可靠侧，并非随机路由。

### 16.4 O1 门禁结论

正式联合门禁结论为 `joint_gate_pass=false`。实现完整性、有限性、温度方向、
路由覆盖、风险富集、双 `T_out` 路由一致性、CDF 来源与配置指纹检查均通过。
唯一失败项是预注册判据：

```text
AP_full >= AP_confidence_only - 0.005
```

train 的差值为 -0.0087854，val 的差值为 -0.0157878，二者都越过了允许下界。
因此不能声称非主类方差项为置信度路由提供了有效增量；相反，它在两个 split
上都降低了教师错误排序 AP。

这个失败不等同于“可靠/不可靠分路无效”。现有结果支持置信度主导的相对风险
路由，但不支持当前 Full 分数中方差项的必要性。不得通过事后放宽 0.005 阈值
追认 O1，也不得带着当前评分进入学生性能实验。

正式门禁产物：

- `runs/diagnostics/phaseO/o1_joint_gate.json`；
- `runs/diagnostics/phaseO/rtc_routing_train.json`；
- `runs/diagnostics/phaseO/rtc_routing_val.json`。

### 16.5 下一步预注册建议：O1.1

建议把下一版训练时风险分数改为 confidence-only：

```text
r_conf(i) = -log(c_i)
u_i = F_train,conf(r_conf(i))
```

其中 `F_train,conf` 仍只由目标训练集的无标签教师输出建立。可靠侧继续向低温
强锐化目标校准，不可靠侧继续向高温平滑目标校准；温度反解、两侧端点、`q`、
KD mask/reduction 和数值机制检查保持不变。评分改变后必须使用新文件名重建
CDF，重新完成完整 train/val 路由诊断，不能覆盖或复用本次 Full-score artifact。

原 `AP_full >= AP_confidence_only - 0.005` 已经产生否定结论并永久保留。O1.1
主分数若定义为 confidence-only，就不能继续用该分数与自身比较作为晋级条件；
这种门禁必然通过，不具可证伪性。O1.1 必须在运行前另行冻结以下非同义门禁：

1. train/val 高风险覆盖率均位于 `[0.15, 0.25]`；
2. 高风险错误率均至少为各 split 全局错误率的 2 倍；
3. 高风险侧教师错误 recall 均不低于 0.70，低风险错误率均低于全局错误率；
4. 45 个有序十分位对中，错误率不发生逆序的 pairwise 单调一致率均不低于 0.90；相等错误率视为一致；
5. nonfinite、fallback、tie、方向违规和双 `T_out` 路由 mismatch 均为 0；
6. 目标残差 p95 小于 `1e-3`，CDF 完整性、来源 SHA 和配置指纹全部通过。

这些阈值是在看过 O1 后为探索性 O1.1 设定的，不能把同一 VOC 上的通过写成独立
确认。真正的确认性证据必须来自未参与此次评分选择的新教师或新数据集，并在
评估前冻结 confidence-only、CDF 策略、`q`、gate 宽度和温度端点。

备选方案是只在训练集内部预注册并选择方差系数，再对 val 做一次确认；但当前
两个 split 都显示 confidence-only 更优，额外系数会增加选择自由度和审稿风险，
因此不作为首选。

O1.1 属于方法定义变化，必须先更新预注册配置并获得确认。在此之前保持：

- O2/O3 锁定；
- 不启动学生训练；
- 保留本次失败产物和阈值，不覆盖、不删除、不改判。

## 17. O1.1：confidence-only 路由正式预注册

冻结状态：本节在任何 O1.1 正式 CDF 构建或路由诊断运行前写入。O1.1 尚未启动；本节只固定假设、配置、产物路径和判据，不包含结果。

### 17.1 假设与证据定位

O1.1 只检验以下缩减假设：教师最大类别置信度定义的相对风险排序，能否在标签不参与路由的前提下，把教师错误稳定富集到相对高风险侧，并驱动可靠侧锐化、不可靠侧平滑。

confidence-only 是查看同一 VOC 上 O1 的 full、confidence-only 和 variance-only 结果后选出的，因此 O1.1 明确属于探索性、非独立确认实验。即使通过，也只能说明该定义在当前教师与 VOC 设置上值得进入后续机制验证，不能写成独立确认、跨数据集泛化证据或“方差项从未有用”。确认性结论必须来自预先冻结方法后的新教师或新数据集。

### 17.2 唯一主风险分数

对原始教师 logits `z_t(i,k)`，固定：

```text
p_assess(i,k) = softmax(z_t(i,k) / T_assess)
c_i           = max_k p_assess(i,k)
r_conf(i)     = -log(clamp(c_i, epsilon, 1-epsilon))

T_assess         = 1.0
reliability_mode = confidence
coefficient_a    = 0.0
```

`coefficient_a=0.0` 必须显式写入配置和产物元数据。它在 confidence 主路由中不参与计算，用 0 是为了消除是否仍隐式保留 O1 方差项的歧义。诊断可以离线报告旧 full/variance 分数作为非活动参考，但不得将其混入 CDF、gate 或温度目标。

冻结训练集 CDF 后：

```text
u_i = F_train,confidence(r_conf(i))
```

`r_conf` 的数值与 CDF 排序不使用类别值、教师正确性或学生信息；但构建器沿用监督分割数据的 GT `ignore-valid` 空间掩码，以排除 ignore/void 与 padding 像素。因此本实现严格说不是完全 label-free，论文中只能表述为“标签类别和正确性不参与风险定义或排序拟合，仅使用训练协议既有的有效像素掩码”。

O1.1 不复用 O1 的 full-score CDF。

### 17.3 冻结路由、温度与随机性

除风险分数和独立产物命名外，其余主配置保持不变：

| 参数 | O1.1 冻结值 |
|---|---:|
| `q` / `w` | 0.8 / 0.05 |
| `T_R` / `T_0` / `T_U` | 0.5 / 1.0 / 2.0 |
| `alpha_R` / `alpha_U` | 1.0 / 1.0 |
| `T_out` | 3.0 |
| 二分次数 | 16 |
| CDF 构建 seed | 1234 |
| train 正式诊断增强 seed | 2025 |
| AP/AUC 排名采样 seed | 3407 |

gate 固定为：

```text
gate_R(i) = max(tanh((q - u_i) / w), 0)
gate_U(i) = max(-tanh((q - u_i) / w), 0)
```

可靠侧只允许 `T_i` 落在 `[T_R,T_0]`，不可靠侧只允许落在 `[T_0,T_U]`。`T_out=3.0` 只进入实际 KD 教师 logits 和目标置信度，不得进入风险 CDF 或路由。train 诊断 seed `2025` 必须与 CDF seed `1234` 独立。

train/val 均须完整扫描。AP/AUC 继续采用每图最多 1,024 个 native-valid 像素、seed `3407` 和 tie-aware 分组阈值实现；覆盖、错误率、precision、recall 与十分位错误率使用全 native-valid 像素 micro 统计。

### 17.4 独立目录、phase 与强制产物

O1.1 统一使用独立目录：

```text
runs/diagnostics/phaseO_o11/
```

所有正式 JSON 的 phase 字段必须精确为 `O1.1`。四个主产物冻结为：

```text
runs/diagnostics/phaseO_o11/voc_train_rtc_confidence_cdf.pt
runs/diagnostics/phaseO_o11/rtc_confidence_routing_train.json
runs/diagnostics/phaseO_o11/rtc_confidence_routing_val.json
runs/diagnostics/phaseO_o11/o11_confidence_gate.json
```

CDF summary、十等分 CSV 等 sidecar 如由脚本生成，也必须留在 `runs/diagnostics/phaseO_o11/`，并采用 confidence 前缀或与主 CDF 同名的 sidecar 命名。禁止覆盖、改写、移动或删除以下 O1 产物：

```text
runs/diagnostics/phaseO/voc_train_rtc_cdf.pt
runs/diagnostics/phaseO/rtc_routing_train.json
runs/diagnostics/phaseO/rtc_routing_val.json
runs/diagnostics/phaseO/o1_joint_gate.json
```

联合门禁必须核对 O1.1 的实际 CDF SHA256、train/val 报告所记录 CDF SHA、teacher/list/source/config 指纹、完整扫描状态及 phase 字段；任一不一致均为结构性失败，不能手工放行。

### 17.5 非同义联合门禁

O1 的 `AP_full >= AP_confidence_only - 0.005` 已永久保留为 O1 的失败结论。O1.1 主分数就是 confidence-only，禁止用 confidence-only 与自身比较作为门禁。O1.1 的 train 与 val 必须分别满足以下全部条件，联合门禁才可为真：

1. 高风险覆盖率位于闭区间 `[0.15, 0.25]`；
2. 高风险侧错误率不低于该 split 全局教师错误率的 2 倍；
3. 高风险侧教师错误 recall 不低于 `0.70`；
4. 低风险侧教师错误率严格低于该 split 全局教师错误率；
5. 从低风险到高风险排列的十个风险十分位组成 45 个 `i<j` 有序对；若 `error_rate[i] <= error_rate[j]` 则该对一致（相等视为通过），pairwise 单调一致率不低于 `0.90`；
6. native-valid 像素数为正；nonfinite、fallback、tie、两侧方向违规、`T_out=1` 与 `T_out=3` 的路由字段 mismatch、正温度缩放后的 argmax mismatch 均严格为 0；
7. 固定 ranking sample 中 active solved nonfallback 像素的目标 log-odds 残差 p95 小于 `1e-3`；
8. train/val 和 CDF 均为完整数据集扫描；CDF 来源、实际文件 SHA256、teacher/list SHA、类别数、`T_assess=1.0`、`reliability_mode=confidence`、`coefficient_a=0.0`、随机种子及关键配置指纹全部一致。

十分位 Spearman 仍须报告，但不作为门禁。原因是低风险多个十分位可能都为零错误；平均秩 Spearman 会惩罚这种理想平台，而 pairwise 一致率把相等视为不违背单调性。本口径在正式 CDF 和正式 train/val 运行前，由 8 图链路 smoke 暴露并完成修订。

AP、AUC、错误 precision、温度分布和十分位明细仍须报告，但 AP/AUC 不作为上述门禁的同义替代。任一阈值失败都必须令 `o11_confidence_gate.json` 中的 `joint_gate_pass=false`；不得事后改阈值、删 split 或挑 seed。

风险富集、召回率与低风险错误率必须由全量 native-valid 的整数 `risk_routing_counts` 独立重算：高风险为 `u>q`，低风险为 `u<q`，边界为 `u==q`。风险十分位逐 bin 保存精确 `teacher_wrong_count`，用于核对人口、错误总数与单调性，不再用十分位浮点率反推路由两侧指标。

联合门禁还必须逐项锁定完整 RTC 配置、CDF/train/ranking seed、每图 ranking cap、batch/workers、train/val 完整规模及 canonical list SHA，并核对预注册 teacher SHA；仅“train 与 val 彼此相同”不足以通过。

O1.1 checker 本身纳入 CDF 的 source SHA 映射；checker 运行时必须核对该 SHA 与当前文件，并在门禁产物中记录自身 SHA。上述内容是在正式 CDF 和正式 train/val 结果产生前的 schema 完整性修正，不改变任何效果门槛。

### 17.6 执行禁令与晋级边界

在 `o11_confidence_gate.json` 正式生成且 `joint_gate_pass=true` 之前：

- 禁止启动 O2/O3；
- 禁止运行任何 RTC 学生 20-iteration、20k 或 80k 训练；
- 禁止生成或宣称 O1.1 student mIoU、checkpoint 或性能结论；
- 禁止覆盖 O1 失败记录，禁止把 O1.1 的探索性通过改写成 O1 通过；
- 实现变更、失败运行和偏离预注册的事项必须先记录在 `2026-07-13_phaseO_rtc_o11_execution_record.md`。

只有 O1.1 的结构检查、train 门禁和 val 门禁全部通过，才允许另行讨论是否开启 O2；该放行也不自动授权任何学生训练。
## 18. O1.1 正式结果与当前决策

状态：2026-07-13 正式完成，独立联合门禁通过；未启动任何学生训练。

| 指标 | Train | Val | 门槛 |
|---|---:|---:|---:|
| 高风险覆盖率 | 0.198395 | 0.210878 | [0.15, 0.25] |
| 高风险错误富集 | 4.9748x | 3.9183x | >=2x |
| 高风险错误 recall | 0.986971 | 0.826277 | >=0.70 |
| 低风险错误率 | 0.000479 | 0.013701 | < 全局 |
| 十分位 pairwise 一致率 | 1.000000 | 1.000000 | >=0.90 |
| 残差 p95 | 7.58e-5 | 7.53e-5 | <1e-3 |

正式 CDF 扫描 10,582 张 train_aug 图像、32,298,651 个 finite native-valid 像素，nonfinite 为 0。train 与 val 的十分位错误率均严格单调上升，所有 fallback、tie、方向违规、路由 mismatch 和正温度 argmax mismatch 均为 0。独立 checker 给出：

```text
joint_gate_pass = true
train_pass      = true
val_pass        = true
```

因此，当前证据支持：confidence-only 风险在本教师与 VOC 上足以稳定建立相对可靠性排序，旧方差项不是该排序成立的必要条件。

仍需保留两项限制：

1. O1.1 是查看 O1 后提出的探索性实验，不能作为跨数据集或跨教师的独立确认；
2. train/val 温度中位数均约为 0.500004，调和均值约为 0.594/0.601。大量像素受到强锐化，后续学生收益仍可能由广泛低温而非空间风险选择解释。

O1.1 的通过只解除风险定义诊断门槛，不自动启动 O2/O3。若后续获批，可靠侧、非可靠侧、强标量低温以及空间/温度分布匹配对照必须分别验证。

详细结果与指纹见：

- [O1.1 正式诊断报告](2026-07-13_phaseO_rtc_o11_diagnostic_report.md)
- [O1.1 执行记录](2026-07-13_phaseO_rtc_o11_execution_record.md)

## 19. O1.2：高风险优先预算路由的当前决策

状态：2026-07-13 已完成方法重构、独立实现、全量 train/val 机制诊断与联合门禁；O1.2-A 联合门禁通过。同日已完成获批范围内的 O1.2-B `neutral` 与 `unreliable_only` 20-step fresh 和终点零步恢复审计；后续实验未启动。

O1.2 保留 O1.1 confidence-only 风险和冻结 CDF，但废止未来训练中的旧单阈值 tanh gate、0.5/2.0 强端点、全像素激活和学生共享空间温度。新的唯一主方案为：

- qR=0.6、qU=0.8，形成 bottom 60% 轻锐化、middle 20% 中性、top 20% 连续平滑；
- 对数温度 logT=-a*gR+b*gU；
- 0.9<=T<=1.5，train 算术均值目标 0.995，调和均值不低于 0.98；
- 在满足预算的可行解中最大化高风险平滑参数 b；
- 空间温度只生成教师校准目标，学生温度固定为 1；
- 第一项学生性能实验只比较 neutral 与 unreliable_only；
- matched scalar 和 within-image shuffle 未完成前，不允许声称空间风险位置具有独立贡献。

工程上必须保持 O1.1 core、CDF builder、diagnose 和 checker 字节不变，新增独立 O1.2 module、diagnose、checker 和 tests。O1.2 复用 O1.1 CDF 的实际 SHA，但使用独立 phaseO_o12 产物目录和来源清单。

由于 qR、qU 和预算是在查看同一 VOC train/val 结果后冻结，O1.2 在 VOC 上明确定位为探索性筛查；跨教师或跨数据集才承担确认性结论。

完整且唯一的 O1.2 规范见：

- [O1.2 高风险优先预算路由预注册](2026-07-13_phaseO_rtc_o12_budgeted_routing_plan.md)

### 19.1 O1.2-A 正式机制诊断

2026-07-13 已完成独立 O1.2 module、diagnose、checker、训练入口和单元测试。完整测试为 103 passed；O1.1 冻结源码 SHA256 保持不变。正式求解得到 a*=0.1053605157、b*=0.3476499170，train 预算残差为 2.79e-10。

| 指标 | Train | Val |
|---|---:|---:|
| mean(T) | 0.995000 | 0.996092 |
| harmonic(T) | 0.988069 | 0.988161 |
| median(T) | 0.982425 | 0.979061 |
| T>1 覆盖率 | 0.198152 | 0.210636 |
| T>1.25 覆盖率 | 0.039553 | 0.046430 |
| 高风险错误富集 | 4.9748x | 3.9183x |
| 高风险错误 recall | 0.986971 | 0.826277 |

独立 checker 的 joint_gate_pass、train_pass 和 val_pass 均为 true。方向、范围、中性区、熵、argmax、student-fixed、teacher-target-only 和 nonfinite 检查全部通过，train 求解缓存与 train 独立复算缓存字节一致。

因此 O1.2-A 已解决“映射事实上接近全局 T=0.6”这一数值机制隐患，但 A 仍只有教师目标层面的证据。O1.2-B 的后续链路 smoke 单独记录在第 19.2 节，不能回填为 A 的机制 gate 证据。

详细记录：

- [O1.2 执行记录](2026-07-13_phaseO_rtc_o12_execution_record.md)
- [O1.2 机制诊断报告](2026-07-13_phaseO_rtc_o12_diagnostic_report.md)

### 19.2 O1.2-B 正式学生链路 smoke

在明确授权的边界内，`neutral` 与 `unreliable_only` 各自完成 20 个 optimizer step 的 fresh smoke，独立 fail-closed checker 均给出 `pass=true`。两路都保存了 iteration 20 的完整训练状态，loss 与 KD 学生-logit 梯度有限且观察到非零梯度，冻结配置、输入指纹和样本顺序契约通过检查。

随后两路分别从各自 iteration-20 终点执行 `resume_audit`：均成功加载完整训练状态，执行 optimizer step 数严格为 0，并通过终点状态一致性检查。该恢复审计只证明终点 checkpoint 可被严格加载，不把 20-step 预算延长为第 21 步。

O1.2-B 只证明这两个变体的学生训练、保存和恢复链路可运行。fresh 使用 `--skip-val`，因此没有 validation、mIoU、效果比较、空间因果、统计显著性或跨数据集泛化结论。O1.2-A 的联合门禁保持原结论，不把 B 的通过混入 A。当前不自动启动 `reliable_only`、`full_budgeted`、scalar、shuffle 等后续 smoke，也不自动启动 20k、C2、C3 或 80k。

详细记录：

- [O1.2-B neutral 与 unreliable_only 20-step smoke 报告](2026-07-13_phaseO_rtc_o12b_smoke_report.md)
