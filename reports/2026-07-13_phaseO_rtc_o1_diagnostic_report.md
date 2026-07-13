# Phase O：RTC-KD 正式 O1 路由诊断报告

- 日期：2026-07-13
- 范围：冻结教师 CDF、train/val 全量路由诊断与联合门禁
- 当前结论：O1 科学门禁未通过，O2/O3 保持关闭
- 学生训练：未启动
- 方法计划：[2026-07-13_phaseO_rtc_method_reconstruction_plan.md](2026-07-13_phaseO_rtc_method_reconstruction_plan.md)
- 执行总记录：[2026-07-13_phaseO_rtc_execution_record.md](2026-07-13_phaseO_rtc_execution_record.md)

## 1. 结论摘要

本轮 O1 分别检验相对风险路由是否有信息量，以及 full score 的非主类方差项是否比仅使用最大置信度提供增量。

- 风险路由得到支持：高风险约 20% 像素在 train/val 上分别覆盖 98.79% 和 82.08% 的教师错误，错误率分别富集 4.98 倍和 3.92 倍；
- 方差增量没有得到支持：full score 的 AP 相对 confidence-only 在 train/val 分别下降 0.008785 和 0.015788，均越过预注册下限 -0.005；
- 数值稳定性、温度方向、二分残差、fallback、tie、非有限值、T_out 路由解耦和数据/源码指纹全部通过；
- 因此失败定位在评分假设，而不是温度求解或实验工程；联合门禁为 `joint_gate_pass=false`，没有启动学生训练。

## 2. 预注册方法与判据

### 2.1 当前 full 风险分数

可靠性只从原始教师 logits 的固定参考分布计算，不使用 KD 的 `T_out`：

```text
p_assess(i,k) = softmax(z_t(i,k) / T_assess),  T_assess = 1
c_i           = max_k p_assess(i,k)
mu_i          = (1 - c_i) / (K - 1)
v_i           = mean_{k != top} (p_assess(i,k) - mu_i)^2
r_full(i)     = -log(c_i) + a * v_i / (1 - c_i + eps)
```

VOC 有 `K=21`，固定 `a=(K-1)^2/2=200`。非主类方差通过显式排除 argmax 类别计算，避免高置信像素上的 float32 消减。

冻结教师完整扫描 train_aug 后建立经验 CDF：

```text
u_i = F_train(r_full(i))
```

`u_i` 越大，表示像素位于训练集参考分布中越高的相对风险尾部。“可靠/不可靠”是相对风险简称，不等价于教师一定正确/错误。

### 2.2 双向目标置信度

实际进入 KD 的教师 logits 为 `z_kd=z_t/T_out`。在 `T_R=0.5`、`T_0=1`、`T_U=2` 三个端点计算 top-vs-rest log-odds：

```text
L_target = L_0
         + alpha_R * gate_R * (L_R - L_0)
         + alpha_U * gate_U * (L_U - L_0)
alpha_R = alpha_U = 1
```

随后用 16 步单调二分反求像素温度。低风险侧只能向低温端点移动并锐化；高风险侧只能向高温端点移动并平滑。

### 2.3 核心科学门槛

```text
AP_full >= AP_confidence_only - 0.005
```

该门槛用于检验方差项是否至少不明显伤害置信度基线，看到正式结果后不得放宽。

## 3. 冻结 CDF 审计

| 项目 | 正式值 |
|---|---|
| 数据与进度 | VOC train_aug，10,582 / 10,582 |
| 原生输出网格 | 64×64 |
| 训练增强 | scale=True，mirror=True，crop=512×512 |
| CDF seed / knots | 1234 / 4,097 |
| native-valid / finite / nonfinite | 32,298,651 / 32,298,651 / 0 |
| `r_full` min / mean / max | 2.893758e-09 / 0.204975 / 5.441330 |
| CDF SHA256 | `40b4454fc919422899e512fed4ece4a428d12544e7ce828a89dc731d12529e39` |

正式文件为 `runs/diagnostics/phaseO/voc_train_rtc_cdf.pt`。教师权重、数据清单、构建参数、完整扫描标志、脚本和 RTC 核心源码指纹均写入元数据，训练入口会再次校验。

第一次全量扫描因 nonmax 方差数值消减和 PyTorch 大张量 quantile 限制而作废。正式 CDF 是修复后重新完整扫描所得，作废扫描未用于诊断或训练。

## 4. 正式数据口径

- train 诊断增强 seed 为 2025，与 CDF seed 1234 独立；
- AP/AUC 排名采样 seed 为 3407，每图最多采样 1,024 个 native-valid 像素；
- train/val 排名样本分别为 10,603,054 和 1,481,403 个像素；
- 覆盖率、错误率、precision、recall 和分箱表使用全部 native-valid 像素的 micro 计数；
- GT 仅用于 O1 离线诊断，正确性定义为教师原生网格 argmax 与 nearest-resized GT 是否一致；
- 该正确性是 KD 网格代理，不等同于标准 full-resolution mIoU。

## 5. 相对风险路由结果

| 指标 | Train | Val |
|---|---:|---:|
| 完整图像数 | 10,582 / 10,582 | 1,449 / 1,449 |
| native-valid / nonfinite | 32,246,990 / 0 | 3,878,674 / 0 |
| 教师 native-grid 错误率 | 0.0294596 | 0.0622352 |
| 可靠侧覆盖率 | 0.8015259 | 0.7906238 |
| 高风险侧覆盖率 | 0.1984741 | 0.2093762 |
| 高风险侧错误 precision | 0.1466368 | 0.2439632 |
| 教师错误 recall | 0.9879156 | 0.8207589 |
| 错误率富集倍数 | 4.9776 | 3.9200 |
| 可靠侧错误率 | 0.0004442 | 0.0141092 |

冻结 train CDF 在 val 上得到 20.94% 高风险覆盖率而非机械等于 20%，反映 val 风险分布有轻微偏移；但 3.92 倍错误富集说明同一 CDF 在 val 上仍有区分能力。

### 5.1 风险十分位

| `u` 区间 | Train 错误率 | Val 错误率 |
|---|---:|---:|
| [0.0, 0.1) | 0 | 0.0001780 |
| [0.1, 0.2) | 0.0000006 | 0.0009045 |
| [0.2, 0.3) | 0.0000016 | 0.0025612 |
| [0.3, 0.4) | 0.0000099 | 0.0051081 |
| [0.4, 0.5) | 0.0000188 | 0.0083403 |
| [0.5, 0.6) | 0.0000834 | 0.0153689 |
| [0.6, 0.7) | 0.0004019 | 0.0297644 |
| [0.7, 0.8) | 0.0030656 | 0.0552470 |
| [0.8, 0.9) | 0.0313688 | 0.1252905 |
| [0.9, 1.0] | 0.2611881 | 0.3514242 |

两个 split 都呈清晰单调上升，因此“相对风险排序含有教师错误信息”这一基础假设成立。

## 6. 排名消融与失败定位

| 分数 | Train AP | Train AUC | Val AP | Val AUC |
|---|---:|---:|---:|---:|
| Full：confidence + variance | 0.3855771 | 0.9634619 | 0.3698922 | 0.9017147 |
| Confidence-only | 0.3943625 | 0.9632829 | 0.3856800 | 0.9024002 |
| Variance-only | 0.3840141 | 0.9633410 | 0.3676450 | 0.9012806 |

| Split | Full AP − confidence AP | 允许下限 | 通过 |
|---|---:|---:|---|
| Train | -0.0087854 | -0.005 | 否 |
| Val | -0.0157878 | -0.005 | 否 |

train 的 Full AUC 仅比 confidence-only 高约 0.000179，但 AP 已越过容忍范围；val 的 AP 和 AUC 都更低。教师错误是少数类，不能用 train AUC 的微小变化覆盖预注册 AP 失败。

最窄结论是：置信度已经承担主要区分作用，当前非主类方差项没有提供稳定增量，反而降低了高风险头部的教师错误排序质量。

## 7. 温度与数值机制审计

| 指标 | Train | Val |
|---|---:|---:|
| active / solved | 32,246,990 / 32,246,990 | 3,878,674 / 3,878,674 |
| fallback / tie / nonfinite | 0 / 0 / 0 | 0 / 0 / 0 |
| 温度方向违规 / argmax 改变 | 0 / 0 | 0 / 0 |
| 目标残差 mean / p95 / max | 3.9135e-05 / 7.5817e-05 / 1.1349e-04 | 3.9701e-05 / 7.5340e-05 / 1.0204e-04 |
| 温度 mean / harmonic mean | 0.766502 / 0.594266 | 0.782602 / 0.599948 |
| 温度 q10 / q50 / q90 | 0.500004 / 0.500004 / 1.945015 | 0.500004 / 0.500004 / 1.959251 |
| `T_out=1` vs `T_out=3` 路由 mismatch | 0 | 0 |

全部像素成功求解，低风险侧没有被误送到平滑方向，高风险侧没有被误送到锐化方向，路由也没有被 `T_out` 改变。因此 O1 失败不是求解器或温度机制异常。

## 8. 泛化性与审稿口径

冻结训练集 CDF 不等同于记住训练标签。CDF 只依赖冻结教师的无标签输出，训练时每个像素也只查询固定统计量；标签仅在 O1 离线评估中判断路由是否有信息量。val 使用同一 train CDF 仍获得 3.92 倍错误富集，是同数据集 train-to-val 转移证据。

跨数据集时，教师置信度分布、类别难度和域偏移都会改变。当前方法应定位为 dataset-adaptive training method：允许在新目标数据集的无标签训练输出上重建 CDF，同时固定 `q`、gate 宽度、温度端点和目标力度。若要声称 zero-shot 泛化，必须另做直接复用 VOC CDF 的对照。

冻结 CDF 的说服力来自：统计量在学生训练前冻结、不使用标签、并在 val 上用同一 CDF 验证风险富集，而不只是报告 train 自洽结果。

## 9. 联合门禁与停止决定

文件、CDF SHA、配置指纹、教师 SHA、native-grid 语义、完整运行、非有限值和路由不变性检查全部通过，且 `errors=[]`。train/val 唯一失败项都是 `full_ap_not_below_conf_by_0p005`。

```text
joint_gate_pass = false
```

- 不启动 O2 六变体 20-iteration smoke；
- 不启动 O3、20k、80k 或三种子实验；
- 不产生、也不报告学生 mIoU；
- 不放宽阈值，不覆盖失败 artifact。

## 10. 下一步建议：重新预注册 O1.1

最小且与数据一致的修改，是把训练时风险分数改为 confidence-only：

```text
r_conf(i) = -log(c_i)
u_i       = F_train,conf(r_conf(i))
```

保留双向目标置信度、`q=0.8`、`w=0.05`、`T_R=0.5`、`T_U=2`、两侧 `alpha=1`、16 步二分和 masked KD。使用新 artifact 名重建 confidence-only CDF，并完整重跑 train/val 路由诊断；不能覆盖本次 full-score CDF，也不能直接进入学生训练。

不能沿用 `AP_full >= AP_confidence_only - 0.005` 作为 O1.1 晋级条件，因为新主分数与 confidence-only 完全相同，会形成必然通过的同义比较。运行前应冻结非同义门禁：

- train/val 高风险覆盖率均在 0.15–0.25；
- 高风险错误率至少为全局错误率的 2 倍，教师错误 recall 不低于 0.70；
- 低风险错误率低于全局错误率，十分位错误率 pairwise 单调一致率不低于 0.90；
- nonfinite、fallback、tie、方向违规与双 `T_out` mismatch 均为 0；
- 目标残差 p95 小于 `1e-3`，CDF 与所有来源/配置指纹检查通过。

confidence-only 是在查看本次 O1 train/val 后选出的探索性修订，因此同一 VOC 上重跑只验证实现和域内路由，不能作为独立确认。确认性证据必须来自未参与评分选择的新教师或新数据集，并在评估前冻结全部方法参数。

若保留方差项，备选做法只能是在看新验证结果前，用训练集内部划分固定系数选择规则，再进行一次 val 确认。当前 train/val 都偏向 confidence-only，因此不作为首选。

## 11. 正式产物

| 产物 | SHA256 |
|---|---|
| `voc_train_rtc_cdf.pt` | `40b4454fc919422899e512fed4ece4a428d12544e7ce828a89dc731d12529e39` |
| `rtc_routing_train.json` | `c51573f23264b89817e39b2d448fec61709b28147e83a0af1feb83f8f420eb98` |
| `rtc_routing_val.json` | `308065649920b33e920dd84353be966f4ead7e2cee91dbd64d263105b9eea383` |
| `o1_joint_gate.json` | `44997118f45a7c1e32f0028d5c50a8d1206e440a6afc4d07a35ae5852a3bd1af` |

配置指纹：`914a07126502eb63c139f38f10749e17633eea9d546c3e3c7a2b7f5700ed115a`。

以上结论只覆盖正式 O1 路由诊断，不构成 RTC-KD 学生性能结论。
