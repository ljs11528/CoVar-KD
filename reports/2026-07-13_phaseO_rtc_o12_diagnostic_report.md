# Phase O1.2 预算约束 confidence-only RTC 正式机制诊断报告

- 日期：2026-07-13
- phase：O1.2
- 风险定义：confidence-only，r=-log(c)
- 正式联合门禁：通过
- 当前完成阶段：O1.2-A 无学生全量机制诊断，以及 O1.2-B 两条 20-step 训练链路 smoke
- 学生训练：仅完成 `neutral` 与 `unreliable_only` 各 20-step；未运行 validation 或 20k
- 当前边界：O1.2-B 两条 fresh smoke 与端点 resume=0 审计已通过；停在 B 后人工审查线，不自动启动其他变体或 20k
- 证据性质：同一 VOC 设置上的探索性机制开发，不是独立确认或跨数据集泛化证据

## 1. 结论先行

O1.2-A 的正式 parameters、train、val 和 joint gate 四份产物均通过独立检查。当前可以确认的是：

1. confidence-only 风险继续把教师错误集中到相对高风险尾部；
2. 新映射没有再对大面积像素执行强低温锐化，而是把可靠侧限制为轻微锐化、把中间区域保持为严格中性、把主要温度预算用于高风险尾部平滑；
3. train 上温度算术均值为 0.9950000003、调和均值为 0.9880688429，val 在不重拟合时分别为 0.9960921506 和 0.9881613751；
4. train/val 的温度方向、目标置信度方向、目标熵方向、教师 argmax、学生 softmax 不变性、数值范围和人口闭合违规均为 0；
5. O1.1 中“算术均值看似尚可、调和均值却约为 0.6，可能等价于广泛强锐化”的数值隐患，在 O1.2 空间教师目标分支上已经被预算约束解决。

这仍然不能证明学生 mIoU 会提高，也不能证明收益来自正确的空间位置。O1.2-B 只有两条 20-step 链路与终点 checkpoint 加载证据，没有 validation、学生预测、matched scalar 或 within-image shuffle 对照。

| 正式部分 | 结果 | 关键说明 |
|---|---|---|
| 参数求解 | 通过 | 完整 train 风险人口，64 次 float64 二分 |
| Train 机制诊断 | 通过 | 10,582 张，32,246,990 个 native-valid 像素 |
| Val 无重拟合诊断 | 通过 | 1,449 张，3,878,674 个 native-valid 像素 |
| 独立联合 checker | 通过 | joint_gate_pass=true |
| 学生训练 | 仅 2×20-step smoke 通过 | 无 validation/mIoU，不能给出效果结论 |

## 2. 方法与口径

### 2.1 风险与冻结 CDF

教师风险只使用原始教师分布在评估温度 1.0 下的最大类别置信度：

~~~text
p_assess(i,k) = softmax(z_t(i,k) / 1.0)
c_assess(i)    = max_k p_assess(i,k)
r_i            = -log(clamp(c_assess(i), 1e-8, 1-1e-8))
u_i            = F_train,confidence(r_i)
~~~

其中 F_train,confidence 是 O1.1 冻结的 train confidence CDF，使用右连续阶梯查询。O1.2 没有重建或覆盖该 CDF：

~~~text
runs/diagnostics/phaseO_o11/voc_train_rtc_confidence_cdf.pt
SHA256 = 8ba8376a03c2f93835339d98670fa357b381b23a22cbf2a7fc48b0c2441d7e69
~~~

风险计算不使用类别值、教师对错或学生输出，但统计人口仍使用 GT valid/ignore 掩码排除 ignore、void 和 padding，因此不能简称为完全 label-free。

需要区分两个“置信度”：

- c_assess 用于风险排序，来自 z_t/1.0；
- 正式 JSON 中的 c_base 和 c_target 用于教师目标诊断，c_base 已包含 T_out=3.0。二者不是同一个数值口径。

### 2.2 三段非对称门函数

~~~text
clip01(x) = min(max(x,0),1)

g_R(u) = clip01((0.6-u)/0.6)^1
g_U(u) = clip01((u-0.8)/0.2)^2
~~~

由此得到三个互斥区域：

- u<0.6：可靠侧，g_R>0，只允许轻微锐化；
- 0.6<=u<=0.8：中性区，g_R=g_U=0，严格 T=1；
- u>0.8：不可靠侧，g_U>0，只允许连续平滑，且平方门把主要强度集中到风险尾部。

### 2.3 预算温度

~~~text
a*    = -log(0.9) = 0.10536051565782628
ell_i = -a* g_R(u_i) + b* g_U(u_i)
T_i   = exp(ell_i)
~~~

固定 a* 后，只使用完整 train 风险人口在 0<=b<=log(1.5) 上求唯一 b*，使：

~~~text
mean_train(T) = 0.995
harmonic_mean_train(T) >= 0.98
~~~

正式解为：

~~~text
b*                       = 0.3476499170064926
64 次二分后的均值残差    = 2.7882274267199136e-10
理论高风险端点 exp(b*)   = 1.4157365376897748
实际观察最大温度          = 1.4145361185073853
~~~

理论端点与实际最大值的微小差异来自正式人口中最大可查询 u 未必精确等于 1，不是温度越界或求解失败。

### 2.4 教师目标单侧校准

O1.2 的空间温度只进入教师目标：

~~~text
z_teacher_base(i,k) = z_t(i,k) / 3.0
q_teacher(i,k)      = softmax(z_teacher_base(i,k) / T_i)
T_effective(i)      = 3.0 * T_i

p_student(i,k)      = softmax(z_s(i,k) / 1.0)
L_O12               = sum_i M_i KL(q_teacher(i) || p_student(i))
                      / sum_i M_i
~~~

教师目标被 detach；空间温度不进入学生 softmax，不乘 T_i 的额外幂次。DDP 训练入口按全局 native-valid 像素数归一化。该定义只描述新增 O1.2 pixel-KL 分支，不能扩展解释为整个 CWD 损失。

## 3. 正式预算与温度分布

### 3.1 参数人口与可行性

参数只由 train 的 32,246,990 个 finite native-valid 风险值求解，nonfinite=0。solve 与 train evaluate 风险缓存 SHA 完全相同：

~~~text
e3d0b16b3082464a25558aa00ab26b319a4020cadce6aebf4055dd84b9d8eade
~~~

独立 checker 重新载入 float32 风险缓存、重新执行 64 次 float64 二分，并确认：

- b* 绝对误差不超过 1e-8；
- 算术均值目标满足 1e-4 容差；
- 调和均值不低于 0.98；
- 理论高风险端点不低于 1.25；
- 4097 点 float64 公式网格全局单调不减，违规数为 0。

### 3.2 温度分位数

| 统计量 | Train | Val，无重拟合 |
|---|---:|---:|
| native-valid | 32,246,990 | 3,878,674 |
| nonfinite | 0 | 0 |
| min | 0.8999999762 | 0.8999999762 |
| q01 | 0.9015446901 | 0.9014287591 |
| q10 | 0.9156452417 | 0.9128624201 |
| q50 | 0.9824246764 | 0.9790610671 |
| q80 | 1.0000000000 | 1.0010589361 |
| q90 | 1.0896940231 | 1.1127897501 |
| q95 | 1.2158285379 | 1.2398405075 |
| q99 | 1.3685089350 | 1.3774102926 |
| max | 1.4145361185 | 1.4145361185 |
| top-risk decile mean | 1.2285796997 | 1.2333745846 |
| 算术均值 A | 0.9950000003 | 0.9960921506 |
| 调和均值 H | 0.9880688429 | 0.9881613751 |

val 完全复用 train 的 a*、b*，parameters_refit=false。val 的 A 没有被强行调回 0.995，但仍处于预注册的 [0.98,1.02] 区间，H 也高于 0.97。

### 3.3 激活覆盖与有效温度

| 统计量 | Train | Val |
|---|---:|---:|
| T=1 覆盖 | 19.982101% | 18.495651% |
| T>1 覆盖 | 19.815177% | 21.063642% |
| T>1.25 覆盖 | 3.955306% | 4.643030% |
| T<0.9 | 0 | 0 |
| T>1.5 | 0 | 0 |

可靠侧约 60% 的像素虽然 T<1，但最低仅为 0.9，且中位数接近 0.98；这与 O1.1 大量像素贴近 T=0.5 的结构不同。T_effective=3T 的 train/val 算术均值分别为 2.9850000000 和 2.9882764507，调和均值分别为 2.9642065272 和 2.9644841238。

正式参数还保存了后续严格同损失标量对照所需的数值：

| 温度图 | 算术均值匹配标量 | 调和均值匹配标量 |
|---|---:|---:|
| unreliable_only | 1.0256612288 | 1.0212646674 |
| full_budgeted | 0.9950000003 | 0.9880688429 |

这些标量只为后续预注册对照保留，本轮不授权启动它们。

## 4. 三个风险区域

### 4.1 精确人口与教师错误

| Split | 区域 | 像素数 | 覆盖率 | 教师错误数 | 教师错误率 | 占全部教师错误 | T 均值 |
|---|---|---:|---:|---:|---:|---:|---:|
| Train | reliable，u<0.6 | 19,413,566 | 60.2027% | 613 | 0.003158% | 0.0645% | 0.949070 |
| Train | neutral，0.6<=u<=0.8 | 6,435,793 | 19.9578% | 11,764 | 0.182790% | 1.2383% | 1.000000 |
| Train | unreliable，u>0.8 | 6,397,631 | 19.8395% | 937,607 | 14.655534% | 98.6971% | 1.129344 |
| Val | reliable，u<0.6 | 2,344,298 | 60.4407% | 12,201 | 0.520454% | 5.0545% | 0.945652 |
| Val | neutral，0.6<=u<=0.8 | 716,449 | 18.4715% | 29,734 | 4.150191% | 12.3178% | 1.000000 |
| Val | unreliable，u>0.8 | 817,927 | 21.0878% | 199,455 | 24.385428% | 82.6277% | 1.137237 |

u=0.6 和 u=0.8 的精确边界人口在 train/val 都为 0，checker 仍独立核对了这些计数。

### 4.2 解释

- 高风险侧不是错误标签，但确实是最应优先降低硬模仿强度的区域；
- reliable 并不等于绝对正确，尤其 val 仍有 12,201 个错误像素；
- val 的 neutral 和 reliable 错误占比高于 train，说明风险排序跨 split 保持相对意义，但不能把 train 风险分区当成确定性正确性判别；
- 三段人口和错误数都与 O1.1 冻结统计闭合，O1.2 没有通过改变人口口径制造更好的数字。

## 5. 教师目标置信度与熵

表中的 c_base 是 T_out=3 且空间 T=1 的教师目标置信度，不是风险定义中的 c_assess。

| Split | 区域 | c_base -> c_target | 置信度变化 | entropy_base -> entropy_target | 熵变化 |
|---|---|---:|---:|---:|---:|
| Train | reliable | 0.821976 -> 0.849558 | +0.027582 | 0.967063 -> 0.834891 | -0.132172 |
| Train | neutral | 0.695818 -> 0.695818 | 0 | 1.441044 -> 1.441044 | 0 |
| Train | unreliable | 0.505756 -> 0.455059 | -0.050697 | 1.848118 -> 2.048945 | +0.200827 |
| Val | reliable | 0.829952 -> 0.858559 | +0.028607 | 0.931291 -> 0.792721 | -0.138570 |
| Val | neutral | 0.696063 -> 0.696063 | 0 | 1.437311 -> 1.437311 | 0 |
| Val | unreliable | 0.499127 -> 0.447298 | -0.051829 | 1.846685 -> 2.054602 | +0.207917 |

平方高风险门确实把主要平滑强度放到了最危险尾部：

| Split / 风险十分位 | T 均值 | c_base -> c_target | entropy_base -> entropy_target |
|---|---:|---:|---:|
| Train bin 0 | 0.907967 | 0.899880 -> 0.937160 | 0.613593 -> 0.413691 |
| Train bin 9 | 1.228580 | 0.423840 -> 0.341345 | 1.977940 -> 2.313915 |
| Val bin 0 | 0.907952 | 0.900750 -> 0.937788 | 0.608771 -> 0.409902 |
| Val bin 9 | 1.233375 | 0.419218 -> 0.337747 | 1.970094 -> 2.306062 |

train/val 的可靠侧置信度下降、不可靠侧置信度上升、可靠侧熵上升和不可靠侧熵下降违规均为 0；教师 argmax mismatch 也为 0。方向机制成立，但这只是目标分布性质，不等于学生学到了更好的决策边界。

## 6. 分层诊断：前景、边界与小目标

这些分层只用于解释，不参与风险、门函数、预算求解或晋级门禁。

### 6.1 前景与背景

| Split | 分层 | 教师错误率 | T 均值 |
|---|---|---:|---:|
| Train | background | 2.0774% | 0.976449 |
| Train | foreground | 4.4891% | 1.027958 |
| Val | background | 2.3191% | 0.973647 |
| Val | foreground | 17.0508% | 1.058335 |

前景获得的平均温度高于背景，尤其 val 前景错误率明显更高。这说明全局 pixel micro 指标容易被背景主导，后续学生结果不能只看全像素模仿率。

### 6.2 边界与内部

| Split | 分层 | 像素数 | 教师错误率 | T 均值 | c_base -> c_target | entropy_base -> entropy_target |
|---|---|---:|---:|---:|---:|---:|
| Train | boundary | 9,422,384 | 9.5826% | 1.063205 | 0.637351 -> 0.614838 | 1.475616 -> 1.560879 |
| Train | interior | 22,824,606 | 0.2063% | 0.966844 | 0.773985 -> 0.792529 | 1.137726 -> 1.046399 |
| Val | boundary | 1,009,838 | 15.5175% | 1.055739 | 0.650768 -> 0.631834 | 1.427449 -> 1.497599 |
| Val | interior | 2,868,836 | 2.9520% | 0.975096 | 0.765268 -> 0.780532 | 1.143998 -> 1.065351 |

边界错误率是内部的约 46.46 倍（train）和 5.26 倍（val），且边界平均 T>1、内部平均 T<1。confidence-only 风险因此确实把更多平滑预算分配给边界困难区域。但边界也可能包含对分割轮廓有用的正确软信息，所以该相关性不能替代学生端的 rescue、retention 和 mIoU 检验。

### 6.3 小目标

| Split | 分层 | 像素数 | 覆盖率 | 教师错误率 | T 均值 | c_base -> c_target | entropy_base -> entropy_target |
|---|---|---:|---:|---:|---:|---:|---:|
| Train | small_object | 27,976 | 0.0868% | 59.3330% | 1.235643 | 0.399662 -> 0.328323 | 2.055091 -> 2.348309 |
| Train | not_small_object | 32,219,014 | 99.9132% | 2.8970% | 0.994791 | 0.734352 -> 0.740966 | 1.235745 -> 1.195728 |
| Val | small_object | 10,475 | 0.2701% | 55.2267% | 1.168845 | 0.480531 -> 0.422210 | 1.869043 -> 2.107208 |
| Val | not_small_object | 3,868,199 | 99.7299% | 6.0908% | 0.995624 | 0.736147 -> 0.742683 | 1.216033 -> 1.175373 |

小目标像素错误率分别约为非小目标的 20.48 倍和 9.07 倍，温度明显偏向平滑，符合“高风险区域减少错误硬监督”的设计方向。不过小目标人口很小、像素高度相关，不能据此声称小目标类别性能会改善。

## 7. O1.1 风险证据复核

O1.2 重新报告并由 checker 从整数计数复算了 O1.1 证据：

| 指标 | Train | Val |
|---|---:|---:|
| 全局教师错误率 | 0.029459618 | 0.062235187 |
| 高风险覆盖率 | 0.198394672 | 0.210877996 |
| 高风险教师错误率 | 0.146555342 | 0.243854280 |
| 高风险错误 recall | 0.986971360 | 0.826276979 |
| 错误富集倍数 | 4.974787630 | 3.918270256 |
| 十分位 pairwise 单调一致率 | 1.000000 | 1.000000 |
| 十分位 Spearman 近似 | 1.000000 | 1.000000 |

十个风险分箱的教师错误率从低到高严格呈单调趋势：

~~~text
Train:
0,
0.000001860,
0.000004981,
0.000016195,
0.000039732,
0.000126117,
0.000491669,
0.003177479,
0.031711367,
0.260882381

Val:
0.000306965,
0.001421968,
0.003157578,
0.005595617,
0.009002327,
0.014987220,
0.028493649,
0.053505198,
0.123974235,
0.351207857
~~~

这部分只能称为 O1.1 证据的复核，不是 O1.2 温度机制的新门禁，也不是新的独立验证。它说明风险顺序值得用于后续探索，但不能把 u>0.8 称为教师错误标签。

## 8. 数值机制与全零违规

train 和 val 的正式 violations 字段逐项均为 0：

| 违规类别 | Train | Val |
|---|---:|---:|
| nonfinite | 0 | 0 |
| reliable T>1 | 0 | 0 |
| neutral T!=1 | 0 | 0 |
| unreliable T<1 | 0 | 0 |
| T(u) 公式单调性 | 0 | 0 |
| reliable 目标置信度下降 | 0 | 0 |
| unreliable 目标置信度上升 | 0 | 0 |
| reliable 目标熵上升 | 0 | 0 |
| unreliable 目标熵下降 | 0 | 0 |
| 教师 argmax mismatch | 0 | 0 |
| 温度越界 | 0 | 0 |
| neutral 目标不一致 | 0 | 0 |
| 空间温度改变学生 softmax | 0 | 0 |
| valid/finite 人口不闭合 | 0 | 0 |
| 风险分箱人口不闭合 | 0 | 0 |
| 教师错误计数不闭合 | 0 | 0 |

补充数值探针：

| 指标 | Train | Val |
|---|---:|---:|
| neutral target 最大绝对误差 | 0 | 0 |
| student softmax 最大绝对误差 | 0 | 0 |
| 两种教师目标探针最大绝对差 | 0.109136969 | 0.109239101 |

学生 softmax 的误差为 0，说明空间温度没有泄漏到学生端；教师目标差异非零，说明探针不是因为整条温度链路失效才得到全零违规。所有类别、前景/背景、边界/内部、小目标以及风险区的人口闭合检查均为 true。

联合 checker 还确认：

- O1.1 四份冻结源码字节级一致；
- CDF、teacher、train/val list 的实际 SHA 与冻结契约一致；
- parameters、train、val 路径均为 canonical path；
- val 没有重新求参；
- train solve/evaluate 风险缓存完全一致；
- parameters、train、val 三个独立 evaluation 均 pass；
- 每个产物都是单进程、单 NPU；solve/train 使用 NPU 0，val 使用 NPU 1，设备编号只作为执行来源记录，不是统计定义。

## 9. 调和均值隐患是否解决

就空间温度数值机制而言，答案是“已解决”；就学生效果而言，答案仍是“未知”。

| 指标 | O1.1 Train | O1.2 Train | O1.1 Val | O1.2 Val |
|---|---:|---:|---:|---:|
| 算术均值 A | 0.766246 | 0.995000 | 0.784605 | 0.996092 |
| 调和均值 H | 0.594183 | 0.988069 | 0.600597 | 0.988161 |
| 中位数 T | 0.500004 | 0.982425 | 0.500004 | 0.979061 |
| 最低 T | 约 0.5 | 0.9 | 约 0.5 | 0.9 |

O1.2 的 mean(1/T)=1/H 约为 1.0121（train）和 1.0120（val），不再存在 O1.1 中 H 约 0.6 所反映的强平均逆温度。可靠侧仍约占 60%，但锐化下限被限制为 0.9，且中性区域有约 18% 至 20% 的严格 T=1；高风险尾部同时获得 T>1 的平滑预算。

因此可以排除“新温度图仍主要是大面积强低温锐化”这一数值解释，但不能排除以下性能解释：

- 轻微的整体温度分布变化本身就足够影响训练；
- 高风险位置不重要，只有温度直方图或均值重要；
- 对可靠侧的轻微锐化仍是主要贡献；
- 新增 O1.2 pixel-KL 与其他固定 CWD 分支的交互决定最终结果。

此外，O1.2 的 H 只解释 teacher-target-only 分支的空间教师目标，不解释学生梯度温度缩放。它与 O1.1 的历史训练语义不是严格的性能同义比较。

## 10. 局限与审查风险

1. q_R=0.6、q_U=0.8、幂次和预算是在查看 O1.1 的 VOC 结果后提出的；同一 VOC-val 只能作为探索性开发集。
2. 当前虽有两个 20-step 学生 checkpoint，但没有 validation、学生预测、rescue、error imitation、teacher-correct retention 或 mIoU 结果。
3. 风险排序与边界、小目标和前景难例高度相关。平滑错误教师监督可能有益，但也可能削弱正确且有价值的细粒度监督。
4. pixel micro 统计受背景、大区域和像素相关性影响；数百万像素不能替代图像级 bootstrap、seed 级重复或类别平衡指标。
5. 风险定义使用 GT valid mask 限定人口，不应宣传为完全 label-free。
6. 本次 val 只证明使用 train 冻结 CDF 和参数时的同数据集无重拟合稳定性，不能推出新数据集或新教师泛化。
7. 换数据集时应保持风险公式、q_R/q_U、幂次、预算目标和温度上下限不变，但在新数据集训练集上重建 confidence CDF，并只用新训练风险人口求 b*；直接复用 VOC CDF 不是预期的 zero-shot 泛化方式。
8. 正式产物记录了 dirty worktree；复现必须依赖产物中的 commit、完整 argv、配置指纹和源码/输入 SHA，不能只依赖 commit 名。
9. 当前尚无 matched scalar 和 within-image shuffle，不能声称空间风险位置具有独立贡献。

## 11. O1.2-A 当时的下一步边界

本节保留 O1.2-A 结束时的历史授权边界；该授权随后已于 2026-07-13 给出并执行，当前状态见第 13 节。O1.2-A joint gate 通过后，第一批只允许：

1. neutral；
2. unreliable_only。

该 smoke 只检查：

- 单 NPU 设备、显存、吞吐和 loss finite；
- teacher-target-only 温度语义；
- 梯度 finite；
- checkpoint 保存与恢复；
- 同 seed 样本顺序及 1-based shuffle 恢复一致；
- 20 iteration 链路完整性。

20 iteration 不产生也不得宣称 mIoU 效果结论。在 O1.2-A 当时，不允许启动 reliable_only、full_budgeted、scalar、shuffle、20k 或 80k。上述 neutral/unreliable_only smoke 后续已按授权完成；本节不改写其预注册前置边界。

## 12. 正式产物与指纹

| 产物 | SHA256 |
|---|---|
| o12_budget_parameters.json | a9006972e165ed6c95e7a966c526927bb9f846fb6c7ec6528f5507dd21b8b5df |
| o12_budget_train.json | 8deb4850a629e7a7b44a6ed52bf86b988857e98bf786ee39e6e948995f448b6e |
| o12_budget_val.json | d1dde29c569df7a6fabda32ec3daa50bb6a16758125768e5059c1762cb0b1a5f |
| o12_joint_gate.json | c1961cfec8f232c1fdf7fc8b90db5a6f4855901317ea2705536eea88cbc1be82 |

联合门禁记录的源码 SHA：

| 源码 | SHA256 |
|---|---|
| utils/rtc_o12_calibration.py | 5e23f3e1bd526017aa2046faff621dd284093d91dead9088dd4d7d9ddb641d9e |
| scripts/diagnostics/diagnose_rtc_o12_budget.py | cc391388f64505abae4cded5ac7b36122018a131b3c90f480290ff275a4cee50 |
| scripts/diagnostics/check_rtc_o12_gate.py | 805a19625d496d3c3864d529e314a49d75584afad69fedf69e28cadc431ce085 |
| train_kd.py | f50a130c17ca3b5f368f848b96c59fd58c408952e75bda447c051589505b316d |

第 1 至 12 节解释上述正式 JSON 中已经生成的 O1.2-A 证据；以下第 13 节追加 O1.2-B 的链路审计结果。追加内容不改变预注册门槛，也不把 smoke 升格为性能实验。

## 13. O1.2-B 20-step 链路诊断追加

### 13.1 接受结果

2026-07-13，`neutral` 与 `unreliable_only` 的 fresh 20-step run，以及各自从 iteration=20 checkpoint 执行的端点 resume=0 审计，四份 final acceptance 均为 `pass=true`：

| 变体/模式 | Acceptance SHA256 | Checkpoint SHA256 | 结果 |
|---|---|---|---|
| neutral fresh | `90eb89fe7a7dff773845152d5d59a7efcc3d61299976137223be6cf7d2615e1c` | `6309cba985d8831586a610c45a4f463ba15d71381924b7093ca9f9fa91a982a9` | 20 optimizer steps，pass |
| neutral endpoint resume=0 | `30fe00fa72b2f20293db2cfca3da7f0f9d960649fef8d153629b3d2b2f496677` | `6b0032efd44202181cb2bf4248209b788ea9573b92541ffe339562804d7d3e44` | 0 optimizer steps，pass |
| unreliable_only fresh | `d8a3363724c7c5d93c7629ba0ebf365a7f73d1c25c53138633e179e21fe623cd` | `b9426976a45ab1ec6396bb68d85f2f43fe80f176d94fde8c5395580a41ea0ed4` | 20 optimizer steps，pass |
| unreliable_only endpoint resume=0 | `af540bf53a0b1646227eb9024a6eb9f5d7bf1af250773c62d1074b77aaa82488` | `8760408718353c0e8123d4a4a5965eb63e8db504e23724d9b57c19ce77362aca` | 0 optimizer steps，pass |

两条 fresh run 的 finite scan 都覆盖 847 个 tensor、19,711,341 个 tensor element 和 7 个 float scalar，结构化 `errors=[]`、`warnings=[]`。step 20 的 KD-only student-logit gradient L2 分别为 neutral `0.00273925`、unreliable_only `0.00282540`，均有限且非零。该证据说明指定 KD 路径具有梯度信号；它不是每个模型参数梯度的逐张量扫描。

### 13.2 路由诊断能说什么

fresh step-20 诊断为：

| 指标 | neutral | unreliable_only |
|---|---:|---:|
| O1.2 KD KL | 0.95004142 | 1.00187660 |
| teacher target entropy | 1.22812260 | 1.26829381 |
| KD-only student-logit gradient L2 | 0.00273925 | 0.00282540 |
| teacher output temperature | 3.0 | 3.0 |
| valid pixel | 43,209 | 43,209 |

这些数字只用于验证两个冻结变体确实走通各自教师目标链路、数值有限且 KD 梯度非零。因为教师目标不同、只有一个 step-20 记录点且未验证，它们不能支持“unreliable_only 更好/更差”、收敛更快或机制有效的结论。

### 13.3 顺序与恢复边界

四份 acceptance 的 canonical order 都是 320 个 dataset index，seed=`1234`，SHA256=`99326472a2e5e2bd42428d4709ff9f8049d2c906c7a6b9e8fa3068cb0439d564`。这证明 canonical 索引顺序一致；没有实际 `sample_names` 序列证据，也没有证明 8-worker 随机增强按位复现。

两个 resume audit 均从对应 fresh iteration=20 checkpoint 严格加载，关键状态与 student-weight SHA 相等，并以 `optimizer_steps=0` 结束。这是终点 checkpoint 的加载审计，不是恢复后 next batch/next step 的实跑证据；不能用单元测试中的 sampler slicing 代替实际 continuation。

### 13.4 环境告警与证据级别

控制台原始日志保留了 CANN owner mismatch 告警；fresh run 保留内部格式禁用后回退 base format 的告警；resume audit 保留旧权重文件格式/torch 兼容性及未来弃用告警。这些没有触发 final checker 的结构化 error/warning，但不得从记录中删除，也不属于方法表现。

本阶段使用 `skip-val`，未生成 validation、预测或 mIoU，也没有 rescue、error imitation、teacher-correct retention。故新增的唯一结论是训练/数值/checkpoint/终点严格加载链路通过；不是性能、收敛、空间位置因果或泛化结论。

当前停在 O1.2-B 后人工审查线，不自动启动 `reliable_only`、`full_budgeted`、scalar、shuffle、20k、C2、C3 或 80k。完整证据见 [O1.2-B 20-step 正式链路 smoke 报告](2026-07-13_phaseO_rtc_o12b_smoke_report.md) 与 [O1.2 执行记录](2026-07-13_phaseO_rtc_o12_execution_record.md)。
