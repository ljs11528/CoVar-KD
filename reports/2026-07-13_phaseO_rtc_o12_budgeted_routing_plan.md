# Phase O1.2：高风险优先的预算约束教师置信度校准计划

- 制定日期：2026-07-13
- 当前状态：O1.2-A 联合门禁通过；O1.2-B 的 `neutral` 与 `unreliable_only` 各 20-step fresh smoke 及 iteration=20 端点 resume=0 审计均通过；停在 B 后人工审查线
- 方法定位：探索性的空间机制因果筛查，不是独立确认实验
- 风险定义：confidence-only，保持 O1.1 不变
- 第一开发底座：CWD，VOC，教师 Tout=3.0
- 主原则：先单独验证高风险保护，再决定是否加入轻微可靠侧锐化

> 本文是 O1.2 的唯一规范定义。若它与 Phase O 主计划中 O1/O1.1 的历史公式冲突，O1.2 以后文为准。O1 的历史产物、失败结论和当时源码指纹不得改写；O1.1 当前四份冻结源码、正式结果和产物必须继续保持字节一致。

## 0. 决策摘要

O1.1 已证明 confidence-only 风险能把教师错误稳定富集到相对高风险侧，但旧温度映射几乎对所有像素激活：

- train/val 高风险侧约覆盖 19.84%/21.09% 像素；
- 高风险侧召回 98.70%/82.63% 的教师错误；
- 高风险侧仍有 85.34%/75.61% 的教师预测正确；
- 温度中位数均约为 0.500004；
- 温度调和均值约为 0.594/0.601，几乎等同历史强标量 T=0.6。

因此 O1.2 不再采用“bottom 80% 强锐化、top 20% 强平滑”的全激活结构，而采用：

1. bottom 60% 仅允许逐渐减弱的轻微锐化；
2. middle 20% 形成严格中性区；
3. top 20% 随风险连续增强平滑；
4. 温度算术均值只允许比 1 少量降低；
5. 同时约束调和均值，避免平均逆温度重新接近标量 T=0.6；
6. 空间温度只校准教师目标，学生端温度固定；
7. 第一项学生实验只比较 neutral 与 unreliable_only。

预注册冻结时不授权 O1.2 学生训练，要求先完成独立实现、单元测试、全量 train/val 机制诊断和联合门禁。该 O1.2-A 前置阶段已完成并通过；随后在 2026-07-13 获得明确授权并完成 O1.2-B 的两条 20-step 正式训练链路 smoke。该授权只覆盖 `neutral`、`unreliable_only` 及各自的端点恢复审计，不自动授权 20k、其他变体或性能结论。

## 1. O1.1 证据与正确解读

### 1.1 三个风险区域

由 O1.1 全量 native-valid 十分位整数统计得到：

| Split | 区域 | 像素覆盖率 | 教师错误率 | 占全部教师错误 |
|---|---|---:|---:|---:|
| Train | bottom 60%，u<0.6 | 0.602027 | 0.0000316 | 0.000645 |
| Train | middle 20%，0.6<=u<=0.8 | 0.199578 | 0.001828 | 0.012383 |
| Train | top 20%，u>0.8 | 0.198395 | 0.146555 | 0.986971 |
| Val | bottom 60%，u<0.6 | 0.604407 | 0.005205 | 0.050545 |
| Val | middle 20%，0.6<=u<=0.8 | 0.184715 | 0.041502 | 0.123178 |
| Val | top 20%，u>0.8 | 0.210878 | 0.243854 | 0.826277 |

这里的等号人口在实际 4097-knot CDF 下可能为空；O1.2 checker 仍必须独立保存并核对边界计数，不能默认为零。

### 1.2 可以支持的判断

当前证据支持：

- 教师的 native-valid 像素微平均错误率总体不高；
- confidence-only 的 top 20% 是教师错误的强富集区；
- top 10% 比第二高十分位更危险，适合使用凸增的平滑强度；
- bottom 60% 相对安全，但不是教师错误的补集；
- 主方法应优先减少高风险侧的错误模仿，而不是继续扩大低风险锐化。

### 1.3 不能支持的判断

当前证据不支持：

- 把 u>0.8 当成教师错误标签；
- 把 u<0.6 当成教师必然正确标签；
- 用数百万相关像素代替图像级或 seed 级统计独立性；
- 从 VOC-val 的结果推出跨数据集泛化；
- 从 O1.1 路由诊断推出学生 mIoU 一定提高。

像素 micro accuracy 可能被背景和大区域主导。O1.2 诊断必须补充类别、边界/内部、小目标等分层报告，但这些标签只用于事后诊断，不进入风险、门函数或预算求解。

### 1.4 探索性边界

qR=0.6、qU=0.8 和预算约束是在查看 O1.1 的 train/val 结果后提出的。因此：

- O1.2 在同一个 VOC-val 上只能作为探索性开发；
- 不能把 VOC-val 写成未见确认集；
- 真正确认性证据必须来自未参与设计的新教师、新数据集或预先冻结的新设置；
- 后续不得再查看学生 mIoU 后修改 qR、qU、指数、端点或预算目标。

## 2. 不变的风险定义与冻结 CDF

### 2.1 confidence-only 风险

风险只由教师原始参考分布计算：

~~~text
p_assess(i,k) = softmax(z_t(i,k) / T_assess)
T_assess       = 1.0
c_i            = max_k p_assess(i,k)
r_i            = -log(clamp(c_i, epsilon, 1-epsilon))
epsilon        = 1e-8
u_i            = F_train,confidence(r_i)
~~~

冻结元数据：

~~~text
reliability_mode        = confidence
coefficient_a           = 0.0
active_terms            = [confidence]
coefficient_a_active    = false
~~~

风险数值和 CDF 排序不使用类别值、教师正确性或学生信息；实现仍沿用监督分割协议的 GT valid/ignore 空间掩码排除 ignore、void 与 padding，因此不能宣称完全 label-free。

### 2.2 精确复用 O1.1 CDF

O1.2 不重建、不覆盖 O1.1 confidence CDF：

~~~text
runs/diagnostics/phaseO_o11/voc_train_rtc_confidence_cdf.pt
SHA256 =
8ba8376a03c2f93835339d98670fa357b381b23a22cbf2a7fc48b0c2441d7e69
~~~

O1.2 启动前必须确认：

- O1.1 joint_gate_pass=true；
- CDF 实际 SHA 与预注册值一致；
- teacher、train list、有效像素语义和类别数与 O1.1 一致；
- O1.1 冻结源码的当前 SHA 与 CDF 元数据一致。

如任一项不一致，O1.2 结构门禁失败。不得生成同名 CDF、修改旧 SHA sidecar 或手工放行。

### 2.3 VOC 冻结输入与扫描协议

| 项目 | 冻结值 |
|---|---|
| 教师 | DeepLabV3-ResNet101 |
| 教师 SHA256 | ac49b2c7720b21d565072e974e4404fcb009ba106288d95d1a4bb25f09c3fe58 |
| 类别数 | 21 |
| train list | dataset/list/voc/train_aug.txt，10,582 张 |
| train list SHA256 | d1326bd532648d73bc4b1bd275434eba81930982dc7401a65a8cecb26c028e24 |
| val list | dataset/list/voc/val.txt，1,449 张 |
| val list SHA256 | cdc1326d12f69ce5153aa5da04a4d8783e146868d42d19f82f44e74b97ac907d |
| ignore label | -1 |
| train augmentation seed | 2025 |
| train 协议 | 512x512，scale=true，mirror=true，batch=4，workers=0 |
| val 协议 | 原始可变分辨率，scale=false，mirror=false，batch=1，workers=0 |
| bootstrap seed | 3407 |
| 预算扫描进程 | 单进程、单 NPU，不使用 DDP |
| train 预期 native-valid | 32,246,990 |
| val 预期 native-valid | 3,878,674 |
| 教师输出网格 | native logits grid |
| GT 对齐 | nearest resize 到 native logits grid |
| CDF 查询 dtype | 与 float32 risk tensor 一致 |
| CDF 查询语义 | 只读调用 O1.1 的 right-continuous step query |

预算参数严格复用 O1.1 正式 train 诊断协议、augmentation seed=2025 和预期 native-valid 人口 32,246,990，不能改用 CDF 构建人口 32,298,651，也不能使用每图 1,024 像素的 ranking sample。预算扫描固定为单进程；workers=0 消除 worker seed 歧义。val 只做无重拟合验证。设备编号不属于统计定义，但正式命令、NPU、进程和运行时间必须写入执行记录。

新模块应只读调用 O1.1 已冻结的 confidence risk、CDF loader 和 query。若未来必须重实现 query，必须用全边界、重复 knot 和随机输入证明逐元素等价后才能替换。

## 3. 非对称风险门函数

定义：

~~~text
clip01(x) = min(max(x, 0), 1)

g_R(u) = clip01((q_R-u)/q_R)^p_R
g_U(u) = clip01((u-q_U)/(1-q_U))^p_U

q_R = 0.6
q_U = 0.8
p_R = 1
p_U = 2
~~~

精确边界语义：

| 风险区域 | 门函数 | 允许的作用 |
|---|---|---|
| u<0.6 | gR>0，gU=0 | 轻微锐化 |
| 0.6<=u<=0.8 | gR=0，gU=0 | 严格中性 |
| u>0.8 | gR=0，gU>0 | 连续平滑 |

设计原因：

- qR 与 qU 分离后，中间区域具有非零测度，不再出现 active rate 接近 100%；
- 可靠侧一次函数使温度随风险逐步回到 1；
- 高风险侧平方函数把主要平滑力度集中到最危险尾部；
- 两个门严格互斥；
- O1.2 不再使用 O1/O1.1 的单阈值 tanh gate 和 w=0.05。

## 4. 预算约束的对数温度

### 4.1 主公式

~~~text
ell_i = -a * g_R(u_i) + b * g_U(u_i)
T_i   = exp(ell_i)
~~~

因此：

~~~text
u<0.6         : 0.9 <= T_i < 1
0.6<=u<=0.8  : T_i = 1
u>0.8         : 1 < T_i <= 1.5
~~~

温度必须关于 u 全局单调不减。

### 4.2 冻结可靠侧上限与高风险参数范围

可靠侧参数不再参与二维搜索，而是预先固定为允许范围内的最强轻锐化端点：

~~~text
a* = -log(0.9) = 0.10536051565782628
~~~

高风险参数范围：

~~~text
0 <= b <= b_max
b_max = log(1.5) = 0.4054651081081644
~~~

由此硬保证：

~~~text
T_min >= 0.9
T_max <= 1.5
~~~

固定 a* 后，只有 b 一个待求参数，避免在同一平均温度下事后选择不同可靠侧/高风险侧组合。任何预算求解失败都不能通过把 T_min 降到 0.5/0.6、把 T_max 提高到 2、放宽调和均值或删除中性区解决。

### 4.3 为什么必须同时约束两种均值

算术均值：

~~~text
A(a,b) = mean_train(T_i)
~~~

调和均值：

~~~text
H(a,b) = 1 / mean_train(1/T_i)
~~~

算术均值回答温度图整体是否偏离 1；调和均值直接约束平均逆温度。当前 O1.1 的主要混淆正是 A 看似尚可、H 却约为 0.6，因此 O1.2 不能只报告 A。

冻结预算：

~~~text
A_target = 0.995
abs(A-A_target) <= 1e-4
H >= 0.98
~~~

这表示像素温度的算术均值只比 1 低 0.5%，同时平均逆温度不能重新产生强全局锐化。

### 4.4 高风险优先的唯一参数求解

参数只用完整 train 风险人口求解，不使用类别标签、教师正确性、学生结果或 val 统计：

~~~text
1. 固定 a* = -log(0.9)
2. 在闭区间 [0, log(1.5)] 内求唯一 b*
3. 使 A(a*,b*) = 0.995
4. 求解后必须额外满足 H(a*,b*) >= 0.98
~~~

A 关于 b 单调增加，因此在固定 a* 后根唯一。该规则等价于先把可靠侧最低温度限制在 0.9，再把剩余的允许温度预算尽可能用于高风险平滑，不保留二维参数选择自由度。1e-4 只是正式 checker 对 A 的数值验收容差，不属于数学可行域。

实现要求：

- float64 累积 A 和 H；
- b 的左端点固定为 0，右端点固定为 log(1.5)；
- 固定执行 64 次二分：若 A(mid)<0.995，则令 left=mid，否则令 right=mid；
- 第 64 次后令 b*=(left+right)/2；
- checker 使用同一完整人口独立执行上述算法，要求参数绝对差不超过 1e-8；
- 参数和预算统计使用第 2.3 节完整 train native-valid 风险人口；
- val 只能使用 train 冻结的 a*、b*，禁止重新求解；
- 若 A=0.995 在闭区间内无根，或求得根后 H<0.98，则 O1.2 失败；
- checker 必须独立重算 a*、b*、预算残差和可行性，不能只核对报告值。

基于当前十分位人口的预注册粗略数值为：

~~~text
a*    = 0.10536051565782628
b*    approximately 0.34788
T_min approximately 0.9000
T_max approximately 1.4161
~~~

该预估不是正式结果。正式 b* 只能由 O1.2 train 预算求解器产生并由独立 checker 重算。

### 4.5 当前人口上的只读预估

使用 O1.1 正式十分位人口并假设每个十分位内部均匀进行数值积分：

| Split | 温度算术均值 | 温度调和均值 |
|---|---:|---:|
| Train | 约 0.9950 | 约 0.9881 |
| Val | 约 0.9956 | 约 0.9879 |

均匀分位近似下：

~~~text
q10 approximately 0.916
q50 approximately 0.983
q80 = 1.000
T(u=0.85) approximately 1.022
T(u=0.90) approximately 1.091
T(u=0.95) approximately 1.216
T(u=1.00) approximately 1.416
~~~

这些数字来自十分位内均匀近似，只用于证明约束具有可行性。它们不能验证正式 mean gate，也不能替代完整 32,246,990 train population 的一维求解与全像素诊断。

## 5. 教师目标单侧校准

### 5.1 目标教师分布

固定教师输出温度：

~~~text
T_out = 3.0
~~~

O1.2 的像素温度只用于构造教师目标：

~~~text
z_teacher_base(i,k) = z_t(i,k) / T_out
q_teacher(i,k)      = softmax(z_teacher_base(i,k) / T_i)
c_target(i)         = max_k q_teacher(i,k)
T_effective(i)      = T_out * T_i
~~~

方向语义：

- u<0.6 时，c_target 不低于 T=1 教师目标置信度；
- 0.6<=u<=0.8 时，教师目标与 T=1 完全一致；
- u>0.8 时，c_target 不高于 T=1 教师目标置信度；
- 所有正温度下教师 argmax 必须保持不变。

O1.2 直接用预算温度定义目标置信度，不再执行旧端点 log-odds 插值和逐像素二分反解。二分反解会恢复同一个 T_i，只增加数值路径和失败模式。

### 5.2 学生端固定温度

~~~text
T_student = 1.0
p_student(i,k) = softmax(z_s(i,k) / T_student)
~~~

空间温度不得进入学生 softmax。这样可以把“教师目标置信度校准”与“学生分布/梯度尺度变化”分开。

### 5.3 O1.2 KD 损失

~~~text
L_O12 =
    sum_i M_i * KL(q_teacher(i) || p_student(i))
    / sum_i M_i
~~~

冻结语义：

~~~text
spatial_temperature_applies_to = teacher_target_only
spatial_temperature_loss_power = not_applicable
teacher_target_detached         = true
student_temperature             = 1.0
~~~

要求：

- DDP 下按全局 native-valid 像素数归一化；
- ignore、void、padding 不进入分子或分母；
- 不乘空间 T_i^gamma；
- 改变 T_i 时，给定学生 logits 的 p_student 必须逐元素不变；
- neutral 且 T_i=1 时必须精确退化到统一 masked teacher-target KD 基线。

### 5.4 唯一训练语义与适用范围

teacher-target-only 只描述新增的 O1.2 pixel-KL 分支，不描述整个 CWD 总损失。训练实现必须同时满足：

1. O1.2 分支中的 raw teacher logits 只除一次 Tout=3.0，禁止再叠加 legacy kd_temperature；
2. teacher 与 student logits 的类别数和空间尺寸必须完全相同，不做隐式插值；不一致时 hard-fail；
3. 先在类别维求 KL 和，再用 native-valid mask 做像素归约；
4. teacher target 在 no-grad 中构造并 detach；
5. neutral 只能称为 O1.2 teacher-only masked pixel-KL baseline，不能与整个 CWD loss、legacy CriterionKD 或历史 CWD 结果混称。

O1.2 学生分支的 DDP 归一化冻结为：

~~~text
N_global = all_reduce(sum_r N_r)
L_rank   = world_size * local_KL_sum / N_global
~~~

这样经过 DDP 的梯度平均后等价于全局有效像素均值。N_global=0 必须返回与 student logits 连通的零损失。

CWD 总目标保持受控底座不变：

~~~text
L_student =
    L_semantic
  + 1.0   * L_O12_pixel_KL
  + 0.001 * L_adv_G
  + 50.0  * L_CWD_feature_T4
  + 3.0   * L_CWD_logit_T4

L_discriminator = 0.1 * L_adv_D
~~~

其中两个 CriterionCWD 分支继续使用固定全局 temperature=4；logit-CWD 使用各变体共同不变的 raw teacher/student logits。它们不接收 O1.2 空间温度。lambda_skd、lambda_ifv、lambda_fitnet、lambda_at、lambda_psd、lambda_csd 全部冻结为 0。

第一开发底座的完整公共配置：

| 项目 | 冻结值 |
|---|---|
| teacher | DeepLabV3-ResNet101 |
| student | DeepLabV3-MobileNetV3-Small |
| student init SHA256 | 47085aa164b2977003221458a2a5fdf5f46f434f5539b164dc71969b4dd4cd75 |
| crop / batch / workers | 512x512 / 16 / 8 |
| devices per run | 1 NPU，distributed=false |
| lr / momentum / weight decay | 0.02 / 0.9 / 1e-4 |
| iterations / log interval | 20,000 / 20 |
| save / val interval | 800 / 800 |
| seed | 1234 |

O1.2 新 launcher 必须保存自身 SHA 和完整 argv。四个旧 Phase O launcher 不属于冻结配置，也不得用于 O1.2。

teacher-only 后，调和均值只解释 O1.2 教师目标分支的平均逆温度，不解释学生梯度缩放。正式报告必须同时给出该分支的 KL、去除教师熵常数后的 cross-entropy 项和 student-logit 梯度范数。

## 6. 冻结分支与对照

所有分支使用同一份 O1.1 CDF 和同一组 train 冻结参数 a*、b*。禁止为每个分支重新匹配预算。

| 分支 | a | b | 目的 |
|---|---:|---:|---|
| neutral | 0 | 0 | 新 teacher-only masked KD 基线 |
| reliable_only | a* | 0 | 单独检验轻微可靠侧锐化 |
| unreliable_only | 0 | b* | 第一优先：单独检验高风险平滑 |
| full_budgeted | a* | b* | 检验双向预算映射 |

空间因果对照：

| 对照 | 冻结规则 | 回答的问题 |
|---|---|---|
| arithmetic-matched scalar | 标量 T=A_train(对应分支) | 是否只由平均温度造成 |
| harmonic-matched scalar | 标量 T=H_train(对应分支) | 是否只由平均逆温度造成 |
| within-image shuffled | 每图 valid 像素内、用 stateless seed 打乱同一温度多重集合 | 风险位置是否重要 |

所有标量和 shuffle 对照也必须使用 teacher-target-only 损失。旧 Phase M2 的 T=0.6 同时改变学生端温度，只能作为历史动机，不能与 O1.2 做严格因果比较。

shuffle_seed 固定为 3407。canonical dataset_index 是去除空行后的 train list 的 0-based 行号。每个排列使用以下唯一算法：

~~~text
payload = ASCII(
  'rtc_o12_shuffle_v1|3407|' +
  str(dataset_index) + '|' + str(global_iteration)
)
digest = SHA256(payload)
seed64 = int.from_bytes(digest[0:8], byteorder='big', signed=false)
seed63 = seed64 mod (2^63-1)
generator = torch.Generator(device='cpu').manual_seed(seed63)
perm = torch.randperm(num_valid, generator=generator, device='cpu')
~~~

禁止使用 Python 内置 hash。global_iteration 采用 1-based 训练步编号，第一次参数更新使用 1；checkpoint 保存“已经完成的 global iteration”，恢复后的下一步使用该整数加 1。

valid_indices 精确定义为 valid_mask 按行优先 flatten 后非零位置的升序索引。对每个 j，排列应用方向冻结为：

~~~text
T_shuffled_flat[valid_indices[j]] =
    T_original_flat[valid_indices[perm[j]]]
~~~

permutation 在 CPU 生成后再移动到温度图设备；num_valid 为 0 或 1 时保持恒等。算法不使用 rank-local RNG，恢复训练后同一 canonical index 和 global_iteration 必须得到相同排列。每图只打乱 valid pixels，无效像素保持 T=1。单元测试必须覆盖映射方向、跨进程、DDP sampler 和 resume 等价性。

同一温度多重集合不保证相同的教师目标变化，因为温度与 logit margin 的配对被改变。所有 shuffle 和 scalar 对照必须额外报告 mean target entropy、mean confidence delta 和 mean KL(q_target || q_base)，不能把“温度直方图相同”写成“目标作用量完全匹配”。

可选的高风险 KD 权重下降不属于 O1.2 主方法。只有 unreliable_only 已证明连续平滑不足时，才能另行预注册；不得与新温度映射同时首次引入。

## 7. 实现隔离与来源契约

### 7.1 历史产物与 O1.1 当前源码边界

O1 的历史 CDF、失败 gate、诊断报告和当时 source SHA 只作为不可改写的历史证据；当前共享源码在 O1.1 开发后已经不同于 O1 当时的 source SHA，不能声称 O1 当前源码仍字节一致。

O1.1 confidence CDF 冻结了以下当前源码 SHA：

- utils/rtc_temperature.py；
- scripts/diagnostics/build_rtc_cdf.py；
- scripts/diagnostics/diagnose_rtc_routing.py；
- scripts/diagnostics/check_rtc_o11_gate.py。

这四个当前文件在 O1.2 中不得格式化或改写，否则 O1.1 CDF 来源契约会失效。允许 O1.2 新模块只读 import 其中已经冻结的 confidence risk、CDF loader 和 query；只读调用不会改变旧 artifact 的 SHA 契约。新模块只实现 gR/gU、预算温度、teacher-only target 和新损失。

### 7.2 新增独立实现

计划新增：

~~~text
utils/rtc_o12_calibration.py
scripts/diagnostics/diagnose_rtc_o12_budget.py
scripts/diagnostics/check_rtc_o12_gate.py
tests/test_rtc_o12_calibration.py
tests/test_rtc_o12_gate.py
~~~

train_kd.py 只能通过新的显式 O1.2 mode 调用新模块；旧 O1/O1.1 mode 的默认行为和源码契约不得被隐式改变。

### 7.3 独立目录

~~~text
runs/diagnostics/phaseO_o12/
runs/kd_baselines_npu/phaseO_o12/
~~~

建议正式诊断产物：

~~~text
runs/diagnostics/phaseO_o12/o12_budget_train.json
runs/diagnostics/phaseO_o12/o12_budget_val.json
runs/diagnostics/phaseO_o12/o12_budget_parameters.json
runs/diagnostics/phaseO_o12/o12_joint_gate.json
~~~

phase 字段必须精确为 O1.2。产物必须保存：

- O1.1 CDF 实际 SHA；
- teacher/list SHA；
- 新 O1.2 module、diagnose、checker、train entry source SHA；
- Git commit 与 dirty 状态；
- 完整配置指纹；
- 参数求解人口、精度、迭代数和约束残差。

## 8. O1.2-A：无学生全量机制诊断

### 8.1 执行顺序

1. 单元测试和 Python 编译；
2. 8 图 NPU smoke；
3. 完整 train 风险扫描并求 a*、b*；
4. 用冻结 a*、b* 完整扫描 train；
5. 不重拟合地完整扫描 val；
6. 独立 checker 重算联合门禁；
7. 门禁通过后才能讨论 20-iteration 学生 smoke。

### 8.2 必须报告

每个 split 报告：

- native-valid、finite、nonfinite 和各风险区人口；
- T 的 min、max、mean、harmonic mean、q01/q10/q50/q80/q90/q95/q99；
- T<0.9、T=1、T>1、T>1.25、T>1.5 的覆盖；
- 每个风险十分位的 T mean、c_base、c_target、entropy_base、entropy_target；
- T_effective=T_out*T 的对应分布；
- 教师 argmax mismatch；
- 目标置信度和熵方向违规；
- 中性区目标与 T=1 基线的最大绝对差；
- 给定学生 logits 时，两个不同温度图下 p_student 的最大绝对差；
- 类别、边界/内部、小目标诊断；
- O1.1 风险富集、召回和十分位单调性复核。

教师正确性和 GT 分层只用于报告，不进入 a*、b*、门函数或温度。

## 9. O1.2 非同义联合门禁

train 与 val 均须完整扫描。联合门禁只在以下全部条件成立时通过。

### 9.1 来源与结构

1. O1.1 CDF 实际 SHA 精确匹配预注册值；
2. O1.1 joint gate 仍为 true；
3. teacher、list、类别数、risk formula 和 valid-mask 语义一致；
4. O1.2 phase、source SHA、配置指纹与 canonical path 一致；
5. a*、b* 只由完整 train 风险人口求得；
6. val 使用完全相同的 a*、b*，没有重拟合；
7. checker 不信任 diagnose 的 all_checks_pass，必须独立重算。

### 9.2 train 温度预算

~~~text
a* = 0.10536051565782628
0 <= b* <= 0.4054651081081644
abs(independently_resolved_b - b*) <= 1e-8
abs(mean(T)-0.995) <= 1e-4
harmonic_mean(T) >= 0.98
min(T) >= 0.9 - 1e-6
max(T) <= 1.5 + 1e-6
q10(T) >= 0.9
q50(T) >= 0.95
mean(T | top risk decile) >= 1.10
~~~

理论高风险端点 exp(b*) 必须不低于 1.25，避免 unreliable 分支退化为近似恒等。

### 9.3 val 无重拟合稳定性

~~~text
0.98 <= mean(T) <= 1.02
harmonic_mean(T) >= 0.97
min(T) >= 0.9 - 1e-6
max(T) <= 1.5 + 1e-6
q10(T) >= 0.9
q50(T) >= 0.95
~~~

不得为了让 val 接近 0.995 而改变参数。

### 9.4 方向、目标与数值语义

机制计算 dtype 和容差冻结为：

~~~text
temperature_map_dtype = float32
budget_accumulator_dtype = float64
tau_T = 1e-6
tau_prob = 1e-6
tau_entropy = 1e-6
tau_student = 1e-7
tau_formula_monotonic = 1e-12
~~~

方向违规按超出对应 tau 才计数；中性区检查 max_abs_error>tau_prob；学生不变性检查 max_abs_error>tau_student。T(u) 单调性由独立 checker 在排序后的 4097 点固定 u 网格上用 float64 公式重算，不能按数据加载顺序检查。

按上述容差定义后，以下违规计数必须严格为 0：

- nonfinite；
- u<0.6 且 T>1；
- 0.6<=u<=0.8 且 T!=1；
- u>0.8 且 T<1；
- T(u) 单调方向违规；
- 可靠侧目标置信度下降；
- 高风险侧目标置信度上升；
- 可靠侧目标熵上升；
- 高风险侧目标熵下降；
- 正温度后的教师 argmax mismatch；
- 温度越界；
- 中性区目标与 T=1 基线不一致；
- 空间温度改变学生 softmax；
- 有效像素、风险分箱或错误计数不闭合。

风险富集、召回率和十分位单调性继续报告并复核，但它们是 O1.1 已建立的证据，不能冒充 O1.2 温度机制的新门禁。

## 10. 单元测试与 smoke

最低测试集合：

1. gR、gU 的边界、幂次和互斥性；
2. T(u) 全局单调不减；
3. a=0、b=0 精确退化为 T=1；
4. neutral/reliable_only/unreliable_only/full 使用同一 a*、b*；
5. 预算求解的 float64 重现性和无可行解拒绝；
6. train 参数不可被 val 重拟合；
7. teacher target 的置信度、熵和 argmax 方向；
8. student softmax 对空间温度严格不变；
9. teacher target detach；
10. masked loss 的 ignore 语义；
11. 单卡和 DDP 全局有效像素归一化；
12. scalar 和 full 在常数温度图下精确等价；
13. within-image shuffle 保持每图 valid 温度多重集合；
14. O1.1 CDF/source/config 错配必须失败；
15. O1.2 checker 自身 source SHA 校验。

8 图 smoke 只验证链路、数值和产物 schema，不能修改正式参数，也不能用于效果结论。

## 11. 分阶段最小学生实验

### 11.1 O1.2-B：20-iteration 链路 smoke

只有 O1.2-A joint gate 通过后才允许执行。

第一批只启动：

- neutral；
- unreliable_only。

检查：

- 设备、显存和吞吐；
- teacher-only 温度语义；
- loss finite；
- 梯度 finite；
- checkpoint 保存/恢复；
- 同 seed 数据顺序一致；
- 20 iteration 不产生或宣称 mIoU 结论。

两者通过后，才允许 smoke reliable_only 和 full_budgeted。

#### 11.1.1 2026-07-13 执行回填

`neutral` 与 `unreliable_only` 已按本节执行，四份 final acceptance 均为 `pass=true`：

| 变体 | fresh 20-step acceptance SHA256 | iteration=20 端点 resume=0 acceptance SHA256 | 设备 |
|---|---|---|---|
| neutral | `90eb89fe7a7dff773845152d5d59a7efcc3d61299976137223be6cf7d2615e1c` | `30fe00fa72b2f20293db2cfca3da7f0f9d960649fef8d153629b3d2b2f496677` | NPU:0 |
| unreliable_only | `d8a3363724c7c5d93c7629ba0ebf365a7f73d1c25c53138633e179e21fe623cd` | `af540bf53a0b1646227eb9024a6eb9f5d7bf1af250773c62d1074b77aaa82488` | NPU:1 |

两次 fresh run 都完成 20 次 optimizer step、保存 iteration=20 checkpoint，并在 step 20 记录有限且非零的 KD-only student-logit 梯度；两个端点恢复审计都严格加载 fresh checkpoint，核对关键状态后以 `optimizer_steps=0` 结束。端点恢复只证明终点 checkpoint 可严格加载及关键状态一致，不等价于已执行恢复后的下一个训练 step。

四条链路的 canonical order SHA256 均为 `99326472a2e5e2bd42428d4709ff9f8049d2c906c7a6b9e8fa3068cb0439d564`。这证明相同 seed 的 canonical 样本索引顺序一致；它不证明 8-worker 随机增强可以按位重放。梯度证据覆盖总 generator/discriminator loss 的有限性硬门禁、step 20 的 KD-only student-logit 梯度，以及更新后 checkpoint 张量有限性；没有逐参数扫描并证明每个参数梯度都有限。

本 smoke 设置 `skip-val`，没有生成预测、mIoU、rescue、error imitation 或 teacher-correct retention。两条 fresh run 的单点 loss/KL 也不能用作变体效果比较，因为教师目标不同且只有一个记录点。完整命令、产物路径、运行身份、显存/吞吐、环境告警和证据边界见 [O1.2-B 20-step 正式链路 smoke 报告](2026-07-13_phaseO_rtc_o12b_smoke_report.md)。

### 11.2 O1.2-C1：高风险分支信号筛查

首个 20k 比较严格限定为 neutral versus unreliable_only。

冻结条件：

- CWD recipe、VOC 和 student/teacher architecture 不变；
- Tout=3.0、T_student=1.0、teacher-target-only KD；
- seed=1234；
- 相同初始化、数据顺序、增强、学习率、损失系数、验证和 checkpoint 频率；
- iteration 20,000 final mIoU 为唯一主性能指标；
- last-10 mean、best mIoU 和 runtime 为次指标，best checkpoint 不进入机制门禁。

唯一主机制指标在 final checkpoint、native KD grid 上定义为：

~~~text
student_rescue_U =
P(student=ground_truth | teacher_wrong, u>0.8)
~~~

error_imitation_U 和 teacher_correct_retention_U 为次级机制指标。

所有差值统一定义为 delta_X(A-B)=X(A)-X(B)，括号中的前者减后者。final checkpoint 固定为 iteration 20,000；所有变体按 canonical val list 顺序缓存预测、GT、teacher prediction 和 u，并保存缓存 SHA。

每张 val 图保存三个指标的 numerator 和 denominator。bootstrap 使用 numpy.random.Generator(PCG64(3407)) 固定生成 10,000 组 index；每组有放回抽取 1,449 张图，再汇总 numerator/denominator 后计算 paired delta。零分母图仍参与图像抽样但不增加汇总分母；若任一完整 bootstrap 样本的汇总 denominator 为 0，则门禁结构失败。95% CI 使用 bootstrap delta 的 2.5%/97.5% percentile 区间，numpy.quantile(method='linear')。所有变体必须使用相同的 bootstrap index 矩阵。

C1 继续条件全部满足才通过：

1. delta_student_rescue_U >= 0.005；
2. delta_student_rescue_U 的 paired-bootstrap 95% 区间下界大于 0；
3. delta_error_imitation_U <= 0；
4. final_mIoU(unreliable_only)-final_mIoU(neutral) >= -0.002；
5. delta_teacher_correct_retention_U >= -0.005；
6. 无数值、运行完整性或配置偏差。

任一项失败即停止，不自动加入可靠侧锐化。

### 11.3 O1.2-C2：高风险位置因果对照

C1 通过后、可靠侧分支启动前，固定运行：

- unreliable_only within-image shuffled；
- unreliable_only arithmetic-matched scalar；
- unreliable_only harmonic-matched scalar。

两个 scalar 的值只由 unreliable_only 正式 train 温度分布冻结。strongest matched scalar 精确定义为二者中 final mIoU 较高者；若完全相等，固定选择 harmonic-matched scalar。不得临时增加 scalar 候选。

C2 空间继续条件：

1. final_mIoU(unreliable_only)-final_mIoU(unreliable_shuffle) >= 0.001；
2. delta_student_rescue_U(unreliable_only-shuffle) >= 0；
3. final_mIoU(unreliable_only)-final_mIoU(strongest_scalar) >= -0.001；
4. 温度、target entropy、confidence delta 和 target KL 的差异完整报告。

若 C1 通过但 C2 失败，只能声称高风险分支的总体目标分布变化可能有效，不能声称正确风险位置有独立作用，也不进入 full spatial 叙事。

### 11.4 O1.2-C3：可靠侧与完整预算映射

只有 C2 通过后才运行：

- reliable_only；
- full_budgeted；
- full arithmetic-matched scalar；
- full harmonic-matched scalar；
- within-image shuffled full map。

Full 的 strongest matched scalar 仍只在预列的 arithmetic/harmonic 两项中按 final mIoU 选择，平局固定选 harmonic。主空间因果差值为 final_mIoU(full_budgeted)-final_mIoU(full_shuffled)。

last10_mIoU(X) 固定为变体 X 在正式 val interval=800 下，iteration 12,800、13,600、14,400、15,200、16,000、16,800、17,600、18,400、19,200、20,000 共十次验证 mIoU 的算术均值。

Full 晋级条件：

1. 相对 neutral 的 final mIoU 至少提高 0.002；
2. last10_mIoU(full_budgeted)-last10_mIoU(neutral) >= 0；
3. delta_student_rescue_U(full_budgeted-neutral) >= 0；
4. full-strongest_matched_scalar >= -0.001 final mIoU；
5. full-full_shuffled >= 0.001 final mIoU；
6. reliable_only 的作用单独报告，不能用 Full 反推；
7. 不得查看结果后改预算、挑 checkpoint 或更换主指标。

如果 Full 与 shuffle 的差值小于 0.001，只能声称温度分布或预算可能有效，不能声称风险空间位置有独立贡献。

### 11.5 O1.2-D：80k 与多 seed

只有 C3 全部通过后才能另行预注册：

- full_budgeted；
- C3 中最强的严格同损失标量对照；
- 至少 seeds 1234、2025、3407；
- 80k；
- 以 seed 为统计单位。

O1.2 本文不自动授权 D 阶段。

## 12. 分层诊断与 mIoU 相关性

训练前和学生验证时均需报告：

- GT 类别；
- foreground/background；
- 边界带/区域内部；
- 小目标/非小目标；
- risk decile；
- teacher-correct/teacher-wrong。

主要目的：

1. 检查 top 20% 是否主要由边界或稀有类别构成；
2. 检查平滑是否削弱了对 mIoU 重要的正确教师监督；
3. 避免用背景占优的 micro accuracy 代替类别平衡结论。

分层定义冻结为：

- 全部分层在 nearest-resized GT 的 native teacher grid 上计算；
- background 为 VOC label 0，foreground 为其他有效类别；
- boundary core 为 8 邻域内存在不同有效 GT 类别的像素；boundary band 是 core 的 Chebyshev 半径 2 像素膨胀；interior 为 valid 减去 boundary band；
- object 使用逐 GT foreground 类别的 8-connected component；面积小于该图 native-valid 像素数 0.5% 的 component 记为 small object；
- 分层结果属于探索性报告，不进入 O1.2 预算、效果或晋级门禁。

这些诊断不得反向用于修改 O1.2 主参数。发现问题只能触发停止或另行预注册 O1.3。

## 13. 跨数据集与确认性实验

新数据集协议：

1. 保持 risk formula、qR、qU、pR、pU、预算目标、温度上下限和参数选择规则不变；
2. 只在新数据集训练图像上重建 confidence CDF；
3. 只用新训练风险人口求预算参数；
4. 不使用新 val 标签重新调参；
5. 重新报告风险富集、错误召回、温度预算和分层机制；
6. 新数据集结果才承担确认 O1.2 泛化的主要职责。

这属于固定算法下的训练统计适配，不是直接复用 VOC CDF 的 zero-shot 泛化。

## 14. 允许与禁止的结论

O1.2-A 通过后只允许说：

- 新温度映射满足预注册预算；
- 高风险侧获得单调增强的教师目标平滑；
- O1.2 pixel-KL 分支中的学生温度已与空间教师目标校准解耦；其他固定 CWD 分支保持共同不变。

O1.2-B 的两条 20-step smoke 通过后只允许补充说：

- `neutral` 与 `unreliable_only` 的 fresh 20-step 正式训练链路均可完成，loss/指定梯度与 checkpoint 数值门禁通过；
- 两个 iteration=20 端点 checkpoint 均可严格加载并完成 resume=0 状态审计；
- 同 seed 的 canonical 样本索引顺序哈希一致。

这些结论仍不是 mIoU、收敛、机制收益、空间因果或泛化结论。

C1 通过后只允许说：

- 高风险平滑在当前单 seed、20k 探索设置下具有或不具有机制信号。

只有 C2 且空间对照通过后才允许说：

- 风险位置可能具有独立于温度直方图和平均尺度的贡献。

禁止：

- 将 O1.2 称为同一 VOC-val 上的独立确认；
- 把高风险称为错误像素；
- 在 matched scalar 与 shuffle 未完成前声称空间自适应有效；
- 用历史 T=0.6 的不同损失定义直接证明 O1.2 优越；
- 用单 seed 20k 声称统计显著或跨数据集泛化。

## 15. 记录模板

每次正式执行必须在独立 O1.2 execution record 中记录：

| 字段 | 内容 |
|---|---|
| Git commit / dirty | 待执行时写入 |
| O1.1 CDF path / SHA | 必填 |
| teacher / list SHA | 必填 |
| O1.2 source SHA | 必填 |
| a* / b* / solver residual | 必填 |
| train A / H | 必填 |
| val A / H | 必填 |
| direction / entropy / argmax violations | 必填 |
| teacher-only/student-fixed contract | 必填 |
| gate result | 必填 |
| run command / PID / NPU | 仅获授权训练后填写 |
| checkpoint / metrics | 仅完整运行后填写 |
| deviations / failures | 必填，不得删除失败记录 |

## 16. O1.2-A 正式结果与当时停止线

状态：2026-07-13 已完成；全量 train/val 机制诊断和独立联合门禁通过；在 O1.2-A 结束时尚未启动学生训练。以下停止线是 O1.2-A 当时的历史记录，已由第 17 节更新。

预算求解固定得到：

~~~text
a*                = 0.10536051565782628
b*                = 0.34764991700649260
b_max             = 0.40546510810816440
train mean(T)     = 0.9950000002788227
train harmonic(T) = 0.9880688428627591
budget residual   = 2.7882274267199136e-10
~~~

全量机制统计如下：

| 指标 | Train | Val |
|---|---:|---:|
| 图像数 / native-valid 像素 | 10,582 / 32,246,990 | 1,449 / 3,878,674 |
| mean(T) | 0.995000 | 0.996092 |
| harmonic(T) | 0.988069 | 0.988161 |
| median(T) | 0.982425 | 0.979061 |
| T>1 覆盖率 | 0.198152 | 0.210636 |
| T>1.25 覆盖率 | 0.039553 | 0.046430 |
| 高风险错误率 | 0.146555 | 0.243854 |
| 高风险错误 recall | 0.986971 | 0.826277 |
| 高风险错误富集 | 4.9748x | 3.9183x |

所有预注册的方向、温度范围、中性区、教师 argmax、学生固定温度、teacher-target-only、熵方向和数值有限性检查均为零违规；train 求解与 train 复算缓存字节一致。联合 checker 给出：

~~~text
joint_gate_pass = true
train_pass      = true
val_pass        = true
~~~

这说明 O1.2 已把 O1.1 接近全局 T=0.6 的大面积强锐化压回到轻预算区间，并让主要平滑预算集中于高风险侧。它只证明映射、预算和教师目标校准机制按预注册工作，不证明学生 mIoU 提升，也不证明空间位置优于匹配标量或打乱对照。高风险侧仍有约 85.34%/75.61% 的教师预测正确，因此只能称为错误富集区，不能称为错误标签。

正式记录：

- [O1.2 执行记录](2026-07-13_phaseO_rtc_o12_execution_record.md)
- [O1.2 机制诊断报告](2026-07-13_phaseO_rtc_o12_diagnostic_report.md)

O1.2-A 当时停止线：

- O1.1 四份冻结源码和正式产物保持不变；
- O1.2-A 已完成，联合门禁通过；
- 没有 O1.2 学生 checkpoint、mIoU 或性能结论；
- Phase N 不自动恢复；
- 不允许使用旧 Phase O 启动脚本运行 O1.2；
- 只有人工审查 O1.2-A 后明确授权，才可进入 O1.2-B 的 20-step neutral 与 unreliable_only smoke；不得自动启动 20k、C2、C3 或 80k。

## 17. O1.2-B 正式结果与当前停止线

状态：2026-07-13 已完成；`neutral` 与 `unreliable_only` 各自的 fresh 20-step smoke 和 iteration=20 端点 resume=0 审计全部通过。详细记录见：

- [O1.2-B 20-step 正式链路 smoke 报告](2026-07-13_phaseO_rtc_o12b_smoke_report.md)
- [O1.2 执行记录](2026-07-13_phaseO_rtc_o12_execution_record.md)
- [O1.2 机制诊断报告](2026-07-13_phaseO_rtc_o12_diagnostic_report.md)

共同执行证据：

- Git commit：`5c93c1eeebfdf58ca4a9c82aca9852cb8a661330`；
- seed：`1234`；batch size：`16`；每条 fresh run：`20` optimizer steps；
- canonical order：320 个索引，SHA256=`99326472a2e5e2bd42428d4709ff9f8049d2c906c7a6b9e8fa3068cb0439d564`；
- neutral fresh checkpoint SHA256=`6309cba985d8831586a610c45a4f463ba15d71381924b7093ca9f9fa91a982a9`；
- unreliable_only fresh checkpoint SHA256=`b9426976a45ab1ec6396bb68d85f2f43fe80f176d94fde8c5395580a41ea0ed4`；
- fresh acceptance 的 finite scan 均覆盖 847 个 tensor、19,711,341 个 tensor element 与 7 个 float scalar，errors/warnings 数组均为空；
- neutral/unreliable_only 的 step-20 KD-only student-logit gradient L2 分别为 `0.00273925`/`0.00282540`，均有限且非零。

运行控制台保留了 CANN 安装目录 owner mismatch 告警；fresh run 还出现内部格式被禁用后回退到 base format 的告警，resume audit 出现旧权重文件格式/兼容性告警。四份 final acceptance 的 `errors=[]`、`warnings=[]`，这些原始环境/序列化告警没有触发接受失败，但必须保留，不能解释为方法效果证据或从记录中删除。

当前停止线：

- O1.2-B 的授权范围已经执行完毕；不自动启动 `reliable_only`、`full_budgeted`、scalar、shuffle、20k、C2、C3 或 80k；
- 不把 20-step 单点 loss、KL、运行速度或显存差异写成变体优劣；
- 不宣称已验证 mIoU、收敛、错误救援、错误模仿、正确教师保留、空间因果或跨数据集泛化；
- 不把端点 resume=0 写成已执行恢复后下一 step，也不把 canonical order hash 写成 worker 随机增强的按位复现；
- 正式参数、CDF、配置、核心源码与 O1.2-A artifact SHA 保持冻结；后续如获新授权，必须继续使用 fail-closed 入口并另行记录，不覆盖 A/B 历史证据。
