# Phase O1.2：高风险优先预算校准执行记录

- 执行日期：2026-07-13
- 记录范围：O1.2-A 全量机制诊断，以及 O1.2-B `neutral`/`unreliable_only` 各 20-step fresh smoke 与端点恢复审计
- 当前状态：O1.2-A 联合门禁通过；O1.2-B 四份 final acceptance 全部通过；停在 B 后人工审查线
- 正式联合门禁：`joint_gate_pass=true`
- 学生训练：仅完成两条 20-step 链路 smoke；没有验证、预测、mIoU、20k 或性能结论
- 规范计划：[2026-07-13_phaseO_rtc_o12_budgeted_routing_plan.md](./2026-07-13_phaseO_rtc_o12_budgeted_routing_plan.md)

> 本文区分“数值/机制门禁通过”“20-step 训练链路通过”和“学生效果成立”。O1.2-A 证明温度映射、教师目标方向与来源契约满足预注册要求；O1.2-B 只证明两条短链路、指定梯度、checkpoint 保存和端点严格加载通过。二者均不证明该方法提高学生 mIoU，也不构成空间因果或跨数据集泛化证据。

## 0. 执行结论

本轮已完成计划允许的 O1.2-A 工程动作：独立实现、测试、两轮 8 图 NPU smoke、完整 train 参数求解、完整 train/val 无重拟合评估，以及独立联合 checker。正式结论如下：

1. 使用完整 train 风险人口 `32,246,990` 求得：

   ~~~text
   a = 0.10536051565782628 = -log(0.9)
   b = 0.3476499170064926
   A_train = mean(T) = 0.9950000002788227
   H_train = harmonic_mean(T) = 0.9880688428627591
   residual = 2.7882274267199136e-10
   ~~~

2. val 不重拟合，直接复用同一 `a,b`：

   ~~~text
   A_val = 0.9960921500775646
   H_val = 0.9881613746466074
   ~~~

   上述 val 两个值取自联合 checker 的独立 NumPy 复算；val 评估 JSON 中以 float32 温度逐像素累计得到的对应值为 `0.9960921506400221` 和 `0.9881613751091604`。两条数值路径均通过既定容差，差异仅来自复算路径与浮点累计细节。

3. 当前映射没有重现 O1.1 的大面积强锐化：train/val 温度中位数分别为 `0.9824246764`/`0.9790610671`，调和均值分别约为 `0.9881`/`0.9882`；同时最危险十分位的平均温度为 `1.22858`/`1.23337`，主要平滑预算集中在高风险尾部。

4. train/val 的 16 类方向、数值与闭合违规计数全部为 `0`；中性区目标误差和学生 softmax 变化最大值均为 `0.0`。

5. 联合 checker 的 23 项顶层联合检查全部为 `true`；4097 点 float64 公式网格的最小相邻温差为 `0.0`，单调违规为 `0`。

6. 第一轮 smoke 暴露了一个真实审计问题：checker 错把“必须使用 NPU:0、三次扫描必须同卡”当作统计门禁。该问题已经修复并由第二轮异卡 smoke 验证；正式 solve/train 使用 NPU:0，val 使用 NPU:1，三份产物分别满足单进程单 NPU，联合门禁通过。

7. 2026-07-13 按明确授权完成 O1.2-B：`neutral` 与 `unreliable_only` 的 fresh 20-step run 以及各自 iteration=20 端点 resume=0 审计均为 final `pass=true`。本阶段没有运行 validation，不产生学生效果或 mIoU 结论。

## 1. 目标边界与证据等级

### 1.1 本轮实际完成

- confidence-only 风险和 O1.1 冻结 CDF 的只读复用；
- O1.2 非对称门函数与一维预算求解；
- teacher-target-only 空间温度语义；
- neutral、reliable_only、unreliable_only、full_budgeted 以及预注册 scalar/shuffle 控制的训练入口实现与契约检查；
- 单元测试、编译检查、NPU 链路 smoke；
- 完整 VOC train/val 机制扫描、分层诊断和联合门禁；
- 失败 smoke、修复、正式运行和产物 SHA 的可追溯记录；
- `neutral` 与 `unreliable_only` 各 20 个 optimizer step 的正式链路 smoke；
- 两个 iteration=20 checkpoint 的严格端点加载、关键状态相等与 resume=0 审计；
- canonical 样本索引顺序、指定梯度、checkpoint 数值、设备/显存/吞吐及环境告警记录。

### 1.2 本轮没有完成，也不允许推断

- 没有 20k/80k 或足以判断收敛的学生训练；
- O1.2-B 只有 step 20 的训练诊断和 checkpoint；没有完整 loss 曲线、validation、学生预测或 mIoU；
- 没有 neutral 与 unreliable_only 的效果比较；两条链路的单点 loss/KL 不具有性能可比性；
- 没有 arithmetic/harmonic scalar 或 within-image shuffle 的学生因果对照；
- 没有新数据集、新教师、多 seed 或确认性实验；
- 不得把 `u>0.8` 写成教师错误标签，也不得把 `u<0.6` 写成必然正确；
- O1.1 的风险富集/召回是已知风险证据，本轮只做复核，不能冒充 O1.2 温度映射带来的新效果；
- VOC train/val 像素是高度相关的 micro 样本，不能当作数百万独立统计单位。

`q_R=0.6`、`q_U=0.8`、门函数指数和预算是在查看 O1.1 VOC 结果后确定的。因此 O1.2-A 仍是同一数据域上的探索性机制开发，不是独立确认实验。

## 2. 冻结方法与公式

### 2.1 confidence-only 风险

教师参考温度固定为 `T_assess=1`：

~~~text
p_assess(i,k) = softmax(z_t(i,k) / T_assess)
c_i           = max_k p_assess(i,k)
r_i           = -log(clamp(c_i, 1e-8, 1-1e-8))
u_i           = F_train,confidence(r_i)
~~~

其中 `F_train,confidence` 是 O1.1 冻结的、right-continuous 的 train confidence CDF。风险本身不使用教师正确性、类别值或学生信息；但扫描使用 GT valid/ignore mask 排除 ignore、void 和 padding，因此不能宣称整个诊断完全 label-free。

### 2.2 三个风险区与门函数

~~~text
clip01(x) = min(max(x,0),1)

g_R(u) = clip01((0.6-u)/0.6)^1
g_U(u) = clip01((u-0.8)/0.2)^2
~~~

| 区域 | 条件 | 作用 |
|---|---|---|
| 相对可靠侧 | `u<0.6` | `g_R>0,g_U=0`，只允许轻微锐化 |
| 中性区 | `0.6<=u<=0.8` | `g_R=g_U=0`，严格 `T=1` |
| 相对不可靠侧 | `u>0.8` | `g_R=0,g_U>0`，随风险凸增平滑 |

正式 train/val 中 `u==0.6` 和 `u==0.8` 的人口及教师错误人口均为 `0`，但实现和 checker 仍显式检查边界语义。

### 2.3 预算温度

~~~text
log T_i = -a*g_R(u_i) + b*g_U(u_i)
T_i     = exp(log T_i)

a       = -log(0.9) = 0.10536051565782628
b       in [0, log(1.5)]
~~~

固定 `a` 后，在完整 train `u` 人口上用 float64 累积并固定执行 64 次二分，唯一求解 `b` 使：

~~~text
mean_train(T) = 0.995
harmonic_mean_train(T) >= 0.98
~~~

val 只能复用 train 的冻结参数，禁止重拟合。

### 2.4 教师目标与学生端解耦

~~~text
T_out                  = 3.0
q_teacher(i)           = softmax(z_t(i) / (T_out*T_i))
p_student(i)           = softmax(z_s(i))
T_effective(i)         = T_out*T_i
L_O12                  = mean_valid KL(q_teacher(i) || p_student(i))
teacher_target_detached = true
~~~

空间温度只校准教师目标，学生温度固定为 `1.0`，不乘空间 `T_i` 的幂次。`T_i` 的预算边界是 `[0.9,1.5]`；`T_effective=3*T_i` 的理论范围相应约为 `[2.7,4.5]`，不能错误地用 `[0.9,1.5]` 审计有效温度。

## 3. 实现、隔离与审计修复

### 3.1 最终实现文件

| 文件 | 作用 |
|---|---|
| [utils/rtc_o12_calibration.py](../utils/rtc_o12_calibration.py) | 门函数、预算求解、温度构造、教师目标、masked KL、标量与 shuffle 工具 |
| [scripts/diagnostics/diagnose_rtc_o12_budget.py](../scripts/diagnostics/diagnose_rtc_o12_budget.py) | solve/evaluate 两阶段 NPU 全量诊断和正式 JSON/NumPy 产物 |
| [scripts/diagnostics/check_rtc_o12_gate.py](../scripts/diagnostics/check_rtc_o12_gate.py) | 不信任 diagnose 布尔值的独立来源、缓存、预算、方向和联合门禁复算 |
| [train_kd.py](../train_kd.py) | 显式 `rtc_o12_teacher_target` 训练入口、artifact fail-closed 契约、变体、确定性数据顺序和 resume 元数据 |
| [tests/test_rtc_o12_calibration.py](../tests/test_rtc_o12_calibration.py) | 公式、预算、目标、损失、shuffle 等模块测试 |
| [tests/test_rtc_o12_gate.py](../tests/test_rtc_o12_gate.py) | checker、来源、缓存、NPU 运行身份和篡改拒绝测试 |
| [tests/test_rtc_o12_train_entry.py](../tests/test_rtc_o12_train_entry.py) | 训练入口、参数契约、数据顺序、checkpoint/resume 和 DDP 语义测试 |

O1.1 冻结的四份源码未被本提交改写；联合 checker 已逐文件验证它们与 O1.1 CDF 来源元数据字节一致。

### 3.2 关键 fail-closed 契约

最终训练入口会校验 canonical CDF/parameters/joint-gate 路径及实际 SHA、O1.1 gate、O1.2 配置指纹、四份核心源码 SHA、冻结输入 SHA、预算解和所有联合检查。类别数或 teacher/student logits 空间尺寸不一致时 hard-fail；不得隐式插值。checkpoint 保存 O1.2 artifact/config/data-order 元数据，恢复时不一致即拒绝。

### 3.3 smoke 暴露并修复的 P1

第一轮 smoke 中，val 明确请求并实际运行在 `npu:1`，但 checker 原先把 `runtime_actual_npu_zero` 和三次扫描同卡作为门禁。这与计划中的“设备编号是执行来源，不是统计定义”冲突，也会把合法的多卡串行调度误判为失败。

修复后的契约是：

- 每份产物必须是 rank 0、world size 1、单进程单 NPU；
- 显式请求的 `npu:<index>` 必须与 `torch.npu.current_device()` 一致；
- NPU 名称必须非空，PID 和 UTC 时间必须有效；
- solve/train/val 可以使用不同的实际 NPU；
- `same_actual_npu` 只记录来源，不参与门禁，`same_actual_npu_is_gate=false`。

第二轮 smoke 使用 solve/train=`npu:0`、val=`npu:1` 通过了该运行身份审计；正式运行沿用同样异卡安排并通过联合门禁。

### 3.4 有效温度审计语义

checker 独立重算 `T_effective=3*T`，并逐项核对 min/max/mean/H/分位数与原温度恰为三倍。正式 train/val 的所有有效温度都大于 `1.5` 是数学上预期的，因为外层 `T_out=3`；这不表示像素预算越界。像素预算边界只适用于 `T`。

## 4. Git、dirty 状态与源码指纹

### 4.1 正式产物记录的快照

| 字段 | 正式值 |
|---|---|
| Git commit | `6362097501bbdab4234bc39a8e04876046f84e6f` |
| 短 SHA / 提交信息 | `6362097` / `实现 Phase O1.2 预算校准与机制门禁` |
| Git dirty | `true` |
| dirty entry count | `4` |
| 配置指纹 | `047ba2158ba290a3b50b63241258a5fb7caa8c365cea75bb94ac5036dcce406f` |

正式运行时的 4 个 dirty 项均为与本机制诊断无关的未跟踪实验 runner：

~~~text
scripts/experiments/kd_baselines_npu/check_phaseO_rtc_runs.py
scripts/experiments/kd_baselines_npu/launch_phaseO_rtc.sh
scripts/experiments/kd_baselines_npu/run_phaseO_rtc.sh
scripts/experiments/kd_baselines_npu/run_phaseO_rtc_variant.sh
~~~

本执行记录在正式产物生成后创建，因此它不在正式 JSON 的 `dirty_entry_count=4` 快照中。

该提交共改动 7 个 O1.2 文件，统计为 `6047 insertions(+), 13 deletions(-)`；未推送 GitHub，按用户要求由用户自行处理远端推送。

### 4.2 正式核心源码 SHA256

| 来源键 | SHA256 |
|---|---|
| `rtc_o12_calibration` | `5e23f3e1bd526017aa2046faff621dd284093d91dead9088dd4d7d9ddb641d9e` |
| `diagnose_rtc_o12_budget` | `cc391388f64505abae4cded5ac7b36122018a131b3c90f480290ff275a4cee50` |
| `check_rtc_o12_gate` | `805a19625d496d3c3864d529e314a49d75584afad69fedf69e28cadc431ce085` |
| `train_entry` | `f50a130c17ca3b5f368f848b96c59fd58c408952e75bda447c051589505b316d` |

正式 checker 验证当前四份文件与产物记录 SHA 完全一致。

## 5. 测试与编译记录

### 5.1 最终测试快照

- 最终完整测试套件：`103 passed, 4 warnings in 8.95s`；
- O1.2 定向测试：`42 passed, 4 warnings in 12.68s`；
- Python 编译检查：diagnose、checker 和 gate 测试文件通过；
- 4 条 warning 均为环境侧 CANN owner mismatch 提示，不是测试失败；
- 第二轮 8 图 NPU smoke 通过除 `formal_full_scan/canonical path` 之外的机制与运行身份检查；这些剩余 false 对临时 smoke 是预期值；
- 随后的正式全量 solve/train/val 和联合 checker 全部通过。

开发过程中还运行过一次更宽的 RTC 兼容回归：`79 passed, 4 warnings in 7.96s`。它发生在最终 NPU-index 门禁修复之前，只作为历史回归记录，不能替代上面的最终定向测试、smoke2 和正式联合门禁。

### 5.2 测试覆盖重点

测试覆盖门函数边界和互斥、公式单调性、一维求解和不可行拒绝、val 禁止重拟合、教师置信度/熵/argmax 方向、学生 softmax 不变性、target detach、ignore mask、常数温度等价、DDP 全局有效像素归一化、within-image shuffle 映射、canonical 数据顺序、checkpoint/resume、来源 SHA 漂移、缓存篡改、NPU 请求/实际设备不一致，以及 checker 自身 SHA 契约。

## 6. 两轮 8 图 smoke：失败与修复链

smoke 只验证链路和 schema；其参数来自 8 图局部人口，既不是正式参数，也不能用于任何效果判断。

| 轮次 | 时间（UTC，约） | 设备 | 结果 | 解释 |
|---|---|---|---|---|
| smoke1 | 10:37-10:38 | solve/train `npu:0`，val `npu:1` | 联合 gate=false | 除临时路径、非全量人口等预期失败外，暴露 NPU:0/同卡硬编码 P1 |
| smoke2 | 10:49-10:50 | solve/train `npu:0`，val `npu:1` | 联合 gate 仍为 false | 只剩 canonical path 与 formal full scan 预期失败；NPU P1 已消失 |

smoke1 的真实问题项包括：val `runtime_actual_npu_zero=false`，以及联合的 `train_val_each_single_process_single_npu`、`solve_train_val_each_single_process_single_npu`、`solve_train_val_same_actual_npu` 被错误判 false。修复后每份 artifact 均验证其请求设备与当前设备一致，跨 artifact 同卡不再是门禁。

smoke2 的 `b=0.39572031796...` 只来自 8 图局部人口，禁止抄作正式结果。正式 `b` 只能使用第 9 节的全量解。

两轮 smoke JSON 的 SHA256：

| 轮次 | artifact | SHA256 |
|---|---|---|
| smoke1 | solve | `ed0a2887b7b87982733cad1ebc7c943486ebfd137252f46822bbac997a761890` |
| smoke1 | train evaluate | `a6d20834b086c4d5264fb36e0e31003bfab80034dac564f58fa00f0470fc432c` |
| smoke1 | val evaluate | `1264ee628178e68212e57ec1123387c201accea49d7da720bc692a2eb7b01a55` |
| smoke1 | joint gate | `6f071cbf2aa1c418345b4e9f9d48c8c33e8543346c120c5d7a8932b8fb8272cf` |
| smoke2 | solve | `5f3bec23a3b315c99ea086adc84bd43ffbd87faf20cfd9254bfcb003c51f6d67` |
| smoke2 | train evaluate | `a78cd04bc97d5a71b61b2bb1557a09e6929818ff7b373073751c582413c77a26` |
| smoke2 | val evaluate | `72617e30f3742f04d53b4004f6d53b6fb2d42753abe63f25e4a5d78963ef8278` |
| smoke2 | joint gate | `928e639618b13ac2fd8887575599034acea2a3cdaa55e884b76cff36917c87e6` |

这些临时文件位于 `/tmp/phaseO_o12_smoke*`，不属于仓库冻结产物；SHA 仅用于保留失败—修复证据。

## 7. 正式命令与扫描协议

工作目录均为 `/home/ma-user/work/ljs`，固定解释器为：

~~~text
/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python
~~~

### 7.1 完整 train 求解

~~~bash
/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python \
  scripts/diagnostics/diagnose_rtc_o12_budget.py \
  --stage solve --split train --device npu:0 --max-images 0 \
  --log-every 100 --strict
~~~

JSON 原始 `argv`：

~~~text
scripts/diagnostics/diagnose_rtc_o12_budget.py --stage solve --split train --device npu:0 --max-images 0 --log-every 100 --strict
~~~

### 7.2 完整 train 冻结参数评估

~~~bash
/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python \
  scripts/diagnostics/diagnose_rtc_o12_budget.py \
  --stage evaluate --split train --device npu:0 --max-images 0 \
  --parameters runs/diagnostics/phaseO_o12/o12_budget_parameters.json \
  --log-every 100 --strict
~~~

### 7.3 完整 val 无重拟合评估

~~~bash
/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python \
  scripts/diagnostics/diagnose_rtc_o12_budget.py \
  --stage evaluate --split val --device npu:1 --max-images 0 \
  --parameters runs/diagnostics/phaseO_o12/o12_budget_parameters.json \
  --no-scale --no-mirror --log-every 100 --strict
~~~

### 7.4 独立联合 checker

~~~bash
/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python \
  scripts/diagnostics/check_rtc_o12_gate.py --strict
~~~

checker JSON 不保存自身完整 `argv`；上式是使用脚本 canonical 默认路径和默认输出的可复现等价命令，不应写成“从 JSON 原样恢复”。

### 7.5 冻结扫描协议

| 项目 | train solve/evaluate | val evaluate |
|---|---|---|
| 图像数 | 10,582 | 1,449 |
| crop | 512x512 | 原始可变分辨率 |
| batch/workers | 4/0 | 1/0 |
| scale/mirror | true/true | false/false |
| seed | 2025 | 2025 |
| 进程 | rank0、world_size1、单 NPU | rank0、world_size1、单 NPU |
| 教师网格 | native logits | native logits |
| valid mask | nearest 到 native logits | nearest 到 native logits |
| CDF 查询 | float32 right-continuous step | 同左 |

扫描协议指纹：solve train=`aa70ededf7df28f60ac3b6399e8dbf97dc2ea4c67e1cb92e34012f9307d2e55a`，evaluate train=`d5328ba475b518d904fecf06d5b9238508f56dc3cef7450f945cf1b06a341bd1`，evaluate val=`252523d9a0c1542785e2f5a815ecd65f2944d8ee8209250cc1c450b64cd07e1d`。

## 8. 正式运行身份、UTC 与耗时

| 阶段 | PID | 请求/当前 NPU | 设备名 | UTC 开始 | UTC 结束 | 耗时 |
|---|---:|---|---|---|---|---:|
| train solve | 2381019 | 0/0 | Ascend910_9392 | 2026-07-13T10:54:25.963653+00:00 | 2026-07-13T10:59:32.714316+00:00 | 306.750683 s |
| train evaluate | 2387410 | 0/0 | Ascend910_9392 | 2026-07-13T10:59:58.504290+00:00 | 2026-07-13T11:05:36.306057+00:00 | 337.801772 s |
| val evaluate | 2394670 | 1/1 | Ascend910_9392 | 2026-07-13T11:06:03.308682+00:00 | 2026-07-13T11:07:09.952398+00:00 | 66.643723 s |

联合 gate 创建时间为 `2026-07-13T11:07:44.638031+00:00`。三份 artifact 的 UTC、PID、rank/world size、请求设备、当前设备和设备名均通过 checker；`same_actual_npu=false` 只是来源记录，不是失败。

## 9. 正式预算求解结果

| 字段 | 正式值 |
|---|---:|
| train 完整人口 | 32,246,990 |
| nonfinite | 0 |
| `a` | 0.10536051565782628 |
| `b` | 0.3476499170064926 |
| `b_max=log(1.5)` | 0.4054651081081644 |
| 目标 A | 0.995 |
| 求解 A | 0.9950000002788227 |
| H | 0.9880688428627591 |
| A 残差 | 2.7882274267199136e-10 |
| 二分次数 | 64 |
| `A(b=0)` | 0.9693387715088079 |
| `A(b=b_max)` | 0.9998279647099281 |
| 可行 | true |
| 理论高风险端点 `exp(b)` | 1.4157365376897748 |

求解输入是完整 float32 `u` 缓存，不是十分位直方图近似。checker 从 NumPy 缓存独立执行同样 64 次二分，得到同一 `b`，绝对差不超过 `1e-8`。

预保存的后续对照标量为：

| 分支 | arithmetic matched | harmonic matched |
|---|---:|---:|
| unreliable_only | 1.0256612287700149 | 1.0212646673765022 |
| full_budgeted | 0.9950000002788227 | 0.9880688428627586 |

这些只是冻结控制参数，当前没有用它们训练学生。

## 10. NumPy 风险缓存独立审计

除 checker 外，本记录还直接加载三份 `.npy` 做了独立只读检查：

| 缓存 | dtype/shape | finite | min/max | 区域人口（R/N/U） | `u==0.6/0.8` |
|---|---|---:|---|---|---|
| train solve | float32 / `(32246990,)` | 32,246,990 | 0 / 0.999755859375 | 19,413,566 / 6,435,793 / 6,397,631 | 0/0 |
| train evaluate | float32 / `(32246990,)` | 32,246,990 | 0 / 0.999755859375 | 19,413,566 / 6,435,793 / 6,397,631 | 0/0 |
| val evaluate | float32 / `(3878674,)` | 3,878,674 | 0 / 0.999755859375 | 2,344,298 / 716,449 / 817,927 | 0/0 |

train solve 与 train evaluate 缓存 SHA 完全相同，逐元素完全相等，`max_abs_diff=0.0`。这证明求解和正式 train 评估使用同一冻结风险人口与扫描顺序。

## 11. train/val 全量结果

### 11.1 全局教师错误与风险证据复核

| Split | valid | 教师错误 | micro 错误率 | top-risk 覆盖 | 错误召回 | 风险富集 | 十分位成对单调率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 32,246,990 | 949,984 | 0.029459618 | 0.198394672 | 0.986971360 | 4.974787630 | 1.0 |
| val | 3,878,674 | 241,390 | 0.062235187 | 0.210877996 | 0.826276979 | 3.918270256 | 1.0 |

这支持“教师总体错误不占多数，但错误高度富集于高风险侧”，不支持“高风险侧等于教师错误”。事实上高风险侧仍有大量教师正确像素：train `6,397,631-937,607=5,460,024`，val `817,927-199,455=618,472`。

### 11.2 温度总体分布

| 指标 | train | val |
|---|---:|---:|
| count | 32,246,990 | 3,878,674 |
| min | 0.8999999762 | 0.8999999762 |
| max | 1.4145361185 | 1.4145361185 |
| arithmetic mean | 0.9950000003 | 0.9960921506 |
| harmonic mean | 0.9880688429 | 0.9881613751 |
| q01 | 0.9015446901 | 0.9014287591 |
| q10 | 0.9156452417 | 0.9128624201 |
| q50 | 0.9824246764 | 0.9790610671 |
| q80 | 1.0000000000 | 1.0010589361 |
| q90 | 1.0896940231 | 1.1127897501 |
| q95 | 1.2158285379 | 1.2398405075 |
| q99 | 1.3685089350 | 1.3774102926 |
| top risk decile mean | 1.2285796997 | 1.2333745846 |

覆盖统计：

| 条件 | train count / coverage | val count / coverage |
|---|---:|---:|
| `T<0.9` | 0 / 0 | 0 / 0 |
| `T==1` | 6,443,626 / 0.199821007 | 717,386 / 0.184956510 |
| `T>1` | 6,389,798 / 0.198151765 | 816,990 / 0.210636420 |
| `T>1.25` | 1,275,467 / 0.039553056 | 180,088 / 0.046430297 |
| `T>1.5` | 0 / 0 | 0 / 0 |

注意 val 的 `q80` 略大于 1，是因为冻结 train CDF 在 val 上的风险人口并非精确均匀，且 val 的严格中性人口覆盖约为 18.50%；这不是 val 重拟合或门函数边界改变。

### 11.3 三个风险区的机制方向

| Split/区域 | count | 教师错误率 | mean T | `c_base -> c_target` | `H_base -> H_target` |
|---|---:|---:|---:|---:|---:|
| train reliable | 19,413,566 | 0.000031576 | 0.949070 | 0.821976 -> 0.849558 | 0.967063 -> 0.834891 |
| train neutral | 6,435,793 | 0.001827902 | 1.000000 | 0.695818 -> 0.695818 | 1.441044 -> 1.441044 |
| train unreliable | 6,397,631 | 0.146555342 | 1.129344 | 0.505756 -> 0.455059 | 1.848118 -> 2.048945 |
| val reliable | 2,344,298 | 0.005204543 | 0.945652 | 0.829952 -> 0.858559 | 0.931291 -> 0.792721 |
| val neutral | 716,449 | 0.041501907 | 1.000000 | 0.696063 -> 0.696063 | 1.437311 -> 1.437311 |
| val unreliable | 817,927 | 0.243854280 | 1.137237 | 0.499127 -> 0.447298 | 1.846685 -> 2.054602 |

表中最后一列的 `H` 表示教师目标熵，不是温度调和均值。可靠侧置信度提高且熵下降；不可靠侧置信度下降且熵上升；中性区逐元素不变，方向与设计一致。

### 11.4 风险十分位单调性

| bin | train count | train 错误率 | train mean T | val count | val 错误率 | val mean T |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3,284,698 | 0 | 0.907967 | 475,624 | 0.000306965 | 0.907952 |
| 1 | 3,225,316 | 0.000001860 | 0.924066 | 423,357 | 0.001421968 | 0.923939 |
| 2 | 3,212,217 | 0.000004981 | 0.940386 | 394,923 | 0.003157578 | 0.940308 |
| 3 | 3,210,850 | 0.000016195 | 0.957063 | 369,575 | 0.005595617 | 0.956963 |
| 4 | 3,221,610 | 0.000039732 | 0.974016 | 345,133 | 0.009002327 | 0.973923 |
| 5 | 3,258,875 | 0.000126117 | 0.991269 | 335,686 | 0.014987220 | 0.991221 |
| 6 | 3,233,884 | 0.000491669 | 1.000000 | 343,831 | 0.028493649 | 1.000000 |
| 7 | 3,201,909 | 0.003177479 | 1.000000 | 372,618 | 0.053505198 | 1.000000 |
| 8 | 3,191,600 | 0.031711367 | 1.029660 | 386,419 | 0.123974235 | 1.029882 |
| 9 | 3,206,031 | 0.260882381 | 1.228580 | 431,508 | 0.351207857 | 1.233375 |

两套错误率均随十分位严格单调增加；温度在可靠侧逐渐回到 1，中性区保持 1，高风险尾部再增强平滑。

### 11.5 有效温度

| 指标 | train `T_effective` | val `T_effective` |
|---|---:|---:|
| min | 2.6999998093 | 2.6999998093 |
| max | 4.2436084747 | 4.2436084747 |
| mean | 2.9849999997 | 2.9882764507 |
| harmonic mean | 2.9642065272 | 2.9644841238 |
| q50 | 2.9472739697 | 2.9371831417 |
| top risk decile mean | 3.6857390995 | 3.7001237548 |

这些值只描述教师 logits 实际除数 `3*T`；学生 softmax 仍固定温度 1。

### 11.6 事后分层诊断

| Split/分层 | count | 教师错误率 | mean T |
|---|---:|---:|---:|
| train background | 20,633,462 | 0.020773925 | 0.976449 |
| train foreground | 11,613,528 | 0.044891268 | 1.027958 |
| train boundary | 9,422,384 | 0.095825855 | 1.063205 |
| train interior | 22,824,606 | 0.002062511 | 0.966844 |
| train small object | 27,976 | 0.593329997 | 1.235643 |
| train not-small | 32,219,014 | 0.028970005 | 0.994791 |
| val background | 2,850,690 | 0.023191227 | 0.973647 |
| val foreground | 1,027,984 | 0.170507518 | 1.058335 |
| val boundary | 1,009,838 | 0.155175385 | 1.055739 |
| val interior | 2,868,836 | 0.029519987 | 0.975096 |
| val small object | 10,475 | 0.552267303 | 1.168845 |
| val not-small | 3,868,199 | 0.060908190 | 0.995624 |

正式 JSON 还保存了 21 个 GT 类别的逐类 count、教师错误率、温度、置信度和熵。class、前景/背景、边界/内部、小目标/非小目标的人口闭合检查均为 true。这些结果提示高风险与边界、小目标、前景困难区域相关，但它们是事后诊断，不进入风险计算、`a,b` 求解或 O1.2-A 门禁，也不能由此推出学生获益。

## 12. 数值、方向和联合门禁

### 12.1 train/val 全部为零的 16 类违规

以下键在 train 和 val 中都精确为 `0`：

~~~text
nonfinite
reliable_temperature_above_one
neutral_temperature_nonunit
unreliable_temperature_below_one
formula_monotonic
reliable_confidence_decrease
unreliable_confidence_increase
reliable_entropy_increase
unreliable_entropy_decrease
teacher_argmax_mismatch
temperature_out_of_bounds
neutral_target_mismatch
student_softmax_changed
population_closure
risk_bin_closure
error_count_closure
~~~

数值 probe：

| Split | neutral target max abs error | student softmax max abs error | teacher target probe max difference |
|---|---:|---:|---:|
| train | 0.0 | 0.0 | 0.1091369689 |
| val | 0.0 | 0.0 | 0.1092391014 |

前两项为零证明中性退化和学生端不变；第三项非零证明空间温度实际改变了教师目标，并非死分支。valid/finite、风险区、十分位、错误数、21 类、前景/背景、边界/内部和小目标分层均闭合。

### 12.2 联合门禁

正式 [o12_joint_gate.json](../runs/diagnostics/phaseO_o12/o12_joint_gate.json) 的 `joint_gate_pass=true`。23 项顶层联合检查全部为 true，覆盖：

- O1.1 四份冻结源码字节一致；
- canonical parameters/train/val 路径；
- CDF、teacher、train/val list 实际 SHA；
- O1.1 gate 仍为 true；
- parameter/train/val 各自独立 gate；
- train solve/evaluate 缓存 SHA 相同；
- 每份 artifact 单进程单 NPU；
- 4097 点 float64 公式单调性。

checker 没有信任 diagnose 的 `all_checks_pass`，而是重新加载 NumPy 缓存、重算 `a,b,A,H`、温度统计、有效温度、分层闭合、方向条件和来源契约。公式网格结果：`grid_points=4097`、`minimum_difference=0.0`、`violation_count=0`、`pass=true`。

## 13. 冻结输入与正式产物 SHA256

### 13.1 输入来源

| 输入 | 路径 | SHA256 |
|---|---|---|
| O1.1 confidence CDF | `runs/diagnostics/phaseO_o11/voc_train_rtc_confidence_cdf.pt` | `8ba8376a03c2f93835339d98670fa357b381b23a22cbf2a7fc48b0c2441d7e69` |
| teacher | `data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth` | `ac49b2c7720b21d565072e974e4404fcb009ba106288d95d1a4bb25f09c3fe58` |
| train list | `dataset/list/voc/train_aug.txt` | `d1326bd532648d73bc4b1bd275434eba81930982dc7401a65a8cecb26c028e24` |
| val list | `dataset/list/voc/val.txt` | `cdc1326d12f69ce5153aa5da04a4d8783e146868d42d19f82f44e74b97ac907d` |

### 13.2 正式产物

| Artifact | SHA256 |
|---|---|
| [o12_budget_parameters.json](../runs/diagnostics/phaseO_o12/o12_budget_parameters.json) | `a9006972e165ed6c95e7a966c526927bb9f846fb6c7ec6528f5507dd21b8b5df` |
| [o12_budget_train.json](../runs/diagnostics/phaseO_o12/o12_budget_train.json) | `8deb4850a629e7a7b44a6ed52bf86b988857e98bf786ee39e6e948995f448b6e` |
| [o12_budget_val.json](../runs/diagnostics/phaseO_o12/o12_budget_val.json) | `d1dde29c569df7a6fabda32ec3daa50bb6a16758125768e5059c1762cb0b1a5f` |
| [o12_joint_gate.json](../runs/diagnostics/phaseO_o12/o12_joint_gate.json) | `c1961cfec8f232c1fdf7fc8b90db5a6f4855901317ea2705536eea88cbc1be82` |
| [o12_budget_train_solve_u.npy](../runs/diagnostics/phaseO_o12/o12_budget_train_solve_u.npy) | `e3d0b16b3082464a25558aa00ab26b319a4020cadce6aebf4055dd84b9d8eade` |
| [o12_budget_train_evaluate_u.npy](../runs/diagnostics/phaseO_o12/o12_budget_train_evaluate_u.npy) | `e3d0b16b3082464a25558aa00ab26b319a4020cadce6aebf4055dd84b9d8eade` |
| [o12_budget_val_evaluate_u.npy](../runs/diagnostics/phaseO_o12/o12_budget_val_evaluate_u.npy) | `855b21d471cd1989c57786f4c7d6e125213e1fcd6d21b8270f697c595f8d9f49` |

正式 JSON 和 NumPy 文件均位于 `runs/diagnostics/phaseO_o12/`。不得用 smoke 临时产物覆盖它们。

## 14. 当前方法解读

当前全量诊断与重构动机一致：教师在 native-valid 像素上总体错误率不高，但错误在 confidence-only 高风险尾部明显富集；因此没有必要对大多数像素继续采用接近 `T=0.6` 的全局强锐化。新映射把可靠侧最低温度限制在 `0.9`，设置严格中性区，并将主要正温度偏移集中到高风险尾部，在 train 上把 `A` 固定到 `0.995` 且保持 `H≈0.988`。

不过，这一诊断只能说明温度预算和方向符合设计。高风险区仍以教师正确像素为多数；平滑会同时削弱正确和错误教师目标。它最终是否减少错误模仿、是否损害 teacher-correct retention、是否改善 mIoU，必须由后续 neutral 对 unreliable_only 的学生实验和 matched-scalar/shuffle 对照回答。

分层结果还显示边界和小目标拥有更高教师错误率与更高平均温度。这可作为后续机制审查重点，但不能反向修改本阶段阈值，也不能把相关性叙述成空间位置的独立因果作用。

## 15. 偏差、限制与审查提示

1. 正式产物的 Git 快照为 dirty，原因已逐项列出；核心源码和所有冻结输入仍由实际 SHA 校验，而不是只信任 commit。
2. 正式 val 在 NPU:1、solve/train 在 NPU:0；这是有意的执行调度，不改变统计定义，且已通过请求设备与实际设备一致性检查。
3. 第一轮 smoke 的联合失败不能删除；它是发现 NPU 门禁定义错误的证据。第二轮 smoke 证明修复，但临时人口仍不满足正式门禁。
4. 当前证据来自 VOC 和同一冻结教师；换数据集时允许按固定算法在新训练集重建 confidence CDF 并重新求预算参数，但不得用新 val 标签调参。当前结果不保证跨数据集泛化。
5. 风险和温度统计是像素 micro 统计；分层统计缓解了背景占优的解释风险，但没有提供图像级、seed 级或跨数据集显著性。
6. O1.2-B 虽生成了两个 20-step 学生 checkpoint，但没有 validation 或学生预测；任何“提高精度”“优于标量”“减少错误模仿”或“空间位置有效”的表述仍超出证据。

## 16. O1.2-A 当时的停止线与授权条件

以下为 O1.2-A 结束时的历史停止线；O1.2-B 后的当前边界见第 17 节。

截至 O1.2-A 记录：

- O1.2-A 实现、正式诊断和联合门禁已经完成；
- 正式参数、CDF、配置、核心源码和 artifact SHA 应保持冻结；
- Phase N 不自动恢复；
- 不使用旧 Phase O launcher 运行 O1.2；
- 不启动 20-iteration 学生 smoke、20k 或 80k；
- 不因查看后续学生结果而修改 `q_R/q_U`、指数、端点、`A_target`、`a/b` 或 CDF；
- 如果核心源码、输入或正式 artifact 发生变化，训练入口与 checker 应 fail-closed；需要变更时另建阶段和新产物，不覆盖本记录的历史证据。

下一步只能在人工审查本记录和正式产物、并获得明确授权后进入 O1.2-B。届时第一批 20-iteration 链路 smoke 只允许 `neutral` 与 `unreliable_only`，且仍不得从 20 iteration 宣称 mIoU 效果。该授权随后已于 2026-07-13 给出并执行完毕；本段保留为 O1.2-A 历史证据。

## 17. O1.2-B：neutral 与 unreliable_only 正式 20-step smoke

### 17.1 最终结果

四条链路的 final acceptance 均为 `pass=true`，对应 artifact 如下：

| 变体/模式 | Acceptance | Acceptance SHA256 | Checkpoint SHA256 | optimizer steps | NPU |
|---|---|---|---|---:|---:|
| neutral fresh 20-step | [acceptance.json](../runs/runtime/kd_baselines_npu/phaseO_o12/o12b_neutral_smoke20_seed1234/acceptance.json) | `90eb89fe7a7dff773845152d5d59a7efcc3d61299976137223be6cf7d2615e1c` | `6309cba985d8831586a610c45a4f463ba15d71381924b7093ca9f9fa91a982a9` | 20 | 0 |
| neutral endpoint resume=0 | [acceptance.json](../runs/runtime/kd_baselines_npu/phaseO_o12/o12b_neutral_smoke20_seed1234_resume_audit/acceptance.json) | `30fe00fa72b2f20293db2cfca3da7f0f9d960649fef8d153629b3d2b2f496677` | `6b0032efd44202181cb2bf4248209b788ea9573b92541ffe339562804d7d3e44` | 0 | 0 |
| unreliable_only fresh 20-step | [acceptance.json](../runs/runtime/kd_baselines_npu/phaseO_o12/o12b_unreliable_only_smoke20_seed1234/acceptance.json) | `d8a3363724c7c5d93c7629ba0ebf365a7f73d1c25c53138633e179e21fe623cd` | `b9426976a45ab1ec6396bb68d85f2f43fe80f176d94fde8c5395580a41ea0ed4` | 20 | 1 |
| unreliable_only endpoint resume=0 | [acceptance.json](../runs/runtime/kd_baselines_npu/phaseO_o12/o12b_unreliable_only_smoke20_seed1234_resume_audit/acceptance.json) | `af540bf53a0b1646227eb9024a6eb9f5d7bf1af250773c62d1074b77aaa82488` | `8760408718353c0e8123d4a4a5965eb63e8db504e23724d9b57c19ce77362aca` | 0 | 1 |

fresh student-weight SHA256 分别为 neutral `87dc4b5a7f8ed6b6285f36b83a15bbb72d17c7e8d1f5a0a27e53324807ab26dd`、unreliable_only `15744fa57c1a38a2c927579ce2e46c8a11bc6f7178f7ab7667cfbbc5d2e9d5a7`；各自 resume audit 的 student-weight SHA 与对应 fresh 完全相同。Git commit 为 `5c93c1eeebfdf58ca4a9c82aca9852cb8a661330`。

### 17.2 训练、数值与运行记录

| 指标 | neutral fresh | unreliable_only fresh |
|---|---:|---:|
| step-20 task loss | 1.7301 | 1.7785 |
| step-20 O1.2 KD loss | 0.9500 | 1.0019 |
| step-20 KD KL | 0.95004142 | 1.00187660 |
| teacher target entropy | 1.22812260 | 1.26829381 |
| KD-only student-logit gradient L2 | 0.00273925 | 0.00282540 |
| optimizer runtime | 11.030566 s | 11.530959 s |
| mean step time | 0.5515 s | 0.5765 s |
| throughput | 29.011786 samples/s | 27.753686 samples/s |
| process peak memory | 12,863 MB | 12,862 MB |
| NPU HBM delta | 12,812 MB | 12,807 MB |

上述 loss/KL/entropy 只用于证明链路有信号且数值有限。两个变体使用不同教师目标，且每条链路只有一个 step-20 记录点，因此不能用这些数字判断性能或机制优劣。两份 fresh acceptance 的 finite scan 均覆盖 847 个 tensor、19,711,341 个 tensor element 和 7 个 float scalar，未发现非有限值；acceptance 的 `errors`、`warnings` 均为空。

### 17.3 顺序、梯度与恢复证据边界

- 四份 acceptance 的 canonical order 均含 320 个索引，seed=`1234`，SHA256=`99326472a2e5e2bd42428d4709ff9f8049d2c906c7a6b9e8fa3068cb0439d564`。这支持同 seed canonical dataset-index 顺序一致，不支持 8-worker 随机增强按位重放；没有把实际 `sample_names` 日志作为证据。
- 梯度门禁证明总 generator/discriminator loss 有限，并在 step 20 记录了有限且非零的 KD-only student-logit 梯度；更新后的 checkpoint 张量也通过有限性扫描。没有逐参数扫描每一个 model parameter 的梯度，因此不得写成“所有参数梯度逐一有限”。
- 两个 resume audit 从对应 iteration=20 fresh checkpoint 严格加载，核对关键状态和 student-weight SHA 后以 `optimizer_steps=0` 结束。它们证明终点 checkpoint 的可加载性及状态一致，不证明恢复后的下一个训练 batch 已实际执行；单元测试中的 sampler slicing 证据也不能替代一次真实 next-step continuation。

### 17.4 原始环境告警

四条控制台日志都保留了 CANN 安装目录和 `ascend_ops_install.info` 的 owner mismatch 告警。fresh run 还记录了在 `allow_internal_format=False` 时回退到 base format 的环境告警；resume audit 记录了旧权重文件格式/当前 torch 兼容性与未来弃用告警。这些告警没有导致 checker 失败，四份 final acceptance 的结构化 `errors=[]`、`warnings=[]`，但原始日志不得删除，也不能把它们当成方法有效或无效的证据。

### 17.5 证据结论与当前停止线

本阶段使用 `skip-val`，没有运行 validation、保存预测或计算 mIoU、student rescue、error imitation、teacher-correct retention。因而当前唯一新增结论是：两条 20-step 训练链路和两个终点 checkpoint 严格加载审计按预注册数值/运行门禁通过。

当前停在 O1.2-B 后人工审查线：

- 不自动启动 `reliable_only`、`full_budgeted`、scalar、shuffle、20k、C2、C3 或 80k；
- 不从 step-20 loss、KL、吞吐或显存差异推断变体效果；
- 不宣称 mIoU、收敛、错误处理收益、空间因果或跨数据集泛化；
- 不修改已冻结的 `q_R/q_U`、指数、端点、预算、CDF 或 O1.2-A 正式 artifact；
- 后续只有在新一轮明确授权后才能执行，并须另行记录，不能覆盖 A/B 历史。

完整 smoke 报告：[O1.2-B 20-step 正式链路 smoke 报告](2026-07-13_phaseO_rtc_o12b_smoke_report.md)。
