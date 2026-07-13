# Phase O：RTC-KD 执行与实验记录

- 创建日期：2026-07-13
- 最后同步：2026-07-13（O1 正式联合门禁完成）
- 当前阶段：O1 已完成，联合门禁未通过；O2 未启动
- 学生训练状态：尚未启动
- 代码版本：`38ec8edafa57c23beac2c4e32d8887745c69b42b`（产物记录为 dirty worktree）
- 方法预注册：[2026-07-13_phaseO_rtc_method_reconstruction_plan.md](2026-07-13_phaseO_rtc_method_reconstruction_plan.md)
- O1 详细诊断：[2026-07-13_phaseO_rtc_o1_diagnostic_report.md](2026-07-13_phaseO_rtc_o1_diagnostic_report.md)

> 结论先行：RTC 的双向温度构造、求解、mask 归约和路由诊断在工程上均通过；高风险侧也确实富集了教师错误。但当前预注册的 full score 在 train 与 val 上都未满足“AP 不得比 confidence-only 低超过 0.005”的科学门槛，因此 `joint_gate_pass=false`。本阶段不放宽门槛，不启动 O2/O3，也没有产生任何学生训练结果。

## 1. 执行原则

1. Phase N 和旧训练保持停止。
2. 先完成 RTC 数学、mask 归约、NPU 算子和全链路诊断，再决定是否进入学生训练。
3. O1 train/val 联合门禁未通过时，不启动 O2 的 20-iteration smoke。
4. O2 六个 smoke 未全部通过时，不启动 O3 20k。
5. RTC、neutral 和标量温度对照统一使用 masked pixel KD。
6. 正式门槛一经看到结果不得事后放宽；任何改分数的方案都视为新的 O1.1 假设，必须重新预注册。

## 2. O0：停止、环境与数据快照

状态：完成。

### 2.1 训练状态

- 未发现 `train_kd.py`、`train_cirkdv2.py`、`torchrun` 或 Phase N 控制进程；
- 两张 Ascend 910 均健康、空闲，无训练进程；
- Phase N 保持停止，未自动恢复。

### 2.2 运行环境

- Python：`/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python`；
- Python 版本：3.11.10；
- PyTorch：2.8.0+cpu，`torch.npu.is_available()=True`；
- NPU 数量：2；CUDA 不可用；
- CANN：`/usr/local/Ascend/cann-8.5.0/set_env.sh`。

### 2.3 数据与权重指纹

| 对象 | 数量/状态 | SHA256 |
|---|---:|---|
| VOC train list | 10,582 张，完整扫描 | `d1326bd532648d73bc4b1bd275434eba81930982dc7401a65a8cecb26c028e24` |
| VOC val list | 1,449 张，完整扫描 | `cdc1326d12f69ce5153aa5da04a4d8783e146868d42d19f82f44e74b97ac907d` |
| Teacher | 可加载 | `ac49b2c7720b21d565072e974e4404fcb009ba106288d95d1a4bb25f09c3fe58` |
| Student ImageNet init | 可加载；本阶段未使用 | `47085aa164b2977003221458a2a5fdf5f46f434f5539b164dc71969b4dd4cd75` |

训练与验证清单中的图像、标签缺失数均为 0。

## 3. 实现与诊断口径

### 3.1 RTC 单一实现源

`utils/rtc_temperature.py` 统一实现：

1. `RTCConfig` 与参数边界校验；
2. 冻结 CDF 保存、加载、SHA256 和设备侧查询；
3. 原始教师 logits 上的固定参考风险分数；
4. 连续、互斥的可靠侧/不可靠侧 gate；
5. top-vs-rest log-odds 目标与固定步数二分反求温度；
6. 每图 valid pixel 内温度 shuffle；
7. DDP 全局有效像素归一化的 masked KD；
8. 温度、覆盖、求解残差、fallback、tie 和方向违规诊断。

高置信像素的非主类方差采用显式排除 argmax 类后的稳定计算，避免 `sum(p²)-confidence²` 的 float32 灾难性消减。

### 3.2 CWD 接入

`train_kd.py` 已完成：

- `raw_teacher_logits` 只用于参考风险与路由；
- `teacher_kd_logits = raw_teacher_logits / T_out` 只用于目标、反求温度和 KD；
- RTC 与所有标量温度对照共用 masked KD；
- DDP 使用全局 valid count，避免各 rank 的 local mean 被等权混合；
- CDF 元数据校验教师、训练清单、类别数、`T_assess`、`a`、评分模式及源文件指纹；
- 跨 rank 校验 CDF SHA256；
- checkpoint 保存 RTC 配置与 CDF 指纹，恢复时拒绝配置漂移；
- 旧 Newton 与 legacy weight 路径保持不变。

### 3.3 O1 诊断口径

入口为：

- `scripts/diagnostics/build_rtc_cdf.py`；
- `scripts/diagnostics/diagnose_rtc_routing.py`。

CDF 与诊断都在教师原生输出网格计算，GT valid mask 使用 nearest resize。错误标签为 native-grid 教师 argmax 与 nearest-resized GT 是否一致；它是 KD 网格上的路由代理，不是标准 full-resolution segmentation accuracy。

覆盖率、错误精度、错误召回及低风险错误率均使用全部 native-valid 像素的 micro 统计；AP/AUC 为控制内存而使用每图最多 1,024 像素、固定 seed `3407` 的采样，并采用对同分数次序不敏感的 tie-aware 分组阈值实现。

## 4. 前置验证、失败扫描与修复

### 4.1 单元测试与设备 smoke

| 检查 | 结果 |
|---|---|
| RTC CPU 单元测试 | 18/18 通过 |
| 旧 CoVar 回归测试 | 3/3 通过 |
| NPU 随机 logits forward/backward | CDF、RTC、16 步二分、masked KD、反向均 finite |
| 8 train + 8 val 全链路 smoke | artifact、summary、诊断 JSON/CSV、方向与残差链路通过 |
| 无 cache 语法编译与 CLI help | 通过 |
| tie-aware 排名回归 | 正负样本同分时 AP=AUC=0.5，输入顺序不影响结果 |
| synthetic nonfinite 计数 | nonfinite 仍计入 full-valid 总体，并保守记为错误 |
| flat SHA 强元数据 | 正确元数据通过，指纹不一致被拒绝 |
| 联合门禁正/负例 | 合法输入通过；错误 CDF SHA 被拒绝 |

### 4.2 首次全量 CDF 扫描作废

首次扫描已经遍历完整 train 并收集约 3,170 万个样本，但在落盘前发现：

1. 旧的减法式 nonmax 方差在极高置信度下会发生 float32 消减；
2. `torch.quantile` 无法处理该规模的输入张量。

该扫描及其临时结果被明确作废，未用于任何方法结论或学生训练。

### 4.3 修复与诊断加固

- nonmax 方差改为显式排除主类后直接求平方偏差；
- CDF knots 改用 CPU NumPy quantile；
- 增加全 native-grid micro counter，不在计数前过滤 nonfinite；
- 明确 fallback、tie 与求解残差统计总体；
- AP/AUC 改为 tie-aware；
- train 诊断使用与 CDF 构建独立的增强 seed `2025`；
- 同时实际构造 `T_out=1` 与 `T_out=3` 路由图并逐像素核对；
- 增加完整运行标识、源文件 SHA、CDF/报告 SHA 与联合门禁检查。

## 5. O1：正式冻结 CDF

状态：完成。

### 5.1 构建配置与完整性

| 项目 | 正式值 |
|---|---|
| Split | VOC train_aug |
| 图像 | 10,582/10,582，full scan |
| Teacher grid | 原生 64×64 logits 网格 |
| 训练变换 | scale=True，mirror=True，crop=512×512 |
| CDF seed | `1234` |
| 类别数 | 21 |
| `T_assess` | 1.0 |
| 方差系数 `a` | 200 |
| Score | full |
| CDF knots | 4,097 |
| native-valid / finite / nonfinite | 32,298,651 / 32,298,651 / 0 |
| 风险分数 min / mean / max | 2.893758e-09 / 0.204975 / 5.441330 |

冻结产物：

- CDF：`runs/diagnostics/phaseO/voc_train_rtc_cdf.pt`，SHA256 `40b4454fc919422899e512fed4ece4a428d12544e7ce828a89dc731d12529e39`；
- summary：`runs/diagnostics/phaseO/voc_train_rtc_cdf.pt.summary.json`，SHA256 `909132e14477432513b025128bb8efb87787470643197dd09e3dfaca436c305e`；
- 创建时间：`2026-07-13T05:25:51.471314+00:00`。

源文件指纹：CDF 构建脚本 `7c9f6d5de606ce0a2e2d4de0af9b728b372e0d2ef6c0121fcd79af3753d8e7c2`，诊断脚本 `1c11ca5537668f6ebe121847e2243bd848fd38271838278d73359b900bb8904b`，RTC 实现 `4986b545536bb9d0417b60f1b57855a4ca1bdbe982364e09224db53a68628224`。

## 6. O1：正式 train/val 路由诊断

状态：完成；工程检查通过，科学门禁失败。

主参数：`q=0.8`、`w=0.05`、`T_R=0.5`、`T_0=1`、`T_U=2`、`alpha=1`、16 步二分、主诊断 `T_out=3`。

### 6.1 路由与错误富集

| 指标 | Train | Val |
|---|---:|---:|
| 图像完整性 | 10,582/10,582 | 1,449/1,449 |
| native-valid 像素 | 32,246,990 | 3,878,674 |
| finite / nonfinite | 32,246,990 / 0 | 3,878,674 / 0 |
| 教师 native-grid proxy 错误率 | 2.945962% | 6.223519% |
| 可靠侧覆盖率 | 80.152588% | 79.062381% |
| 不可靠侧覆盖率 | 19.847412% | 20.937619% |
| 不可靠侧错误精度 | 14.663683% | 24.396320% |
| 不可靠侧错误召回 | 98.791559% | 82.075894% |
| 错误富集倍数 | 4.977554× | 3.920020× |
| 可靠侧错误率 | 0.044416% | 1.410924% |

这说明风险排序能够把错误明显集中到不可靠侧，路由方向本身具有信息量。

### 6.2 排名指标与唯一失败项

| 排名分数 | Train AP | Train AUC | Val AP | Val AUC |
|---|---:|---:|---:|---:|
| full：confidence + variance | 0.385577 | 0.963462 | 0.369892 | 0.901715 |
| confidence-only | 0.394362 | 0.963283 | 0.385680 | 0.902400 |
| variance-only | 0.384014 | 0.963341 | 0.367645 | 0.901281 |

预注册门槛为：

```text
AP_full >= AP_confidence - 0.005
```

| Split | Full AP | Confidence-only AP | Full − Confidence | 差值门槛 | 相对门槛余量 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Train | 0.385577 | 0.394362 | -0.008785 | -0.005 | -0.003785 | 失败 |
| Val | 0.369892 | 0.385680 | -0.015788 | -0.005 | -0.010788 | 失败 |

full score 在两个 split 上都超过容忍范围地弱于 confidence-only。variance-only 也在两边低于 confidence-only；这构成一致的证据，不能用 train AUC 的极小上升抵消 AP 门禁失败。

### 6.3 温度求解与工程不变量

| 指标 | Train | Val |
|---|---:|---:|
| active / solved | 32,246,990 / 32,246,990 | 3,878,674 / 3,878,674 |
| fallback / tie | 0 / 0 | 0 / 0 |
| 方向违规 | 0 | 0 |
| 教师 argmax 改变 | 0 | 0 |
| 残差 mean / p95 / max | 3.914e-05 / 7.582e-05 / 1.135e-04 | 3.970e-05 / 7.534e-05 / 1.020e-04 |
| 温度 mean / harmonic mean | 0.766502 / 0.594266 | 0.782602 / 0.599948 |
| 温度 q10 / q50 / q90 | 0.500004 / 0.500004 / 1.945015 | 0.500004 / 0.500004 / 1.959251 |
| `T_out=1` vs `T_out=3` 路由图 mismatch | 0 | 0 |

因此，失败不是二分未收敛、fallback、方向反转、argmax 改变或 `T_out` 污染路由造成的；失败定位在当前 full 风险分数的科学假设。

### 6.4 正式产物与联合门禁

| 产物 | SHA256 |
|---|---|
| `rtc_routing_train.json` | `c51573f23264b89817e39b2d448fec61709b28147e83a0af1feb83f8f420eb98` |
| `rtc_routing_val.json` | `308065649920b33e920dd84353be966f4ead7e2cee91dbd64d263105b9eea383` |
| `o1_joint_gate.json` | `44997118f45a7c1e32f0028d5c50a8d1206e440a6afc4d07a35ae5852a3bd1af` |

- 联合配置指纹：`914a07126502eb63c139f38f10749e17633eea9d546c3e3c7a2b7f5700ed115a`；
- 联合门禁错误列表：空；
- train 与 val 的结构、完整性、数值和不变量检查均通过；
- train 与 val 唯一失败项均为 `full_ap_not_below_conf_by_0p005`；
- 最终：`joint_gate_pass=false`。

## 7. O2/O3 状态

状态：未启动，且当前未授权启动。

原预注册 O2 变体为 `neutral`、`reliable_only`、`unreliable_only`、`full`、`scalar_0p6_gamma0` 和 `shuffled`。由于 O1 联合门禁失败：

- 未运行任何 20-iteration 学生 smoke；
- 未运行任何 20k 或 80k 学生训练；
- 没有可报告的 student mIoU、loss 或 checkpoint；
- 不得把工程通过误写为方法通过。

## 8. 当前决定与后续边界

- O0/O0b：通过；
- O1 工程链路：通过；
- O1 科学门禁：失败；
- O2/O3：保持关闭。

看到正式结果后将 AP 容忍度从 0.005 放宽，或只引用错误富集而忽略 full-vs-confidence 对照，都会破坏预注册的可证伪性。当前结论必须保留。

如果继续，应先另立并预注册 O1.1；最小候选是把 confidence-only 作为新的风险分数基线，并重新冻结对应 CDF。原 full-vs-confidence AP 判据已经完成且失败，不能在新主分数等于 confidence-only 时继续充当门禁，否则会成为必然通过的同义比较。

O1.1 应改用预先冻结的非同义检查：风险覆盖、相对全局错误率的富集、教师错误 recall、低风险错误率、十分位单调性，以及有限性、温度方向、残差、`T_out` 解耦和 artifact 指纹。具体阈值记录在方法计划第 16.5 节。

由于 confidence-only 是看过本次 O1 train/val 后作出的探索性修订，在同一 VOC 上重跑只能验证实现和域内路由，不能声称独立确认。确认性证据必须来自未参与评分选择的新教师或新数据集。若仍保留方差项，其归一化与系数也必须在看新结果前固定。以上仅为下一阶段建议，本记录没有启动 O1.1。
