# Phase M2：匹配标量温度因果对照预注册与运行记录

- 文档创建时间：`2026-07-12`（Asia/Shanghai）
- 当前状态：`已预注册，待启动`
- 实验目的：判断 Phase M 中 Newton CoVar 的增益来自像素级温度分配，还是主要来自整体降低 KD 温度造成的锐化。
- 本阶段范围：VOC、CWD 配方、seed `1234`、`20000` iterations；新增标量 KD 温度 `T=0.5` 和 `T=0.6`，并与既有 `T=1.0` 和 Newton CoVar 结果比较。
- 预注册约束：启动训练前冻结本页的配置、主次指标和决策规则。若必须修改，须先在“偏差与变更记录”中写明时间、原因和影响，不得根据结果事后改口径。

## 1. 已完成实验与已知事实

### 1.1 Phase L：CWD 80k 三种子稳定性

来源：`reports/2026-07-11_phaseL_cwd_seed_stability.md`

| Seed | Best mIoU | Best iter | Final mIoU | 完成 |
|---:|---:|---:|---:|---|
| `1234` | `0.6640` | `79200` | `0.6610` | 是 |
| `2025` | `0.6600` | `73600` | `0.6580` | 是 |
| `3407` | `0.6650` | `75200` | `0.6640` | 是 |

- 原报告聚合：best `0.6630 +/- 0.0022`，final `0.6610 +/- 0.0024`。
- 上述 `+/-` 来自现有汇总脚本的总体标准差；后续论文统计统一预注册为样本标准差（`ddof=1`），按当前三次结果重算约为 best `0.6630 +/- 0.0026`、final `0.6610 +/- 0.0030`。
- Phase L 的 CWD 与旧 CoVar/CIRKD 使用不同基础配方，只能做方法级背景参考，不能作为本阶段的受控因果对照。

### 1.2 Phase M：CWD + CoVar 20k 单种子筛选

来源：`reports/2026-07-11_phaseM_cwd_covar_triage.md`

| 变体 | KD 温度机制 | Teacher output temp | Best mIoU | Best iter | Final mIoU | Runtime | 完成 |
|---|---|---:|---:|---:|---:|---|---|
| 历史 CWD | 标量 `T=1.0` | `1.0` | `0.6390` | `18400` | `0.6360` | `2:26:24` | 是 |
| CWD fixed | 标量 `T=1.0` | `3.0` | `0.6430` | `19200` | `0.6420` | `2:24:59` | 是 |
| CWD Newton CoVar | 像素级 `T(x)` | `3.0` | `0.6460` | `20000` | `0.6460` | `2:31:14` | 是 |

- 受控比较 CoVar 相对 fixed `T=1.0`：best `+0.0030`，final `+0.0040`。
- Phase M 的训练日志仅保留三位小数，因此表中的四位小数不是额外精度；本阶段解释小差异时必须保留该限制。
- 既有受控对照日志：
  - fixed `T=1.0`：`runs/logs/kd_baselines_npu/phaseM_cwd_covar/triage_20k/cwd_tout3_fixed_20k_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
  - Newton CoVar：`runs/logs/kd_baselines_npu/phaseM_cwd_covar/triage_20k/cwd_covar_newton_tout3_20k_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`

### 1.3 为什么必须增加 matched scalar controls

来源：`reports/2026-07-07_phaseH_h3_rt_distribution.md`

- Newton CoVar 的像素温度均值/中位数为 `0.5815 / 0.5000`。
- `90.35%` 的采样有效像素位于下界 `T_min=0.5`，`90.41%` 的像素满足 `T<0.75`。
- 因此，当前 CoVar 相对标量 `T=1.0` 的增益可能混合了两种效应：
  1. 全局低温锐化；
  2. 根据可靠性进行的空间自适应温度分配。
- 标量 `T=0.5` 对照匹配温度分布的主峰/中位数，标量 `T=0.6` 对照近似匹配均值。只有 CoVar 在这些对照下仍有稳定优势，才能把增益主要归因于像素级自适应机制。

## 2. 预注册实验问题与假设

### 2.1 主要问题

在完全相同的 CWD 训练配方、teacher output temp、数据顺序、seed 和训练预算下，Newton CoVar 的 `final mIoU` 是否高于最强的匹配标量温度对照？

### 2.2 假设

- `H0`：CoVar 相对 `T=0.5/0.6` 的 Phase M 增益可由整体低温锐化解释，像素级温度分配没有可辨识的额外收益。
- `H1`：CoVar 在匹配温度尺度后仍提高主指标，并且训练末段表现不劣于最强标量对照，支持像素级温度分配具有额外收益。

本阶段是单种子 20k triage，不做统计显著性结论；它只用于决定是否值得进入 80k 多种子成对验证。

## 3. 冻结配置

### 3.1 新增运行矩阵

| Run ID | 机制 | `--kd-temperature` | `--teacher-output-temp` | Seed | Iterations | 计划设备 |
|---|---|---:|---:|---:|---:|---|
| `cwd_tout3_kdtemp0p5_20k_seed1234` | 固定标量 KD 温度 | `0.5` | `3.0` | `1234` | `20000` | NPU `0` |
| `cwd_tout3_kdtemp0p6_20k_seed1234` | 固定标量 KD 温度 | `0.6` | `3.0` | `1234` | `20000` | NPU `1` |

既有 `T=1.0` 与 CoVar 不重跑；比较时直接使用 §1.2 中的 Phase M 产物。若发现代码、数据、teacher/student 权重或公共配方已变化，必须停止跨运行直接比较，并把四个变体在同一快照下重跑。

### 3.2 所有新增运行必须一致的公共配方

| 项目 | 冻结值 |
|---|---|
| 设备后端 | Ascend NPU；每个变体单卡、`NPROC_PER_NODE=1` |
| Python | `/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python` |
| 数据集 | VOC / VOCAug |
| 数据目录 | `/home/ma-user/work/ljs/dataset/VOCAug/` |
| Teacher | DeepLabV3 + ResNet-101 |
| Student | DeepLabV3 MobileNet SS-Seg + MobileNetV3-Small |
| Teacher 权重 | `/home/ma-user/work/ljs/data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth` |
| Student 初始化 | `/home/ma-user/work/ljs/data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth` |
| Crop / batch / workers | `512x512` / `16` / `8` |
| Optimizer 公共参数 | learning rate `0.02`、momentum `0.9`、weight decay `1e-4` |
| 训练预算 | `20000` iterations |
| 日志 / 保存 / 验证间隔 | `20 / 800 / 800` iterations |
| 随机种子 | `1234` |
| CWD 损失权重 | `lambda_kd=1.0`、`lambda_d=0.1`、`lambda_adv=0.001`、`lambda_cwd_fea=50.0`、`lambda_cwd_logit=3.0` |
| 验证 | 启用；共预期 `25` 次定期验证，包含 iteration `20000` |

新增标量运行不得启用 `--use-covar`。除 `--kd-temperature` 分别为 `0.5` 和 `0.6` 外，二者以及既有 Phase M fixed 运行的命令行和运行环境应保持一致。

### 3.3 对照中保持不变的 CoVar 配置

既有 CoVar 行使用：`temp_mode=newton`、`base=1.0`、`min=0.5`、`max=8.0`、`temperature_power=2.0`、`eta=0.6`、`max_iter=8`、`hessian_eps=1e-5`、`max_step=0.25`、`reliability_mode=full`。本阶段不调这些超参数。

### 3.4 产物路径

- Checkpoint 根目录：`/home/ma-user/work/ljs/data/winycg/checkpoints/kd_baselines_npu/phaseM2_scalar_temperature/`
- 指标日志根目录：`/home/ma-user/work/ljs/runs/logs/kd_baselines_npu/phaseM2_scalar_temperature/triage_20k/`
- `T=0.5` 指标日志：`/home/ma-user/work/ljs/runs/logs/kd_baselines_npu/phaseM2_scalar_temperature/triage_20k/cwd_tout3_kdtemp0p5_20k_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- `T=0.6` 指标日志：`/home/ma-user/work/ljs/runs/logs/kd_baselines_npu/phaseM2_scalar_temperature/triage_20k/cwd_tout3_kdtemp0p6_20k_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt`
- 启动器 stdout/stderr、PID 文件和最终 checkpoint 的实际路径在启动后填入 §6。

## 4. 预注册指标与比较方法

### 4.1 主指标

- `final mIoU`：iteration `20000` 验证得到的 mIoU。
- 主比较：`Newton CoVar final mIoU - max(T=0.5 final mIoU, T=0.6 final mIoU)`。
- 标量 `T=1.0` 作为历史受控锚点报告，但不参与“最强匹配标量”的选择。

### 4.2 次指标

- `best mIoU` 及对应 iteration；
- 最后 `10` 次验证 mIoU 的算术平均值（预期为 iterations `12800, 13600, ..., 20000`；以实际日志中最后 10 个有效验证点为准）；
- 运行时间与秒/iteration；
- 完整性：必须同时出现 `Iters: 20000/20000` 和 `Total training time:`，且最后一次验证可解析；
- 健康检查：无 NaN/Inf、无异常退出、验证次数符合预期。

若主指标与次指标方向冲突，以 `final mIoU` 作为阶段决策的首要依据，并把冲突明确记录，不得用 best 替代 final 宣称胜出。

### 4.3 精度与统计口径

- 原始日志能提供多少精度就报告多少精度，不得通过格式化虚构额外有效位。
- Phase M2 是同 seed 的成对配置比较；先报告逐运行值和有符号 delta，不对单种子结果给出置信区间或显著性结论。
- 后续三种子阶段报告 seed-wise paired delta、均值、样本标准差（`ddof=1`）和置信区间；主指标仍为 final mIoU。
- 所有用于汇总的解析脚本、命令及其 Git commit 必须随结果记录。

## 5. 预注册决策规则

令 `S*` 为 `T=0.5` 和 `T=0.6` 中 final mIoU 更高的标量对照，令 `Delta_final = CoVar_final - S*_final`。

1. 先按 final mIoU 选择 `S*`。若二者 final 完全相同，则按最后 10 次验证均值选择；若仍相同，保留两者并以更强的次指标表现作为文字说明，不进行有利于 CoVar 的任意挑选。
2. 若 `Delta_final >= +0.002`，且 CoVar 的最后 10 次验证均值不低于 `S*`，则判为“值得升级”：进入 80k、seeds `1234/2025/3407` 的 CoVar vs `S*` 成对实验。
3. 若 `Delta_final <= -0.002`，则判为“标量温度更强”：停止自动升级 CoVar，优先修正论文因果表述，并分析低温锐化机制。
4. 若 `-0.001 <= Delta_final <= +0.001`，则判为“20k 实质持平”：不据此主张像素级自适应收益；先做保持温度直方图但打乱空间位置的 permutation control，或增加一个 20k seed 后再决定。
5. 若 delta 落在上述规则未覆盖的过渡区间，或 final 与最后 10 次均值方向冲突，则判为“不确定”：先补 seed `2025` 的 20k 成对复核，不直接升级完整 80k 三种子实验。
6. 任一新增运行不完整、配置漂移或关键日志不可解析时，不做性能结论；先修复并从同一初始条件完整重跑受影响变体。

阈值以当前日志约 `0.001` 的分辨率制定，只用于资源分配而非统计显著性声明。

## 6. 启动、PID、日志与状态表

> 本节由启动实验的操作者在启动后立即回填。PID 必须用 `kill -0 <PID>` 与实际进程核验；不能仅凭 PID 文件判断运行中。

| Run ID | 启动命令（完整或脚本 + 环境变量） | NPU | PID / PID 文件 | stdout/stderr | 指标日志 | 启动时间 | 结束时间 | 状态 |
|---|---|---:|---|---|---|---|---|---|
| `cwd_tout3_kdtemp0p5_20k_seed1234` | `待填写` | `0` | `待填写` | `待填写` | `runs/logs/kd_baselines_npu/phaseM2_scalar_temperature/triage_20k/cwd_tout3_kdtemp0p5_20k_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt` | `待填写` | `待填写` | `未启动` |
| `cwd_tout3_kdtemp0p6_20k_seed1234` | `待填写` | `1` | `待填写` | `待填写` | `runs/logs/kd_baselines_npu/phaseM2_scalar_temperature/triage_20k/cwd_tout3_kdtemp0p6_20k_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt` | `待填写` | `待填写` | `未启动` |

允许的状态值：`未启动`、`运行中`、`已完成`、`失败`、`已停止`、`配置无效待重跑`。

### 启动前快照（待填写）

| 项目 | 值 |
|---|---|
| Git commit | `待填写` |
| `git status --short` 摘要 | `待填写` |
| 启动脚本 | `待填写` |
| Python / torch / torch_npu 版本 | `待填写` |
| `npu-smi info` 摘要 | `待填写` |
| 数据列表 checksum | `待填写` |
| Teacher 权重 checksum | `待填写` |
| Student 初始化权重 checksum | `待填写` |

## 7. 结果回填表

| 变体 | Final mIoU（主） | Best mIoU | Best iter | Last-10 mean | 验证次数 | Runtime | 完整 |
|---|---:|---:|---:|---:|---:|---|---|
| 标量 `T=1.0`（既有） | `0.6420` | `0.6430` | `19200` | `待解析` | `25` | `2:24:59` | 是 |
| 标量 `T=0.5` | `待填写` | `待填写` | `待填写` | `待填写` | `待填写` | `待填写` | `待填写` |
| 标量 `T=0.6` | `待填写` | `待填写` | `待填写` | `待填写` | `待填写` | `待填写` | `待填写` |
| Newton CoVar（既有） | `0.6460` | `0.6460` | `20000` | `待解析` | `25` | `2:31:14` | 是 |

### 预注册比较（待填写）

| 比较 | Final delta | Best delta | Last-10 mean delta | 结论 |
|---|---:|---:|---:|---|
| `T=0.5 - T=1.0` | `待填写` | `待填写` | `待填写` | `待填写` |
| `T=0.6 - T=1.0` | `待填写` | `待填写` | `待填写` | `待填写` |
| `CoVar - T=0.5` | `待填写` | `待填写` | `待填写` | `待填写` |
| `CoVar - T=0.6` | `待填写` | `待填写` | `待填写` | `待填写` |
| `CoVar - S*` | `待填写` | `待填写` | `待填写` | `待填写` |

- 选中的最强匹配标量 `S*`：`待填写`
- 命中的 §5 决策规则：`待填写`
- 下一步动作：`待填写`

## 8. 监控与同步清单

- [ ] 启动前完成配置/权重/数据 checksum 和 Git 快照。
- [ ] 两个运行启动后立即回填命令、设备、PID、stdout/stderr、启动时间和状态。
- [ ] 运行中检查进程、NPU 利用率、最新 iteration、loss 是否有限、验证是否按 `800` iterations 出现。
- [ ] 运行结束后核验 `20000/20000`、总训练时间、最后验证、checkpoint 和退出码。
- [ ] 用同一个解析器计算 final、best、best iter、last-10 mean 和所有 delta。
- [ ] 把结果、异常和决策回填本页；不得只在终端或聊天中留记录。
- [ ] 更新总实验记录 `reports/2026-07-11_AAAI_paper_experiment_record.md` 的 Phase L/M/M2 状态。
- [ ] 检查 `git status`，同步新增脚本、报告与必要的可复现元数据；训练 checkpoint 和大日志按项目既有策略保存，不误提交大文件。

## 9. 偏差与变更记录

| 时间 | 变更/异常 | 原因 | 对可比性的影响 | 处理 |
|---|---|---|---|---|
| `待填写` | `无` | `—` | `—` | `—` |

若发生 OOM、NPU 故障、数据读取错误、日志截断、代码热修改或重启，必须逐项记录。任何改变 seed、训练预算、batch size、数据、权重、损失权重、验证频率或温度定义的操作都视为配置偏差，不能与既有 Phase M 直接合并比较。
