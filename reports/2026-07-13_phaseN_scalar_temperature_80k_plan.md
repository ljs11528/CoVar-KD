# Phase N：标量温度 80k 成对确认实验预注册与运行记录

- 文档创建时间：`2026-07-13`（Asia/Shanghai）
- 启动前状态：`ready（未启动）`
- 实验目的：确认 Phase M2 中标量 `T=0.6` 相对 `T=1.0` 的 20k final mIoU 优势，能否在论文主预算 80k 下保持。
- 本阶段不测试像素级空间自适应，也不包含 CoVar；唯一受控变量是标量 logit KD temperature。
- 本页在启动训练前冻结主次指标、配置、路径、完整性要求和多种子晋级规则。若启动后必须修改，须先在“偏差与中断记录”中登记时间、原因和影响。

## 1. 前置证据与实验问题

Phase M2 在相同 CWD、`Tout=3.0`、seed `1234`、20k 配方下得到：

| 标量温度 | Best mIoU | Final mIoU | Last-10 mean |
|---:|---:|---:|---:|
| `T=1.0` | `0.643000` | `0.642000` | `0.626800` |
| `T=0.6` | `0.653381` | `0.653381` | `0.624513` |

`T=0.6 - T=1.0` 的 20k delta 为 best `+0.010381`、final `+0.011381`、last-10 mean `-0.002287`。主指标与末段均值方向冲突，因此 Phase N 不能仅用 best 或单个 20k 终点下结论；必须按预注册的 80k final 主指标及末段稳定性共同判断。

主要问题：在其他条件完全一致时，`T=0.6` 的 iteration-80000 final mIoU 是否高于 `T=1.0`，并且其最后 10 次验证均值是否支持同方向的长训练收益？

## 2. 冻结运行矩阵

| Run ID | 机制 | `--kd-temperature` | `--teacher-output-temp` | Seed | Iterations | 设备 | 启动前状态 |
|---|---|---:|---:|---:|---:|---:|---|
| `cwd_tout3_kdtemp1p0_80k_seed1234` | 固定标量 KD 温度 | `1.0` | `3.0` | `1234` | `80000` | NPU `0` | `ready` |
| `cwd_tout3_kdtemp0p6_80k_seed1234` | 固定标量 KD 温度 | `0.6` | `3.0` | `1234` | `80000` | NPU `1` | `ready` |

启动器会先在相同设备上各运行 20-iteration、`--skip-val` smoke；只有两路 smoke 都完整退出，才允许启动 80k pair。两路主实验同时启动，避免把时间段或设备负载差异混入温度比较。

## 3. 公共配置

| 项目 | 冻结值 |
|---|---|
| 设备后端 | Ascend NPU；每个变体单卡、`NPROC_PER_NODE=1` |
| Python | `/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python` |
| 数据集 | Pascal VOC / VOCAug；21 类 |
| 数据目录 | `/home/ma-user/work/ljs/dataset/VOCAug/` |
| Teacher | DeepLabV3 + ResNet-101 |
| Student | DeepLabV3 MobileNet SS-Seg + MobileNetV3-Small |
| Teacher 权重 | `/home/ma-user/work/ljs/data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth` |
| Student 初始化 | `/home/ma-user/work/ljs/data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth` |
| Crop / batch / workers | `512x512` / `16` / `8` |
| Optimizer 公共参数 | learning rate `0.02`、momentum `0.9`、weight decay `1e-4` |
| 训练预算 | `80000` iterations |
| 日志 / 保存 / 验证间隔 | `20 / 800 / 800` iterations |
| 随机种子 | `1234` |
| CWD 损失权重 | `lambda_kd=1.0`、`lambda_d=0.1`、`lambda_adv=0.001`、`lambda_cwd_fea=50.0`、`lambda_cwd_logit=3.0` |
| Teacher output temperature | `3.0` |
| CoVar | 禁用；不得传入 `--use-covar` |
| 唯一受控差异 | `--kd-temperature 1.0` vs `--kd-temperature 0.6` |
| 验证 | 启用；预期 100 次定期验证，包含 iteration `80000` |

不得根据运行中指标调整学习率、预算、验证频率、损失权重、温度、seed、数据或权重。任一公共配置漂移都使 pair 失去直接可比性，须停止并从统一快照重跑受影响的比较。

## 4. 主次指标与解析口径

### 4.1 主指标

- `final mIoU`：iteration `80000` 的验证 mIoU。
- 主比较：`Delta_final = final(T=0.6) - final(T=1.0)`。

### 4.2 次指标

- `best mIoU` 及对应 iteration；
- 最后 10 次有效验证 mIoU 的算术平均值，预期对应 iterations `72800, 73600, ..., 80000`；
- `Delta_best` 与 `Delta_last10`，方向均定义为 `T=0.6 - T=1.0`；
- 运行时间与 sec/iteration；
- 完整性和健康状态。

主指标与次指标方向冲突时，必须同时报告，不得用 best 替代 final 宣称胜出。Phase N 仅有一个 seed，不计算显著性、置信区间或跨 seed 标准差；后续三种子阶段统一报告 seed-wise paired delta、均值、样本标准差（`ddof=1`）及置信区间。

统一使用 `scripts/experiments/kd_baselines_npu/summarize_phaseN_scalar_temperature_80k.py` 解析两路日志。原始日志能提供多少精度就报告多少精度，不通过格式化虚构有效位。

## 5. 完整性、checkpoint 与断点规则

每路必须同时满足：

1. 最新训练 session 出现 `Iters: 80000/80000`；
2. 有可解析的 iteration-80000 验证和 `Total training time:`；
3. 无 NaN/Inf、traceback 或异常退出；
4. 预期 100 次验证均可解析；
5. final/best 模型 checkpoint 和 latest/best training state 均存在。

`training_state` 可恢复 model/module、optimizer、iteration 和主进程 RNG 状态，但不保存 DataLoader worker/prefetch 游标。因此断点续训不是 bit-exact。发生任何中断都必须记录中断时间、iteration、设备、原因、恢复 checkpoint 和恢复时间，并谨慎解释与无中断对照的微小差异：

- 若仅一方中断，不能把恢复后的细小 delta 当成严格配对因果证据；
- 若恢复结果位于晋级阈值附近、主次指标冲突或会改变晋级决定，应从统一初始条件重跑受影响 pair；
- 不得隐藏 appended/restarted session；汇总器只解析日志中的最新 session，报告必须同时说明恢复历史。

## 6. 多种子晋级规则

只有两路 80k 均完整且配置一致时才应用以下规则：

1. 若 `Delta_final >= +0.002` 且 `Delta_last10 >= 0`，判为“长预算低温收益成立，值得晋级”：保留 seed `1234`，继续成对运行 seeds `2025`、`3407`，形成 80k 三种子证据。
2. 若 `Delta_final <= -0.002`，判为“低温在长预算下更差”：停止自动多种子晋级，保留 Phase M2 作为短预算现象。
3. 若 `-0.001 <= Delta_final <= +0.001`，判为“80k 实质持平”：不主张 `T=0.6` 优势；先补 seed `2025` 的 80k 成对复核，不直接运行 seed `3407`。
4. 若 delta 落在上述未覆盖的过渡区间，或 `Delta_final >= +0.002` 但 `Delta_last10 < 0`，判为“不确定”：先补 seed `2025` 的 80k 成对复核，再决定是否运行 seed `3407`。
5. 任一运行不完整、配置漂移或关键日志不可解析时，不应用性能规则；先修复并完整重跑受影响 pair。

阈值用于资源分配，不代表统计显著性。补充 seed 应交换两种温度的 NPU 分配，以降低固定设备混杂；即使晋级，论文结论也必须以三种子 paired delta 为准。

## 7. 命令、路径、PID 与状态占位

- 启动命令：`bash scripts/experiments/kd_baselines_npu/launch_phaseN_scalar_temperature_80k.sh`
- 编排脚本：`scripts/experiments/kd_baselines_npu/run_phaseN_scalar_temperature_80k.sh`
- 变体脚本：`scripts/experiments/kd_baselines_npu/run_phaseN_cwd_scalar_variant.sh`
- 监控脚本：`scripts/experiments/kd_baselines_npu/monitor_phaseN_scalar_temperature_80k.sh`
- 汇总脚本：`scripts/experiments/kd_baselines_npu/summarize_phaseN_scalar_temperature_80k.py`
- Checkpoint 根目录：`data/winycg/checkpoints/kd_baselines_npu/phaseN_scalar_temperature_80k/`
- 严格完成态 helper：`scripts/experiments/kd_baselines_npu/phaseN_scalar_temperature_common.sh`
- 指标日志根目录：`runs/logs/kd_baselines_npu/phaseN_scalar_temperature_80k/`
- Controller PID：`data/winycg/checkpoints/kd_baselines_npu/phaseN_scalar_temperature_80k/phaseN_scalar_temperature_80k.pid`
- Controller stdout/stderr：`data/winycg/checkpoints/kd_baselines_npu/phaseN_scalar_temperature_80k/phaseN_scalar_temperature_80k.nohup.log`

| Run ID | NPU | Worker PID / 文件 | stdout/stderr | 指标日志 | Checkpoint 目录 | 启动时间 | 结束时间 | 状态 |
|---|---:|---|---|---|---|---|---|---|
| `cwd_tout3_kdtemp1p0_80k_seed1234` | `0` | `待启动`; `data/winycg/checkpoints/kd_baselines_npu/phaseN_scalar_temperature_80k/runtime/main_80k/kdtemp1p0.pid` | `data/winycg/checkpoints/kd_baselines_npu/phaseN_scalar_temperature_80k/runtime/main_80k/kdtemp1p0.nohup.log` | `runs/logs/kd_baselines_npu/phaseN_scalar_temperature_80k/main_80k/cwd_tout3_kdtemp1p0_80k_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt` | `data/winycg/checkpoints/kd_baselines_npu/phaseN_scalar_temperature_80k/main_80k/cwd_tout3_kdtemp1p0_80k_seed1234/` | `待启动` | `待填写` | `ready` |
| `cwd_tout3_kdtemp0p6_80k_seed1234` | `1` | `待启动`; `data/winycg/checkpoints/kd_baselines_npu/phaseN_scalar_temperature_80k/runtime/main_80k/kdtemp0p6.pid` | `data/winycg/checkpoints/kd_baselines_npu/phaseN_scalar_temperature_80k/runtime/main_80k/kdtemp0p6.nohup.log` | `runs/logs/kd_baselines_npu/phaseN_scalar_temperature_80k/main_80k/cwd_tout3_kdtemp0p6_80k_seed1234/deeplabv3_mobilenet_ssseg_resnet101_mobilenetv3_small_log.txt` | `data/winycg/checkpoints/kd_baselines_npu/phaseN_scalar_temperature_80k/main_80k/cwd_tout3_kdtemp0p6_80k_seed1234/` | `待启动` | `待填写` | `ready` |

允许的状态：`ready`、`smoke_running`、`running`、`complete`、`failed`、`stopped`、`invalid_requires_rerun`。

## 8. 启动前快照（`2026-07-13T08:07:46+08:00`）

| 项目 | 值 |
|---|---|
| Git commit | `5cfc1e8eae72d281a1604d560a055b33f405a432`（Phase N 预注册与脚本提交） |
| `git status --short` 摘要 | 快照时为空；无已跟踪修改、无未跟踪文件 |
| Python / torch / torch_npu | `3.11.10` / `2.8.0+cpu` / `2.8.0.post2`；解释器已在编排器中冻结为 `/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python` |
| `npu-smi info` 摘要 | 2 x Ascend 910，均 `Health OK`、AICore `0%`；无运行中的 NPU process；可用磁盘约 `140 GB` |
| 数据列表 checksum | `dataset/list/voc/train_aug.txt`: `d1326bd532648d73bc4b1bd275434eba81930982dc7401a65a8cecb26c028e24`; `dataset/list/voc/val.txt`: `cdc1326d12f69ce5153aa5da04a4d8783e146868d42d19f82f44e74b97ac907d` |
| Teacher 权重 checksum | `ac49b2c7720b21d565072e974e4404fcb009ba106288d95d1a4bb25f09c3fe58` |
| Student 初始化权重 checksum | `47085aa164b2977003221458a2a5fdf5f46f434f5539b164dc71969b4dd4cd75`；fresh ImageNet init |
| 启动器/检查器验证 | Phase N shell `bash -n`: PASS；汇总器与 `train_kd.py` `py_compile`: PASS；strict helper 正例、错温度、数字前缀、旧完成+新 partial、NaN、checkpoint 缺失反例：PASS；当前环境未安装 `shellcheck` |

## 9. 结果占位

| 变体 | Final mIoU（主） | Best mIoU | Best iter | Last-10 mean | 验证次数 | Runtime | 完整 |
|---|---:|---:|---:|---:|---:|---|---|
| 标量 `T=1.0` | `待填写` | `待填写` | `待填写` | `待填写` | `待填写` | `待填写` | `待填写` |
| 标量 `T=0.6` | `待填写` | `待填写` | `待填写` | `待填写` | `待填写` | `待填写` | `待填写` |

| 比较（`T=0.6 - T=1.0`） | Delta | 方向/解释 |
|---|---:|---|
| Final mIoU | `待填写` | `待填写` |
| Best mIoU | `待填写` | `待填写` |
| Last-10 mean | `待填写` | `待填写` |

- 命中的 §6 晋级规则：`待填写`
- 下一步动作：`待填写`

## 10. 执行与同步清单

- [x] Phase M2 结果和因果警告已完成记录。
- [x] Phase N 主次指标、配置、产物路径和多种子晋级规则已在启动前冻结。
- [x] 启动前回填 Git、环境、NPU、数据和权重快照。
- [ ] 双卡 smoke 均通过后，核验 80k pair 的实际 PID、设备、命令、日志和启动时间。
- [ ] 运行中定期同步 iteration、最近验证、sec/iter、ETA 与健康状态。
- [ ] 结束后核验 `80000/80000`、100 次验证、总训练时间、checkpoint/training state 和退出状态。
- [ ] 使用冻结汇总器生成 `reports/2026-07-13_phaseN_scalar_temperature_80k.md`，回填本页和总实验记录。
- [ ] 检查 Git diff/status，只同步代码、脚本、报告和必要的复现元数据，不误提交 checkpoint 或大日志。

## 11. 偏差与中断记录

| 时间 | 变更/中断 | 原因 | 对可比性的影响 | 处理 |
|---|---|---|---|---|
| `2026-07-13T08:07:46+08:00` | 启动前无配置偏差或中断 | 不适用 | 无 | 预注册、代码、环境、数据与权重快照均已冻结 |
