# P8 / P9 单卡实验执行记录

状态：正式队列已于 2026-09-11 21:05（北京时间）在后台启动，控制进程 PID 2776。当前进入 CE-only seed 1234，尚无本次 80k 最终结果。最终 P8 / P9 报告将直接从已完成的训练日志生成。

## 已有 CE-only 结果

| 原双卡 run | 已完成步数 | 20k mIoU (%) | 40k | 60k | 80k |
|---|---:|---:|---:|---:|---:|
| seed 1234 | 80000 | 56.097996 | 58.335817 | 60.959673 | 63.076729 |
| seed 2025 | 22980 | 54.828501 | — | — | — |
| seed 3407 | 未启动 | — | — | — | — |

记录来自 runs/covar_match/P8_ce_only_baseline。seed 1234 的四个验证点、四份里程碑状态和训练结束记录齐全；2025 不完整，预定 retry1 尚无产物。第二模型对仅有既往双卡单步 smoke，没有正式 80k run。

## 本次协议

2026-09-11 用户确认采用单卡，并记录与 P7 的差异。三条 CE-only 均重新开始；旧双卡 seed 1234 不并入新均值。

| 项目 | 固定设置 |
|---|---|
| 执行 | 一张 H100 80GB，单进程 |
| Seeds / 训练 | 1234、2025、3407；每条 fresh 80k |
| Student 初始化 | 对应 ImageNet backbone + 每个 seed 的新分割头 |
| Teacher | 同一 DeepLabV3-ResNet101 VOC checkpoint |
| Optimizer | SGD，lr=0.02，momentum=0.9，weight decay=0.0001 |
| Schedule | 原 train_kd.py poly schedule |
| Batch / crop / workers | 全局 16 / 512×512 / 4 |
| 增强 | 原 VOC 随机缩放 0.5:0.1:2.0、随机裁剪和镜像 |
| 验证 / 保留状态 | 20k、40k、60k、80k；主终点为 80k final mIoU |
| CE-only | MobileNetV3-Small；L=L_CE；KD 和其余辅助权重均为 0，aux=False |
| 第二模型对 | R101 → MobileNetV3-Large；CE+KD；teacher-only temperature，student T=1，无 T² 补偿 |

环境为项目内 .venv-p8-p9，复用 PyTorch 2.2.2 / CUDA 12.1。依赖清单为 requirements-p8-p9-single-gpu.txt；启动时保存软件版本、GPU 信息及权重、训练源文件的 SHA-256 到 protocol.json。

P7 使用双卡 DDP。单卡把 head 的 SyncBatchNorm 改为 BatchNorm，backbone 每卡 batch 从 8 变为 16；RandomSampler、rank/worker 随机流及验证归约也不同。单卡验证每张图一次，旧双卡 sampler 对奇数大小验证集补齐。原 P7 日志未充分记录软件版本，不能证明软件版本一致。

因此，P7 KD − 本次 CE 是同编号 seed 的跨协议差值，不能单独识别 KD 净收益、负迁移或温度选择的实际收益。P9 内部各温度采用一致的单卡协议；P9 与 P7 的差异同时包含学生容量与执行协议变化。

## 执行顺序与预算

1. 三条 CE-only 80k 完成后，生成 P8 报告。
2. 第二模型对运行 T={0.25, 0.5, 1.0, 1.5, 2.0} × 三 seed，共 15 条。
3. δ=0.2 pp 固定于运行前。阶段 1 样本均值最高点为 T=1.0，或最高点为 T=0.5/1.5 且 {0.5,1.0,1.5} 至少两点进入近优集合时，才新增 T=0.75/1.25 × 三 seed。
4. 基础预算 18 条正式 run，最多 24 条。两条 20-step smoke 单独归档，不计入性能结果。

队列不覆盖不完整输出，也不自动恢复中断的正式 run。单卡验证补充一条 Overall 汇总日志，供四个里程碑统一解析。

启动：

    .venv-p8-p9/bin/python -u scripts/experiments/covar_match/run_p8_p9_single_gpu.py --smoke
    .venv-p8-p9/bin/python -u scripts/experiments/covar_match/run_p8_p9_single_gpu.py

控制状态与日志位于 runs/covar_match/P8_P9_single_gpu/runtime。正式 CE 和 KD 产物分别位于 P8_ce_only_baseline_single_gpu 与 P9_pair2_temperature_response_single_gpu。

## 已完成的工程验证

39 项相关测试通过，包括 CE/KD 温度语义、协议不一致拒收、条件补点，以及单卡和双卡验证汇总日志。两条独立 20 步短程训练均以退出码 0 完成：CE 的 KD loss=0；Large KD 在第 20 步的 KD loss=1.7168，损失均为有限值。

共享盘在两次短程训练的数据加载器退出时出现 Errno 16 临时文件清理告警。告警发生在退出清理阶段，两条训练均有完整结束记录；保留日志，不据此删文件。正式结果仍必须通过 80k、四个验证点及四份状态文件核验。

移除了 losses/dsd.py 中未使用的 turtle 导入，解决无图形库环境无法导入训练器的问题。详细检查证据见 validation_checks.json。

## 报告与证据范围

最终报告保留每个 seed 的四个验证点、80k 均值、样本标准差、各 seed 温度赢家和 δ-近优集合，导出 Markdown、JSON 和 CSV。阶段 1 结果与第二阶段决策单独保存。

不预设第二模型对将出现稳定或不稳定赢家。固定 teacher 时，每个 T 的 CoVar 坐标相同；坐标近优集合重合本身不能证明温度规则可迁移。

分类 KD 中温度与 optimizer、teacher 预训练/微调等组件的交互已有系统研究：[Frank and Davis, 2026](https://arxiv.org/abs/2603.02430)。本实验分析 VOC dense prediction 的温度响应、可识别性和 CoVar 坐标。

本机下载目录：/Users/ljs/research_2026/covar_kd_2026/remote/downloads/P8_P9_single_gpu/。本执行记录不代替未完成的实验结论。
