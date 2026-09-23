# 第二教师—学生模型对：H20 实验批次

2026-09-16 21:11（北京时间）启动，2026-09-22 11:52 完成：15/15 条均为完整 80k，预设规则判定无需补点。最终均值最高点为 T=2.0（69.305603%），δ=0.2 pp 近优集合为 {1.5, 2.0}。详见[最终结果报告](final_report.md)和[完整性核验](completion_audit.json)。下文保留启动协议及执行记录。

Teacher 为 DeepLabV3-ResNet101，student 为 DeepLabV3-MobileNetV3-Large；原 P7 student 为 MobileNetV3-Small。主问题是该模型对在 VOC dense prediction 上的温度响应、近优集合和逐 seed 排名是否稳定。

## 固定设置

| 项目 | 设置 |
|---|---|
| 第一阶段 | T={0.25, 0.5, 1.0, 1.5, 2.0} × seeds {1234, 2025, 3407}，15 条 |
| 初始化 | 每条从同一 ImageNet backbone 权重和对应 seed 的新分割头开始 |
| 训练 | 每条 fresh 80,000 步，global batch=16，crop=512×512，workers=4 |
| 优化器 | SGD，lr=0.02，momentum=0.9，weight decay=0.0001；原 poly schedule |
| 数据增强 | 原 VOC 随机缩放、随机裁剪、镜像 |
| 目标 | CE+KD，lambda_kd=1；teacher-only T，student T=1，无 T² 补偿；其余辅助损失为 0 |
| 验证 | 20k、40k、60k、80k，保留每个里程碑状态；主终点为 80k final mIoU |
| 近优阈值 | δ=0.2 个百分点，沿用已有预设 |
| 执行 | 每条单进程、单 GPU；GPU 6 跑 seeds 1234、3407，GPU 7 跑 seed 2025 |

GPU 4、5 在准备期间被其他任务占用，首次资源检查拒绝启动，未产生训练 run。随后在空闲的 GPU 6、7 完成短程检查并启动正式队列。控制器先并行执行 1234、2025 两个 seed，各自按温度顺序运行；两者结束后执行 3407。

只有 15 条均完整且通过既有协议核验后才决定第二阶段：阶段一均值最高点为 T=1.0；或最高点为 T=0.5/1.5 且 {0.5,1.0,1.5} 至少两点进入 δ-近优集合时，补 T={0.75,1.25} × 三 seed。否则结束。第一阶段表格和决定独立归档，最多 21 条正式 run。

## 验证和实际环境

29 项测试通过，覆盖温度/损失定义、单卡验证归约、条件补点、分批调度和已有产物保护。三个 seed 的独立 20 步 KD 检查均正常完成，损失有限。这些检查不进入性能统计。

正式队列的首两条 T=0.25 在启动检查时均已超过 40/80000，损失有限。训练前核对了原协议记录的训练源文件和 teacher/student 权重 SHA-256。

详见 [启动核验](validation_checks.json)、[实际运行协议](protocol.json) 和 [环境版本](environment-freeze.txt)。本批次复用已安装的 Python 3.10.20、PyTorch 2.0.1+cu118 环境，并记录全部实际版本。

旧 H100 入口在本次核查中连续两次连接失败，其最新完成数无法确认。已有下载记录和旧报告仍保留。本批次的全部温度使用 H20 上统一的 fresh 80k 设置，旧 H100 结果不并入均值；三条已经完成的 P8 CE 不重跑。

## 报告口径

完整报告保留每条 run 的四个验证点、80k 均值和样本标准差、逐 seed 温度赢家与 δ-近优集合、均值赢家相对其余温度的配对差值，以及近优温度对应的 CoVar 坐标。阶段一输出 `P9_stage1.{md,json,csv}` 和 `stage2_decision.json`；必要补点完成后更新 `P9_temperature.{md,json,csv}`。

P7 为双卡 DDP，本批次为单卡，存在 head SyncBN→BN、backbone 本地 batch、采样随机流和验证归约差异，且 P7 的完整软件环境未锁定。因此跨模型对差异包含容量与执行协议的共同变化，不能单独归因于 student capacity。三个 seed 的有限网格结果用于描述本实验条件下的可重复性，不能推出“最优蒸馏温度通常不可识别”。固定 teacher 时，同 T 的 CoVar 坐标相同；近优区域重合本身不验证选择规则可迁移。

分类 KD 中温度与 optimizer、teacher 训练等组件的交互已有系统研究：[Frank and Davis, 2026](https://arxiv.org/abs/2603.02430)。本报告重点是 dense prediction 的响应、可识别性和 CoVar complexity，不将温度网格搜索本身作为创新结论。

## 路径和运行

服务器：`ssh lyf_H200_141G`，实际主机 `H20d`，显卡 `NVIDIA H20-3e`。

```bash
cd /home/lyf/research_2026/covar_kd_2026/CoVar-KD-pair2
source ../env/bin/activate
# 本批次已完成，请勿重复启动。
# 本批次的启动命令：
python -B -u scripts/experiments/covar_match/run_p9_h20.py --gpus 6 7
```

控制队列曾运行于 tmux 会话 `covar-p9-h20`，完成后已退出，状态为 COMPLETE。状态与日志为 `runs/covar_match/P9_h20/runtime/`，训练产物为 `runs/covar_match/P9_pair2_temperature_response_h20/`。启动时控制器 PID 为 1895621，仅作历史记录。

本机报告下载目录：`/Users/ljs/research_2026/covar_kd_2026/remote/downloads/P9_h20/`。代码和报告提交到 `codex/p8-p9-single-gpu`。2026-09-23 已完成 15 条训练、60 个验证点和 75 份状态的独立核验，并重算最终统计；所有正式 run 均完整，无中断 run 被纳入统计。
