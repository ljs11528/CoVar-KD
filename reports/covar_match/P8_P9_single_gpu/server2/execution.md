# P9 第二模型对：新服务器执行记录

状态：截至 2026-09-12 20:00（北京时间），第一阶段已完成 1/15 条 80k run。当前 T=0.5、seed 1234 已到 64,420/80,000，后台控制进程 PID 661 正常运行。正式队列于 2026-09-12 10:48 启动。

## 实验设计

保持 DeepLabV3-ResNet101 teacher，student 使用 DeepLabV3-MobileNetV3-Large；原 P7 student 为 MobileNetV3-Small。

| 项目 | 固定设置 |
|---|---|
| 第一阶段 | T={0.25, 0.5, 1.0, 1.5, 2.0} × seeds {1234, 2025, 3407}，共 15 条 |
| 初始化 / 长度 | 每条 ImageNet backbone + 同 seed 新分割头；fresh 80k |
| 优化器 | SGD，lr=0.02，momentum=0.9，weight decay=0.0001；原 poly schedule |
| 数据 | VOC Aug，global batch=16，crop=512×512，workers=4；原随机缩放、裁剪和镜像 |
| 蒸馏 | CE+KD，lambda_kd=1；teacher-only T，student T=1，无 T² 补偿，其余辅助权重为 0 |
| 验证和状态保存 | 20k / 40k / 60k / 80k；主终点为 80k final mIoU |
| 第二阶段 | 仅满足预定规则时，增加 T={0.75, 1.25} × 三 seed，共 6 条 |
| 近优阈值 | δ=0.2 个百分点，沿用启动前的设置 |

补点规则：第一阶段样本均值赢家为 T=1.0；或赢家为 T=0.5/1.5，且 {0.5,1.0,1.5} 中至少两个点进入 δ-近优集合。第一阶段表格、原始轨迹和 gate 决策在补点前单独存档。本次第二模型对共 15 条完整基础 run，最多 21 条；先前完成的三条 CE 单独保留。

## 新服务器和中断产物

新主机为 szu-a21144257891921920377453，项目仍在 /share/home/tm1156348881820000/a1156346300/covar-kd。单张 H100 80GB、Python 3.10.14、PyTorch 2.2.2、CUDA 12.1、cuDNN 8902、NumPy 1.26.4、OpenCV 4.7.0 均与原单卡记录一致。

旧实例的首条 Large KD（T=0.25、seed 1234）日志停在 22,060 步，并保留 20k 里程碑状态。原产物留在 runs/covar_match/P9_pair2_temperature_response_single_gpu；该中断尝试不计入性能汇总。本轮所有 P9 条件均从初始权重开始，使用独立输出目录，保持各温度采用一致的 fresh 80k 协议。

训练器、模型、数据增强、权重和原队列模块保持原有 SHA-256；新入口复用原训练命令与报告函数，独立管理 P9 输出、运行状态和条件补点。两个入口使用同一队列锁，遇到已有不完整输出时保留产物并停止该队列。

## 已完成核验

39 项相关测试通过，包括训练命令一致性、锁定文件变化拒绝、运行环境变化拒绝、中断产物保留，以及 15/21 条条件队列与第一阶段存档。

独立 20 步 Large KD 检查以退出码 0 完成，损失为有限值，第 20 步 KD loss=1.7226。短程检查仅验证执行，不作为性能结果。三条 CE 的完整性和原始文件哈希也已在启动前检查。

证据见 [validation_checks.json](validation_checks.json) 和 [protocol.json](protocol.json)。

## 结果和解释范围

本次产物复核结果如下；mIoU 单位为 %，未完成的 80k 结果记为“待完成”。

| T | seed | 20k | 40k | 60k | 80k | 状态 |
|---|---|---:|---:|---:|---:|---|
| 0.25 | 1234 | 58.273399 | 63.536787 | 67.013323 | 68.958694 | 完整 80k |
| 0.5 | 1234 | 60.474288 | 62.677807 | 65.160006 | 待完成 | 训练中（64,420 步快照） |

首条 run 的四份里程碑状态可在 CPU 上正常加载，内部步数与文件名一致，模型参数均为有限值，并包含优化器和随机数状态。当前 run 已保存的三份里程碑也通过上述检查。三条 CE 完整性、CE 报告与日志哈希，以及十个锁定源码/权重哈希复核通过。证据见 [artifact_audit.json](artifact_audit.json)。

当前只有一条完整 P9 run，温度排名、三 seed 近优集合和补点决策尚未形成。

已完成的 [CE-only 基线](../P8_ce_only.md) 80k 均值为 61.845306%，样本标准差为 0.188929 个百分点。其三 seed 四个验证点保留在上一阶段报告，本轮不重跑 CE。

P9 完整结果将包含每个 seed 的四个验证点、80k 均值与样本标准差、逐 seed 温度赢家、δ-近优集合及补点决策。在结果形成前，不判断 A/B/C，也不预设最优温度存在或不存在。

P7 使用双卡 DDP，本次 P9 使用单卡，包含 head SyncBN→BN、backbone 本地 batch 8→16、采样随机流和验证归约差异；因此 P7/P9 差异同时包含容量和执行协议变化。同编号 seed 不能单独消除这些混杂。P9 内部温度比较使用一致的单卡协议。

固定 teacher 时，同一 T 的 CoVar 坐标相同。报告比较两个模型对的近优温度所对应的 CoVar 区域；坐标重合本身不证明温度选择规则可以迁移。分析范围为 VOC dense prediction、CoVar complexity 与温度可识别性。[分类 KD 温度交互研究](https://arxiv.org/abs/2603.02430) 为相关背景。

## 产物位置

- 训练输出：runs/covar_match/P9_pair2_temperature_response_single_gpu_server2
- 控制状态：runs/covar_match/P9_single_gpu_server2/runtime
- 报告：reports/covar_match/P8_P9_single_gpu/server2
- 本机下载：/Users/ljs/research_2026/covar_kd_2026/remote/downloads/P8_P9_single_gpu/server2

正式队列入口：

    .venv-p8-p9/bin/python -B -u scripts/experiments/covar_match/continue_p9_single_gpu.py

已运行的队列应通过上述状态和日志查看；再次运行入口会因已有状态或队列锁拒绝覆盖。
