# P10 启动记录（历史快照）

记录时间：2026-09-27T00:43:04.144493+00:00。
控制器状态：WAITING_GPU；已完成正式实验 0/33。

- 服务器：lyf_H200_141G / H20d；项目目录 /home/lyf/research_2026/covar_kd_2026/CoVar-KD-pair2。
- 复用 P9 Python 3.10.20、PyTorch 2.0.1+cu118 环境。原 P9 锁定的训练源码、权重及运行时版本全部核验一致。
- tmux：covar-p10-h20；控制器 PID：1518399；GPU候选为6、7。
- CPU测试：58 passed，0 failed；两个 warning 来自旧双卡 validation 回归测试。
- 正式训练前的七条20-step GPU smoke 尚未开始；两张候选 GPU 均有其它任务驻留。控制器已运行并等待空闲，不抢占或终止已有任务。
- 基础实验24条，加最小2×2中三个新格子×三seed共9条，总计33条fresh80k。
- 运行器按阶段产生报告并逐条审计检查点；最终报告和图表完成后下载、核验并推送指定分支。
- 本记录是启动时快照。实时状态以远程 runs/covar_match/P10_h20/runtime/state.json 为准；启动记录不代表已完成训练。

## 目录迁移

2026-09-27 已迁移到 /data/lyf/common/covar_kd/CoVar-KD-pair2，环境为 /data/lyf/common/covar_kd/env/bin/python。上方启动记录为迁移前历史快照；当前路径、全量哈希核验和队列恢复信息见 [migration.md](migration.md)。
