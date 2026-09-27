# P10：同协议基础实验与最小损失对照

登记日期：2026-09-27。固定复用 P9_h20 的环境和训练源码；所有新实验 fresh 80k，单 GPU、batch=16、crop=512、workers=4、SGD lr=0.02/momentum=0.9/weight decay=0.0001、原 poly schedule，20k/40k/60k/80k 验证，80k final mIoU 为主终点。固定 seed 1234、2025、3407。教师为同一 DeepLabV3-R101；学生为 MobileNetV3-Small 或 Large，使用各自 ImageNet backbone 和同编号 seed 的新分割头。

## 登记矩阵与执行顺序

| 阶段 | 新增实验 | 数量 |
|---|---|---:|
| 1 | Small、Large 各 CE-only × 三 seed | 6 |
| 1 | Small teacher-only T∈{0.25,0.5,1,1.5,2} × 三 seed | 15 |
| 2 | Large teacher-only T=4 × 三 seed | 3 |
| 3 | Large T=2 的温度/尺度对照，三个新增格子 × 三 seed | 9 |
| 合计 | 新增独立训练 | **33** |

复用 P9_h20 已完成并审计通过的 Large 五点 × 三 seed，不重跑、不改写、不合并旧 H100 结果。P7 双卡和 P8 H100 CE 仅保留为历史记录，不用于本轮受控增益。

## 最小 2×2 对照

所有格子的目标为 CE + c·mean_valid KL(softmax(z_t/2) || softmax(z_s/T_s))；CE 和有效像素集合一致。

| student T_s | 总 KD 系数 c=1 | 总 KD 系数 c=4 |
|---|---|---|
| 1 | 复用 P9 teacher-only T=2 | 新增三 seed，teacher_only、lambda_kd=4 |
| 2 | 新增三 seed，masked、lambda_kd=1 | 新增三 seed，masked、lambda_kd=4 |

masked 分支显式设置 covar_kd_temp_power=0，使用总系数表达尺度，避免双重乘 T²。T_s=2、c=4 与 T=2 的标准共享温度 T² KD 等价。固定 T=2 是基于已观察 P9 结果选择的代表设置，属于局部敏感性实验，不是未观察温度上的确认实验，不足以声称标准共享温度拥有同样的完整响应曲线。未做梯度范数匹配。

## 报告与边界

- 在共同五点网格报告两种容量的各 seed 结果、均值、样本 SD、KD−CE 配对差值、温度响应交互、赢家和 δ=0.2 pp 均值近优集合。均值差、seed 赢家和集合均为描述性结果，不据三 seed 声称统计显著性、等效性或总体非可识别性。
- 单独报告 Large T=4−T=2 和 T=4−CE；若 T=4 进入扩展六点网格的 δ 近优集合，标记上边界未闭合。不自动补 T=8、不补 0.75/1.25；原 P9 stage2_decision.json 保持原样。T=4 不进入近优集合也不证明连续空间的最优点。
- 2×2 对照报告两种温度模式内的系数效应、相同系数下的模式效应和配对交互。
- 留一 seed 选择损失仅用于探索性稳定性诊断；仍使用同一 VOC validation，不解释为独立测试泛化。跨学生同编号 seed 不声称随机流逐位相同。
- 自动输出 Markdown、JSON、逐次验证 CSV 和两张 PNG/PDF 图；完成后再进行图表视觉复核，下载到本机并核验 SHA-256，提交指定 GitHub 分支。

## 执行与完整性

入口为 `scripts/experiments/covar_match/run_p10_h20.py`。`--prepare --gpus 6 7` 登记环境、源码、数据列表和权重哈希，不占用 GPU。默认 GPU 6、7；仅在显存低于 1024 MiB、利用率不超过 10% 且连续空闲至少 30 秒后启动，每张 GPU 同时最多一个训练；不会停止或挪动已有任务。

正式训练前执行七条 20-step smoke，覆盖两个 CE、Small KD、Large T=4 和三个损失对照。所有 smoke 通过才进入阶段 1。每个阶段全部完成后再进入下一阶段。训练源码和依赖版本必须与 P9 登记一致。

每条新 run 检查日志参数、完整迭代、四个验证点、有限 loss 和指标；CPU 加载四份里程碑和 latest 状态，核验配置、optimizer、RNG 和有限模型张量。失败时保存原始输出并停止追加任务；不自动覆盖或重跑部分轨迹。继续前须检查失败原因和残留 worker，使用 `--continue` 仅跳过已审计完整实验。

运行目录为 `runs/covar_match/P10_h20`，报告目录为 `reports/covar_match/P10_h20`。本轮不引入新自适应策略、不新增数据集，也不扩展其它温度网格。
