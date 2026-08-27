# P4a：Task-Aligned Region Temperature 实验报告

## 结论先行

- P4a 相对固定 T=1.5 的 final mIoU 变化为 -0.343633 个百分点，数值上低于强基线。
- 当前比较是同一协议下各一个 seed=1234 的 20k run；没有独立的 run-to-run 噪声估计，因此不把纯数值差异写成统计显著差异。
- P3A 的近乎精确 one-step 排序只证明局部代理有效；P4a 检验的是贪心区域选择能否转化为长期参数训练收益，两者不混同。

## 唯一变量与实现门禁

- 数据/模型：Pascal VOC，DeepLabV3-ResNet101 → DeepLabV3-MobileNetV3-Small。
- 两组均为 20k iterations、global batch 16、双 GPU、seed 1234、CE + 1.0×KL、student T=1、无 T²、teacher output T=1，其他 KD 分支关闭。
- A 复用 P1 已完成的固定 T=1.5 run；B 仅把教师 target 温度改为每个原生 logits 网格 8×8 region 的 hard argmax。
- 候选集合固定为 {0.5, 0.75, 1.0, 1.25, 1.5, 2.0}；每区至少 16 个有效像素。稀疏区回退 T=1.5；浮点完全并列时选择离 T=1.5 最近的候选。
- selector 在 no_grad/detach 下运行；没有额外模型 forward、真实 one-step update、二阶梯度、margin gate 或可学习模块。

## 主结果

| 方法 | best mIoU (%) | final mIoU (%) | final pixAcc (%) | 验证点 | 训练时间 |
|---|---:|---:|---:|---:|---:|
| 固定 T=1.5 | 60.861409 | 60.861409 | 90.041566 | 1 | 1:07:05.689746 |
| Task-aligned 8×8 | 60.517776 | 60.517776 | 89.898586 | 1 | 1:07:26.791419 |

- best mIoU 差：-0.343633 pp；final mIoU 差：-0.343633 pp。
- 两个正式协议都只在 20k 做一次 validation，故本报告中的 best=final；没有用更密的验证频率改变 P1 契约。

## 训练损失（每 20 iteration 的同口径日志点）

| 方法 | mean CE | final CE | mean KD | final KD | 日志点 |
|---|---:|---:|---:|---:|---:|
| 固定 T=1.5 | 0.431715 | 0.272800 | 0.356667 | 0.224400 | 1000 |
| Task-aligned 8×8 | 0.443114 | 0.255900 | 0.399602 | 0.235700 | 1000 |

## Selector 诊断

- 全程温度分布（eligible region）：T=0.5: 56.37%, T=0.75: 21.46%, T=1: 7.65%, T=1.25: 3.89%, T=1.5: 4.80%, T=2: 5.83%。
- eligible regions：15,759,237；稀疏回退：297,022/16,056,259 (1.85%)；浮点完全并列：2,837,042 (18.00%)。
- mean margin：0.00278871；P(margin≥1e-4)=35.05%；P(margin≥1%×|selected A|)=20.61%。
- mean [A(selected)−A(T=1.5)]=0.00794521；按 P3A 的 η=0.01 线性刻度，对应预测 one-step gain uplift=0.0000794521。
- 选中教师目标的 mean (r_c, r_v, r)=(0.028090, 0.150179, 0.178268)；r=r_c+r_v 数值一致。

### 学生状态阶段

| 阶段 | 温度分布 | high margin (≥1e-4) | ΔA vs T=1.5 | mean r |
|---|---|---:|---:|---:|
| early 1–4k | T=0.5: 47.71%, T=0.75: 24.75%, T=1: 11.01%, T=1.25: 4.16%, T=1.5: 5.02%, T=2: 7.36% | 35.37% | 0.00625786 | 0.184883 |
| middle 4k–12k | T=0.5: 55.41%, T=0.75: 22.36%, T=1: 7.77%, T=1.25: 3.84%, T=1.5: 4.86%, T=2: 5.77% | 34.98% | 0.00774350 | 0.177792 |
| late 12k–20k | T=0.5: 61.68%, T=0.75: 18.92%, T=1: 5.85%, T=1.25: 3.80%, T=1.5: 4.63%, T=2: 5.13% | 34.96% | 0.00899008 | 0.175440 |

## 如何解释本轮

- P4a 比固定 T=1.5 低 0.343633 pp。按预先决策树，本轮操作性结论是停止 region-wise adaptive temperature；不做仅为“基本相同”结果预留的 margin-aware 回退，也不扩展其它温度设计。

- 单 seed 边界意味着不能宣称该下降具有统计普遍性；但本轮没有产生继续该方向所需的正向证据。
- early→late，T=0.5 占比由 47.71% 升至 61.68%、mean r 由 0.184883 降至 0.175440、ΔA 由 0.00625786 升至 0.00899008；但 final mIoU 仍变化 -0.343633 pp。这是局部一阶对齐/复杂度轨迹与长期蒸馏收益解耦的直接证据。

无论本轮长期结果方向如何，训练内 ΔA 非负仅是 hard argmax 对其自身一阶目标的代数结果，不应被当作 mIoU 改善的保证。

## 论文定位与边界

- SCKD 已从多任务优化和梯度相似性角度做 student-customized KD，并覆盖语义分割；DTKD 已研究基于 teacher–student sharpness 差异的样本级动态温度。因此不能声称“首次 student-aware KD”。
- 当前可检验的新意应限定为：密集预测中的区域级、任务方向驱动温度选择，以及 CoVar complexity 与 teachability 的理论/实验解耦。
- 这是 VOC、单 teacher–student 对、单 seed、20k 的最小实验；不外推到其它数据集、80k 或统计显著性。

来源：[SCKD (ICCV 2021, CVF)](https://openaccess.thecvf.com/content/ICCV2021/html/Zhu_Student_Customized_Knowledge_Distillation_Bridging_the_Gap_Between_Student_and_ICCV_2021_paper.html)；[DTKD (arXiv:2404.12711)](https://arxiv.org/abs/2404.12711)。

## 执行门禁

- P1 基线契约：pass。
- P4a 日志/迭代/验证/候选集合/统计覆盖/有限性：pass。
