# CoVar Match P0–P3 实验执行总报告

- 基线代码提交：f102a71
- 数据/架构：Pascal VOC；DeepLabV3-ResNet101 teacher；DeepLabV3-MobileNetV3-Small student。
- 正式训练仅发生在 P1；P0、P2、P3 均为冻结模型或既有检查点上的离线诊断。
- 四阶段执行门禁：全部通过。

## 1. 主线结论

| 主张 | 证据 | 状态 |
|---|---|---|
| r 理论与实现一致 | 一/二阶闭式导数对 autograd 与有限差分均通过；VOC val 全量数值有限 | 通过 |
| T 能改变教师输出复杂度 | mean r 从 0.103372 变到 0.353386 | 支持 |
| 最低复杂度不一定最好 | 最低 r 温度 T=0.50；最佳 mIoU 温度 T=1.50 | 支持 |
| teachability 依赖学生状态 | early→late 区域 oracle 改变 42.9973% | 支持 |
| CoVar gap 预测 teachability | overall top-1=12.1258%，mean Spearman=-0.089139 | 不支持 |
| 状态匹配带来最终蒸馏收益 | P0–P3 没有训练自适应匹配策略 | 尚未验证 |

## 2. P0：理论—实现一致性门禁

- VOC val：1449 张；有效像素 244886857。
- 闭式一阶导 vs autograd max abs：1.465e-14。
- 闭式二阶导 vs autograd max abs：1.181e-13。
- 采样像素至少一次 r 下降：4.8934%；轨迹转折：4.7587%。

## 3. P1：全局 teacher-only 温度扫描

- 20k iterations，global batch 16，双 GPU，seed 1234；学生温度固定 1，无 T²，所有其它 KD 分支关闭。

| T | mean r | mIoU (%) | pixAcc (%) |
|---:|---:|---:|---:|
| 0.50 | 0.103372 | 59.499210 | 89.774579 |
| 0.75 | 0.155276 | 60.375696 | 89.903975 |
| 1.00 | 0.206608 | 59.384930 | 89.687377 |
| 1.50 | 0.293892 | 60.861409 | 90.041566 |
| 2.00 | 0.353386 | 60.368663 | 90.048164 |

- 最佳温度相对最低复杂度温度的 mIoU 差：+1.362199 个百分点。
- 该结果只支持固定全局 teacher-target 温度会改变短程蒸馏结果；不等同于区域自适应策略收益。

## 4. P2：区域 oracle 与学生状态

- 固定抽样 VOC val 200 张；候选缓存 166950 行。
- oracle 为 8×8 原生 logit 区域中，归一化 teacher-only KD 梯度一步更新后的监督 CE 最大降幅。

| 状态 | iteration | 区域数 | mean oracle gain | oracle gain>0 |
|---|---:|---:|---:|---:|
| early | 4000 | 9275 | 2.066373e-02 | 97.6388% |
| middle | 12000 | 9275 | 1.400226e-02 | 94.3720% |
| late | 20000 | 9275 | 1.140068e-02 | 93.1968% |

- early→late exact agreement：57.0027%；adjacent agreement：83.9569%。
- 这是标签可见的一步局部 oracle，只是机制诊断。

## 5. P3：gap 对 teachability 的预测

| 分数 | overall top-1 | adjacent | mean regret | mean Spearman |
|---|---:|---:|---:|---:|
| teacher_min_r | 58.4151% | 76.6110% | 3.067790e-03 | 0.651139 |
| scalar_r_gap | 11.9173% | 22.6235% | 2.626861e-03 | -0.199782 |
| vector_covar_gap | 12.1258% | 24.4816% | 2.399578e-03 | -0.089139 |
| teacher_student_kl | 14.5481% | 36.3702% | 2.110622e-03 | 0.206509 |

- overall 最强 top-1：teacher_min_r；最低 mean regret：teacher_student_kl。
- r_c/r_v 缩放来自同一无标签分析缓存，没有独立校准集；P3 结果属于探索性机制证据。

## 6. 测试验证

- P0–P3 聚焦测试：17 passed。
- 完整测试套件：132 passed，1 failed。
- 唯一失败原因：缺少历史冻结 artifact runs/diagnostics/phaseO_o11/voc_train_rtc_confidence_cdf.pt；项目内无副本。
- 远端系统 pytest 的自动插件 anyio 与 pytest 版本不兼容，测试使用 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1；未修改依赖。

## 7. 可复现入口与产物

- P0：scripts/diagnostics/covar_metric_theory_audit.py
- P1：scripts/experiments/covar_match/run_p1_teacher_only_temperature.sh
- P1 汇总：scripts/diagnostics/summarize_p1_teacher_only_temperature.py
- P2：scripts/diagnostics/region_teachability_oracle.py
- P3：scripts/diagnostics/covar_gap_teachability.py
- 分阶段详细报告：reports/covar_match/P0_metric_theory_audit.md、P1_teacher_only_temperature_scan.md、P2_region_teachability.md、P3_covar_gap_teachability.md。

## 8. 结论边界

- 本轮完成了从 r 一致性、温度效应、全局蒸馏结果、状态依赖 oracle 到 gap 预测的 P0–P3 链条。
- 若要闭合“教师复杂度与学生能力匹配 → 最终蒸馏收益”，下一步仍需把不使用标签的匹配规则放回训练，并与最强全局 T 对照；本轮没有执行该训练。
- 所有支持/不支持判断均由实际结果条件生成，负结果和混合结果不改写为正结论。
