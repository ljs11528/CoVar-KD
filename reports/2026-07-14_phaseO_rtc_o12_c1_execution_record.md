# Phase O1.2-C1：neutral 与 unreliable_only 正式 20k 执行协议及启动前审计记录

- 记录日期：2026-07-14
- 阶段：O1.2-C1
- 实验性质：20k 信号筛选，不是最终方法确认实验
- 正式分支：`neutral`、`unreliable_only`
- 启动方式：两支 fresh、并行、各占一张物理 NPU
- 对照目的：只验证“在高风险侧平滑教师目标”是否改善教师错误处理，同时控制整体性能损失
- 前置依据：[O1.2 预算路由计划](./2026-07-13_phaseO_rtc_o12_budgeted_routing_plan.md)与 [O1.2-B smoke 报告](./2026-07-13_phaseO_rtc_o12b_smoke_report.md)
- 启动前审计时间：2026-07-14 08:32:59（Asia/Shanghai）

> 本文在正式启动前封存实验身份、统计口径和停止线。封存后立即通过专用 launcher 启动；实际 PID、UTC 时间、Git commit、argv、NPU 证据、训练日志和 acceptance 以 `runs/runtime/kd_baselines_npu/phaseO_o12_c1/` 内的不可覆盖运行证据为准。训练完成前不填写结果，也不据中途曲线修改门槛。

## 0. 本轮决策

本轮只运行 O1.2-C1 的两个 20k 分支：

1. `neutral`：所有有效像素保持 `T_pixel=1`；
2. `unreliable_only`：仅在冻结 train CDF 判定的高风险侧 `u>0.8` 连续提高教师像素温度，其他位置保持 `T_pixel=1`。

两支都从同一个 ImageNet student init fresh 启动。不得续接 O1.2-B 的 20-step checkpoint，因为 B 的 `max_iterations=20` 对应不同的 polynomial learning-rate 轨迹，续接会破坏 20k 公平比较。

本轮不运行：

- `reliable_only`；
- `full_budgeted`；
- matched scalar；
- within-image shuffle；
- 80k；
- 多 seed；
- Phase O1.2-C2/C3/D。

即使 C1 最终通过，自动控制器也只停止并等待人工审查，不自动进入 C2。

## 1. 研究问题与可解释边界

当前诊断表明教师错误主要富集在 confidence-only 高风险侧，而教师整体错误并不占多数。因此本轮不再尝试大范围强锐化，改为先回答更窄的问题：

> 在总体温度预算接近中性的条件下，只平滑冻结高风险侧，能否减少学生对错误教师标签的模仿、增加学生纠正教师错误的概率，同时不明显伤害 final mIoU 和高风险侧教师正确像素的保留率？

C1 若通过，只能说明高风险平滑分支出现值得继续验证的信号。它不能独立证明：

- 风险空间位置优于 matched scalar；
- 高风险位置本身具有因果作用；
- 可靠侧锐化有效；
- full map 有效；
- 跨数据集或跨 seed 泛化。

这些问题分别留给 C2、C3 和后续另行预注册的确认性实验。

## 2. 冻结训练配置

| 项目 | 冻结值 |
|---|---|
| 数据集 | VOC train_aug；canonical population 10,582 |
| teacher | DeepLabV3-ResNet101 |
| student | DeepLabV3-MobileNetV3-Small |
| crop / batch / workers | 512x512 / 16 / 8 |
| seed | 1234 |
| max iterations | 20,000 |
| log / save / val interval | 20 / 800 / 800 |
| validation | 开启；VOC val 1,449 张 |
| lr / momentum / weight decay | 0.02 / 0.9 / 1e-4 |
| KD mode | `rtc_o12_teacher_target` |
| teacher outer temperature | 3.0 |
| student KD temperature | 1.0 |
| lambda KD / adversarial G / D | 1.0 / 0.001 / 0.1 |
| lambda CWD feature / logit | 50.0 / 3.0；CWD 分支仍固定 T=4 |
| 其余 KD loss | SKD、IFV、FitNet、AT、PSD、CSD 全部为 0 |
| 进程拓扑 | 每支 world size 1，进程内只见一张 NPU，`local_rank=0` |
| 物理卡绑定 | `neutral -> NPU0`；`unreliable_only -> NPU1` |

教师目标为：

~~~text
q_teacher = softmax(raw_teacher_logits / (3.0 * T_pixel))
p_student = softmax(student_logits)
~~~

其中空间温度只改变教师目标，不改变学生 softmax 温度，也不乘额外 `T^gamma`。

`unreliable_only` 使用冻结预算解，高风险侧温度上限为 1.5；正式 train 诊断中的 arithmetic mean 为约 1.025661，harmonic mean 为约 1.021265。`neutral` 精确为 1。该设计的重点是局部处理高风险教师错误，而不是把平均温度大幅降低。

## 3. Fresh 与数据顺序契约

两支命令均不得包含 `--resume` 或 `--skip-val`。共同 student init SHA256 为：

~~~text
47085aa164b2977003221458a2a5fdf5f46f434f5539b164dc71969b4dd4cd75
~~~

20k canonical sample-order 合同覆盖 320,000 个训练索引，固定 SHA256：

~~~text
10e600fd87537bba4329a7a90473bd71931e5fe2c775345af4dc483c3b9f8c5c
~~~

final checker 必须验证 checkpoint v4、iteration 20,000、world size 1、完整 args/O1.2 metadata、数据顺序 SHA 和 `next_canonical_index_offset=320000`。任何缺失、恢复痕迹或配置偏差都使该分支失败。

## 4. 性能指标

每 800 iteration 验证一次，共必须有 25 个完整 block：800、1,600、……、20,000。单 NPU 日志中每个 block 只取最后一条 `Sample: 1449` 的累计 pixAcc/mIoU。

- 主性能指标：iteration 20,000 final mIoU；
- 次指标：25 个 block 的 best mIoU；
- 次指标：固定 iteration 12,800 至 20,000 的最后 10 个 block 的 mIoU 算术均值。

所有值必须为有限的 `[0,1]` 比例。缺失、重复、乱序或不完整 block 均为结构失败；不得改用 best checkpoint 作为机制评估 checkpoint。

## 5. 高风险侧机制指标

final evaluator 按 canonical VOC val 顺序逐图推理，不做随机增强。学生加载 iteration 20,000 的 `training_state_latest.pth` 中 `student`，教师加载冻结正式 checkpoint。

学生和教师 raw logits 必须具有完全相同的 native KD grid，不允许插值 logits。GT 只用 nearest resize 到该 native grid。对每个像素定义：

~~~text
g = nearest-resized ground-truth
s = argmax(raw_student_logits)
t = argmax(raw_teacher_logits)
c = clamp(max(softmax(raw_teacher_logits)), 1e-8, 1-1e-8)
r = -log(c)
u = frozen_train_CDF(r)
V = (g != -1)
U = V and (u > 0.8)
W = U and (t != g)
C = U and (t == g)
~~~

三个指标固定为：

~~~text
student_rescue_U = sum(1[W and s == g]) / sum(1[W])
error_imitation_U = sum(1[W and s == t]) / sum(1[W])
teacher_correct_retention_U = sum(1[C and s == g]) / sum(1[C])
~~~

当教师错误时，学生若预测为既非 GT、也非 teacher label 的第三类错误，该像素不进入 rescue 或 imitation 分子，但仍保留在共同分母。所有差值固定为：

~~~text
delta = unreliable_only - neutral
~~~

每图保存 int64 numerator/denominator，并缓存 name、native shape、student/teacher prediction、nearest GT、valid mask 和 float32 `u`。两支的 name、顺序、shape、teacher prediction、GT、valid mask、`u` 和三个 denominator 必须逐项一致，否则 paired 比较结构失败。

## 6. Paired bootstrap 与 C1 门禁

共享 bootstrap 矩阵在查看 C1 结果前已生成：

| 项目 | 冻结值 |
|---|---|
| RNG | `numpy.random.Generator(PCG64(3407))` |
| shape | `(10000, 1449)` |
| dtype | `int32` |
| 取值 | `[0,1449)` |
| 抽样单位 | 图像；每组有放回抽 1,449 张 |
| 置信区间 | paired delta 的 2.5% / 97.5% percentile |
| quantile method | `linear` |
| NPY SHA256 | `de2b18873dcd9f05f2d1d7acd9c0d94088680fb009441a501b8ba31ee8ce10b5` |

每个 bootstrap replicate 先分别汇总各分支的 numerator/denominator、计算比例，再求 paired delta。任一完整 replicate 在任一指标或分支上汇总 denominator 为 0，均使结构门禁失败。

C1 只有以下六项全部满足才算出现继续信号：

1. `delta_student_rescue_U >= 0.005`；
2. `delta_student_rescue_U` 的 paired-bootstrap 95% CI 下界严格大于 0；
3. `delta_error_imitation_U <= 0`；
4. `final_mIoU(unreliable_only)-final_mIoU(neutral) >= -0.002`；
5. `delta_teacher_correct_retention_U >= -0.005`；
6. 无数值、运行完整性或配置偏差。

任一项失败即停在 C1。即使六项通过，gate 也固定输出 `automatic_c2_launch=false`，等待人工复核。

## 7. 启动链、输出与故障联动

正式入口：

~~~bash
bash scripts/experiments/kd_baselines_npu/launch_phaseO_o12_c1_20k.sh
~~~

只读监控：

~~~bash
bash scripts/experiments/kd_baselines_npu/monitor_phaseO_o12_c1_20k.sh
~~~

输出隔离如下：

| 分支 | checkpoint | logger | runtime evidence |
|---|---|---|---|
| neutral | `data/winycg/checkpoints/kd_baselines_npu/phaseO_o12_c1/o12c1_neutral_20k_seed1234/` | `runs/kd_baselines_npu/phaseO_o12_c1/o12c1_neutral_20k_seed1234/` | `runs/runtime/kd_baselines_npu/phaseO_o12_c1/o12c1_neutral_20k_seed1234/` |
| unreliable_only | `data/winycg/checkpoints/kd_baselines_npu/phaseO_o12_c1/o12c1_unreliable_only_20k_seed1234/` | `runs/kd_baselines_npu/phaseO_o12_c1/o12c1_unreliable_only_20k_seed1234/` | `runs/runtime/kd_baselines_npu/phaseO_o12_c1/o12c1_unreliable_only_20k_seed1234/` |

controller 的 PID、状态和总日志位于：

~~~text
runs/runtime/kd_baselines_npu/phaseO_o12_c1/controller.pid
runs/runtime/kd_baselines_npu/phaseO_o12_c1/controller.status
runs/runtime/kd_baselines_npu/phaseO_o12_c1/controller.log
~~~

所有正式目录均使用不存在前检和 lock，拒绝覆盖。两个 worker 各自使用独立 `setsid` 进程组；任一分支异常退出时，pair controller 立即终止另一个进程组，10 秒后仍未退出才 KILL。controller 收到 INT/TERM 时同时清理两支。

## 8. 启动前审计

### 8.1 前置 acceptance

O1.2-B 的四份 final acceptance 均已通过且被冻结：

| 分支 | 模式 | acceptance SHA256 |
|---|---|---|
| neutral | fresh | `90eb89fe7a7dff773845152d5d59a7efcc3d61299976137223be6cf7d2615e1c` |
| neutral | resume audit | `30fe00fa72b2f20293db2cfca3da7f0f9d960649fef8d153629b3d2b2f496677` |
| unreliable_only | fresh | `d8a3363724c7c5d93c7629ba0ebf365a7f73d1c25c53138633e179e21fe623cd` |
| unreliable_only | resume audit | `af540bf53a0b1646227eb9024a6eb9f5d7bf1af250773c62d1074b77aaa82488` |

### 8.2 工具与预注册 SHA

| 文件 | SHA256 |
|---|---|
| O1.2 预算路由计划 | `c6ac659aea7019d8c2faed88ccdd596678e909129e3a468942cde921d6a6d8b9` |
| single runner | `9f878abf36e8ad0c28500a45c1c7c27ac817680e1181866446532813f3d7325a` |
| pair controller | `3e54ea0d062ac977344a655f3f627a547ef6fe0497d0f1327ae062eb6037f32d` |
| background launcher | `6630392707eabda5cf71ce1f71251ec58ad1551bf3004af2a52108e19294f051` |
| read-only monitor | `9480506b6e68c0f72c70cf9b38996a0df4d436d512fe02e3f159b9faa840c303` |
| single-run checker | `cc2a1cb482208d762a23f8643cd15b8c98ca176e01f35b26886a16e6ea880ae9` |
| final evaluator | `c1869ae2e1a3316e3c4c99b11eb3cd54dd4ad23987f43a817aaa5d6898119f05` |
| paired gate | `92fa52fd602a31962edb3d0a79d296f1d82cf8682020a093a6b6ea22e333fda1` |
| checker tests | `26f0123233181c2798f0ef34a321152325170357e9215cc5bb884fdd629e3f22` |
| evaluator/gate tests | `27aaa8fd5dd048df81f3d02745434c64e5ad0b14fcee549c6b9b948965eccdbf` |

### 8.3 静态、单测与 NPU evaluator smoke

启动前已完成：

- 四个 shell 脚本 `bash -n`：通过；
- checker、evaluator、gate 与两份测试 `py_compile`：通过；
- C1 两份定向测试：`16 passed`；
- 项目完整回归测试：`119 passed`；
- 负向 CLI、错卡绑定、绕过 pair 授权、额外参数：均被拒绝；
- pair 的首错 PID、兄弟进程组 TERM 和 setsid PID 对接 mock：通过；
- read-only monitor 在未运行态测试：通过，未创建正式训练产物；
- NPU0 一图 evaluator forward smoke：student/teacher raw logits 均为 `[1,21,46,63]`、shape 相同、全部有限、canonical sample name 一致；
- NPU0/NPU1 mapping：物理卡、逻辑卡和 Chip ID 分别严格对应 0/1；
- NPU0/NPU1 health：`OK`；
- 启动前两卡均无设备进程。

测试中保留了已知的 CANN 路径 owner warning 和旧 teacher 权重格式兼容 warning；它们不是数值结果，也不等价于零 warning。正式 checker 仍会拒绝 traceback、NaN、Inf、缺失日志和配置偏差。

### 8.4 Git 边界

本轮只做本地提交，不推送 GitHub。四个旧 Phase O 未跟踪脚本继续隔离，不使用、不修改、不 stage：

~~~text
scripts/experiments/kd_baselines_npu/check_phaseO_rtc_runs.py
scripts/experiments/kd_baselines_npu/launch_phaseO_rtc.sh
scripts/experiments/kd_baselines_npu/run_phaseO_rtc.sh
scripts/experiments/kd_baselines_npu/run_phaseO_rtc_variant.sh
~~~

正式启动时 Git dirty 状态必须精确只含这四项；launcher、checker 和 final acceptance 都会复核。运行期间不得修改 tracked 文件或改变 Git commit，否则 final acceptance 失败。

## 9. 训练完成后的记录顺序

训练结束后按以下顺序继续，不能跳步：

1. 两支 single-run final acceptance 均通过；
2. 分别在 iteration 20,000 checkpoint 上运行 native-grid evaluator，生成 packed cache 与 summary；
3. paired gate 复算 cache、共享 bootstrap 和六项门槛；
4. 写正式 C1 结果报告，完整记录 25 点验证曲线、final/best/last10、三个机制指标及 CI、运行时间和异常；
5. 人工审查后再决定是否另行授权 C2。

在步骤 1 至 4 完成前，不对 `unreliable_only` 是否有效作结论，也不使用中途最好曲线提前选择方法。
