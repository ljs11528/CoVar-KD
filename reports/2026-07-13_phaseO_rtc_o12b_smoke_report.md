# Phase O1.2-B：neutral 与 unreliable_only 20-step 学生训练链路 smoke 报告

- 执行日期：2026-07-13
- 阶段：O1.2-B
- 实验性质：20-step 学生训练链路 smoke，不是性能实验
- fresh 分支：`neutral`、`unreliable_only`
- 恢复审计：两支各一次 completed-checkpoint `resume_audit`
- 最终状态：四份 final acceptance 全部通过
- 前置依据：[O1.2 预算路由计划](./2026-07-13_phaseO_rtc_o12_budgeted_routing_plan.md)与 [O1.2-A 正式机制诊断](./2026-07-13_phaseO_rtc_o12_diagnostic_report.md)

> 本阶段只回答训练入口、数值、设备、checkpoint 和恢复链路能否按冻结契约工作。20 iteration 没有 validation，也没有学生 mIoU 结果，不能用于判断 `unreliable_only` 是否优于 `neutral`。

## 0. 结论

O1.2-A 联合门禁通过后，本轮按授权只运行了 `neutral` 与 `unreliable_only` 两个 20-step fresh smoke，并分别运行一次 0-step `resume_audit`。独立审计结论如下：

1. 两个 fresh 运行都从同一冻结初始化开始，在单进程、单 NPU、seed 1234 下完成恰好 20 个 optimizer step；训练和 checker 退出码均为 0。
2. 两个 fresh 运行各生成 checkpoint v4，iteration 均为 20；checkpoint 中所有受检张量和浮点状态均为有限值。
3. iteration 20 的 teacher-target KD、cross-entropy、教师熵和 KD-only student-logit 梯度均为有限值；两支的 KD-only student-logit 梯度范数均大于 0。
4. 两个 `resume_audit` 都从对应 fresh checkpoint 的 iteration 20 完整恢复，执行 0 个额外 optimizer step；恢复前后的学生权重 SHA 完全一致。
5. 四份 final acceptance 均为 `pass=true`、`errors=[]`、`warnings=[]`，acceptance 中登记的全部证据文件 SHA 已独立逐文件复算，未发现错配。
6. 原始 console 和 checker 日志并非字面上的“零 warning”：存在 CANN 所有者、NPU tensor format 和旧权重格式兼容性 warning。它们没有触发门禁失败，但必须作为环境证据保留。
7. 没有运行 validation，没有 pixAcc 或 mIoU 评估。恢复日志中的 `best_mIoU=0.000000` 只是 skip-val checkpoint 的未验证元数据，不是性能结果。

当前停止在 O1.2-B 人工审查线，不自动启动 `reliable_only`、`full_budgeted`、20k、80k、scalar 或 shuffle 实验。

## 1. 本阶段目标与证据边界

### 1.1 实际检查内容

本次 smoke 检查：

- O1.2 teacher-target-only 训练入口能否加载正式 CDF、预算参数和 joint gate；
- `neutral` 与 `unreliable_only` 是否能使用同一冻结 recipe 完成 20 个更新；
- loss、教师目标统计和显式 KD-only student-logit 梯度是否为有限值；
- 单 NPU 显存、进程、吞吐和完整 argv 是否可追溯；
- checkpoint v4、optimizer、判别器、CWD 模块、RNG 和 O1.2 元数据是否能够保存；
- completed checkpoint 是否能够按相同来源、配置和数据顺序契约恢复；
- 恢复审计是否保持 0 个额外 optimizer step。

### 1.2 本阶段不回答的问题

本阶段不回答：

- `unreliable_only` 是否改善学生 mIoU；
- 高风险平滑是否减少错误模仿；
- 两支 iteration-20 loss 的差异是否具有统计或因果意义；
- 空间风险位置是否优于 matched scalar 或 shuffled 对照；
- VOC 之外的数据集、其他教师或多 seed 是否泛化；
- 20-step sample-order SHA 是否等价于数据增强逐比特重放。

## 2. 冻结训练配置

| 项目 | 冻结值 |
|---|---|
| 数据集 | VOC train_aug |
| teacher | DeepLabV3-ResNet101 |
| student | DeepLabV3-MobileNetV3-Small |
| crop / batch / workers | 512x512 / 16 / 8 |
| seed | 1234 |
| 设备 | 每个运行 1 个进程、1 张可见 NPU，`local_rank=0` |
| max iterations | 20 |
| log / save / val interval | 20 / 800 / 800 |
| validation | `skip_val=true` |
| lr / momentum / weight decay | 0.02 / 0.9 / 1e-4 |
| KD mode | `rtc_o12_teacher_target` |
| teacher outer temperature | 3.0 |
| student KD temperature | 1.0 |
| O1.2 temperature power | 无，不乘 `T^gamma` |
| lambda KD / adversarial G / D | 1.0 / 0.001 / 0.1 |
| lambda CWD feature / logit | 50.0 / 3.0，两个 CWD 分支仍固定 T=4 |
| 其余 KD loss | SKD、IFV、FitNet、AT、PSD、CSD 均为 0 |

O1.2 教师目标保持为：

~~~text
q_teacher = softmax(raw_teacher_logits / (3.0 * T_pixel))
p_student = softmax(student_logits)
~~~

其中：

- `neutral`：所有有效像素精确使用 `T_pixel=1`；
- `unreliable_only`：可靠侧和中性区保持 `T_pixel=1`，只在 `u>0.8` 的高风险侧连续提高温度；
- 两支使用相同的 CDF 和正式预算解 `b=0.3476499170064926`；
- 空间温度只改变教师目标，不改变学生 softmax 温度。

## 3. 实现、输入与目录隔离

### 3.1 执行源码

| 文件 | SHA256 |
|---|---|
| [run_phaseO_o12_smoke_variant.sh](../scripts/experiments/kd_baselines_npu/run_phaseO_o12_smoke_variant.sh) | `40c87164d3ed79d5df536e3118650451304916254197d3eac044416d23b7d507` |
| [check_phaseO_o12_smoke.py](../scripts/experiments/kd_baselines_npu/check_phaseO_o12_smoke.py) | `e7e74564d8c2d231fc03e4e6fc50b8ecd5f4f0165138fa1cea3c66f23770aff4` |
| [utils/rtc_o12_calibration.py](../utils/rtc_o12_calibration.py) | `5e23f3e1bd526017aa2046faff621dd284093d91dead9088dd4d7d9ddb641d9e` |
| [diagnose_rtc_o12_budget.py](../scripts/diagnostics/diagnose_rtc_o12_budget.py) | `cc391388f64505abae4cded5ac7b36122018a131b3c90f480290ff275a4cee50` |
| [check_rtc_o12_gate.py](../scripts/diagnostics/check_rtc_o12_gate.py) | `805a19625d496d3c3864d529e314a49d75584afad69fedf69e28cadc431ce085` |
| [train_kd.py](../train_kd.py) | `f50a130c17ca3b5f368f848b96c59fd58c408952e75bda447c051589505b316d` |

本轮没有使用四个隔离的旧 Phase O launcher/checker。运行时 Git commit 为 `5c93c1eeebfdf58ca4a9c82aca9852cb8a661330`；dirty 状态精确包含那四个旧未跟踪脚本，且已由 checker 校验，没有把它们纳入 O1.2-B 命令。

### 3.2 冻结输入

| 输入 | SHA256 |
|---|---|
| teacher checkpoint | `ac49b2c7720b21d565072e974e4404fcb009ba106288d95d1a4bb25f09c3fe58` |
| student ImageNet init | `47085aa164b2977003221458a2a5fdf5f46f434f5539b164dc71969b4dd4cd75` |
| O1.1 confidence CDF | `8ba8376a03c2f93835339d98670fa357b381b23a22cbf2a7fc48b0c2441d7e69` |
| O1.2 budget parameters | `a9006972e165ed6c95e7a966c526927bb9f846fb6c7ec6528f5507dd21b8b5df` |
| O1.2 joint gate | `c1961cfec8f232c1fdf7fc8b90db5a6f4855901317ea2705536eea88cbc1be82` |
| O1.1 gate | `47ff2f1f2ea68a4e50375bfa7efc7221c8197703372d5d8f22dfec9032088d3a` |
| VOC train_aug list | `d1326bd532648d73bc4b1bd275434eba81930982dc7401a65a8cecb26c028e24` |

### 3.3 四个独立输出目录

| 运行 | checkpoint 目录 | log 目录 | runtime evidence 目录 |
|---|---|---|---|
| neutral fresh | `data/winycg/checkpoints/kd_baselines_npu/phaseO_o12/o12b_neutral_smoke20_seed1234/` | `runs/kd_baselines_npu/phaseO_o12/o12b_neutral_smoke20_seed1234/` | `runs/runtime/kd_baselines_npu/phaseO_o12/o12b_neutral_smoke20_seed1234/` |
| unreliable fresh | `data/winycg/checkpoints/kd_baselines_npu/phaseO_o12/o12b_unreliable_only_smoke20_seed1234/` | `runs/kd_baselines_npu/phaseO_o12/o12b_unreliable_only_smoke20_seed1234/` | `runs/runtime/kd_baselines_npu/phaseO_o12/o12b_unreliable_only_smoke20_seed1234/` |
| neutral resume | `data/winycg/checkpoints/kd_baselines_npu/phaseO_o12/o12b_neutral_smoke20_seed1234_resume_audit/` | `runs/kd_baselines_npu/phaseO_o12/o12b_neutral_smoke20_seed1234_resume_audit/` | `runs/runtime/kd_baselines_npu/phaseO_o12/o12b_neutral_smoke20_seed1234_resume_audit/` |
| unreliable resume | `data/winycg/checkpoints/kd_baselines_npu/phaseO_o12/o12b_unreliable_only_smoke20_seed1234_resume_audit/` | `runs/kd_baselines_npu/phaseO_o12/o12b_unreliable_only_smoke20_seed1234_resume_audit/` | `runs/runtime/kd_baselines_npu/phaseO_o12/o12b_unreliable_only_smoke20_seed1234_resume_audit/` |

每个 runtime 目录保存完整 NUL 分隔 argv、provenance、PID、Git 状态、NPU mapping、周期性 `npu-smi`、console、prefinal/final checker 输出和 acceptance。目录创建使用覆盖保护；本轮没有复用或覆盖既有输出。

## 4. 运行拓扑、环境与时间边界

### 4.1 环境

| 项目 | 记录值 |
|---|---|
| Python | `/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python` |
| Python version | 3.11.10 |
| PyTorch | 2.8.0+cpu |
| torch_npu | 2.8.0.post2 |
| NPU 可用性 | true |
| 进程内可见 NPU 数 | 1 |
| world size / rank / local rank | 1 / 0 / 0 |

`torch_version=2.8.0+cpu` 是该 torch/torch_npu 组合记录的发行标识；实际运行设备、进程 PID 和 HBM 占用由 NPU mapping 与 `npu-smi` 交叉确认。

### 4.2 时间、设备和退出状态

下表时间为 UTC；北京时间为 UTC+8。

| 运行 | 物理 NPU | PID | start UTC | training end UTC | sealed end UTC | 训练/最终状态 |
|---|---:|---:|---|---|---|---|
| neutral fresh | 0 | 2736739 | 2026-07-13 15:55:51.066586448 | 15:56:19.766860654 | 15:56:29.225271518 | 0 / 0 |
| unreliable fresh | 1 | 2736723 | 2026-07-13 15:55:51.067035466 | 15:56:21.131637909 | 15:56:30.066634416 | 0 / 0 |
| neutral resume | 0 | 2741174 | 2026-07-13 15:58:23.410596524 | 15:58:41.064736913 | 15:58:49.873700031 | 0 / 0 |
| unreliable resume | 1 | 2741317 | 2026-07-13 15:58:25.876623483 | 15:58:42.410863368 | 15:58:51.585470433 | 0 / 0 |

四份 provenance 均满足：

~~~text
training_exit_status=0
checker_exit_status=0
exit_status=0
exit_reason=completed_and_checked
~~~

fresh 两支并行运行于不同物理 NPU；该调度只用于链路 smoke，不能把两卡吞吐差异解释为方法效果。审计结束时四个 PID 均已退出，NPU AICore 回到 0，未发现残留训练进程。

## 5. Fresh 20-step 训练结果

### 5.1 iteration 20 数值

| 指标 | neutral | unreliable_only |
|---|---:|---:|
| optimizer steps | 20 | 20 |
| LR | 0.001349 | 0.001349 |
| Task loss | 1.7301 | 1.7785 |
| KD loss | 0.9500 | 1.0019 |
| Adv G loss | 0.0050 | 0.0048 |
| Adv D loss | 0.0000 | 0.0000 |
| CWD feature loss | 5.8314 | 5.8964 |
| CWD logit loss | 4.0377 | 4.0249 |
| SKD / IFV / AT / FitNet / PSD / CSD | 全部 0 | 全部 0 |
| O1.2 branch KL mean | 0.95004142 | 1.00187660 |
| O1.2 cross-entropy mean | 2.17816402 | 2.27017049 |
| O1.2 teacher entropy mean | 1.22812260 | 1.26829381 |
| KD-only student-logit grad L2 | 0.00273925 | 0.00282540 |
| native KD valid pixels | 43,209 | 43,209 |
| teacher outer T | 3.0 | 3.0 |

这些 loss 和诊断来自唯一的 iteration-20 日志点，是该 iteration 当前 batch 的值，不是 20 个 iteration 的算术均值，也不是收敛曲线。两支数值不能据此排序。

打印精度下的 KL 闭合为：

~~~text
neutral:
2.17816402 - 1.22812260 = 0.95004142

unreliable_only:
2.27017049 - 1.26829381 = 1.00187668
reported KL = 1.00187660
absolute closure residual = 8e-8
~~~

两者均满足 checker 的 `2e-5` 闭合容差。

### 5.2 运行时间和显存

| 指标 | neutral | unreliable_only |
|---|---:|---:|
| logger reported total time | 11.030566 s | 11.530959 s |
| seconds / iteration | 0.5515 | 0.5765 |
| samples / second | 29.011786 | 27.753686 |
| provenance training wall time | 28.700274 s | 30.064602 s |
| process peak memory | 12,863 MB | 12,862 MB |
| device HBM baseline | 3,100 MB | 2,882 MB |
| device HBM peak | 15,912 MB | 15,689 MB |
| device HBM delta | 12,812 MB | 12,807 MB |
| own-process NPU observations | 13 | 15 |

logger time只覆盖 `Trainer.train()`，provenance wall time还包含 Python/NPU 初始化、模型构建和保存，因此两者定义不同。吞吐和显存只用于确认链路可运行，不是性能基准。

### 5.3 checkpoint 有限值扫描

两个 fresh checkpoint 均为：

| 项目 | 值 |
|---|---:|
| checkpoint version | 4 |
| completed iteration | 20 |
| finite tensors | 847 |
| finite tensor elements | 19,711,341 |
| NumPy arrays | 0 |
| finite float scalars | 7 |

该扫描覆盖学生、CWD、FitNet、判别器及 optimizer 状态中可遍历的张量/浮点值。它证明已保存状态有限，但不等价于逐参数记录每一步梯度。

## 6. 0-step resume audit

| 指标 | neutral resume | unreliable_only resume |
|---|---:|---:|
| source checkpoint SHA | `6309cba985d8831586a610c45a4f463ba15d71381924b7093ca9f9fa91a982a9` | `b9426976a45ab1ec6396bb68d85f2f43fe80f176d94fde8c5395580a41ea0ed4` |
| output checkpoint SHA | `6b0032efd44202181cb2bf4248209b788ea9573b92541ffe339562804d7d3e44` | `8760408718353c0e8123d4a4a5965eb63e8db504e23724d9b57c19ce77362aca` |
| optimizer steps | 0 | 0 |
| logger reported time | 0.459907 s | 0.456523 s |
| provenance wall time | 17.654140 s | 16.534240 s |
| process peak memory | 1,055 MB | 1,061 MB |
| device HBM delta | 1,005 MB | 1,005 MB |

每份恢复日志都同时出现：

~~~text
completed_iteration=20
Resumed full training state ... iteration=20
Continuing training from iteration 20
~~~

并且没有任何 `Iters:` 日志。checker 逐项比较源 checkpoint 与恢复后 checkpoint 中的：

- student；
- CriterionCWD；
- CriterionFitNet；
- discriminator；
- optimizer；
- discriminator optimizer；
- `rtc_o12`；
- `rtc_o12_data_order`。

以上状态均相等。完整 checkpoint SHA 不相等是预期的，因为 resume 输出中的 argv、`resume`、save/log 路径等容器元数据不同；方法状态和 optimizer 状态的等价由逐项比较保证。

fresh 与 resume 的独立学生权重文件 SHA 精确相同：

| 分支 | fresh / resume 共同 student SHA256 |
|---|---|
| neutral | `87dc4b5a7f8ed6b6285f36b83a15bbb72d17c7e8d1f5a0a27e53324807ab26dd` |
| unreliable_only | `15744fa57c1a38a2c927579ce2e46c8a11bc6f7178f7ab7667cfbbc5d2e9d5a7` |

本次恢复发生在已完成的 iteration 20，因此只证明完整状态能够加载并保持不变，不证明 iteration 1-19 任意中断点的实际续训轨迹能够逐比特重现。

## 7. 数据顺序证据

四份 checkpoint 的数据顺序契约都记录：

~~~text
algorithm=rtc_o12_canonical_order_v1
seed=1234
canonical_population=10582
batch_size=16
max_iterations=20
total_canonical_indices=320
complete_order_sha256=99326472a2e5e2bd42428d4709ff9f8049d2c906c7a6b9e8fa3068cb0439d564
~~~

两个 fresh 分支因此使用相同的 320 个 canonical dataset index 及相同顺序；resume checkpoint 的 `completed_iteration=20`、`next_global_iteration=null`、`next_canonical_index_offset=320` 与已完成状态一致。

该 SHA 只覆盖 canonical dataset index 序列，不覆盖 DataLoader worker 内的随机增强、图像变换输出、NPU kernel 或完整数值轨迹。workers 固定为 8，但本次没有缓存每次 augmentation tensor，也没有从中途 checkpoint 继续处理下一 batch。因此只能声称“canonical 样本顺序一致”，不能声称“augmentation bitwise replay”或整条训练逐比特重现。

## 8. Acceptance 与关键产物 SHA

### 8.1 四份 final acceptance

| 运行 | acceptance | SHA256 | pass / errors / warnings |
|---|---|---|---|
| neutral fresh | [acceptance.json](../runs/runtime/kd_baselines_npu/phaseO_o12/o12b_neutral_smoke20_seed1234/acceptance.json) | `90eb89fe7a7dff773845152d5d59a7efcc3d61299976137223be6cf7d2615e1c` | true / [] / [] |
| neutral resume | [acceptance.json](../runs/runtime/kd_baselines_npu/phaseO_o12/o12b_neutral_smoke20_seed1234_resume_audit/acceptance.json) | `30fe00fa72b2f20293db2cfca3da7f0f9d960649fef8d153629b3d2b2f496677` | true / [] / [] |
| unreliable fresh | [acceptance.json](../runs/runtime/kd_baselines_npu/phaseO_o12/o12b_unreliable_only_smoke20_seed1234/acceptance.json) | `d8a3363724c7c5d93c7629ba0ebf365a7f73d1c25c53138633e179e21fe623cd` | true / [] / [] |
| unreliable resume | [acceptance.json](../runs/runtime/kd_baselines_npu/phaseO_o12/o12b_unreliable_only_smoke20_seed1234_resume_audit/acceptance.json) | `af540bf53a0b1646227eb9024a6eb9f5d7bf1af250773c62d1074b77aaa82488` | true / [] / [] |

### 8.2 每次运行的关键文件

| 运行 | checkpoint SHA | logger SHA | console SHA | provenance SHA |
|---|---|---|---|---|
| neutral fresh | `6309cba985d8831586a610c45a4f463ba15d71381924b7093ca9f9fa91a982a9` | `dae2b1b78cbcde0e8754d4d6998ff13bbc679cbe15652ac6f92fd75a2950c0aa` | `21fe3907a760acdbb9945b1be1b68ecfb721c9f7e37b2766cd6836a4105a573f` | `a30ef6476f21fa3c3501b7c2f2925040f4864b68716130b23cc64eeb0615cf66` |
| neutral resume | `6b0032efd44202181cb2bf4248209b788ea9573b92541ffe339562804d7d3e44` | `0d2637178f48de0f3353729e8c40e7a94805cb164f645616ced2201695ce7275` | `578a211049b5c0e0de1bf8a289e7484e27ea5714b1e1ca81ab813ba0ef3ed032` | `172ce145da4ceda5fbbe6034c5eb9af550d6310313b6fa1556c5d40b2830abe6` |
| unreliable fresh | `b9426976a45ab1ec6396bb68d85f2f43fe80f176d94fde8c5395580a41ea0ed4` | `fa5ae28a7d3ebb12ae34281de832864e79fc372961d418e495b83418b05a0b3f` | `df77fe1dcfa9a6faa723ca9747903292148a65b9b6a442eea72d344ff24041e0` | `0ee6d9e51f6a978a96ab72c0f634fdace3f7fee7b090b4dacedcfeb5267ff1f5` |
| unreliable resume | `8760408718353c0e8123d4a4a5965eb63e8db504e23724d9b57c19ce77362aca` | `71fe64edae52afeef3fc0a2d91171e15c4b970a49ccdd66fdadd482e01e60baf` | `de306e2e8e9ea69a1b3bb9e2d18ac7a61b70612e470fb6001a1258b44dafbf21` | `9bfe2a46055ca2f75aff51546cd1b756d2deb3c00d94002d1ff3b6f60c1b53c1` |

每份 acceptance 还记录 student weights、`npu_smi`、NPU mapping、`argv.nul`、Git status、prefinal acceptance 和 prefinal checker log 的 SHA。独立审计已对这些清单逐文件复算，四份运行均为 `file_sha_mismatches=[]`。

## 9. Warning、错误与 validation 审计

### 9.1 Acceptance 语义

四份 final acceptance 的：

~~~text
pass=true
errors=[]
warnings=[]
~~~

表示没有 checker 定义的来源、配置、数值、梯度、checkpoint、恢复、设备或证据完整性异常。它不表示所有原始 stderr 都没有出现字符串 `Warning`。

### 9.2 原始日志中的环境兼容性 warning

独立扫描发现：

1. 四份 training console 都出现 CANN 安装目录/`ascend_ops_install.info` 所有者与当前用户不一致的 `torch_npu` 环境 warning；
2. 两份 fresh console 各出现一次 NPU internal-format fallback warning：tensor 在 `allow_internel_format=False` 下退回 base format；
3. 两份 resume console 各出现一次旧权重文件格式兼容性 warning；
4. prefinal/final checker 自身加载 checkpoint 时也记录了同类 CANN 和旧权重格式 warning。

这些 warning 被原样保存在 console/checker 证据中，没有被删除。四份运行仍满足：

- 无 Traceback；
- 无 `RuntimeError`、`AssertionError`、`FloatingPointError` 或 `ValueError`；
- 无 NaN 或 Inf；
- training、prefinal checker 和 final checker 退出码均为 0；
- checkpoint 全状态有限值扫描通过。

因此准确表述是“没有门禁级、数值级或训练失败 warning；存在已记录的环境/格式兼容性 warning”，不能写成“原始日志完全无 warning”。

### 9.3 无 validation 或 mIoU 结果

四份命令均使用 `--skip-val`，日志中不存在：

- `Start validation`；
- `Overall validation`；
- pixAcc；
- validation mIoU。

两个 resume 日志会打印：

~~~text
best_mIoU=0.000000
~~~

这是 checkpoint 中 `best_pred=0.0` 的恢复信息。因为 fresh 运行从未执行 validation，它不表示模型获得了 0 mIoU，更不能当作性能指标。

## 10. 梯度证据的严格范围

本轮显式梯度诊断是 iteration 20 的：

~~~text
torch.autograd.grad(kd_loss, student_logits)
~~~

它证明 O1.2 KD 分支对当前 student logits 的梯度有限且非零。训练主路径的 `losses.backward()` 和 optimizer step 也顺利完成，checkpoint 参数均有限。

但本轮没有逐参数保存或扫描每个 student parameter、CWD parameter、判别器 parameter 的梯度。因此不能从这一个 student-logit 梯度范数外推为：

- 所有参数在所有 20 步都有非零梯度；
- 每个模块的梯度范数都已验证；
- 两支梯度大小差异代表优化优势。

## 11. 当前证据能支持的结论

可以支持：

1. O1.2 正式 artifact 和 teacher-target-only 入口能够在单 NPU 上完成 20-step 学生更新；
2. `neutral` 和 `unreliable_only` 使用相同 seed、canonical sample order、初始化和公共训练 recipe；
3. iteration 20 的 O1.2 教师目标统计、KD loss 和显式 student-logit 梯度为有限值；
4. checkpoint v4 能保存 O1.2 来源、配置、模型、optimizer 和数据顺序状态；
5. completed checkpoint 能被完整加载，0-step 恢复不会改变受检方法状态和学生权重；
6. 两支链路在约 12.9 GB 进程峰值显存下可运行。

不能支持：

1. `unreliable_only` 提升或降低学生精度；
2. iteration-20 loss 较大或较小代表最终效果；
3. sample-order SHA 证明 augmentation 或完整训练逐比特重放；
4. 一个 student-logit 梯度范数证明所有参数梯度正确；
5. 两张不同 NPU 上的轻微吞吐差异是算法开销差异；
6. 高风险空间位置、可靠侧锐化、matched scalar 或跨数据集泛化已经成立。

## 12. 停止线

截至本报告：

- O1.2-B 授权范围内的 `neutral` 和 `unreliable_only` fresh 20-step smoke 已完成；
- 两支 completed-checkpoint 0-step `resume_audit` 已完成；
- 四份 final acceptance 已封存并通过；
- 没有运行 validation、20k、80k、scalar、shuffle、`reliable_only` 或 `full_budgeted`；
- 没有学生 mIoU 或方法效果结论；
- 所有训练 PID 已退出，当前没有残留训练进程。

下一步必须先人工审查本报告和四份 acceptance，再依据预注册计划另行明确授权。O1.2-B 的通过不会自动启动任何后续实验，也不会授权查看 smoke 后修改 CDF、风险阈值、预算参数、温度公式或主指标。
