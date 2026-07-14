# Phase O1.2-C1：高风险侧平滑 20k 正式结果与机制门禁报告

- 执行日期：2026-07-14
- 阶段：O1.2-C1
- 对照：`unreliable_only - neutral`
- 训练：两支 fresh、seed 1234、20,000 iteration、各 25 次 VOC val
- final evaluator：canonical VOC val 1,449 张，native KD grid
- 统计：图像级 paired bootstrap，10,000 次，PCG64(3407)
- 最终结论：**C1 joint gate 不通过，停止；不授权 C2**
- 预注册依据：[O1.2 预算路由计划](./2026-07-13_phaseO_rtc_o12_budgeted_routing_plan.md)
- 执行协议：[O1.2-C1 启动前审计记录](./2026-07-14_phaseO_rtc_o12_c1_execution_record.md)

> 本报告只解释当前冻结实现、当前完整 KD recipe、VOC 和 seed 1234 下的 C1 结果。它否定的是“当前 confidence-only top-20% 平滑映射已经表现出继续信号”，不是对所有高风险处理方法的普遍否定。

## 0. 结论

两支训练均完成并通过结构验收，但 `unreliable_only` 没有实现预期的教师错误处理机制：

1. final mIoU 从 `0.648326` 降至 `0.643941`，差值 `-0.004385`，低于预注册下限 `-0.002`；
2. 高风险且教师错误像素上的学生纠错率从 `0.411832` 降至 `0.406698`，差值 `-0.005134`，与要求的至少 `+0.005` 方向相反；
3. 同一人口上的错误教师模仿率从 `0.479291` 升至 `0.483452`，差值 `+0.004161`，与要求的非正增量方向相反；
4. 高风险但教师正确像素的保持率下降 `-0.004427`，仍在 `-0.005` 容差内；
5. 纠错率、错误模仿率和教师正确保持率的 paired-bootstrap 95% CI 都跨过 0，没有形成稳定的正向机制证据；
6. 六项门禁中四项失败、两项通过，正式 gate 为 `pass=false`、`joint_gate_pass=false`；
7. controller、两支训练、checkpoint、evaluator 和 paired gate 均按结构契约完成，没有数值或运行完整性失败；
8. 按预注册停止线，不运行 C2 matched-scalar/shuffle，也不运行可靠侧或 full map。

## 1. 执行完整性

### 1.1 训练状态

| 项目 | neutral | unreliable_only |
|---|---:|---:|
| optimizer steps | 20,000 | 20,000 |
| validation blocks | 25/25 | 25/25 |
| final acceptance | pass | pass |
| acceptance errors / warnings | 0 / 0 | 0 / 0 |
| physical NPU | 0 | 1 |
| training PID | 3395537 | 3395533 |
| reported total time | 2:31:47.532391 | 2:32:27.287478 |
| seconds / iteration | 0.4554 | 0.4574 |
| peak own-process memory | 13,217 MB | 13,217 MB |

Pair controller 于北京时间 08:38:01 启动，在 11:11:21 以 `both_final_acceptances_passed`、exit status 0 完成。两支 final checkpoint 均为 v4、iteration 20,000；训练数据顺序、冻结输入、argv、PID、物理卡和 checkpoint finite scan 已由 single-run checker 验证。

### 1.2 Final native-grid evaluator

两份 evaluator 均以 exit status 0 完成：

| 项目 | neutral | unreliable_only |
|---|---:|---:|
| canonical images | 1,449 | 1,449 |
| packed native-grid pixels（含 ignore/invalid） | 4,104,672 | 4,104,672 |
| native-valid pixels | 3,878,674 | 3,878,674 |
| evaluator wall time | 57.165 s | 56.706 s |
| images / second | 25.348 | 25.553 |
| summary pass / errors / warnings | true / 0 / 0 | true / 0 / 0 |

两份 cache 的 sample name、顺序、native shape、teacher prediction、nearest GT、valid mask、float32 `u` 和三个机制分母逐项一致。学生预测允许不同。任一不一致原本都会使 paired gate 结构失败，本轮未触发。

## 2. 20k 性能结果

### 2.1 主、次性能指标

| 指标 | neutral | unreliable_only | U-N |
|---|---:|---:|---:|
| final mIoU | 0.648326 | 0.643941 | **-0.004385** |
| best mIoU | 0.648326 | 0.643941 | -0.004385 |
| last-10 mean mIoU | 0.622748 | 0.623165 | +0.000417 |
| final pixAcc | 0.914791 | 0.914411 | -0.000380 |

预注册 C1 性能条件为：

~~~text
final_mIoU(unreliable_only) - final_mIoU(neutral) >= -0.002
~~~

实际差值为 `-0.004385`，比下限再低 `0.002385`，因此性能条件明确失败。两支的 best 都是 final checkpoint，没有 best/final 选择冲突。

last-10 mean 略微有利于 `unreliable_only`，但不能覆盖主指标失败。最后五个验证点中 `unreliable_only` 全部落后，均值差为 `-0.002502`，所以 final 落后不是一个孤立的单点反转。

### 2.2 25 个预注册验证点

| iteration | neutral | unreliable_only | U-N |
|---:|---:|---:|---:|
| 800 | 0.126466 | 0.124474 | -0.001992 |
| 1,600 | 0.250495 | 0.259078 | +0.008583 |
| 2,400 | 0.354394 | 0.344194 | -0.010200 |
| 3,200 | 0.430220 | 0.420951 | -0.009269 |
| 4,000 | 0.426066 | 0.403829 | -0.022237 |
| 4,800 | 0.490658 | 0.488525 | -0.002133 |
| 5,600 | 0.498509 | 0.507355 | +0.008846 |
| 6,400 | 0.530731 | 0.527087 | -0.003644 |
| 7,200 | 0.519727 | 0.534471 | +0.014744 |
| 8,000 | 0.551490 | 0.560962 | +0.009472 |
| 8,800 | 0.553935 | 0.548014 | -0.005921 |
| 9,600 | 0.564112 | 0.576132 | +0.012020 |
| 10,400 | 0.583223 | 0.574905 | -0.008318 |
| 11,200 | 0.587195 | 0.577458 | -0.009737 |
| 12,000 | 0.595694 | 0.582498 | -0.013196 |
| 12,800 | 0.592701 | 0.604949 | +0.012248 |
| 13,600 | 0.605535 | 0.611841 | +0.006306 |
| 14,400 | 0.592522 | 0.606097 | +0.013575 |
| 15,200 | 0.620559 | 0.613738 | -0.006821 |
| 16,000 | 0.625349 | 0.616719 | -0.008630 |
| 16,800 | 0.626697 | 0.626556 | -0.000141 |
| 17,600 | 0.632719 | 0.630019 | -0.002700 |
| 18,400 | 0.638722 | 0.636457 | -0.002265 |
| 19,200 | 0.644352 | 0.641335 | -0.003017 |
| 20,000 | 0.648326 | 0.643941 | -0.004385 |

`unreliable_only` 在 25 点中领先 8 次、落后 17 次；差值范围为 `[-0.022237,+0.014744]`。中途曲线波动较大，说明挑选单个中途 checkpoint 会产生不稳定结论，也进一步支持保持预注册 final checkpoint。

## 3. 高风险人口组成

机制人口来自 frozen train CDF 的严格条件 `u>0.8`：

| 人口 | 像素数 | 占比 |
|---|---:|---:|
| packed native grid | 4,104,672 | 全部 grid 的 100% |
| native-valid V | 3,878,674 | 全部 grid 的 94.494% |
| 高风险 U | 817,927 | V 的 21.088%；全部 grid 的 19.927% |
| U 中教师错误 W | 199,455 | U 的 24.385% |
| U 中教师正确 C | 618,472 | U 的 75.615% |

该 selector 在 val native-valid 人口中选出约 21.1%，接近预期的高风险尾部；19.927% 是相对包含 ignore/invalid 的全部 packed grid 的占比。它不是“教师错误选择器”：高风险侧仍有约四分之三像素是教师正确的，正确/错误像素数之比约为 3.10:1。因此，对整个 `u>0.8` 人口统一平滑时，受到处理的正确教师像素远多于错误教师像素。这是当前方法最关键的解释风险。

## 4. Final 机制指标

差值方向固定为 `unreliable_only-neutral`。

| 指标 | neutral | unreliable_only | U-N | paired-bootstrap 95% CI | 门槛 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| student_rescue_U | 0.411832 | 0.406698 | **-0.005134** | [-0.021868, +0.014881] | delta >= +0.005 且 CI 下界 > 0 | 失败 |
| error_imitation_U | 0.479291 | 0.483452 | **+0.004161** | [-0.008420, +0.016266] | delta <= 0 | 失败 |
| teacher_correct_retention_U | 0.806793 | 0.802366 | -0.004427 | [-0.010993, +0.002393] | delta >= -0.005 | 通过 |

三个 CI 都跨过 0，所以当前单 seed 不能支持这些机制差值具有稳定非零效应。更重要的是，两个针对教师错误的主方向都没有达到预期：纠错率点估计下降，错误模仿率点估计上升。

### 4.1 像素计数解释

在共同的 199,455 个 `W = high-risk and teacher-wrong` 像素上：

| 学生结果 | neutral | unreliable_only | 计数变化 U-N |
|---|---:|---:|---:|
| 预测 GT，纠正教师 | 82,142 | 81,118 | **-1,024** |
| 预测错误 teacher label | 95,597 | 96,427 | **+830** |
| 预测第三类错误 | 21,716 | 21,910 | +194 |

三类互斥且合计为共同分母。当前平滑没有把错误教师像素从“模仿教师”转向“预测 GT”；观察到的变化恰好相反。

在共同的 618,472 个 `C = high-risk and teacher-correct` 像素上，学生保持正确的计数从 498,979 降至 496,241，减少 2,738 个。该差值的比例仍在预注册容差内，且 CI 跨 0，不能单独宣称稳定伤害；但它与 final mIoU 的负向结果一致。

## 5. 六项联合门禁

| 检查 | 结果 |
|---|---|
| delta_student_rescue_U >= 0.005 | 失败 |
| student_rescue_U paired CI 下界 > 0 | 失败 |
| delta_error_imitation_U <= 0 | 失败 |
| final mIoU delta >= -0.002 | 失败 |
| delta_teacher_correct_retention_U >= -0.005 | 通过 |
| 数值、运行完整性和配置一致 | 通过 |

正式 paired gate 以预期的科学失败 exit status 1 结束，并正常封存 JSON 和 bootstrap delta；这不是脚本运行事故。输出固定记录：

~~~text
pass=false
joint_gate_pass=false
next_stage_authorized=false
automatic_c2_launch=false
action=stop_for_manual_review
~~~

## 6. 方法解释

### 6.1 当前可以支持的结论

1. confidence-only `r=-log(c)` 的 train CDF 在本次 val 上构造出约 21.1% 的高风险人口；
2. `u>0.8` 不是高精度教师错误检测器：该人口中的教师错误率为 24.385%，其余 75.615% 仍是教师正确像素；
3. 当前 top-20% 连续平滑映射没有提高学生纠正错误教师的概率，也没有降低错误教师模仿；
4. 当前配置对 final mIoU 有超过容忍线的负面影响；
5. 因此当前 `unreliable_only` 映射没有进入空间因果对照 C2 的必要信号。

### 6.2 不能支持的结论

本轮不能证明：

- 所有高风险平滑都无效；
- confidence 与教师错误完全无关；
- 任何额外可靠性信号都不会提高错误检测精度；
- 其他数据集、教师、学生或 seed 下结论相同；
- 某个类别、边界、小目标或前景区域是性能下降的确定原因。

### 6.3 可能原因，均属于待验证假设

1. **风险富集不等于错误定位。** top-20% 中错误率为 24.385%，统一处理会同时平滑约三倍数量的正确教师像素。
2. **教师目标已经有 outer T=3。** 在此基础上继续提高高风险侧温度，可能削弱了仍然有用的类别相对结构，而不仅是错误 hard target。
3. **完整 recipe 仍含其他教师约束。** CWD feature/logit 分支保持不变，当前 teacher-target 分支的局部平滑不等于总教师监督在同一位置被等比例削弱。
4. **单 seed 的机制差值不稳定。** 三个 bootstrap CI 都跨 0，不能把点估计解释为普遍规律；但门禁所需的正向证据同样不存在。
5. **类别均衡伤害可能大于整体像素伤害。** final pixAcc 只下降 0.000380，而 mIoU 下降 0.004385；这提示前景、稀有类或边界可能受影响，但必须由分层诊断确认。

## 7. 决策与后续边界

### 7.1 当前决策

- 停止 O1.2-C1；
- 不启动 C2 的 unreliable shuffle 或 arithmetic/harmonic matched scalar；
- 不启动 reliable_only、full_budgeted、80k 或多 seed；
- 不通过修改 checkpoint、门槛、bootstrap seed 或高风险阈值挽救当前实验。

### 7.2 若另行重构 O1.3

更合理的方向不是简单扩大平滑强度，而是提高“教师真正出错”定位的精度。可在另行预注册后评估：

1. 将 confidence 与独立错误线索结合，例如增强一致性、teacher/student disagreement、边界或区域稳定性；
2. 让平滑强度与估计错误概率连续关联，避免对 top-20% 内的大量正确像素统一处理；
3. 先做 GT-only 离线诊断验证 selector precision/recall，再授权学生训练；
4. 单独检查完整 KD recipe 中各教师约束的局部作用，避免只修改一个分支却把结果解释为总教师监督变化；
5. 任何新定义均重建独立 train CDF/参数并重新预注册，不能用本轮 val 标签直接调参。

## 8. 证据文件与 SHA256

### 8.1 训练 acceptance 与 checkpoint

| 文件 | SHA256 |
|---|---|
| neutral final acceptance | `3065ec5e9397007d5b148ab3b57312cf386fa7ebad2bbb20b2716078060c4c2e` |
| unreliable_only final acceptance | `96c0e059c214493edf7975a6a69ba0a8fca7876de5397ce954618b9404b7e287` |
| neutral final checkpoint | `18b47764f49b5b6c0b500ec9855efe0f26bd539c6ad304d0bb8ccff727a85281` |
| unreliable_only final checkpoint | `0b0e7bd6bdca8b366e161fb7ff17b6d281d1dd10820585a0700a9b3c074d2523` |

### 8.2 Evaluator 与 paired gate

| 文件 | SHA256 |
|---|---|
| neutral cache | `fea9c13eddbcaa78810a0743db63e6cc3c88fc1bd0122a46297d155b14648b57` |
| unreliable_only cache | `4208cf2eac9476497309aec5d4e238730f764c76a5d9a22d5c2f4f96546d5dab` |
| neutral summary | `2872d1a70df1f21fe6dc9e7f8fc165f6b072ed0779bbe2db8a88ab6c9ebef597` |
| unreliable_only summary | `017c898d6f250c3e49b80e9289771762746f41e904c32b91daf4401c488f00d5` |
| neutral evaluator log | `faf781bd6e0c3a2f5e42202783945fb489c7496925f71121f2b6d1400d59807c` |
| unreliable_only evaluator log | `008bae0636b6ffd2ebc6b5bd548ecf804d6f58050daf50ec222a8e41430ee4a9` |
| paired bootstrap deltas | `f6d81ac83c33697c321d5d8b61de8853e9dace39f76b8168861b454f775a2b88` |
| paired gate JSON | `fb037cd1d2cd3d16df4172b18e4d3738d20f045750dce0d6137e5b30ac7f7c35` |
| paired gate log | `fb037cd1d2cd3d16df4172b18e4d3738d20f045750dce0d6137e5b30ac7f7c35` |

冻结来源保持为：

| 来源 | SHA256 |
|---|---|
| plan | `c6ac659aea7019d8c2faed88ccdd596678e909129e3a468942cde921d6a6d8b9` |
| bootstrap indices | `de2b18873dcd9f05f2d1d7acd9c0d94088680fb009441a501b8ba31ee8ce10b5` |
| evaluator source | `c1869ae2e1a3316e3c4c99b11eb3cd54dd4ad23987f43a817aaa5d6898119f05` |
| paired gate source | `92fa52fd602a31962edb3d0a79d296f1d82cf8682020a093a6b6ea22e333fda1` |

本报告不修改冻结 plan。

## 9. Git 状态说明

训练启动和 final acceptance 封存时，Git commit 为 `f282b27d97879bb8cf9096efc9ccdc3ce9dc5b9f`，dirty 状态精确为四个旧 Phase O 脚本未跟踪。训练完成后、evaluator 启动前，外部 Git index 已把这四个脚本显示为 staged `A`。本轮 evaluator/gate 未读取、修改或使用它们；tracked evaluator、gate 和 plan 相对 `f282b27` 无差异且 SHA 匹配。

报告提交时必须保留这四个外部暂存项，不能误带入 C1 结果提交，也不能擅自取消暂存。
