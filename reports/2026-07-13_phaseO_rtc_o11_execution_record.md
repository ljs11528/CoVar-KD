# Phase O1.1：confidence-only RTC 路由执行记录

- 日期：2026-07-13
- phase：`O1.1`
- 当前状态：正式 CDF、train/val 全量诊断与独立联合门禁均已完成
- 联合门禁：`joint_gate_pass=true`
- 学生训练状态：未启动；O2/O3 仍等待人工审查决定
- 方法计划：[2026-07-13_phaseO_rtc_method_reconstruction_plan.md](2026-07-13_phaseO_rtc_method_reconstruction_plan.md)
- 诊断报告：[2026-07-13_phaseO_rtc_o11_diagnostic_report.md](2026-07-13_phaseO_rtc_o11_diagnostic_report.md)
- O1 失败记录：[2026-07-13_phaseO_rtc_execution_record.md](2026-07-13_phaseO_rtc_execution_record.md)

## 1. 目标、边界与结论状态

O1.1 在查看 O1 结果后提出，属于探索性而非独立确认实验。它只回答：教师最大类别置信度定义的相对风险，能否在当前教师与 VOC 上稳定富集教师错误，并驱动方向正确、数值稳定的锐化/平滑温度路由。

固定主风险：

```text
p_assess(i,k) = softmax(z_t(i,k) / 1.0)
c_i           = max_k p_assess(i,k)
r_conf(i)     = -log(clamp(c_i, 1e-8, 1-1e-8))
u_i           = F_train,confidence(r_conf(i))

reliability_mode = confidence
coefficient_a    = 0.0
```

`coefficient_a=0` 显式进入配置与元数据，但不参与 confidence 风险计算。旧 full/variance 分数只作为离线诊断参考。

正式结果通过全部预注册结构、效果与数值门禁。该通过只允许把 confidence-only 风险视为值得进入下一阶段讨论的候选，不等价于授权学生训练，也不构成跨数据集泛化证据。

## 2. 标签使用的准确口径

风险值、CDF 排序和温度路由不使用标签类别、教师正确性或学生信息。实现仍沿用监督分割数据中的 GT `ignore-valid` 空间掩码，以排除 ignore/void 与 padding 像素。

因此后续论文与报告统一使用：

> 标签类别和教师正确性不参与风险定义或排序拟合；仅使用训练协议既有的有效像素掩码限定统计人口。

不得再把当前实现简称为完全 label-free。

## 3. 冻结配置

| 项目 | 正式值 |
|---|---:|
| `T_assess` | 1.0 |
| `reliability_mode` | confidence |
| `coefficient_a` | 0.0 |
| `q / w` | 0.8 / 0.05 |
| `T_R / T_0 / T_U` | 0.5 / 1.0 / 2.0 |
| `alpha_R / alpha_U` | 1.0 / 1.0 |
| `T_out` | 3.0 |
| 二分次数 | 16 |
| CDF seed | 1234 |
| train 诊断增强 seed | 2025 |
| ranking seed | 3407 |
| CDF 每图采样上限 | 4096 |
| 诊断 ranking 每图上限 | 1024 |
| CDF knots | 4097 |
| workers | 0 |
| train batch | 4 |
| val batch | 1 |
| 正式 `max_images` | 0，完整扫描 |

## 4. 执行前冻结快照

### 4.1 代码与环境

- Git commit：`38ec8edafa57c23beac2c4e32d8887745c69b42b`
- 正式 CDF 元数据记录：dirty=true，dirty entry count=17
- Python：`/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python`
- 设备：两张 Ascend 910；CDF/train 使用 NPU 0，val 使用 NPU 1
- 正式执行前未发现学生训练进程

| 冻结源码 | SHA256 |
|---|---|
| `scripts/diagnostics/build_rtc_cdf.py` | `c88bdfcb885cde01cbf437e2e7751c8eab510067aacf530b469df808ae6604dd` |
| `scripts/diagnostics/diagnose_rtc_routing.py` | `f838f25b70b8eadfc982873057e5fb68c54c81089a32e9121fd664e02916f9ef` |
| `scripts/diagnostics/check_rtc_o11_gate.py` | `55553ec523ac2c2a979470b8542f31878462bbe5a919e0c52132edfbba4eb256` |
| `utils/rtc_temperature.py` | `01b7b6e6aa0d513561332510347b52ea9411330dfb0f2da54abdc36f2375fe59` |

checker 自身纳入 CDF source SHA，并在运行时核对当前文件 SHA。

### 4.2 数据与教师

| 输入 | 数量 | SHA256 |
|---|---:|---|
| VOC train_aug list | 10,582 | `d1326bd532648d73bc4b1bd275434eba81930982dc7401a65a8cecb26c028e24` |
| VOC val list | 1,449 | `cdc1326d12f69ce5153aa5da04a4d8783e146868d42d19f82f44e74b97ac907d` |
| DeepLabV3-ResNet101 teacher | 303 MB | `ac49b2c7720b21d565072e974e4404fcb009ba106288d95d1a4bb25f09c3fe58` |

## 5. 实现、审计与前置验证

### 5.1 实现要点

- O1.1 使用独立目录 `runs/diagnostics/phaseO_o11/`；
- O1 与 O1.1 的 CDF、诊断和 checker 产物互不覆盖；
- confidence 公式元数据固定为：
  - `reliability_definition_id=neg_log_top1_confidence_v1`
  - `active_terms=[confidence]`
  - `coefficient_a_active=false`
  - `reliability_epsilon=1e-8`
- 全 native-valid 风险路由保存精确整数 `risk_routing_counts`；
- 高风险、低风险和边界分别定义为 `u>q`、`u<q`、`u==q`；
- 每个风险十分位保存精确 `teacher_wrong_count`；
- checker 从整数计数重算 coverage、富集、recall 和低风险错误率；
- Spearman 只报告，十分位门禁采用 45 对 pairwise 单调一致率；
- checker 独立锁定完整 RTC 配置、seed、ranking cap、canonical list/teacher SHA、CDF 与源码 SHA；
- checker 不信任 split JSON 的 `all_checks_pass`。

### 5.2 正式前发现并修正的问题

正式结果产生前完成了两轮 schema 审计：

1. 将会惩罚零错误平台的 Spearman 门禁改为 pairwise 单调一致率，门槛仍为 0.90；
2. 修正 checker 对公式字段名的早期错配；
3. 用 exact routing counts 消除从十分位浮点率反推 `u==q` 边界的歧义；
4. 补齐完整配置、随机种子、ranking cap、canonical 输入和 checker self-SHA 契约；
5. 修正文档中“完全无标签”的过度表述。

这些修正均发生在正式 CDF、正式 train 和正式 val 结果产生之前，不改变效果门槛。

### 5.3 测试与最终 smoke

- Python 编译检查：通过；
- `git diff --check`：通过；
- RTC、诊断与 checker 相关测试：39/39 通过；
- 最终 8 图 CDF/train/val NPU smoke：通过；
- smoke checker 正确给出 `joint_gate_pass=false`，失败原因是 `max_images=8`、非完整扫描等正式性条件，而不是 schema 或数值异常；
- 所有 smoke 产物位于 `/tmp`，不作为正式结果。

## 6. 正式命令

### 6.1 Confidence CDF

```bash
/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python \
  scripts/diagnostics/build_rtc_cdf.py \
  --phase O1.1 --data dataset/VOCAug \
  --list-path dataset/list/voc/train_aug.txt \
  --teacher-model deeplabv3 --teacher-backbone resnet101 \
  --teacher-pretrained data/winycg/cirkd/teachers/deeplabv3_resnet101_voc_best_model.pth \
  --num-classes 21 --crop-size 512 512 --ignore-label -1 \
  --device npu:0 --batch-size 4 --workers 0 --max-images 0 \
  --max-pixels-per-image 4096 --num-quantiles 4097 \
  --assess-temperature 1.0 --reliability-mode confidence \
  --coefficient-a 0 --seed 1234 --log-every 100 \
  --output runs/diagnostics/phaseO_o11/voc_train_rtc_confidence_cdf.pt
```

### 6.2 Train 与 val 诊断

两条命令共同固定 `T_assess=1`、`q/w=0.8/0.05`、`T_R/T_0/T_U=0.5/1/2`、`alpha_R/U=1`、`T_out=3`、16 次二分、seed 2025、ranking seed 3407、ranking cap 1024、`max_images=0` 和 `--strict`。

train 使用 `train_aug.txt, npu:0, batch_size=4`；val 使用 `val.txt, npu:1, batch_size=1`。两者均写入固定独立 JSON 路径，完整 argv 已保存在各正式 JSON 的 `input_provenance.argv`。

### 6.3 联合门禁

```bash
/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python \
  scripts/diagnostics/check_rtc_o11_gate.py \
  --train-json runs/diagnostics/phaseO_o11/rtc_confidence_routing_train.json \
  --val-json runs/diagnostics/phaseO_o11/rtc_confidence_routing_val.json \
  --output runs/diagnostics/phaseO_o11/o11_confidence_gate.json \
  --strict
```

## 7. Confidence CDF 正式结果

- 完成时间：2026-07-13 07:12:38 UTC，15:12:38 CST；
- 图像：10,582/10,582；
- native-valid：32,298,651；
- finite：32,298,651；
- nonfinite：0；
- CDF 样本：32,298,651；
- knots：4,097；
- `r_conf` min/mean/max：-0.0 / 0.025293196 / 1.559070706；
- 完整扫描：true；
- source、teacher、list 与配置检查：通过；
- CDF SHA256：`8ba8376a03c2f93835339d98670fa357b381b23a22cbf2a7fc48b0c2441d7e69`；
- summary SHA256：`3c0e5f5f660ec03be454c497b95ca168e69d1145389373860b405944fd27a145`。

## 8. Train 与 val 正式结果

| 指标 | Train | Val |
|---|---:|---:|
| 图像 | 10,582/10,582 | 1,449/1,449 |
| native-valid | 32,246,990 | 3,878,674 |
| finite / nonfinite | 32,246,990 / 0 | 3,878,674 / 0 |
| 教师准确率，native KD-grid proxy | 0.970540 | 0.937765 |
| 全局教师错误率 | 0.029460 | 0.062235 |
| 高风险覆盖率 | 0.198395 | 0.210878 |
| 高风险错误 precision | 0.146555 | 0.243854 |
| 高风险错误 recall | 0.986971 | 0.826277 |
| 高风险错误富集 | 4.974788x | 3.918270x |
| 低风险错误率 | 0.000479 | 0.013701 |
| pairwise 单调一致率 | 1.000000 | 1.000000 |
| Spearman，仅报告 | 1.000000 | 1.000000 |
| split `all_checks_pass` | true | true |

### 8.1 精确风险计数

| 计数 | Train | Val |
|---|---:|---:|
| 教师错误总数 | 949,984 | 241,390 |
| 高风险像素，`u>0.8` | 6,397,631 | 817,927 |
| 高风险教师错误 | 937,607 | 199,455 |
| 低风险像素，`u<0.8` | 25,849,359 | 3,060,747 |
| 低风险教师错误 | 12,377 | 41,935 |
| 边界像素，`u==0.8` | 0 | 0 |

### 8.2 风险十分位错误率

| bin | Train | Val |
|---:|---:|---:|
| 0 | 0.000000000 | 0.000306965 |
| 1 | 0.000001860 | 0.001421968 |
| 2 | 0.000004981 | 0.003157578 |
| 3 | 0.000016195 | 0.005595617 |
| 4 | 0.000039732 | 0.009002327 |
| 5 | 0.000126117 | 0.014987220 |
| 6 | 0.000491669 | 0.028493649 |
| 7 | 0.003177479 | 0.053505198 |
| 8 | 0.031711367 | 0.123974235 |
| 9 | 0.260882381 | 0.351207857 |

十个分箱均非空；train 与 val 均为 45/45 对一致。

### 8.3 排名参考

AP/AUC 不参与联合门禁。

| 分数 | Train AP | Train AUC | Val AP | Val AUC |
|---|---:|---:|---:|---:|
| confidence 主分数 | 0.394362 | 0.963283 | 0.385680 | 0.902400 |
| 旧 full 参考 | 0.385577 | 0.963462 | 0.369892 | 0.901715 |
| 旧 variance 参考 | 0.384014 | 0.963341 | 0.367645 | 0.901281 |

confidence AP 相对旧 full 参考：train +0.008785，val +0.015788。

## 9. 温度与数值机制

| 指标 | Train | Val |
|---|---:|---:|
| 温度 mean | 0.766246 | 0.784605 |
| 温度 harmonic mean | 0.594183 | 0.600597 |
| q10 / q50 | 0.500004 / 0.500004 | 0.500004 / 0.500004 |
| q90 / p95 | 1.945137 / 1.992287 | 1.961479 / 1.993889 |
| 温度 min / max | 0.500004 / 1.999184 | 0.500004 / 1.999153 |
| active / solved | 32,246,990 / 32,246,990 | 3,878,674 / 3,878,674 |
| 残差 mean | 3.91e-5 | 3.99e-5 |
| 残差 p95 | 7.58e-5 | 7.53e-5 |
| fallback / tie | 0 / 0 | 0 / 0 |
| 可靠/不可靠方向违规 | 0 / 0 | 0 / 0 |
| `T_out=1/3` 六个路由字段 mismatch | 全 0 | 全 0 |
| 正温度 argmax mismatch | 0 | 0 |

温度中位数几乎等于 `T=0.5`，调和均值约为 0.59/0.60。这证明强锐化分支实际生效，但也保留了“广泛低温锐化可能解释未来收益”的重要替代解释。O1.1 本身没有消除该混淆。

## 10. 独立联合门禁

- 完成时间：2026-07-13 07:20:30 UTC，15:20:30 CST；
- `joint_gate_pass=true`；
- `train_evaluation.pass=true`；
- `val_evaluation.pass=true`；
- `errors=[]`；
- config fingerprint：`93021fef3f8e14325179ff23e6da4ef70be5b27c75a5bacf9ca7c973675bc94b`。

联合结构检查全部通过：

- train/val JSON 均可加载；
- 两个 split 均通过独立门禁；
- CDF 路径、报告 SHA 与实际文件 SHA 一致；
- train/val critical config 完全一致且等于冻结配置；
- canonical teacher/list、数据规模、seed、ranking cap 和 batch/workers 匹配；
- CDF metadata、source SHA、native correctness 语义一致；
- checker 当前 SHA 与 CDF/诊断冻结记录一致；
- 精确风险计数、十分位人口和错误总数可加且一致。

## 11. 正式产物与 SHA256

| 产物 | SHA256 |
|---|---|
| `voc_train_rtc_confidence_cdf.pt` | `8ba8376a03c2f93835339d98670fa357b381b23a22cbf2a7fc48b0c2441d7e69` |
| CDF summary JSON | `3c0e5f5f660ec03be454c497b95ca168e69d1145389373860b405944fd27a145` |
| CDF SHA sidecar | `d7760d9e727cb3bc37b09909900c582e6578d2759439bb0aeaffde919c8dd749` |
| train routing JSON | `312c1a1d309df0406402b07fbb94401d265ccc89dd062314004ac8fdf191c116` |
| train bins CSV | `ea4d8b8f56318de39be4a5520f95fe6cd4ba3dad78706cc128adad2728ba2d41` |
| train primary deciles CSV | `3976cb670df11293849083a27d43448573f1c5148c627a93b4bd5893844127a5` |
| val routing JSON | `1decff5e04b585a2cec3172213a55570ea5bc555eac3a9e00a8802b077049863` |
| val bins CSV | `fdce53f30912e1a7285fe55842c77a52dce6bd3a4f7ed084f59054e24abf78e9` |
| val primary deciles CSV | `11c0aeaab711281477bfeb22f1cc5bc656118d949cf9205f2b9d14ae66a16f3f` |
| joint gate JSON | `47ff2f1f2ea68a4e50375bfa7efc7221c8197703372d5d8f22dfec9032088d3a` |

## 12. 当前判断与后续边界

当前证据支持：

1. confidence-only 风险在当前设置上具有明显、稳定、单调的错误富集能力；
2. 旧方差项不是建立该排序所必需的；
3. 冻结 CDF 在 val 上没有出现覆盖率或单调性失控；
4. 温度反解、方向约束、`T_out` 解耦和数值稳定性均满足要求。

当前证据不支持：

1. confidence-only 在新数据集或新教师上必然保持同样效果；
2. 高风险像素等于教师错误；
3. RTC 学生一定优于固定低温；
4. 学生收益一定来自空间选择，而不是大量像素被推到 `T≈0.5`。

因此，本次执行到 O1.1 为止。未启动 O2/O3、20-iteration、20k 或 80k 学生训练。是否进入后续实验，等待对诊断报告的人工审查。
