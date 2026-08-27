# Paired multi-seed：固定温度与 P4a 复核

## 结论先行

- P4a 的 paired difference 有正有负，未形成一致正收益。可写为：P4a fails to yield a consistent improvement.
- n=3，仅报告 paired mean 与样本标准差；不计算或强调 p-value。
- d_complexity 的逐 seed 符号为正、负、正，paired mean +0.346881 pp、样本 SD 0.941712 pp；T=1.5 数值上胜出 2/3，但并非逐 seed 稳定优于最低复杂度 T=0.5。
- d_adaptive 的逐 seed 符号为负、正、负，paired mean +0.073015 pp、样本 SD 0.639808 pp。
- 三种方法均只在 20k 做一次 validation，因此表中 best=final。

## 协议

- seed=1234 复用既有正式结果；新 seed=2025、3407 沿用仓库既有约定。
- 每个新 seed 只跑固定 T=0.5、固定 T=1.5、P4a 8×8，均为 20k。
- 数据、模型、global batch=16、双 GPU、CE+KD、优化器及验证频率保持不变。

## 原始结果与 paired difference

| seed | T=0.5 best/final | T=1.5 best/final | P4a best/final | d_complexity final (pp) | d_adaptive final (pp) |
|---:|---:|---:|---:|---:|---:|
| 1234 | 59.499210/59.499210 | 60.861409/60.861409 | 60.517776/60.517776 | +1.362199 | -0.343633 |
| 2025 | 60.292262/60.292262 | 59.794337/59.794337 | 60.604030/60.604030 | -0.497925 | +0.809693 |
| 3407 | 61.496580/61.496580 | 61.672950/61.672950 | 61.425936/61.425936 | +0.176370 | -0.247014 |

定义：d_complexity=mIoU(T=1.5)−mIoU(T=0.5)；d_adaptive=mIoU(P4a)−mIoU(T=1.5)。

## Paired 汇总

| contrast | best mean ± sample SD (pp) | final mean ± sample SD (pp) |
|---|---:|---:|
| T=1.5 − T=0.5 | +0.346881 ± 0.941712 | +0.346881 ± 0.941712 |
| P4a − T=1.5 | +0.073015 ± 0.639808 | +0.073015 ± 0.639808 |

## 边界

- 本轮只复核 seed 稳定性，不新增机制解释或自适应设计。
- 不据 n=3 声称统计显著性，也不把数值下降改写成“始终下降”。
- 未运行其它温度、margin gate、其它 region size、P2–P5、跨设置实验或短时程轨迹。
