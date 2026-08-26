# P3：CoVar gap 是否预测区域 teachability

- 本阶段不训练；完全复用 P2 的区域×状态×候选温度缓存。
- 预测规则统一为分数越小越优；oracle 为 P2 的一步监督 CE gain 最大温度。
- 二维 CoVar gap 使用全分析缓存的总体标准差缩放 r_c/r_v；不使用标签拟合权重。
- 方向契约：pred_idx = argmin(raw_cost)。
- 相关性契约：raw_rho = spearmanr(raw_cost, gain)；aligned_rho = spearmanr(-raw_cost, gain)。
- 方向恒等式最大绝对误差：0.000e+00；aligned_rho = -raw_rho。
- 执行门禁：通过

## 预测结果

| 状态 | 分数 | top-1 | adjacent | mean regret | median regret | mean aligned rho | median aligned rho |
|---|---|---:|---:|---:|---:|---:|---:|
| early | min teacher r | 54.1671% | 72.4636% | 4.216970e-03 | 0.000000e+00 | 0.604833 | 0.942857 |
| early | scalar abs(r_t-r_s) | 11.5040% | 20.4636% | 3.329953e-03 | 6.016113e-04 | -0.280821 | -0.657143 |
| early | 2D CoVar gap | 12.3235% | 21.5526% | 3.039325e-03 | 3.627254e-04 | -0.191438 | -0.142857 |
| early | teacher-student KL | 13.4447% | 30.6415% | 2.685213e-03 | 4.409440e-05 | 0.076785 | 0.085714 |
| middle | min teacher r | 59.5256% | 78.3612% | 2.390223e-03 | 0.000000e+00 | 0.676860 | 0.985611 |
| middle | scalar abs(r_t-r_s) | 11.9569% | 22.5876% | 2.251641e-03 | 4.952264e-04 | -0.201799 | -0.142857 |
| middle | 2D CoVar gap | 12.0647% | 24.5499% | 2.037548e-03 | 2.922871e-04 | -0.083402 | -0.085714 |
| middle | teacher-student KL | 14.6846% | 37.2183% | 1.791193e-03 | 2.416810e-05 | 0.225748 | 0.428571 |
| late | min teacher r | 61.5526% | 79.0081% | 2.596178e-03 | 0.000000e+00 | 0.671725 | 0.985611 |
| late | scalar abs(r_t-r_s) | 12.2911% | 24.8194% | 2.298988e-03 | 4.011189e-04 | -0.116728 | -0.115954 |
| late | 2D CoVar gap | 11.9892% | 27.3423% | 2.121861e-03 | 2.455082e-04 | 0.007424 | 0.028571 |
| late | teacher-student KL | 15.5148% | 41.2507% | 1.855460e-03 | 1.616147e-05 | 0.316996 | 0.485714 |
| overall | min teacher r | 58.4151% | 76.6110% | 3.067790e-03 | 0.000000e+00 | 0.651139 | 0.985611 |
| overall | scalar abs(r_t-r_s) | 11.9173% | 22.6235% | 2.626861e-03 | 4.960982e-04 | -0.199782 | -0.142857 |
| overall | 2D CoVar gap | 12.1258% | 24.4816% | 2.399578e-03 | 2.983673e-04 | -0.089139 | -0.085714 |
| overall | teacher-student KL | 14.5481% | 36.3702% | 2.110622e-03 | 2.738810e-05 | 0.206509 | 0.428571 |

## 结论与边界

- overall 最强 top-1 分数：min teacher r。
- overall 最低 mean regret 分数：teacher-student KL。
- min teacher r 实际偏向最低温度；其高 top-1 与 oracle 在 T=0.5 的多数质量一致，不能单独视为细粒度排序能力。
- 二维 CoVar gap 未同时满足 overall top-1 高于随机六选一和 mean Spearman 为正。
- 二维分解同时提高了相对标量 r-gap 的 top-1，并降低 mean regret。
- 二维 CoVar gap 相对标量 gap 的 top-1 差为 +0.2084%，mean regret 差为 -2.272827e-04。
- 分量尺度在同一分析缓存上估计，结果属于机制诊断；没有独立校准集或跨数据集验证。
- 这些分数只预测局部一步 teachability，不直接等价于完整蒸馏训练后的 mIoU 收益。
