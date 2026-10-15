#!/usr/bin/env python3
"""Generate a Chinese research report from the fully audited analysis."""
import json
import statistics
from pathlib import Path
from evaluate import OUT

LOCAL=Path('/Users/ljs/research_2026/covar_kd_2026/remote-8-3090/execution_posthoc_20261015')
LABEL={'ce':'CE','teacher_c1':'Teacher c=1','shared_c1':'Shared c=1','teacher_c4':'Teacher c=4','shared_c4':'Shared c=4'}
DS={'voc':'VOC','cityscapes':'Cityscapes'}

def main():
    a=json.loads((OUT/'analysis_summary.json').read_text());audit=json.loads((OUT/'final_audit.json').read_text())
    assert a['status']==audit['status']=='PASS'
    lines=['# VOC 与 Cityscapes 免重训分析报告','',
        '2026 年 10 月 15 日。已完成两套数据共 30 个学生最终检查点及 2 个教师模型的全验证集推理，以及 18 个既有 CE 中间检查点的局部更新诊断。没有新增完整训练，也没有写入模型检查点。',
        '', '核心结论是：现有实验应保留，后续应收敛论文主张并复用已有基线。增大 KD 系数在 VOC 与 Cityscapes 上产生相反的最终效果，Teacher 与 Shared 也没有稳定的跨数据集排序。这批结果支持数据与训练协议相关的经验结论，尚不能替代完整自适应 CoVar 方法的验证。', '',
        '## 最终结果与验证口径','',
        '学生均为 DeepLabV3 MobileNetV3 Large，教师为对应数据集的 DeepLabV3 ResNet101。Teacher 使用教师温度 2、学生温度 1；Shared 使用教师和学生温度 2。c 是显式 KD 系数，Shared 不再额外乘一次 T²。CE 不使用 KD。', '',
        '主表沿用原记录的 80k 最终 mIoU，未选择最佳检查点。均值和样本标准差来自 1234、2025、3407 三个训练种子。', '',
        '| 数据集 | 条件 | seed1234 | seed2025 | seed3407 | 均值 ± SD |',
        '|---|---|---:|---:|---:|---:|']
    for row in a['p0']['conditions']:
        lines.append(f"| {DS[row['dataset']]} | {LABEL[row['condition']]} | "+' | '.join(f"{row['by_seed'][str(s)]:.3f}" for s in [1234,2025,3407])+f" | {row['mean']:.3f} ± {row['sample_sd']:.3f} |")
    maxerr=max(x['max_miou_reproduction_error_pp'] for x in a['datasets'].values())
    lines+=['',f'所有 30 个学生检查点均通过复算，mIoU 与原记录的最大差异为 {maxerr:.8f} 个百分点，低于预设的 0.001 个百分点。区域统计使用浮点 logits 的 argmax 和 int64 混淆矩阵独立计算。', '',
        f'![最终成绩与三个训练种子]({LOCAL}/figures/final_results.png)', '',
        'Cityscapes 的 12 个 KD 结果全部优于对应种子的 CE。Teacher c=4 与 Shared c=4 分别平均提升 1.810、1.738 个百分点；两者仅相差 0.072 个百分点，三个种子的胜负并不一致。VOC 的最佳 KD 均值为 Teacher c=1，但仍比 CE 低 0.092 个百分点。', '',
        '| 协议 | VOC | Cityscapes |','|---|---:|---:|',
        '| 类别数 | 21 | 19 |','| 训练图像数 | 10582 | 2975 |','| 验证图像数 | 1449 | 500 |',
        '| 全局 batch | 16 | 8 |','| 裁剪尺寸 | 512×512 | 512×1024 |','| 初始学习率 | 0.02 | 0.01 |',
        '| 迭代数 | 80000 | 80000 |',f'| 名义训练样本遍历次数 | {80000*16/10582:.1f} | {80000*8/2975:.1f} |','',
        '教师权重及上述训练配置均有差异，因此不能把效果反转完全归因于数据集。样本遍历次数按迭代数×batch÷训练图像数计算，随机增强使其不等同于重复相同输入。', '',
        f'![四个固定阶段的验证曲线]({LOCAL}/figures/validation_curves.png)','',
        '曲线阴影为三个种子的样本标准差。VOC Teacher c=4、seed2025 在 20k／40k／60k／80k 分别为 45.371／56.688／59.155／61.816，低分从较早阶段就存在，后续持续改善。该种子保留在所有汇总中。原有 30 项实验的 12 万条已记录损失均为有限值，20 步一次的日志序列完整，恢复实验按保存步数拼接。', '',
        '## 类别与教师错误区域','',
        '以下比较 c=4 减 c=1，各类别先做同种子差值，再在三个种子上求均值。“上升类别数”是类别均值为正的数量，不是统计显著性检验。', '',
        '| 数据集 | 模式 | 最终 mIoU 变化 | 上升类别数 | 绝对变化最大的三个类别 |','|---|---|---:|---:|---|']
    for dataset,ds in a['datasets'].items():
        for mode in ['teacher','shared']:
            row=next(x for x in ds['class_contrasts'] if x['contrast']==f'{mode}_c4 - {mode}_c1')
            largest=sorted(row['classes'],key=lambda x:abs(x['mean_delta_pp']),reverse=True)[:3]
            desc='；'.join(f"{x['name']} {x['mean_delta_pp']:+.3f}" for x in largest)
            lines.append(f"| {DS[dataset]} | {mode.title()} | {row['mean_miou_delta_pp']:+.3f} pp | {row['positive_mean_classes']}/{len(row['classes'])} | {desc} |")
    lines+=['','Cityscapes 的事后类别分解显示，主要增益集中在有效像素占比较低的类别：','']
    city=a['datasets']['cityscapes']
    for mode in ['teacher','shared']:
        r=next(x for x in city['class_contrasts'] if x['contrast']==f'{mode}_c4 - {mode}_c1')
        top=r['largest_gains'][:3];contribution=sum(x['mean_delta_pp'] for x in top)/19
        fraction=sum(city['class_ground_truth_pixel_fraction'][x['name']] for x in top)
        assert all(x['positive_seeds']==3 for x in top)
        lines.append(f"- {mode.title()}：{'、'.join(x['name'] for x in top)} 合计占验证集有效像素的 {fraction*100:.3f}%，按类别平均的算术分解贡献 {contribution:.3f} pp，即总 mIoU 增量的 {contribution/r['mean_miou_delta_pp']*100:.1f}%；这些类别在三个种子上均提升。")
    lines+=['','这说明类别平均的 mIoU 与按像素汇总的平均 CE、像素准确率可能对同一变化给出不同侧重。完整类别图同时保留下降类别，不把事后选出的类别当成独立验证。','',f'![系数改变对每个类别的影响]({LOCAL}/figures/class_coefficient_effects.png)','',
        '误差线为同种子类别差值的样本标准差。完整逐类均值、标准差和三个种子的差值保存在分析数据中。', '',
        '教师错误区域的纠正率定义为 P(学生=真值｜教师≠真值)；错误模仿率为 P(学生=教师｜教师≠真值)；教师正确区域的保持率为 P(学生=真值｜教师=真值)。以下均为 c=4 减 c=1 的平均变化，单位为百分点。', '',
        '| 数据集 | 模式 | 错误区域纠正率 | 错误模仿率 | 教师正确区域保持率 |','|---|---|---:|---:|---:|']
    for dataset,ds in a['datasets'].items():
        lookup={r['condition']:r for r in ds['regional_summary']}
        for mode in ['teacher','shared']:
            values=[lookup[mode+'_c4']['metrics'][m]['mean']-lookup[mode+'_c1']['metrics'][m]['mean'] for m in ['correction_rate','wrong_imitation_rate','teacher_correct_retention']]
            lines.append(f'| {DS[dataset]} | {mode.title()} | '+' | '.join(f'{x:+.3f}' for x in values)+' |')
    lines+=['',
        'VOC 的两种模式在 c=4 下都减少了对错误教师的模仿，同时降低了教师正确区域的保持率。因此，VOC 的整体退步不能简单解释为“复制了更多教师错误”。区域指标的分母不同，且它们属于像素准确率分解，不能直接当成 mIoU 的可加分解。', '',
        '对于 Teacher 模式，像素准确率的精确计数分解如下：', '',
        '| 数据集 | 教师错误区域的贡献 | 教师正确区域的贡献 | 总像素准确率变化 |','|---|---:|---:|---:|']
    for dataset,ds in a['datasets'].items():
        r=next(x for x in ds['pixel_accuracy_decomposition'] if x['mode']=='teacher')
        lines.append(f"| {DS[dataset]} | {r['teacher_wrong_region_contribution_pp']:+.4f} pp | {r['teacher_correct_region_contribution_pp']:+.4f} pp | {r['total_pixel_accuracy_delta_pp']:+.4f} pp |")
    lines+=['',
        f'![各条件在教师正确和错误区域的表现]({LOCAL}/figures/teacher_regions.png)','',
        '## 中间检查点诊断','',
        '每套数据使用三个 CE 种子的 20k、40k、60k 检查点。每个种子固定 16 对训练图像小批次，每批 2 张；两批图像互不重叠，三个种子共使用 192 张不同训练图像，未使用验证图像选择条件。同一批次及增强跨条件和检查点复用。', '',
        '每次虚拟更新均恢复原 SGD 动量、学生参数和模型缓冲区，冻结 BN 与 dropout，直接反向传播完整 CE+c×KD 目标。诊断使用 FP32 并关闭两种 TF32，以减少数值近似对微小差值的影响。它不重放训练模式或完整 KD 轨迹。', '',
        '主指标为更新后另一批图像的 CE 减去对应 CE-only 更新后的 CE。下表进一步比较 c=4 与 c=1，使用检查点保存的原学习率；正值表示 c=4 的局部 CE 更高。数字为三种子均值，单位为 CE 损失。', '',
        '| 数据集 | 模式 | 20k | 40k | 60k |','|---|---|---:|---:|---:|']
    positive=0
    for dataset,ds in a['datasets'].items():
        for mode in ['teacher','shared']:
            rows=[next(x for x in ds['probe']['contrasts'] if x['checkpoint']==step and x['lr_multiplier']==1 and x['contrast']==f'{mode}_c4 - {mode}_c1') for step in [20000,40000,60000]]
            positive+=sum(x['mean']>0 for x in rows)
            lines.append(f'| {DS[dataset]} | {mode.title()} | '+' | '.join(f"{x['mean']:+.5f}" for x in rows)+' |')
    lines+=['',f'上述 12 个模式与阶段组合中，{positive} 个组合的平均局部 CE 在 c=4 下更高。每个均值仍只有三个训练种子；不同阶段不是新增独立重复。完整记录包括原学习率及其十分之一，并保留各种子的值。', '',
        '梯度范数控制仅将 Shared 的 KD 参数梯度范数匹配到 Teacher c=1，不改变 CE 或已有动量。Teacher c=1 自身就是参照，不重复计作一个独立条件。下表为原学习率下 Teacher c=1 减去范数匹配 Shared 的局部指标，正值表示 Teacher 的 CE 更高。', '',
        '| 数据集 | 20k | 40k | 60k |','|---|---:|---:|---:|']
    for dataset,ds in a['datasets'].items():
        values=[next(x for x in ds['probe']['contrasts'] if x['checkpoint']==step and x['lr_multiplier']==1 and x['contrast']=='teacher_c1 - shared_norm') for step in [20000,40000,60000]]
        lines.append(f'| {DS[dataset]} | '+' | '.join(f"{x['mean']:+.5f} ± {x['sample_sd']:.5f}" for x in values)+' |')
    lines+=['',f'![局部更新诊断与学习率敏感性]({LOCAL}/figures/local_update_diagnostics.png)','',
        '局部诊断表明更新尺度会明显影响 CE 变化，但一次更新的 CE 与 80k 后的 mIoU 是不同指标，且学生起点是 CE 模型。它适合排查尺度和方向差异，不能直接作为完整训练温度或系数的选择器。尤其不能根据局部 CE 变差就否定已经测得的 Cityscapes c=4 最终收益。', '',
        '## 后续实验安排','',
        '1. 保留全部已完成种子和完整二乘二对照，不重跑低分种子、不重新扫描旧温度网格。当前新增推理和局部诊断均已完成，无后续训练排队。',
        '2. 后续新方法在 Cityscapes 上应与已有 Teacher c=4、Shared c=4 对照，在 VOC 上至少同时对照 CE 和 Teacher c=1。这样可以避免只选择较弱的 KD 基线。',
        '3. 把当前结论限定为温度放置与损失尺度在两套固定协议下的行为，避免“Teacher-only 普遍更好”“增大系数普遍有害”以及仅由数据集造成反转的表述。',
        '4. 若论文继续以完整自适应 CoVar 为主方法，主要缺口是该方法本身的同协议主表、机制对照和计算开销，而不是重做现有 scalar 对照。方法冻结且训练配置完全一致时，单个新方法的两数据集三种子主表新增量为 6 次训练；当前未安排这些训练，也不自动重启此前未通过门槛的自适应分支。', '',
        '## 验证与复现','',
        f"最终检查确认 {audit['unchanged_input_files']} 个已登记输入文件及源文件的哈希未改变；30 个学生和 2 个教师的逐图统计均完整，混淆矩阵与区域计数一致。两个数据集各有 144 对检查点批次诊断、1440 条 KD 条件记录，合计 2880 条。计入 CE-only 分支，共执行 3456 次随后恢复状态的正式虚拟更新，工程检查不计入这些数目。",
        '', '工程检查曾发现 Shared 分支需要将有效掩码按最近邻缩放到输出网格，已在正式诊断开始前修正，并通过解析 KL 梯度及状态恢复检查；旧脚本和失败记录保存在 engineering 目录。原推理结果和训练检查点未因此重跑或改写。', '',
        f"新增分析产物约 {audit['analysis_output_bytes']/1024**2:.1f} MiB，主要是教师类别预测缓存和逐图统计；未保存全量 logits。运行始终限定 GPU 0、2，最多两卡。", '',
        f'- [完整统计与种子差值]({LOCAL}/analysis_summary.json)',
        f'- [最终完整性检查]({LOCAL}/final_audit.json)',
        f'- [冻结输入与抽样协议]({LOCAL}/registration.json)',
        f'- [原训练日志检查]({LOCAL}/training_log_audit.json)',
        f'- [VOC 原始诊断记录]({LOCAL}/probe/voc/raw_results.json)',
        f'- [Cityscapes 原始诊断记录]({LOCAL}/probe/cityscapes/raw_results.json)', '',
        '服务器完整缓存与统计目录：`/data1/youwu/jingshi/kd/CoVar-KD-3090/reports/P13_CrossDataset_Posthoc_20261015`。本地保留报告、图表、逐图统计和诊断记录；教师预测缓存留在服务器复用。','']
    (OUT/'analysis_report_zh.md').write_text('\n'.join(lines))
    print('REPORT_READY',OUT/'analysis_report_zh.md')

if __name__=='__main__':main()
