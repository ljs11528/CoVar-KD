#!/usr/bin/env python3
"""Seed-level statistics and standalone research figures from frozen results."""
import argparse
import datetime as dt
import json
from pathlib import Path
import statistics
from evaluate import ROOT,OUT,write,sha

CONDITIONS=['ce','teacher_c1','shared_c1','teacher_c4','shared_c4']
LABELS=['CE','Teacher c=1','Shared c=1','Teacher c=4','Shared c=4']
COLORS=['#667085','#175CD3','#12A594','#B42318','#E68A00']
CLASSES={
 'voc':['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'],
 'cityscapes':['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle']}
PAIRS=[('teacher_c1','ce'),('shared_c1','ce'),('teacher_c4','ce'),('shared_c4','ce'),
       ('teacher_c4','teacher_c1'),('shared_c4','shared_c1'),('teacher_c1','shared_c1'),('teacher_c4','shared_c4')]

def figures_p0(reg,p0):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(dest/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(dest/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for ax,(dataset,ds) in zip(axes,reg['datasets'].items()):
        rows={r['condition']:r for r in p0['conditions'] if r['dataset']==dataset}
        vals=np.array([[rows[c]['by_seed'][str(s)] for s in reg['seeds']] for c in CONDITIONS])
        x=np.arange(5)
        for j,color in enumerate(COLORS):ax.errorbar(j,vals.mean(1)[j],yerr=vals.std(1,ddof=1)[j],fmt='D',color=color,capsize=5,markersize=7)
        for i,s in enumerate(reg['seeds']):ax.scatter(x+(i-1)*.14,vals[:,i],s=17,color='black',alpha=.65,zorder=3)
        ax.set_xticks(x,LABELS,rotation=25,ha='right');ax.set_ylabel('Final 80k mIoU (%)');ax.set_title(dataset.upper() if dataset=='voc' else 'Cityscapes')
        ax.set_ylim(58 if dataset=='voc' else 69,75);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Matched 3090 cohorts: mean ± sample SD, three training seeds',y=1.03);fig.tight_layout();save(fig,'final_results')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for ax,(dataset,ds) in zip(axes,reg['datasets'].items()):
        for condition,label,color in zip(CONDITIONS,LABELS,COLORS):
            rows=[r for r in ds['runs'] if r['condition']==condition]
            values=np.array([[v['miou_percent'] for v in r['validations']] for r in rows]);mean=values.mean(0);sd=values.std(0,ddof=1)
            ax.plot([20,40,60,80],mean,'o-',color=color,label=label,markersize=4);ax.fill_between([20,40,60,80],mean-sd,mean+sd,color=color,alpha=.09)
        ax.set_title(dataset.upper() if dataset=='voc' else 'Cityscapes');ax.set_xlabel('Training iteration (thousands)');ax.set_ylabel('Validation mIoU (%)');ax.grid(alpha=.15)
    axes[1].legend(fontsize=8);fig.tight_layout();save(fig,'validation_curves')
    return save

def main(p0_only):
    import numpy as np
    reg=json.loads((OUT/'registration.json').read_text());p0=json.loads((OUT/'p0_summary.json').read_text())
    save=figures_p0(reg,p0)
    if p0_only:return
    import matplotlib.pyplot as plt
    result=dict(status='PASS',created_at=dt.datetime.now(dt.timezone.utc).isoformat(),p0=p0,datasets={},
        registration_sha256=sha(OUT/'registration.json'))
    for dataset,ds in reg['datasets'].items():
        p=OUT/'inference'/dataset
        assert json.loads((p/'state.json').read_text())['status']=='PASS'
        teacher=json.loads((p/'teacher.json').read_text())
        runs=[json.loads((p/(r['id']+'.json')).read_text()) for r in ds['runs']]
        assert len(runs)==15 and all(r['status']=='PASS' for r in runs)
        lookup={(r['condition'],r['seed']):r for r in runs}
        columns=['correction_rate','wrong_imitation_rate','teacher_correct_retention','pixacc_percent']
        regions=[]
        for condition in CONDITIONS:
            summary=dict(condition=condition,metrics={})
            for metric in columns:
                multiplier=1 if metric=='pixacc_percent' else 100
                values={s:lookup[condition,s][metric]*multiplier for s in reg['seeds']}
                delta={s:(lookup[condition,s][metric]-lookup['ce',s][metric])*multiplier for s in reg['seeds']}
                summary['metrics'][metric]=dict(by_seed=values,mean=statistics.mean(values.values()),sample_sd=statistics.stdev(values.values()),
                    paired_ce_delta_pp=statistics.mean(delta.values()),paired_ce_delta_by_seed=delta)
            regions.append(summary)
        for r in runs:
            w=r['teacher_wrong_pixels'];c=r['teacher_correct_pixels']
            predicted_accuracy=(w*r['correction_rate']+c*r['teacher_correct_retention'])/(w+c)*100
            assert abs(predicted_accuracy-r['pixacc_percent'])<1e-10
        error_fraction=runs[0]['teacher_wrong_pixels']/(runs[0]['teacher_wrong_pixels']+runs[0]['teacher_correct_pixels'])
        region_lookup={r['condition']:r['metrics'] for r in regions}
        decompositions=[]
        for mode in ['teacher','shared']:
            low,high=region_lookup[mode+'_c1'],region_lookup[mode+'_c4']
            wrong=error_fraction*(high['correction_rate']['mean']-low['correction_rate']['mean'])
            correct=(1-error_fraction)*(high['teacher_correct_retention']['mean']-low['teacher_correct_retention']['mean'])
            total=high['pixacc_percent']['mean']-low['pixacc_percent']['mean']
            assert abs(wrong+correct-total)<1e-10
            decompositions.append(dict(mode=mode,teacher_wrong_region_contribution_pp=wrong,
                teacher_correct_region_contribution_pp=correct,total_pixel_accuracy_delta_pp=total))
        class_contrasts=[]
        for a,b in PAIRS:
            values=np.array([np.array(lookup[a,s]['class_iou_percent'])-np.array(lookup[b,s]['class_iou_percent']) for s in reg['seeds']])
            mean=values.mean(0);sd=values.std(0,ddof=1)
            class_rows=[dict(name=name,mean_delta_pp=float(mean[i]),sample_sd=float(sd[i]),
                by_seed={s:float(values[j,i]) for j,s in enumerate(reg['seeds'])},positive_seeds=int((values[:,i]>0).sum())) for i,name in enumerate(CLASSES[dataset])]
            class_contrasts.append(dict(contrast=a+' - '+b,classes=class_rows,positive_mean_classes=int((mean>0).sum()),
                all_seed_positive_classes=int((values>0).all(0).sum()),mean_miou_delta_pp=float(mean.mean()),
                largest_gains=sorted(class_rows,key=lambda x:x['mean_delta_pp'],reverse=True)[:5],
                largest_losses=sorted(class_rows,key=lambda x:x['mean_delta_pp'])[:5]))
        probe=json.loads((OUT/'probe'/dataset/'summary.json').read_text())
        assert probe['status']=='PASS' and probe['rows']==1440
        assert all(len(x['seed_means'])==3 for x in probe['summaries'])
        # Counts/intervals are computed across training seeds, never pixels.
        result['datasets'][dataset]=dict(teacher={k:v for k,v in teacher.items() if k!='prediction_file_sha256'},
            runs=runs,regional_summary=regions,class_contrasts=class_contrasts,probe=probe,
            max_miou_reproduction_error_pp=max(abs(r['reproduction_delta_pp']) for r in runs),
            total_inference_seconds=teacher['seconds']+sum(r['seconds'] for r in runs),
            teacher_error_fraction=error_fraction,pixel_accuracy_decomposition=decompositions,
            class_ground_truth_pixel_fraction={name:float(np.asarray(teacher['confusion_matrix']).sum(1)[i]/np.asarray(teacher['confusion_matrix']).sum()) for i,name in enumerate(CLASSES[dataset])})
    write(OUT/'analysis_summary.json',result)
    fig,axes=plt.subplots(1,2,figsize=(12,8))
    for ax,(dataset,ds) in zip(axes,result['datasets'].items()):
        for offset,contrast,color,label in [(-.18,'teacher_c4 - teacher_c1',COLORS[3],'Teacher'),(.18,'shared_c4 - shared_c1',COLORS[4],'Shared')]:
            row=next(x for x in ds['class_contrasts'] if x['contrast']==contrast)
            values=[x['mean_delta_pp'] for x in row['classes']];errors=[x['sample_sd'] for x in row['classes']]
            ax.barh(np.arange(len(values))+offset,values,height=.34,xerr=errors,color=color,alpha=.8,label=label,error_kw=dict(elinewidth=.7,capsize=2))
        ax.set_yticks(np.arange(len(CLASSES[dataset])),CLASSES[dataset]);ax.invert_yaxis();ax.axvline(0,color='black',linewidth=.8)
        ax.set_xlabel('Class IoU change: c=4 minus c=1 (pp)');ax.set_title(dataset.upper() if dataset=='voc' else 'Cityscapes');ax.legend();ax.grid(axis='x',alpha=.12)
    fig.tight_layout();save(fig,'class_coefficient_effects')
    fig,axes=plt.subplots(2,3,figsize=(12,7.4))
    for row,(dataset,ds) in zip(axes,result['datasets'].items()):
        for ax,metric,title in zip(row,columns[:3],['Correction on teacher errors','Imitation on teacher errors','Accuracy where teacher is correct']):
            for i,c in enumerate(CONDITIONS):
                r=next(x for x in ds['regional_summary'] if x['condition']==c)['metrics'][metric]
                ax.errorbar(i,r['mean'],yerr=r['sample_sd'],fmt='o',color=COLORS[i],capsize=3)
            ax.set_xticks(range(5),['CE','T1','S1','T4','S4']);ax.set_title(title,fontsize=10);ax.set_ylabel(dataset+' (%)');ax.grid(alpha=.15)
    fig.tight_layout();save(fig,'teacher_regions')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for row,(dataset,ds) in zip(axes,result['datasets'].items()):
        for ax,lm in zip(row,[1.0,.1]):
            for condition,color,label in zip(CONDITIONS[1:]+['shared_norm'],COLORS[1:]+['#9333EA'],LABELS[1:]+['Shared norm matched']):
                groups=[next(r for r in ds['probe']['summaries'] if (r['condition'],r['checkpoint'],r['lr_multiplier'])==(condition,s,lm)) for s in [20000,40000,60000]]
                ax.errorbar([20,40,60],[r['mean'] for r in groups],yerr=[r['sample_sd'] for r in groups],fmt='o-',color=color,capsize=3,label=label,markersize=3)
            ax.axhline(0,color='black',linewidth=.7);ax.set_title(f'{dataset}; saved LR × {lm}');ax.set_xlabel('CE checkpoint iteration (thousands)');ax.set_ylabel('Next-batch CE minus CE-only update');ax.grid(alpha=.15)
    axes[0,1].legend(fontsize=8);fig.tight_layout();save(fig,'local_update_diagnostics')
    print(json.dumps({'status':'PASS','runs':30,'probe_rows':2880,
        'max_error_pp':{d:x['max_miou_reproduction_error_pp'] for d,x in result['datasets'].items()}},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--p0-only',action='store_true');a=p.parse_args();main(a.p0_only)
