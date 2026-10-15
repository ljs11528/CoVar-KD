#!/usr/bin/env python3
"""Register immutable P11/P12 evidence for the approved no-retraining analysis."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import statistics

ROOT = Path('/data1/youwu/jingshi/kd/CoVar-KD-3090')
OUT = ROOT/'reports/P13_CrossDataset_Posthoc_20261015'
SEEDS = [1234, 2025, 3407]
CONDITIONS = ['ce', 'teacher_c1', 'shared_c1', 'teacher_c4', 'shared_c4']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2), b''): h.update(b)
    return h.hexdigest()

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)

def condition(spec):
    if spec['coefficient'] == 0: return 'ce'
    assert spec['temperature'] == 2
    return ('teacher' if spec['mode']=='teacher_only' else 'shared')+'_c'+str(int(spec['coefficient']))

def main():
    import torch
    if (OUT/'registration.json').exists(): raise RuntimeError('Existing registration preserved')
    out = dict(created_at=dt.datetime.now(dt.timezone.utc).isoformat(), status='REGISTERED',
        seeds=SEEDS, conditions=CONDITIONS, datasets={}, source_sha256={},
        protocol=dict(no_full_training=True, max_gpu_count=2, allowed_gpus=[0,2],
            endpoint='final 80000; never best checkpoint', inference_batch_size=1,
            inference_precision='original FP32, cuDNN TF32 on, matmul TF32 off',
            metric='independent int64 confusion matrix; original validation protocol; tolerance 0.001 percentage points',
            correction_rate='P(student==GT | teacher!=GT, valid)',
            wrong_imitation_rate='P(student==teacher | teacher!=GT, valid)',
            teacher_correct_retention='P(student==GT | teacher==GT, valid)',
            probe_checkpoints=[20000,40000,60000], probe_batch_size=2, pairs_per_seed=16,
            probe_sampling_seed=20261014, probe_lr_multipliers=[1.0,0.1],
            probe_conditions=['ce','teacher_c1','teacher_c4','shared_c1','shared_c4','shared_norm'],
            norm_control='shared KD parameter-gradient L2 norm matched to teacher c1; detached coefficient',
            probe_precision='FP32, both TF32 flags off; eval BN/dropout; direct full-objective backward',
            probe_optimizer='actual SGD; original saved momentum restored before every virtual step',
            statistical_unit='3 training seeds; stages and image pairs are repeated measurements, not extra seeds',
            limits=['Different teacher weights, batch sizes, crops and learning rates confound dataset attribution.',
                    'Post hoc diagnostic; no new training, no model or temperature selection on validation.',
                    'CE checkpoint local virtual steps do not replay KD trajectories or establish trained mIoU causality.']))
    for dataset, report_name, runtime_name, count in [
        ('voc','P11_3090_20261002','runtime-large',1449),
        ('cityscapes','P12_Cityscapes_20261008','runtime-cityscapes-large',500)]:
        report=ROOT/'reports'/report_name; runtime=ROOT/runtime_name
        state=json.loads((report/'state.json').read_text())
        results=json.loads((report/'results.json').read_text())
        assert state['status']=='REGISTERED_COMPLETE' and not state['failures']
        selected=[x for x in results['runs'] if x['spec']['student']=='large']
        assert len(selected)==15
        audits=[]
        for p in (ROOT/'runs'/report_name).rglob('audit.json'):
            try: audits.append((p,json.loads(p.read_text())))
            except (ValueError,OSError): pass
        rows=[]; input_hashes={str(report/'results.json'):sha(report/'results.json'),
                               str(report/'state.json'):sha(report/'state.json')}
        for result in selected:
            assert result['status']=='PASS' and all(result['checks'].values())
            assert [v['iteration'] for v in result['validations']]==[20000,40000,60000,80000]
            matches=[(p,a) for p,a in audits if a.get('spec',{}).get('id')==result['spec']['id']
                     and a.get('checkpoint_sha256')==result['checkpoint_sha256'] and a.get('status')=='PASS']
            assert len(matches)==1,(result['spec']['id'],len(matches))
            audit_path,audit=matches[0]; run=audit_path.parent
            latest=run/'checkpoints/training_state_latest.pth'
            assert sha(latest)==result['checkpoint_sha256']
            checkpoints={str(step):str(run/'checkpoints'/f'training_state_iter{step:06d}.pth') for step in [20000,40000,60000,80000]}
            for p in checkpoints.values(): assert Path(p).is_file(),p
            current=torch.load(checkpoints['80000'],map_location='cpu')
            assert current['iteration']==80000 and current['args']['seed']==result['spec']['seed']
            assert all(bool(torch.isfinite(x).all()) for x in current['student'].values())
            frozen_args=current['args']
            for p in [str(audit_path),str(latest),checkpoints['80000']]:input_hashes[p]=sha(p)
            if condition(result['spec'])=='ce':
                for step in ['20000','40000','60000']:input_hashes[checkpoints[step]]=sha(checkpoints[step])
            logs=[p for p in run.rglob('*') if p.is_file() and p.suffix in ['.log','.txt'] and sha(p)==result['log_sha256']]
            assert len(logs)>=1,(str(run),logs)
            input_hashes[str(logs[0])]=result['log_sha256']
            rows.append(dict(id=result['spec']['id'],seed=result['spec']['seed'],condition=condition(result['spec']),
                final_miou_percent=result['final_miou_percent'],validations=result['validations'],
                run_root=str(run),audit=str(audit_path),log=str(logs[0]),checkpoints=checkpoints,args=frozen_args))
            del current
        assert {(x['seed'],x['condition']) for x in rows}=={(s,c) for s in SEEDS for c in CONDITIONS}
        representative=next(x for x in rows if x['condition']=='ce' and x['seed']==1234)
        args=representative['args']; train_list=runtime/('dataset/list/voc/train_aug.txt' if dataset=='voc' else 'dataset/list/cityscapes/train.lst')
        val_list=runtime/('dataset/list/voc/val.txt' if dataset=='voc' else 'dataset/list/cityscapes/val.lst')
        names=train_list.read_text().splitlines(); selected_indices=random.Random(20261014).sample(range(len(names)),192)
        samples={}
        for si,seed in enumerate(SEEDS):
            samples[str(seed)]=[dict(pair=i,same=selected_indices[si*64+i*4:si*64+i*4+2],
                next=selected_indices[si*64+i*4+2:si*64+i*4+4],augmentation_seed=2026101400+si*32+i*2) for i in range(16)]
        for p in [train_list,val_list,Path(args['teacher_pretrained'])]:input_hashes[str(p)]=sha(p)
        assert len(val_list.read_text().splitlines())==count
        for p in runtime.rglob('*.py'):out['source_sha256'][str(p)]=sha(p)
        out['datasets'][dataset]=dict(runtime=str(runtime),report=str(report),data=args['data'],
            classes=21 if dataset=='voc' else 19,val_count=count,train_list=str(train_list),val_list=str(val_list),
            teacher=args['teacher_pretrained'],runs=rows,samples=samples,input_sha256=input_hashes,
            sampled_train_entries={str(i):names[i] for i in selected_indices})
    write(OUT/'registration.json',out)
    summary=[]; contrasts=[]
    pairs=[('teacher_c1','ce'),('shared_c1','ce'),('teacher_c4','ce'),('shared_c4','ce'),
           ('teacher_c1','shared_c1'),('teacher_c4','shared_c4'),('teacher_c4','teacher_c1'),('shared_c4','shared_c1')]
    for dataset,ds in out['datasets'].items():
        vals={(r['condition'],r['seed']):r['final_miou_percent'] for r in ds['runs']}
        for c in CONDITIONS:
            v=[vals[c,s] for s in SEEDS]
            summary.append(dict(dataset=dataset,condition=c,by_seed=dict(zip(SEEDS,v)),mean=statistics.mean(v),sample_sd=statistics.stdev(v),
                delta_vs_ce=statistics.mean(vals[c,s]-vals['ce',s] for s in SEEDS)))
        for a,b in pairs:
            v=[vals[a,s]-vals[b,s] for s in SEEDS]
            contrasts.append(dict(dataset=dataset,contrast=a+' - '+b,by_seed=dict(zip(SEEDS,v)),mean=statistics.mean(v),sample_sd=statistics.stdev(v)))
    write(OUT/'p0_summary.json',dict(status='PASS',conditions=summary,paired_contrasts=contrasts,
        completed_large_runs=30,all_checks_passed=True,registration_sha256=sha(OUT/'registration.json')))
    print(json.dumps({'status':'PASS','runs':30,'report':str(OUT),'summary':summary},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
