#!/usr/bin/env python3
"""Restored-state virtual SGD probes on existing CE checkpoints, never training."""
import argparse
import copy
import datetime as dt
import importlib
import json
import logging
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
from evaluate import ROOT,OUT,write,sha,gpu_setup

def run(dataset,gpu,smoke):
    import numpy as np
    import torch
    import torch.nn.functional as F
    registration=json.loads((OUT/'registration.json').read_text());ds=registration['datasets'][dataset]
    work=OUT/('probe_smoke' if smoke else 'probe')/dataset;work.mkdir(parents=True,exist_ok=True)
    if (work/'summary.json').exists():raise RuntimeError('Completed probe preserved')
    if not smoke:
        gate=json.loads((OUT/'probe_smoke'/dataset/'summary.json').read_text())
        assert gate['status']=='PASS' and gate['source_sha256']==sha(__file__)
    torch.set_num_threads(1);torch.manual_seed(20261014)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0,ds['runtime']);os.chdir(ds['runtime'])
    module=importlib.import_module('train_kd_3090');module.logger=logging.getLogger('p13-probe')
    from utils.teacher_only_kd import teacher_target_kd_loss
    from utils.rtc_temperature import masked_temperature_kd_loss
    ce_runs={r['seed']:r for r in ds['runs'] if r['condition']=='ce'}
    args=argparse.Namespace(**ce_runs[1234]['args'])
    args.batch_size=2;args.workers=0;args.max_iterations=1;args.resume=None
    args.student_pretrained_base='None';args.student_pretrained='None'
    args.log_dir=str(work/'constructor_logs');args.save_dir=str(work/'unused_checkpoints')
    args.device='cuda';args.device_type='cuda';args.distributed=False;args.num_gpus=1;args.local_rank=0
    trainer=module.Trainer(args)
    trainer.t_model.eval();trainer.s_model.eval();trainer.criterion_cwd.eval();trainer.criterion_fitnet.eval()
    if dataset=='voc':
        from dataset.voc import VOCDataTrainSet
        data=VOCDataTrainSet(ds['data'],ds['train_list'],max_iters=None,crop_size=tuple(args.crop_size),scale=True,mirror=True,ignore_label=-1)
    else:
        from dataset.cityscapes import CSTrainValSet
        data=CSTrainValSet(ds['data'],ds['train_list'],max_iters=None,crop_size=tuple(args.crop_size),scale=True,mirror=True,ignore_label=-1)
    parameters=tuple(trainer.s_model.parameters());parameter_names=[n for n,_ in trainer.s_model.named_parameters()]
    samples=ds['samples'];seeds=[1234] if smoke else registration['seeds'];steps=[20000] if smoke else [20000,40000,60000]
    n_pairs=1 if smoke else 16;rows=[];baselines=[];integrity=[];started=time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    def kd(logits,teacher,targets,mode):
        valid=targets!=-1
        if valid.shape[-2:]!=logits.shape[-2:]:
            valid=F.interpolate(valid.float().unsqueeze(1),size=logits.shape[-2:],mode='nearest').squeeze(1)>.5
        if mode=='teacher':return teacher_target_kd_loss(logits,teacher,2.0,valid)
        temperature=torch.full((logits.shape[0],*logits.shape[-2:]),2.0,dtype=logits.dtype,device=logits.device)
        return masked_temperature_kd_loss(logits,teacher,temperature,valid,temperature_power=0.0)

    def norm(gs):
        return float(torch.stack([g.detach().double().square().sum() for g in gs if g is not None]).sum().sqrt())

    def cosine(a,b):
        return float(torch.stack([(x.detach().double()*y.detach().double()).sum() for x,y in zip(a,b) if x is not None and y is not None]).sum())/max(norm(a)*norm(b),1e-30)

    def equal(a,b):
        if torch.is_tensor(a):return torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b

    def batch(indices,seed):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        items=[data[i] for i in indices]
        x=torch.stack([torch.from_numpy(i[0]) for i in items]).cuda()
        y=torch.stack([torch.from_numpy(i[1]).long() for i in items]).cuda()
        assert bool(((y==-1)|((y>=0)&(y<ds['classes']))).all()) and bool((y!=-1).any())
        return x,y,[i[2] for i in items]

    # Loss/gradient formula tests, with ignore masks and native-grid mask resize.
    torch.manual_seed(20261014)
    t=torch.randn(2,ds['classes'],5,7,dtype=torch.float64,device='cuda')
    target=torch.ones(2,10,14,dtype=torch.int64,device='cuda');target[:,0:3,0:4]=-1
    mask=F.interpolate((target!=-1).float().unsqueeze(1),size=(5,7),mode='nearest').squeeze(1).bool()
    for mode,ts in [('teacher',1.0),('shared',2.0)]:
        for coefficient in [1.0,4.0]:
            z=torch.randn_like(t,requires_grad=True);loss=coefficient*kd(z,t,target,mode)
            q=F.softmax(t/2,1);prob=F.softmax(z/ts,1)
            manual=coefficient*(q*(q.log()-F.log_softmax(z/ts,1))).sum(1)[mask].mean()
            gradient=torch.autograd.grad(loss,z)[0]
            expected=coefficient*(prob-q)*mask[:,None]/mask.sum()/ts
            error=float((gradient-expected).abs().max())
            assert torch.allclose(loss,manual,atol=1e-12,rtol=1e-10) and error<1e-12
            integrity.append(dict(check='masked_KL_formula',mode=mode,coefficient=coefficient,max_gradient_error=error))
    del z,t,target,loss,prob,q,gradient,expected,manual
    for seed in seeds:
        for step in steps:
            path=ce_runs[seed]['checkpoints'][str(step)]
            assert sha(path)==ds['input_sha256'][path]
            state=torch.load(path,map_location='cpu')
            assert state['iteration']==step and state['args']['seed']==seed and state['args']['lambda_kd']==0
            trainer.s_model.load_state_dict(state['student'],strict=True)
            for key in ['criterion_cwd','criterion_fitnet']:getattr(trainer,key).load_state_dict(state[key],strict=True)
            trainer.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
            module.move_optimizer_state_to_device(trainer.optimizer,trainer.device)
            initial={key:{k:v.detach().clone() for k,v in getattr(trainer,key).state_dict().items()} for key in ['s_model','criterion_cwd','criterion_fitnet']}
            initial_opt=copy.deepcopy(trainer.optimizer.state_dict())
            assert all(g['lr']>0 and g['momentum']==.9 and g['weight_decay']==.0001 for g in initial_opt['param_groups'])
            # Every optimizer tensor must match its parameter shape after loading.
            for group in trainer.optimizer.param_groups:
                for p in group['params']:
                    for value in trainer.optimizer.state[p].values():
                        if torch.is_tensor(value):assert value.shape==p.shape and bool(torch.isfinite(value).all())

            def restore():
                for key,value in initial.items():getattr(trainer,key).load_state_dict(value,strict=True)
                trainer.optimizer.load_state_dict(copy.deepcopy(initial_opt));trainer.optimizer.zero_grad(set_to_none=True)

            for pi in range(n_pairs):
                dest=work/'pairs'/f'{seed}_{step}_{pi:02d}.json'
                if dest.exists():
                    old=json.loads(dest.read_text());assert old['status']=='PASS' and old['source_sha256']==sha(__file__)
                    rows.extend(old['rows']);baselines.extend(old['baselines']);continue
                chosen=samples[str(seed)][pi]
                x,y,names=batch(chosen['same'],chosen['augmentation_seed'])
                nx,ny,next_names=batch(chosen['next'],chosen['augmentation_seed']+1)
                assert not set(names)&set(next_names)
                restore()
                with torch.no_grad():
                    tlog=trainer.t_model(x)[0]
                    before_same=float(trainer.criterion(trainer.s_model(x)[0],y))
                    before_next=float(trainer.criterion(trainer.s_model(nx)[0],ny))
                logits=trainer.s_model(x)[0];ce=trainer.criterion(logits,y)
                cgrad=tuple(None if g is None else g.detach() for g in torch.autograd.grad(ce,parameters,retain_graph=True,allow_unused=True))
                gradients={};losses={}
                for mode in ['teacher','shared']:
                    loss=kd(logits,tlog,y,mode);losses[mode]=float(loss.detach())
                    gradients[mode]=tuple(None if g is None else g.detach() for g in torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True))
                norms={mode:norm(g) for mode,g in gradients.items()};cnorm=norm(cgrad)
                assert all(math.isfinite(v) and v>0 for v in [cnorm,*norms.values()])
                scale=norms['teacher']/norms['shared'];cosines={m:cosine(cgrad,g) for m,g in gradients.items()}
                mode_cosine=cosine(gradients['teacher'],gradients['shared'])
                # No gradient arithmetic is used for actual updates; each branch
                # recomputes the complete CE+c*KD objective with autograd.
                del logits,ce,loss,cgrad,gradients
                pair_rows=[];pair_baselines=[]
                for lr_multiplier in [1.0,.1]:
                    for condition,mode,coefficient in [('ce',None,0.),('teacher_c1','teacher',1.),('teacher_c4','teacher',4.),
                        ('shared_c1','shared',1.),('shared_c4','shared',4.),('shared_norm','shared',scale)]:
                        restore()
                        for g in trainer.optimizer.param_groups:g['lr']*=lr_multiplier
                        output=trainer.s_model(x)[0];objective=trainer.criterion(output,y)
                        if mode is not None:objective=objective+coefficient*kd(output,tlog,y,mode)
                        assert bool(torch.isfinite(objective));objective.backward();trainer.optimizer.step()
                        del output,objective
                        with torch.no_grad():
                            same=float(trainer.criterion(trainer.s_model(x)[0],y));nxt=float(trainer.criterion(trainer.s_model(nx)[0],ny))
                            update_norm=math.sqrt(sum(float((p-initial['s_model'][name]).double().square().sum()) for name,p in zip(parameter_names,parameters)))
                        assert math.isfinite(same+nxt+update_norm)
                        if condition=='ce':
                            ce_same,ce_next=same,nxt
                            pair_baselines.append(dict(seed=seed,checkpoint=step,pair=pi,lr_multiplier=lr_multiplier,
                                before_same_ce=before_same,before_next_ce=before_next,after_same_ce=same,after_next_ce=nxt,update_norm=update_norm))
                        else:
                            pair_rows.append(dict(dataset=dataset,seed=seed,checkpoint=step,pair=pi,condition=condition,
                                same_names=names,next_names=next_names,lr_multiplier=lr_multiplier,saved_lr=initial_opt['param_groups'][0]['lr'],
                                coefficient=coefficient,kd_loss=losses[mode],ce_gradient_norm=cnorm,kd_gradient_norm=norms[mode],
                                effective_kd_ce_norm_ratio=coefficient*norms[mode]/cnorm,kd_ce_cosine=cosines[mode],teacher_shared_cosine=mode_cosine,
                                before_same_ce=before_same,before_next_ce=before_next,ce_only_same_ce=ce_same,ce_only_next_ce=ce_next,
                                same_ce=same,next_ce=nxt,same_delta_vs_ce=same-ce_same,next_delta_vs_ce=nxt-ce_next,update_norm=update_norm))
                restore()
                assert all(equal(getattr(trainer,k).state_dict(),v) for k,v in initial.items())
                assert equal(trainer.optimizer.state_dict(),initial_opt)
                record=dict(status='PASS',source_sha256=sha(__file__),checkpoint_sha256=ds['input_sha256'][path],rows=pair_rows,
                    baselines=pair_baselines,exact_parameter_buffer_optimizer_restore=True)
                write(dest,record);rows.extend(pair_rows);baselines.extend(pair_baselines)
                write(work/'state.json',dict(status='RUNNING',dataset=dataset,gpu=gpu,seed=seed,checkpoint=step,pair=pi,
                    completed_pairs=len(rows)//10,total_pairs=len(seeds)*len(steps)*n_pairs,elapsed_seconds=time.monotonic()-started,pid=os.getpid()))
                print('PROBE_PAIR_PASS',dataset,seed,step,pi,flush=True)
                del x,y,nx,ny,tlog
            assert sha(path)==ds['input_sha256'][path]
            del state,initial,initial_opt
    summaries=[];contrasts=[]
    for step in steps:
        for lm in [1.0,.1]:
            for condition in ['teacher_c1','teacher_c4','shared_c1','shared_c4','shared_norm']:
                group=[r for r in rows if (r['checkpoint'],r['lr_multiplier'],r['condition'])==(step,lm,condition)]
                seed_means={s:statistics.mean(r['next_delta_vs_ce'] for r in group if r['seed']==s) for s in seeds}
                summaries.append(dict(checkpoint=step,lr_multiplier=lm,condition=condition,seed_means=seed_means,
                    mean=statistics.mean(seed_means.values()),sample_sd=statistics.stdev(seed_means.values()) if len(seeds)>1 else None,
                    mean_effective_kd_ce_norm_ratio=statistics.mean(r['effective_kd_ce_norm_ratio'] for r in group),
                    mean_kd_ce_cosine=statistics.mean(r['kd_ce_cosine'] for r in group)))
            for a,b in [('teacher_c4','teacher_c1'),('shared_c4','shared_c1'),('teacher_c1','shared_c1'),('teacher_c4','shared_c4'),('teacher_c1','shared_norm')]:
                lookup={(r['seed'],r['pair'],r['condition']):r['next_delta_vs_ce'] for r in rows if r['checkpoint']==step and r['lr_multiplier']==lm}
                values={s:statistics.mean(lookup[s,i,a]-lookup[s,i,b] for i in range(n_pairs)) for s in seeds}
                contrasts.append(dict(checkpoint=step,lr_multiplier=lm,contrast=a+' - '+b,seed_means=values,
                    mean=statistics.mean(values.values()),sample_sd=statistics.stdev(values.values()) if len(seeds)>1 else None))
    assert len(rows)==len(seeds)*len(steps)*n_pairs*10
    write(work/'raw_results.json',dict(status='PASS',rows=rows,baselines=baselines))
    summary=dict(status='PASS',dataset=dataset,source_sha256=sha(__file__),registration_sha256=sha(OUT/'registration.json'),
        rows=len(rows),seeds=seeds,checkpoints=steps,pairs_per_seed=n_pairs,integrity=integrity,summaries=summaries,contrasts=contrasts,
        exact_restore=True,elapsed_seconds=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        completed_at=dt.datetime.now(dt.timezone.utc).isoformat())
    write(work/'summary.json',summary);write(work/'state.json',dict(status='PASS',rows=len(rows),elapsed_seconds=summary['elapsed_seconds']))
    print('PROBE_COMPLETE',dataset,'smoke',smoke,len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['voc','cityscapes'],required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();lock=gpu_setup(a.gpu)
    try:run(a.dataset,a.gpu,a.smoke)
    except Exception as e:
        write(OUT/('probe_smoke' if a.smoke else 'probe')/a.dataset/'failure.json',dict(status='FAILED',error=repr(e),source_sha256=sha(__file__)))
        raise
