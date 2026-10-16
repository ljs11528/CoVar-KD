"""Fixed P2 streams, workload and decisions; no validation scores as inputs."""
import hashlib
import itertools
import math
import random
import statistics

DATASETS=('voc','cityscapes')
SEEDS=(1234,2025,3407)
SOURCES=('ce','shared_c1')
COEFFICIENTS=(0,1,4)
HORIZONS=(1,16,128)
STREAM_SEEDS=(2026101601,2026101602,2026101603,2026101604)

def stream(eligible,batch,dataset,seed,flow,steps):
    """Per-step paired draws, excluding A/B; same stream for CE and KD sources."""
    if flow not in range(4) or batch>len(eligible) or steps<1:raise ValueError('Invalid stream')
    key=f'P2:{dataset}:{seed}:20000:{STREAM_SEEDS[flow]}'
    derived=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')%(2**31)
    rng=random.Random(derived)
    return [dict(indices=rng.sample(eligible,batch),augmentation_seed=(derived+131*i)%(2**31),
                 rng_seed=(derived+90000000+131*i)%(2**31)) for i in range(steps)]

def state_id(d,seed,source):return f'{d}_{seed}_{source}'

def tasks():
    probes=[];forks=[]
    for d,s,source in itertools.product(DATASETS,SEEDS,SOURCES):
        sid=state_id(d,s,source)
        for c in COEFFICIENTS:
            for flow in range(3):
                for mode in ('train','eval'):
                    probes.append(dict(id=f'probe_{sid}_c{c}_flow{flow}_{mode}',kind='update',phase='probe',
                        dataset=d,seed=s,source=source,coefficient=c,stream=flow,mode=mode,
                        steps=128 if mode=='train' else 1,horizons=list(HORIZONS) if mode=='train' else [1],repeat=0))
            probes.append(dict(id=f'replay_{sid}_c{c}_flow0_train',kind='update',phase='probe',dataset=d,seed=s,
                source=source,coefficient=c,stream=0,mode='train',steps=128,horizons=list(HORIZONS),repeat=1))
            forks.append(dict(id=f'fork_{sid}_c{c}',kind='update',phase='fork',dataset=d,seed=s,source=source,
                coefficient=c,stream=3,mode='train',steps=2000,horizons=[2000],repeat=0))
    return probes,forks

def counts():
    probes,forks=tasks()
    return dict(source_states=12,train_rollouts=144,eval_one_step=108,probe_endpoints=540,
                baseline_endpoints=12,actual_2k_forks=36,validation_endpoints=36,
                rollout_optimizer_steps=sum(t['steps'] for t in probes),fork_optimizer_steps=72000,
                full_80k_trajectories=0)

def best_candidate(values,metric,previous,epsilon):
    if set(values)!={0,1,4} or any(not math.isfinite(v) for v in values.values()):raise ValueError('Incomplete/nonfinite candidates')
    best=min(values.values()) if metric.endswith('ce') else max(values.values())
    tied=[c for c,v in values.items() if abs(v-best)<=epsilon]
    return previous if previous in tied else min(tied)

def decisions(rows,p0_bands,dataset,source):
    """All three streams and explicit repeat1 are mandatory; no val input."""
    lookup={(r['mode'],r['coefficient'],r['stream'],r['repeat'],r['horizon']):r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('Duplicate probe endpoints')
    previous=0 if source=='ce' else 1
    numerical={}
    for h in HORIZONS:
        for metric in ('pixel_ce','class_ce','soft_miou','hard_miou'):
            spans=[]
            for c in COEFFICIENTS:
                a=lookup[('train',c,0,0,h)]['scores'][metric]
                b=lookup[('train',c,0,1,h)]['scores'][metric]
                spans.append(abs(a-b))
            numerical[f'{h}:{metric}']=max(p0_bands[f'{dataset}:train:{metric}'],5*max(spans),1e-6)
    rules={}
    for name,mode,metric in (('eval_pixel_ce_1','eval','pixel_ce'),('train_pixel_ce_1','train','pixel_ce'),('train_soft_miou_1','train','soft_miou')):
        values={c:statistics.mean(lookup[(mode,c,f,0,1)]['scores'][metric] for f in range(3)) for c in COEFFICIENTS}
        eps=p0_bands[f'{dataset}:eval:{metric}'] if mode=='eval' else numerical[f'1:{metric}']
        rules[name]=dict(coefficient=best_candidate(values,metric,previous,eps),candidate_means=values,deadband=eps)
    eps=numerical['128:hard_miou']
    values={c:statistics.mean(lookup[('train',c,f,0,128)]['scores']['hard_miou'] for f in range(3)) for c in COEFFICIENTS}
    margins={c:[lookup[('train',c,f,0,128)]['scores']['hard_miou']-lookup[('train',previous,f,0,128)]['scores']['hard_miou'] for f in range(3)] for c in COEFFICIENTS if c!=previous}
    eligible=[c for c,v in margins.items() if all(x>eps for x in v)]
    chosen=previous
    if eligible:
        best=max(values[c] for c in eligible);chosen=min(c for c in eligible if best-values[c]<=eps)
    rules['train_hard_miou_128_consistent']=dict(coefficient=chosen,previous=previous,deadband=eps,
        eligible=eligible,candidate_means=values,stream_margins=margins,abstained=not eligible)
    return dict(rules=rules,numerical_bands=numerical,previous=previous)

def gate(seed_regret_reductions):
    if set(seed_regret_reductions)!=set(DATASETS) or any(len(v)!=3 for v in seed_regret_reductions.values()):raise ValueError('Three seed aggregates required per dataset')
    rows={d:dict(mean_reduction_pp=statistics.mean(v),improved_seeds=sum(x>0 for x in v),seed_reductions_pp=v) for d,v in seed_regret_reductions.items()}
    return dict(passed=all(r['mean_reduction_pp']>=.10 and r['improved_seeds']>=2 for r in rows.values()),
                datasets=rows,role='Practical P4 resource rule, not a significance test',p4_launched=False)
