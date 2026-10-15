"""Pure protocol/task definitions, independent of CUDA and model construction."""
from __future__ import annotations

import hashlib
import itertools
import random

SEEDS = (1234, 2025, 3407)
STAGES = (20000, 60000)
DATASETS = ("voc", "cityscapes")
STREAM_SEEDS = (2026101601, 2026101602, 2026101603)
PRIMARY = ("pixel_ce", "class_ce", "soft_miou", "hard_miou")


def stream_entry(eligible, batch, dataset, seed, stage, stream):
    text = f"P15:{dataset}:{seed}:{stage}:{STREAM_SEEDS[stream]}"
    derived = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % (2**31)
    rng = random.Random(derived)
    return {"indices": rng.sample(eligible, batch), "augmentation_seed": derived,
            "rng_seed": (derived+90000000) % (2**31), "stream_seed": STREAM_SEEDS[stream]}


def branch(name, bn, dropout, condition, **extra):
    return dict(name=name, bn=bn, dropout=dropout, condition=condition,
                lr_multiplier=1., zero_momentum=False, **extra)


def tasks():
    result=[]
    for d,t,m in itertools.product(DATASETS,STAGES,("eval","train")):
        branches=[branch(f"{c}_r{r}", "batch" if m=="train" else "running",m=="train",c,
                         repeat=r, buffer_policy="native")
                  for c,r in itertools.product(("ce","shared_c1","shared_c4"),range(3))]
        result.append(dict(id=f"p0_{d}_1234_{t}_{m}",phase="p0",dataset=d,seed=1234,stage=t,
                           stream="legacy",mode=m,branches=branches,monitor="A"))
    for d,s,t,flow in itertools.product(DATASETS,SEEDS,STAGES,range(3)):
        branches=[branch(f"factor_{bn}_drop{int(drop)}_{c}",bn,drop,c,buffer_policy="both")
                  for bn,drop,c in itertools.product(("running","batch"),(False,True),("ce","shared_c1","shared_c4"))]
        branches += [branch(f"buffer_only_drop{int(drop)}","batch",drop,"ce",buffer_only=True,
                            buffer_policy="native") for drop in (False,True)]
        if s==1234:
            for mode,c in itertools.product(("eval","train"),("ce","shared_c1","shared_c4")):
                b=branch(f"lr01_{mode}_{c}","batch" if mode=="train" else "running",mode=="train",c,
                         buffer_policy="native")
                b["lr_multiplier"]=.1
                branches.append(b)
            for c in ("shared_c1","shared_c4"):
                b=branch(f"no_momentum_eval_{c}","running",False,c,buffer_policy="native")
                b["zero_momentum"]=True
                branches.append(b)
            for mode,c in itertools.product(("eval","train"),("teacher_c1","shared_norm")):
                branches.append(branch(f"norm_{mode}_{c}","batch" if mode=="train" else "running",
                                       mode=="train",c,buffer_policy="native"))
        result.append(dict(id=f"p1_{d}_{s}_{t}_flow{flow}",phase="p1",dataset=d,seed=s,stage=t,
                           stream=flow,branches=branches,monitor="B",directions=s==1234,
                           discovery=s==1234 and flow==0))
    return result


def expected_counts():
    rows=tasks();p0=[r for r in rows if r["phase"]=="p0"];p1=[r for r in rows if r["phase"]=="p1"]
    return {"p0_updates":sum(len(r["branches"]) for r in p0),"p0_primary_endpoints":72,
            "p0_cross_gpu_endpoints":16,"p1_factorial_updates":432,"p1_factorial_and_buffer_endpoints":720,
            "p1_additional_lr_updates":72,"p1_zero_momentum_updates":24,"p1_norm_updates":48,
            "p1_direction_endpoints":48,"p1_total_optimizer_updates":576,"p1_total_endpoints":912,
            "p1_buffer_only_forwards":72,"p1_tasks":len(p1),"p0_tasks":len(p0)}


def calibration(records, cross_records=()):
    """Resource gate: resolve c4-c1 pixel CE at all 8 predeclared anchors/modes."""
    grouped={}
    for r in records:
        key=(r["dataset"],r["stage"],r["mode"])
        grouped.setdefault(key,{}).setdefault(r["condition"],[]).append(r)
    if len(grouped)!=8 or any(set(v)!={"ce","shared_c1","shared_c4"} or any(len(x)!=3 for x in v.values()) for v in grouped.values()):
        raise ValueError("P0 incomplete or duplicate replay coverage")
    rows=[];bands={}
    for (d,t,m),group in sorted(grouped.items()):
        for metric in PRIMARY:
            spans={c:max(r["scores"][metric] for r in rs)-min(r["scores"][metric] for r in rs) for c,rs in group.items()}
            key=f"{d}:{m}:{metric}"
            bands[key]=max(bands.get(key,0.),5*max(spans.values()),1e-6)
        a=[r["scores"]["pixel_ce"] for r in group["shared_c4"]]
        b=[r["scores"]["pixel_ce"] for r in group["shared_c1"]]
        contrast=sum(a)/3-sum(b)/3
        span=max(max(a)-min(a),max(b)-min(b))
        rows.append(dict(dataset=d,stage=t,mode=m,contrast=contrast,repeat_span=span,
                         resolved=abs(contrast)>max(5*span,1e-6)))
    cross={}
    for r in cross_records:
        cross.setdefault((r["dataset"],r["stage"]),[]).append(r)
    if len(cross)!=4 or any({r["score_gpu"] for r in v}!={0,1,2,3} or len(v)!=4 for v in cross.values()):
        raise ValueError("P0 cross-GPU calibration is incomplete")
    for (d,t),rs in cross.items():
        for metric in PRIMARY:
            span=max(r["scores"][metric] for r in rs)-min(r["scores"][metric] for r in rs)
            for m in ("eval","train"):
                key=f"{d}:{m}:{metric}";bands[key]=max(bands[key],5*span)
    for row in rows:
        row["deadband"]=bands[f"{row['dataset']}:{row['mode']}:pixel_ce"]
        row["resolved"]=abs(row["contrast"])>row["deadband"]
    return {"pass":all(r["resolved"] for r in rows),"rows":rows,"numerical_bands":bands,
            "role":"P0 measurement-resolution resource rule, not a significance test",
            "p2_or_full_training_authorized":False}
