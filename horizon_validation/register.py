"""Freeze Maker-only P2 source states, streams and validation byte manifests."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import copy

from utility_study.io import read_json,write_json,sha256
from utility_study.register import canonical_files,validate_checkpoint_metadata
from .contracts import DATASETS,SEEDS,SOURCES,stream,counts

def code_hashes(code):
    files=[]
    for folder in ('horizon_validation','probe_validity','utility_study'):files.extend((code/folder).glob('*.py'))
    files.extend([code/'tests/test_horizon_validation.py',code/'reports/cvpr2027/p2_execution_20261016.md',code.parent/'PCOS.py'])
    return {str(p):sha256(p) for p in sorted(files)}

def register(root,out,code,policy):
    import torch
    root,out,code=map(lambda p:Path(p).resolve(),(root,out,code))
    if str(root)!='/tmp/liujinshi-covar-kd-20261015':raise ValueError('Maker-only model execution')
    if (out/'registration.json').exists():raise RuntimeError('Preserve existing registration; use a new revision')
    p15path=root/'reports/P15_ProbeValidity_20261015_r2/registration.json';p15=read_json(p15path)
    if sha256(p15path)!='5ee9c365f701c17cd47c874b34372a7d4fbe7c345ced9f223d71f7513d156caf':raise ValueError('P15 parent differs')
    if read_json(p15path.parent/'workflow.json')['status']!='COMPLETED':raise ValueError('P1 incomplete')
    parent=read_json(p15['parent_registration'])
    reg=dict(schema_version=1,status='REGISTERED',created_at=datetime.now(timezone.utc).isoformat(),
        root=str(root),output=str(out),code=str(code),parent_registration=str(p15path),parent_registration_sha256=sha256(p15path),
        source_files=code_hashes(code),counts=counts(),allowed_gpus=[0,1,2,3],resource_policy=read_json(policy),
        precision='FP32; TF32 off; cuDNN benchmark off; full student batches; CPU saved activations',
        allocation='All four cards with 8192 MiB free admission, 6144 MiB allocator cap; preserve all keepalives',
        teacher_inference_batch=2,stage=20000,schedule_total=80000,scoring_memory_cap_mib=6144,update_memory_cap_mib=6144,
        activation_offload=True,activation_pin_memory=True,p0_bands=read_json(p15path.parent/'p0_calibration.json')['numerical_bands'],
        gradient_validation='Read-only concatenated finite check; identical Boolean predicate, no gradient mutation',
        engineering_reference={},
        source_checkpoint_writes=0,full_80k_trajectories=0,forbidden_phases=['p3','p4','80k_full_training'],datasets={})
    for d in DATASETS:
        cfg=copy.deepcopy(p15['datasets'][d]);old=parent['datasets'][d]
        cfg['runs']=[r for r in old['runs'] if r['condition'] in SOURCES]
        if len(cfg['runs'])!=6:raise ValueError('Need six source runs per dataset')
        for run in cfg['runs']:
            p=run['checkpoints']['20000'];h=sha256(p)
            if h!=old['input_sha256'][p]:raise ValueError('Source checkpoint changed: '+p)
            state=torch.load(p,map_location='cpu',weights_only=False);validate_checkpoint_metadata(state,run,20000);del state
            cfg['input_sha256'][p]=h
        files=canonical_files(d,cfg);excluded={x['index'] for pool in cfg['pools'].values() for x in pool}
        eligible=[i for i in range(len(files)) if i not in excluded];cfg['streams']={}
        for seed in SEEDS:
            for f in range(4):
                rows=stream(eligible,cfg['matched_batch'],d,seed,f,2000 if f==3 else 128)
                path=out/'streams'/f'{d}_{seed}_flow{f}.json';write_json(path,rows)
                cfg['streams'][f'{seed}_{f}']=dict(path=str(path),sha256=sha256(path),steps=len(rows))
        val_list=Path(cfg['runtime'])/'dataset/list'/('voc/val.txt' if d=='voc' else 'cityscapes/val.lst')
        val=[]
        for row in val_list.read_text().splitlines():
            if not row.strip():continue
            if d=='voc':img=Path(cfg['data'])/'JPEGImages'/(row.strip()+'.jpg');label=Path(cfg['data'])/'SegmentationClass'/(row.strip()+'.png')
            else:
                a,b=row.split();img=Path(cfg['data'])/a;label=Path(cfg['data'])/b
            val.append(dict(img=str(img),label=str(label)))
        if len(val)!=(1449 if d=='voc' else 500):raise ValueError('Incomplete official validation list')
        def bind(v):return dict(v,image_sha256=sha256(v['img']),label_sha256=sha256(v['label']))
        with ThreadPoolExecutor(max_workers=8) as pool:val=list(pool.map(bind,val))
        cfg['validation']=dict(list=str(val_list),list_sha256=sha256(val_list),items=val,count=len(val))
        for pool in cfg['pools'].values():
            for item in pool:
                if sha256(item['img'])!=item['image_sha256'] or sha256(item['label'])!=item['label_sha256']:raise ValueError('Pool bytes changed')
        reg['datasets'][d]=cfg
        print('REGISTERED_P2_DATASET',d,'SOURCES=6','VAL='+str(len(val)),flush=True)
    reference=root/'reports/P2_HorizonValidation_20261016'
    if sha256(reference/'registration.json')!='73a7158498e403520b964865ae7e3150ba357cf3afd23c40249d93f2fa6156f8':raise ValueError('Unexpected engineering reference')
    for d in DATASETS:
        for source in SOURCES:
            r=read_json(reference/'bundles'/f'bench_{d}_{source}.json')
            if r['status']!='PASS':raise ValueError('Incomplete engineering reference')
            e=r['endpoints'][0]
            if sha256(e['state_file'])!=e['state_file_sha256']:raise ValueError('Reference state changed')
            reg['engineering_reference'][f'{d}_{source}']=e
    write_json(out/'registration.json',reg)
    return reg
