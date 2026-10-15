"""Freeze P0/P1 inputs on Maker without starting a model experiment."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime,timezone
from pathlib import Path

from utility_study.io import read_json,sha256,write_json
from utility_study.register import canonical_files,label_presence,monitoring_indices
from .contracts import DATASETS,SEEDS,STAGES,stream_entry,tasks,expected_counts


def code_files(code):
    paths=[]
    for folder in ("probe_validity","utility_study"):
        paths.extend((code/folder).glob("*.py"))
    paths += [code/"tests/test_probe_validity.py",code/"tests/test_utility_study.py",
              code/"reports/cvpr2027/followup_protocol_20261015.md",code/"reports/cvpr2027/p15_execution_20261015.md",code.parent/"PCOS.py"]
    return {str(p):sha256(p) for p in sorted(paths)}


def main(root,out,code,resource_policy):
    root,out,code=Path(root).resolve(),Path(out).resolve(),Path(code).resolve()
    if root!=Path('/tmp/liujinshi-covar-kd-20261015'):raise ValueError("This protocol only executes on Maker")
    target=out/"registration.json"
    if target.exists():
        reg=read_json(target)
        if reg["source_files"]!=code_files(code):raise ValueError("Registered execution code changed")
        return reg
    parent=root/"reports/P14_Utility_Maker_20261015_r3/registration.json"
    old=read_json(parent)
    if sha256(parent)!="ecc86a1da30a97f156fc6d864df02bcd16b6f4556f28360d5cff20a08ac8b22e":raise ValueError("Unexpected P14 parent")
    reg=dict(schema_version=1,status="REGISTERED",created_at=datetime.now(timezone.utc).isoformat(),
             parent_registration=str(parent),parent_registration_sha256=sha256(parent),
             root=str(root),output=str(out),code=str(code),source_files=code_files(code),
             authorized_phases=["p0","p1"],forbidden_phases=["p2","p3","p4","80k_full_training"],
             allowed_gpus=[0,1,2,3],resource_policy=read_json(resource_policy),tasks=tasks(),counts=expected_counts(),
             precision="FP32, TF32 off, cuDNN benchmark off; original nondeterministic kernels, measured in P0",
             scoring_memory_cap_mib=6144,update_memory_cap_mib=6144,activation_offload=True,
             activation_offload_policy="torch.autograd.graph.save_on_cpu(pin_memory=False); CUDA forward/backward, exact dtype storage",
             teacher_execution_order="Frozen eval teacher first, then student; assert teacher preserves CPU/CUDA RNG",
             teacher_inference_batch=2,teacher_chunking="Per-image frozen CNN; verify batch2 versus batch1 before formal updates",
             p0_rng="P14 registered first full matched batch; repeat RNG exactly",
             pool_B_seed=20261016,pool_B_size=256,pool_B_per_class=8,
             execution_split="Full-state float32 virtual endpoints serialized on NVMe, then evaluated independently; no batch changes",
             primary_p0="Resolve c4-c1 pixel CE against max(5*replay/cross-card endpoint range,1e-6) at all 8 dataset/stage/modes",
             p0_other_metrics="Numerical bands recorded; no forced significance claim for unresolved endpoints",
             random_streams_are_independent_training_seeds=False,source_checkpoint_writes=0,
             virtual_state_policy="Ephemeral NVMe endpoint .pth files; never overwrite original checkpoints; scientific JSON archived",
             function_sampling="256 uniform valid pixels/image using fixed image-index-derived seed; not pooled full-pixel KL",
             datasets={})
    for d in DATASETS:
        cfg=copy.deepcopy(old["datasets"][d]);files=canonical_files(d,cfg)
        a=set(x["index"] for x in cfg["monitor"])
        p0_indices=set()
        for t in STAGES:p0_indices.update(cfg["updates"][f"1234_{t}_matched_train"][0]["indices"])
        candidates=[(i,x) for i,x in enumerate(files) if i not in a|p0_indices]
        with ThreadPoolExecutor(max_workers=8) as pool:
            presence=list(pool.map(lambda row:label_presence(row[1],d),candidates))
        selected=monitoring_indices(presence,cfg["classes"],seed=20261016)
        b=[]
        for j in selected:
            i,item=candidates[j]
            b.append(dict(item,index=i,classes=presence[j],image_sha256=sha256(item["img"]),label_sha256=sha256(item["label"])))
        b.sort(key=lambda x:x["index"])
        assert not a&{x["index"] for x in b} and len(b)==256
        excluded=a|{x["index"] for x in b};eligible=[i for i in range(len(files)) if i not in excluded]
        streams={}
        for s in SEEDS:
            for t in STAGES:
                for flow in range(3):streams[f"{s}_{t}_{flow}"]=stream_entry(eligible,cfg["matched_batch"],d,s,t,flow)
        p0={str(t):dict(cfg["updates"][f"1234_{t}_matched_train"][0]) for t in STAGES}
        for row in p0.values():
            row["rng_seed"]=row["augmentation_seed"]+90000000
            assert not set(row["indices"])&excluded
        hashes={}
        runs=[r for r in cfg["runs"] if r["condition"]=="ce"]
        for run in runs:
            for t in STAGES:
                p=run["checkpoints"][str(t)];actual=sha256(p)
                if actual!=cfg["input_sha256"][p]:raise ValueError("CE checkpoint mismatch: "+p)
                hashes[p]=actual
        for p,h in cfg["input_sha256"].items():
            if Path(p).suffix==".py" or p in (cfg["train_list"],cfg["teacher"]):
                if sha256(p)!=h:raise ValueError("Frozen runtime/shared input changed: "+p)
                hashes[p]=h
        for item in cfg["monitor"]+b:
            if sha256(item["img"])!=item["image_sha256"] or sha256(item["label"])!=item["label_sha256"]:raise ValueError("Monitoring bytes changed")
        cfg.update(pools={"A":cfg.pop("monitor"),"B":b},p0_entries=p0,streams=streams,input_sha256=hashes,runs=runs)
        cfg.pop("updates");cfg.pop("input_origins",None);cfg.pop("input_hash_provenance",None)
        cfg["pool_B_class_image_counts"]=[sum(k in x["classes"] for x in b) for k in range(cfg["classes"])]
        reg["datasets"][d]=cfg
        print("REGISTERED_P15_DATASET",d,"A=256 B=256 CE_STATES=6",flush=True)
    out.mkdir(parents=True,exist_ok=True);write_json(target,reg)
    print("P15_REGISTRATION_PASS",sha256(target),flush=True)
    return reg
