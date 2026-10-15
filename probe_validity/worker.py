"""Separate full-batch updates from low-memory, native-resolution scoring."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import os
from pathlib import Path
import sys
import time

from utility_study.io import read_json,sha256,write_json


def now():return datetime.now(timezone.utc).isoformat()


def save_state(path,state):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    torch.save({"student":state},tmp);tmp.replace(path)
    return sha256(path)


def update(reg,out,job,gpu,preflight=False):
    import math
    import torch
    from .runtime import Runtime
    from .core import cpu_state,fingerprint,split_hashes,old_buffers,set_update_mode,module_inventory,gradient_stats
    from utility_study.core import Snapshot,isolated_rng,ce_loss,kd_loss,gradient_norm,virtual_outcome
    worker=out/"workers"/job["id"];worker.mkdir(parents=True,exist_ok=True)
    rt=Runtime(reg,job["dataset"],worker,"update")
    run,checkpoint=rt.source(job["seed"],job["stage"])
    model,opt=rt.student,rt.optimizer;named=tuple(model.named_parameters());parameters=tuple(p for _,p in named)
    names={name for name,_ in named};modules=module_inventory(model)
    cfg=rt.cfg
    entry=cfg["p0_entries"][str(job["stage"])] if job["stream"]=="legacy" else cfg["streams"][f"{job['seed']}_{job['stage']}_{job['stream']}"]
    x,y=rt.update_batch(entry)
    batch_sha=fingerprint({"images":x,"labels":y});baseline=Snapshot.capture(model,opt)
    original_hash=split_hashes(baseline.model,names);original_optimizer=fingerprint(baseline.optimizer)
    teacher_chunk_error=None
    if job["phase"]=="engineering":
        with torch.no_grad():
            batch2=rt.teacher(x[:2])[0]
            singles=torch.cat([rt.teacher(x[i:i+1])[0] for i in range(2)])
            teacher_chunk_error=float((batch2-singles).abs().max())
            if teacher_chunk_error>1e-4:raise RuntimeError("Frozen teacher chunk equivalence exceeds tolerance")
            del batch2,singles
        baseline.verify(model,opt)
    outputs=[];direction_cache={};started=time.monotonic()
    for spec in job["branches"]:
        generated=out/"generated"/job["id"]/(spec["name"]+".json")
        if generated.exists() and not job.get("directions"):
            old=read_json(generated)
            if old["registration_sha256"]!=sha256(out/"registration.json"):raise ValueError("Existing branch registration differs")
            if old["status"]!="PASS":raise RuntimeError("Preserve the existing failed numerical branch; do not rerun it as a success")
            for e in old["endpoints"]:
                if sha256(e["state_file"])!=e["state_file_sha256"]:raise RuntimeError("Endpoint state changed")
            outputs.extend(old["endpoints"]);continue
        start=time.monotonic();metadata=None;endpoints=[];alpha=0.;stats=None;step_state=None;kd_cpu=None
        with virtual_outcome(model,opt) as outcome:
            set_update_mode(model,spec["bn"],spec["dropout"])
            with isolated_rng(entry["rng_seed"]):
                for group in opt.param_groups:
                    group["lr"]=run["args"]["lr"]*(1-(job["stage"]+1)/80000.)**.9*spec["lr_multiplier"]
                if spec["zero_momentum"]:
                    for value in opt.state.values():
                        if "momentum_buffer" in value:value["momentum_buffer"].zero_()
                opt.zero_grad(set_to_none=True)
                bx,by=(x[:2],y[:2]) if spec.get("engineering_batch2") else (x,y)
                offload=spec.get("activation_offload",reg["activation_offload"])
                if job["phase"]!="engineering" and not offload:raise ValueError("Formal P15 branches use registered activation storage")
                if spec.get("buffer_only"):
                    with torch.no_grad():model(bx)
                else:
                    condition=spec["condition"]
                    teacher=None
                    if condition!="ce":
                        rng_before=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
                        with torch.no_grad():teacher=torch.cat([rt.teacher(bx[i:i+reg["teacher_inference_batch"]])[0] for i in range(0,len(bx),reg["teacher_inference_batch"])])
                        if not torch.equal(rng_before[0],torch.get_rng_state()) or not torch.equal(rng_before[1],torch.cuda.get_rng_state()):
                            raise RuntimeError("Frozen teacher unexpectedly consumed RNG")
                    torch.cuda.empty_cache()
                    storage=torch.autograd.graph.save_on_cpu(pin_memory=False) if offload else nullcontext()
                    with storage:
                        logits=model(bx)[0];task=ce_loss(logits,by)
                        knowledge=kd_loss(logits,teacher,by,"teacher" if condition.startswith("teacher") else "shared") if condition!="ce" else None
                    if condition!="ce":
                        placement="teacher" if condition.startswith("teacher") else "shared"
                        alpha=4. if condition.endswith("c4") else 1.
                        if condition=="shared_norm":
                            with torch.autograd.graph.save_on_cpu(pin_memory=False):other=kd_loss(logits,teacher,by,"teacher")
                            ag=torch.autograd.grad(other,parameters,retain_graph=True,allow_unused=True)
                            bg=torch.autograd.grad(knowledge,parameters,retain_graph=True,allow_unused=True)
                            an,bn=gradient_norm(ag),gradient_norm(bg)
                            if min(an,bn)<=0 or not math.isfinite(an+bn):raise FloatingPointError("Invalid norm control")
                            alpha=an/bn;del ag,bg,other
                        gc=torch.autograd.grad(task,parameters,retain_graph=True,allow_unused=True)
                        gk=torch.autograd.grad(knowledge,parameters,retain_graph=True,allow_unused=True)
                        stats=gradient_stats(named,gc,gk,alpha,opt.state)
                        if job.get("directions") and condition=="shared_c1" and spec["name"].startswith("factor_") and ((spec["bn"]=="running" and not spec["dropout"]) or (spec["bn"]=="batch" and spec["dropout"])):
                            kd_cpu={name:g.detach().cpu().clone() for (name,_),g in zip(named,gk) if g is not None}
                        del gc,gk
                        objective=task+alpha*knowledge
                    else:
                        gc=torch.autograd.grad(task,parameters,retain_graph=True,allow_unused=True)
                        stats=gradient_stats(named,gc,(None,)*len(parameters),0.,opt.state)
                        del gc
                        objective=task
                    if not bool(torch.isfinite(objective)):raise FloatingPointError("Nonfinite objective")
                    objective.backward()
                    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):raise FloatingPointError("Nonfinite gradient")
                    opt.step()
                    metadata={"local_ce":float(task.detach()),"objective":float(objective.detach()),"coefficient":alpha,"first_gradient":stats,
                              "batch_size":len(bx),"activation_offload":offload,"teacher_rng_preserved":True}
                    logits=task=objective=teacher=knowledge=None
                step_state=cpu_state(model)
                if any(v.is_floating_point() and not bool(torch.isfinite(v).all()) for v in step_state.values()):
                    raise FloatingPointError("Nonfinite updated model state")
                counters={k:int(v)-int(baseline.model[k]) for k,v in step_state.items() if k.endswith("num_batches_tracked")}
                if any(v!=(1 if spec["bn"]=="batch" else 0) for v in counters.values()):raise RuntimeError("Unexpected per-layer BN update count")
                if spec.get("buffer_only") and fingerprint({k:v for k,v in step_state.items() if k in names})!=original_hash["parameters_sha256"]:raise RuntimeError("Buffer-only changed parameters")
                per_step=[]
                for name in names:
                    delta=(step_state[name].double()-baseline.model[name].double()).norm().item()
                    denom=baseline.model[name].double().norm().item()
                    per_step.append({"name":name,"delta_norm":delta,"relative_step":delta/max(denom,1e-30)})
                clean=old_buffers(step_state,baseline.model,names)
                policies=[("original",clean),("updated",step_state)] if spec["buffer_policy"]=="both" and spec["bn"]=="batch" else [("native",step_state)]
                if job.get("directions") and spec["name"].startswith("factor_") and ((spec["bn"]=="running" and not spec["dropout"]) or (spec["bn"]=="batch" and spec["dropout"])):
                    key="train" if spec["bn"]=="batch" else "eval"
                    if spec["condition"]=="ce":direction_cache.setdefault(key,{})["ce_state"]=clean
                    if kd_cpu is not None:direction_cache.setdefault(key,{})["kd_grad"]=kd_cpu
                for policy,state in policies:
                    name=spec["name"]+"__"+policy
                    state_file=out/"states"/job["id"]/(name+".pth")
                    state_sha=save_state(state_file,state)
                    endpoints.append(dict(id=job["id"]+"__"+name,task=job["id"],dataset=job["dataset"],seed=job["seed"],stage=job["stage"],
                        phase=job["phase"],stream=job["stream"],mode=job.get("mode"),condition=spec["condition"],repeat=spec.get("repeat"),
                        branch=spec,buffer_policy=policy,state_file=str(state_file),state_file_sha256=state_sha,
                        update_gpu=gpu,monitor=job["monitor"],**split_hashes(state,names)))
        baseline.verify(model,opt)
        if fingerprint(opt.state_dict())!=original_optimizer:raise RuntimeError("Optimizer restoration hash differs")
        record=dict(status=outcome["status"],reason=outcome["reason"],registration_sha256=sha256(out/"registration.json"),
                    checkpoint_sha256=cfg["input_sha256"][checkpoint],job=job["id"],spec=spec,batch_sha256=batch_sha,
                    update=metadata,bn_counter_deltas=counters if step_state is not None else None,
                    parameter_steps=per_step if step_state is not None else None,state_restored=True,
                    original_state=original_hash,optimizer_restored_sha256=original_optimizer,
                    endpoints=endpoints,seconds=time.monotonic()-start,at=now())
        if generated.exists():
            previous=read_json(generated)
            write_json(out/"attempts"/job["id"]/(spec["name"]+"_previous_"+str(time.time_ns())+".json"),previous)
        write_json(generated,record);outputs.extend(endpoints)
        if outcome["status"]!="PASS":raise RuntimeError("NUMERICAL_DIVERGENCE: "+str(outcome))
        write_json(worker/"progress.json",dict(status="RUNNING",job=job["id"],last_branch=spec["name"],endpoints=len(outputs),pid=os.getpid(),gpu=gpu))
        print("UPDATE_PASS",job["id"],spec["name"],flush=True)
    if job.get("directions"):
        for mode,value in direction_cache.items():
            for sign in (-1,1):
                state={k:v.clone() for k,v in value["ce_state"].items()}
                lr=run["args"]["lr"]*(1-(job["stage"]+1)/80000.)**.9
                for name,gradient in value["kd_grad"].items():state[name].add_(gradient,alpha=-sign*.1*lr)
                name=f"direction_{mode}_{'plus' if sign>0 else 'minus'}"
                path=out/"states"/job["id"]/(name+".pth")
                outputs.append(dict(id=job["id"]+"__"+name,task=job["id"],dataset=job["dataset"],seed=job["seed"],stage=job["stage"],
                    phase="p1",stream=job["stream"],mode=mode,condition="directional_shared",repeat=None,
                    branch={"name":name,"coefficient_perturbation":sign*.1,"base":"CE step with original buffers"},
                    buffer_policy="original",state_file=str(path),state_file_sha256=save_state(path,state),update_gpu=gpu,monitor="B",**split_hashes(state,names)))
    if sha256(checkpoint)!=cfg["input_sha256"][checkpoint]:raise RuntimeError("Source checkpoint mutated")
    comparisons=[]
    if job["phase"]=="engineering":
        for bn in ("running","batch"):
            for drop in (False,True):
                label=f"check_{bn}_drop{int(drop)}"
                pair=[]
                for storage in ("native","offload"):
                    e=next(e for e in outputs if e["branch"]["name"]==label+"_"+storage and e["buffer_policy"]=="native")
                    pair.append(torch.load(e["state_file"],map_location="cpu",weights_only=False)["student"])
                error=max(float((pair[0][k].double()-pair[1][k].double()).abs().max()) for k in pair[0])
                if error>1e-4:raise RuntimeError("Activation storage changes reference mini-batch state beyond engineering tolerance")
                comparisons.append(dict(bn=bn,dropout=drop,reference_batch=2,max_state_absolute_error=error))
    summary=dict(status="PASS",job=job["id"],registration_sha256=sha256(out/"registration.json"),endpoints=outputs,
                 module_inventory=modules,seconds=time.monotonic()-started,update_gpu=gpu,
                 peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                 original_checkpoints_written=0,virtual_states_are_temporary=True,activation_storage_checks=comparisons,
                 teacher_chunk_max_absolute_error=teacher_chunk_error)
    write_json(out/"bundles"/(job["id"]+".json"),summary)
    return summary


def teacher(reg,out,job,gpu):
    import numpy as np
    from PIL import Image
    import torch
    from utility_study.core import resized_logits
    from .runtime import Runtime
    rt=Runtime(reg,job["dataset"],out/"workers"/job["id"],"teacher")
    folder=out/"teachers"/job["dataset"]/job["pool"];folder.mkdir(parents=True,exist_ok=True)
    records=[]
    with torch.no_grad():
        for i,item in enumerate(rt.cfg["pools"][job["pool"]]):
            x,y=rt.monitor_image(item)
            hard=resized_logits(rt.teacher(x)[0],y).argmax(1)[0].byte().cpu().numpy()
            path=folder/(str(item["index"])+".png");Image.fromarray(hard).save(path)
            records.append(dict(index=item["index"],path=str(path),sha256=sha256(path)))
            if (i+1)%32==0:print("TEACHER_PROGRESS",job["dataset"],job["pool"],i+1,flush=True)
    receipt=dict(status="PASS",dataset=job["dataset"],pool=job["pool"],images=records,gpu=gpu,
                 teacher_sha256=rt.cfg["input_sha256"][rt.cfg["teacher"]],registration_sha256=sha256(out/"registration.json"))
    write_json(folder/"receipt.json",receipt);return receipt


def score(reg,out,job,gpu):
    import numpy as np
    from PIL import Image
    import torch
    from .runtime import Runtime
    from .core import DenseMetrics,fingerprint
    rt=Runtime(reg,job["dataset"],out/"workers"/job["id"],"score")
    rt.source(job["seed"],job["stage"])
    cfg=rt.cfg;pool=cfg["pools"][job["pool"]]
    if job.get("preflight"):pool=pool[:2]
    baseline_root=out/"engineering/baselines" if job.get("preflight") else out/"baselines"
    baseline_path=baseline_root/job["dataset"]/(f"{job['seed']}_{job['stage']}_{job['pool']}")
    refs=None
    if not job.get("baseline"):
        baseline=read_json(baseline_path.with_suffix(".json"))
        if baseline["registration_sha256"]!=sha256(out/"registration.json"):raise RuntimeError("Baseline belongs to another registration")
        sample_path=baseline_path.with_suffix(".samples.pt")
        if sha256(sample_path)!=baseline["samples_sha256"]:raise RuntimeError("Baseline function sample bytes changed")
        refs=torch.load(sample_path,map_location="cpu",weights_only=False)
    receipt=read_json(out/"teachers"/job["dataset"]/job["pool"]/"receipt.json")
    if receipt["registration_sha256"]!=sha256(out/"registration.json") or receipt["teacher_sha256"]!=cfg["input_sha256"][cfg["teacher"]]:
        raise RuntimeError("Teacher cache provenance differs")
    teacher_files={r["index"]:r for r in receipt["images"]}
    endpoints=[None] if job.get("baseline") else job["endpoints"]
    all_scores=[]
    for endpoint in endpoints:
        dest=baseline_path.with_suffix(".json") if endpoint is None else out/"scores"/(endpoint["id"]+".json")
        if dest.exists() and not job.get("cross_gpu"):
            old=read_json(dest)
            if old["registration_sha256"]!=sha256(out/"registration.json"):raise RuntimeError("Existing score registration changed")
            all_scores.append(old);continue
        if endpoint is not None:
            if sha256(endpoint["state_file"])!=endpoint["state_file_sha256"]:raise RuntimeError("Virtual endpoint changed")
            state=torch.load(endpoint["state_file"],map_location="cpu",weights_only=False)["student"]
            rt.student.load_state_dict(state,strict=True)
            if fingerprint(rt.student.state_dict())!=endpoint["full_state_sha256"]:raise RuntimeError("Serialized state did not load exactly")
            del state
        rt.student.eval();before_hash=fingerprint(rt.student.state_dict());metric=DenseMetrics(cfg["classes"]);started=time.monotonic()
        with torch.no_grad():
            for i,item in enumerate(pool):
                x,y=rt.monitor_image(item)
                tf=teacher_files[item["index"]]
                if sha256(tf["path"])!=tf["sha256"]:raise RuntimeError("Teacher cache changed")
                with Image.open(tf["path"]) as f:q=torch.from_numpy(np.array(f,dtype=np.int64))[None]
                logits=rt.student(x)[0]
                metric.add(logits,y,q,None if refs is None else refs[i],sample_seed=20261016+item["index"])
                del x,y,logits,q
                if (i+1)%32==0:
                    write_json(out/"workers"/job["id"]/"progress.json",dict(status="RUNNING",image=i+1,total=len(pool),gpu=gpu,endpoint=None if endpoint is None else endpoint["id"],pid=os.getpid()))
        scores=metric.result()
        if len(pool)==256 and not scores["all_classes_present"]:raise RuntimeError("Missing monitoring class")
        if fingerprint(rt.student.state_dict())!=before_hash:raise RuntimeError("Monitoring mutated model state")
        result=dict(status="PASS",registration_sha256=sha256(out/"registration.json"),dataset=job["dataset"],seed=job["seed"],stage=job["stage"],pool=job["pool"],
                    monitor_images=len(pool),score_gpu=gpu,scores=scores,seconds=time.monotonic()-started,at=now(),
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                    monitor_state_unchanged=True)
        if endpoint is None:
            sample_path=baseline_path.with_suffix(".samples.pt");sample_path.parent.mkdir(parents=True,exist_ok=True)
            temporary=sample_path.with_name(sample_path.name+".tmp")
            torch.save(metric.sample_logp,temporary);temporary.replace(sample_path)
            result.update(samples_sha256=sha256(sample_path),samples_file=str(sample_path))
        else:result.update({k:v for k,v in endpoint.items() if k!="state_file"})
        if job.get("cross_gpu"):dest=out/"cross_gpu"/(endpoint["id"]+f"__gpu{gpu}.json")
        write_json(dest,result);all_scores.append(result)
        print("SCORE_PASS",job["id"],"baseline" if endpoint is None else endpoint["id"],flush=True)
    return {"status":"PASS","job":job["id"],"score_records":len(all_scores),"gpu":gpu,
            "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,"peak_reserved_mib":torch.cuda.max_memory_reserved()/2**20}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True);parser.add_argument("--job",type=Path,required=True);parser.add_argument("--gpu",type=int,required=True)
    args=parser.parse_args();reg=read_json(args.out/"registration.json");job=read_json(args.job)
    if args.gpu not in reg["allowed_gpus"]:raise ValueError("Unallocated GPU")
    from .resources import acquire
    handles=[]
    try:
        handles,samples=acquire(args.gpu,reg["resource_policy"],"score" if job["kind"]=="score" else "update")
        os.environ.update(CUDA_VISIBLE_DEVICES=samples[-1]["uuid"],OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
        for name in ("RANK","LOCAL_RANK","WORLD_SIZE"):os.environ.pop(name,None)
        import cv2
        import torch
        torch.set_num_threads(1);cv2.setNumThreads(1)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        cap=reg["scoring_memory_cap_mib"] if job["kind"]=="score" else reg["update_memory_cap_mib"]
        torch.cuda.set_per_process_memory_fraction(cap*2**20/torch.cuda.get_device_properties(0).total_memory)
        result={"update":update,"teacher":teacher,"score":score}[job["kind"]](reg,args.out,job,args.gpu)
        result.update(admission=samples,pid=os.getpid(),at=now(),keepalive_processes_signalled=0,
                      registration_sha256=sha256(args.out/"registration.json"),job_sha256=sha256(args.job))
        write_json(args.out/"done"/(job["id"]+".json"),result)
    except Exception as error:
        text=str(error)
        resource=any(x in text for x in ("RESOURCE_UNAVAILABLE","out of memory","temporarily unavailable","Resource temporarily unavailable"))
        write_json(args.out/"attempts"/job["id"]/(str(time.time_ns())+".json"),dict(status="RESOURCE_RETRY" if resource else "FAILED",error=repr(error),at=now(),gpu=args.gpu,pid=os.getpid()))
        if resource:raise SystemExit(75)
        raise
    finally:
        for handle in handles:handle.close()


if __name__=="__main__":main()
