"""Full-batch P2 rollouts, independent scoring and blinded 2k forks on Maker."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import copy
import os
import time

from utility_study.io import read_json,write_json,sha256
from .contracts import state_id

def now():return datetime.now(timezone.utc).isoformat()

def runtime(reg,job,folder,role):
    from probe_validity.runtime import Runtime
    selected=copy.deepcopy(reg)
    selected['datasets'][job['dataset']]['runs']=[r for r in reg['datasets'][job['dataset']]['runs'] if r['condition']==job['source']]
    return Runtime(selected,job['dataset'],folder,role)

def save(path,value):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');torch.save(value,tmp);tmp.replace(path)
    return sha256(path)

def load_checked(endpoint):
    import torch
    if sha256(endpoint['state_file'])!=endpoint['state_file_sha256']:raise RuntimeError('Endpoint bytes changed')
    return torch.load(endpoint['state_file'],map_location='cpu',weights_only=False)

def publish_endpoint(out,endpoint):
    """Keep any published resource-attempt endpoint and its immutable state."""
    path=out/'endpoints'/endpoint['task']/f"h{endpoint['horizon']}.json"
    if path.exists():
        old=read_json(path)
        if old['registration_sha256']!=endpoint['registration_sha256'] or old['job_sha256']!=endpoint['job_sha256']:raise RuntimeError('Endpoint history provenance changed')
        write_json(out/'endpoint_history'/endpoint['task']/(f"h{endpoint['horizon']}_"+str(time.time_ns())+'.json'),old)
    write_json(path,endpoint)

def finite_gradients(model):
    """One read-only concatenated check avoids hundreds of tiny CUDA reductions."""
    import torch
    gradients=[p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
    if not gradients:raise RuntimeError('No student gradients')
    return bool(torch.isfinite(torch.cat(gradients)).all())

def update(reg,out,job,gpu):
    import torch
    from utility_study.core import Snapshot,isolated_rng,ce_loss,kd_loss,_clone_cpu,rng_state,restore_rng
    from probe_validity.core import cpu_state,fingerprint,set_update_mode,split_hashes
    folder=out/'workers'/job['id'];folder.mkdir(parents=True,exist_ok=True)
    rt=runtime(reg,job,folder,'update');run,original=rt.source(job['seed'],20000)
    model,opt=rt.student,rt.optimizer;baseline=Snapshot.capture(model,opt)
    parameter_names={n for n,_ in model.named_parameters()};source_hash=fingerprint(baseline.model)
    meta=rt.cfg['streams'][f"{job['seed']}_{job['stream']}"]
    if sha256(meta['path'])!=meta['sha256']:raise RuntimeError('Stream changed')
    entries=read_json(meta['path']);endpoints=[];times=[];resume_step=0
    registration_sha=sha256(out/'registration.json');job_sha=sha256(out/'jobs'/(job['id']+'.json'))
    if job['phase']=='fork':
        if sha256(out/'choices.json')!=job['choices_sha256']:raise RuntimeError('Blind choices changed')
        if read_json(out/'choices_backup.json')['status']!='PASS':raise RuntimeError('Choices not durably verified')
        prior=sorted((out/'checkpoints'/job['id']).glob('step_*.json'))
        if prior:
            receipt=read_json(prior[-1])
            if receipt['registration_sha256']!=registration_sha or receipt['job_sha256']!=job_sha:raise RuntimeError('Resume provenance differs')
            state=load_checked(receipt);resume_step=receipt['step']
            model.load_state_dict(state['student'],strict=True);opt.load_state_dict(state['optimizer'])
            for key in ('criterion_cwd','criterion_fitnet'):getattr(rt.trainer,key).load_state_dict(state[key])
            restore_rng(state['rng']);del state
    # Regenerate only already-published milestones; never mix task registrations.
    for h in job['horizons']:
        p=out/'endpoints'/job['id']/f'h{h}.json'
        if p.exists():
            e=read_json(p)
            if e['registration_sha256']!=registration_sha or e['job_sha256']!=job_sha:raise RuntimeError('Old endpoint provenance differs')
            if sha256(e['state_file'])!=e['state_file_sha256']:raise RuntimeError('Old endpoint changed')
            endpoints.append(e)
    started=time.monotonic();attempt_id=str(os.getpid())+'_'+str(time.time_ns())
    try:
        set_update_mode(model,'batch' if job['mode']=='train' else 'running',job['mode']=='train')
        for step in range(resume_step+1,job['steps']+1):
            tick=time.monotonic();entry=entries[step-1]
            x,y=rt.update_batch(entry)
            with isolated_rng(entry['rng_seed']):
                for group in opt.param_groups:group['lr']=run['args']['lr']*(1-(20000+step)/80000.)**.9
                opt.zero_grad(set_to_none=True);teacher=None
                if job['coefficient']:
                    rb=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
                    with torch.no_grad():teacher=torch.cat([rt.teacher(x[i:i+2])[0] for i in range(0,len(x),2)])
                    if not torch.equal(rb[0],torch.get_rng_state()) or not torch.equal(rb[1],torch.cuda.get_rng_state()):raise RuntimeError('Teacher consumed RNG')
                with torch.autograd.graph.save_on_cpu(pin_memory=reg['activation_pin_memory']):
                    logits=model(x)[0];task=ce_loss(logits,y)
                    loss=task if teacher is None else task+job['coefficient']*kd_loss(logits,teacher,y,'shared')
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite objective')
                loss.backward()
                if not finite_gradients(model):raise FloatingPointError('Nonfinite gradient')
                opt.step();torch.cuda.synchronize()
                loss_value=float(loss.detach());del x,y,teacher,logits,task,loss
            elapsed=time.monotonic()-tick;times.append(elapsed)
            if step in job['horizons']:
                state=cpu_state(model)
                if any(v.is_floating_point() and not bool(torch.isfinite(v).all()) for v in state.values()):raise FloatingPointError('Nonfinite state')
                counters={k:int(v)-int(baseline.model[k]) for k,v in state.items() if k.endswith('num_batches_tracked')}
                if any(v!=(step if job['mode']=='train' else 0) for v in counters.values()):raise RuntimeError('BN counter differs from rollout horizon')
                path=out/'states'/job['id']/attempt_id/f'h{step}.pth'
                e={k:job[k] for k in ('dataset','seed','source','coefficient','stream','mode','repeat','phase')}
                e.update(id=job['id']+f'__h{step}',task=job['id'],horizon=step,state_file=str(path),state_file_sha256=save(path,{'student':state}),
                    registration_sha256=registration_sha,job_sha256=job_sha,update_gpu=gpu,
                    source_state_sha256=source_hash,**split_hashes(state,parameter_names))
                publish_endpoint(out,e)
                endpoints=[old for old in endpoints if old['horizon']!=step]+[e];del state
            if job['phase']=='fork' and step%250==0:
                path=out/'checkpoints'/job['id']/f'step_{step:04d}.pth'
                state=dict(student=cpu_state(model),optimizer=_clone_cpu(opt.state_dict()),rng=rng_state(),
                    criterion_cwd=_clone_cpu(rt.trainer.criterion_cwd.state_dict()),criterion_fitnet=_clone_cpu(rt.trainer.criterion_fitnet.state_dict()),
                    iteration=20000+step,args=run['args'],p2_job=job,registration_sha256=registration_sha)
                receipt=dict(step=step,state_file=str(path),state_file_sha256=save(path,state),registration_sha256=registration_sha,job_sha256=job_sha,
                    persist=step in (1000,2000),at=now())
                write_json(path.with_suffix('.json'),receipt);del state
            if step%8==0 or step==job['steps']:
                write_json(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),gpu=gpu,step=step,total=job['steps'],
                    objective=loss_value,seconds_per_step_recent=sum(times[-16:])/len(times[-16:]),at=now()))
                print('STEP',job['id'],step,flush=True)
    finally:
        baseline.restore(model,opt);baseline.verify(model,opt)
    if sha256(original)!=rt.cfg['input_sha256'][original]:raise RuntimeError('Original source changed')
    import statistics
    result=dict(status='PASS',job=job['id'],phase=job['phase'],dataset=job['dataset'],endpoints=endpoints,steps=job['steps'],resumed_from=resume_step,
        seconds=time.monotonic()-started,step_seconds=dict(mean=statistics.mean(times) if times else None,
        steady_median=statistics.median(times[4:] or times) if times else None,measured_steps=len(times)),
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        complete_state_restored=True,source_checkpoint_writes=0,gpu=gpu)
    write_json(out/'bundles'/(job['id']+'.json'),result);return result

def score(reg,out,job,gpu):
    import torch
    from utility_study.core import PooledMetrics,resized_logits
    from probe_validity.core import fingerprint
    rt=runtime(reg,job,out/'workers'/job['id'],'score');rt.source(job['seed'],20000)
    registration_sha=sha256(out/'registration.json');results=[]
    if job.get('validation'):
        if sha256(out/'choices.json')!=job['choices_sha256']:raise RuntimeError('Validation requires unchanged blind choices')
        if read_json(out/'choices_backup.json')['status']!='PASS':raise RuntimeError('Choices not persisted')
    for e in job.get('endpoints',[None]):
        sid=state_id(job['dataset'],job['seed'],job['source'])
        dest=(out/'validation'/f'{sid}_c{e["coefficient"]}.json') if job.get('validation') else ((out/'baselines'/(sid+'.json')) if e is None else out/'scores'/(e['id']+'.json'))
        if dest.exists():
            r=read_json(dest)
            if r['registration_sha256']!=registration_sha:raise RuntimeError('Score registration changed')
            results.append(r);continue
        if e is not None:
            state=load_checked(e)['student'];rt.student.load_state_dict(state,strict=True)
            if fingerprint(rt.student.state_dict())!=e['full_state_sha256']:raise RuntimeError('Loaded endpoint differs')
            del state
        rt.student.eval();before=fingerprint(rt.student.state_dict());start=time.monotonic()
        if job.get('validation'):
            cfg=rt.cfg['validation']
            if sha256(cfg['list'])!=cfg['list_sha256']:raise RuntimeError('Validation list changed')
            if len(rt.trainer.val_loader.dataset)!=cfg['count']:raise RuntimeError('Incomplete validation loader')
            expected={x['img']:x for x in cfg['items']}
            for item in expected.values():
                if sha256(item['img'])!=item['image_sha256'] or sha256(item['label'])!=item['label_sha256']:raise RuntimeError('Validation bytes changed')
            confusion=torch.zeros((rt.cfg['classes'],rt.cfg['classes']),dtype=torch.int64);count=0
            with torch.no_grad():
                for image,target,_ in rt.trainer.val_loader:
                    image,target=image.cuda(),target.long().cuda();pred=resized_logits(rt.student(image)[0],target).argmax(1)
                    valid=target!=-1;y=target[valid].cpu();p=pred[valid].cpu();n=rt.cfg['classes']
                    if bool(((y<0)|(y>=n)).any()):raise ValueError('Invalid validation label')
                    confusion+=torch.bincount(n*y+p,minlength=n*n).reshape(n,n);count+=len(image)
                    if count%32==0:write_json(out/'workers'/job['id']/'progress.json',dict(image=count,total=cfg['count'],gpu=gpu,pid=os.getpid(),at=now()))
            if count!=cfg['count']:raise RuntimeError('Validation image coverage differs')
            union=confusion.sum(0)+confusion.sum(1)-confusion.diag()
            if not bool((union>0).all()):raise RuntimeError('Missing validation class')
            scores=dict(hard_miou=float((confusion.diag().double()/union).mean()),hard_iou=(confusion.diag().double()/union).tolist(),confusion=confusion.tolist())
            total=count
        else:
            metric=PooledMetrics(rt.cfg['classes']);pool=rt.cfg['pools']['B']
            with torch.no_grad():
                for i,item in enumerate(pool):
                    x,y=rt.monitor_image(item);metric.add(rt.student(x)[0],y);del x,y
                    if (i+1)%32==0:write_json(out/'workers'/job['id']/'progress.json',dict(image=i+1,total=len(pool),endpoint=None if e is None else e['id'],gpu=gpu,pid=os.getpid(),at=now()))
            scores=metric.result();total=len(pool)
            if not scores['all_classes_present']:raise RuntimeError('Missing monitoring class')
        if fingerprint(rt.student.state_dict())!=before:raise RuntimeError('Scoring changed model state')
        r=dict(status='PASS',registration_sha256=registration_sha,dataset=job['dataset'],seed=job['seed'],source=job['source'],scores=scores,
            images=total,seconds=time.monotonic()-start,gpu=gpu,at=now(),monitor_state_unchanged=True)
        if e is not None:r.update({k:v for k,v in e.items() if k!='state_file'})
        if job.get('validation'):r['choices_sha256']=job['choices_sha256']
        write_json(dest,r);results.append(r);print('SCORE_PASS',job['id'],dest.name,flush=True)
    return dict(status='PASS',job=job['id'],score_records=len(results),gpu=gpu)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--job',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    reg=read_json(a.out/'registration.json');job=read_json(a.job)
    if a.gpu not in reg['allowed_gpus']:raise ValueError('Unauthorized GPU')
    from probe_validity.resources import acquire
    handles=[]
    try:
        handles,admission=acquire(a.gpu,reg['resource_policy'],'score' if job['kind']=='score' else 'update')
        os.environ.update(CUDA_VISIBLE_DEVICES=admission[-1]['uuid'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        import cv2
        import torch
        torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(6144*2**20/torch.cuda.get_device_properties(0).total_memory)
        result=(update if job['kind']=='update' else score)(reg,a.out,job,a.gpu)
        result.update(registration_sha256=sha256(a.out/'registration.json'),job_sha256=sha256(a.job),pid=os.getpid(),at=now(),admission=admission,keepalive_processes_signalled=0)
        write_json(a.out/'done'/(job['id']+'.json'),result)
    except Exception as error:
        resource=any(t in str(error) for t in ('RESOURCE_UNAVAILABLE','out of memory','temporarily unavailable','Resource temporarily unavailable'))
        write_json(a.out/'attempts'/job['id']/(str(time.time_ns())+'.json'),dict(status='RESOURCE_RETRY' if resource else 'FAILED',error=repr(error),at=now(),gpu=a.gpu))
        if resource:raise SystemExit(75)
        raise
    finally:
        for h in handles:h.close()

if __name__=='__main__':main()
