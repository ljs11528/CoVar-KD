"""P2 detached four-card queue: probes -> frozen choices -> 2k forks -> val."""
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import argparse
import fcntl
import json
import os
import statistics
import subprocess
import sys
import threading
import time

from utility_study.io import read_json,write_json,sha256
from probe_validity.resources import inventory,admissible,available_host_mib
from probe_validity.archive import snapshot,store
from .contracts import tasks,state_id,decisions,gate,DATASETS,SEEDS,SOURCES

def now():return datetime.now(timezone.utc).isoformat()

class Queue:
    def __init__(self,out,durable):
        self.out,self.durable=Path(out),Path(durable);self.reg=read_json(self.out/'registration.json')
        self.busy={};self.pending={};self.finished=set();self.retry_after={};self.ooms=defaultdict(int)
        self.backup_lock=threading.Lock();self.stop=threading.Event()
        self.state=dict(status='PREPARING',phase='engineering',pid=os.getpid(),started_at=now(),gpus=[0,1,2,3],
            registration_sha256=sha256(self.out/'registration.json'),keepalive_processes_signalled=0,old_server_model_commands=0,full_80k_trajectories=0)

    def add(self,job):
        identity=job['id']
        if identity in self.finished or identity in self.pending or any(r['job']['id']==identity for r in self.busy.values()):return
        p=self.out/'jobs'/(identity+'.json')
        if p.exists() and read_json(p)!=job:raise RuntimeError('Job changed: '+identity)
        if not p.exists():write_json(p,job)
        done=self.out/'done'/(identity+'.json')
        if done.exists():
            r=read_json(done)
            if r['status']!='PASS' or r['job_sha256']!=sha256(p) or r['registration_sha256']!=self.state['registration_sha256']:raise RuntimeError('Completed job provenance differs')
            self.finished.add(identity)
        else:self.pending[identity]=job

    def backup(self):
        with self.backup_lock:
            for p in (self.out/'checkpoints').rglob('step_*.json'):
                r=read_json(p)
                if not r['persist']:continue
                key=r['state_file_sha256'];dest=self.durable/'checkpoint_objects'/key[:2]/key
                receipt=self.out/'checkpoint_archive'/key[:2]/(key+'.json')
                if receipt.exists():continue
                if sha256(r['state_file'])!=key:raise RuntimeError('Checkpoint changed before archive')
                data=Path(r['state_file']).read_bytes();store(dest,data)
                write_json(receipt,dict(status='PASS',object=str(dest),sha256=key,source=r['state_file'],at=now()))
            return snapshot(self.out,self.durable)

    def background(self):
        while not self.stop.wait(180):
            try:self.backup()
            except Exception as e:write_json(self.out/'backup_failure.json',dict(at=now(),error=repr(e)))

    def tick(self):
        for g,r in list(self.busy.items()):
            code=r['child'].poll()
            if code is None:continue
            r['log'].close();job=r['job'];del self.busy[g]
            if code==0:
                done=read_json(self.out/'done'/(job['id']+'.json'))
                if done['status']!='PASS' or done['job_sha256']!=sha256(self.out/'jobs'/(job['id']+'.json')):raise RuntimeError('Invalid job completion')
                self.finished.add(job['id'])
            elif code==75:
                attempts=sorted((self.out/'attempts'/job['id']).glob('*.json'))
                error=read_json(attempts[-1])['error']
                if 'out of memory' in error:
                    self.ooms[job['id']]+=1
                    if self.ooms[job['id']]>=3:raise RuntimeError('Three memory failures of unchanged task; preserve and audit: '+job['id'])
                self.pending[job['id']]=job;self.retry_after[job['id']]=time.monotonic()+30
            else:raise RuntimeError(f'Worker failure {code}: '+job['id'])
        cards=inventory()
        for g in self.reg['allowed_gpus']:
            if g in self.busy:continue
            candidates=[j for j in self.pending.values() if j.get('pinned_gpu',g)==g and self.retry_after.get(j['id'],0)<=time.monotonic()
                and admissible(cards[g],self.reg['resource_policy'],'score' if j['kind']=='score' else 'update')]
            active=sum(r['job']['kind']=='update' for r in self.busy.values())
            if available_host_mib()<32768+(active+1)*16384:candidates=[j for j in candidates if j['kind']!='update']
            if not candidates:continue
            candidates.sort(key=lambda j:(j.get('priority',1),j['kind']=='score',j['id']))
            job=candidates[0];del self.pending[job['id']]
            path=self.out/'logs'/(job['id']+'.log');path.parent.mkdir(parents=True,exist_ok=True);log=path.open('ab',buffering=0)
            child=subprocess.Popen([sys.executable,'-u','-m','horizon_validation.worker','--out',str(self.out),'--job',str(self.out/'jobs'/(job['id']+'.json')),'--gpu',str(g)],
                cwd=self.reg['code'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            self.busy[g]=dict(job=job,child=child,log=log,started_at=now())
        self.state.update(updated_at=now(),running={str(g):dict(job=r['job']['id'],kind=r['job']['kind'],pid=r['child'].pid,started_at=r['started_at']) for g,r in self.busy.items()},
            pending=len(self.pending),completed_jobs=len(self.finished),gpu_memory={str(g):{k:v for k,v in c.items() if k!='processes'} for g,c in cards.items()})
        write_json(self.out/'workflow.json',self.state)

    def drain(self,jobs):
        for j in jobs:self.add(j)
        target={j['id'] for j in jobs}
        while not target<=self.finished:self.tick();time.sleep(3)

    def baseline(self,d,s,source):return dict(id='base_'+state_id(d,s,source),kind='score',phase='baseline',dataset=d,seed=s,source=source,priority=2)

    def engineering(self):
        benches=[]
        for g,(d,source) in enumerate((('voc','ce'),('cityscapes','ce'),('voc','shared_c1'),('cityscapes','shared_c1'))):
            benches.append(dict(id=f'bench_{d}_{source}',kind='update',phase='engineering',dataset=d,seed=1234,source=source,coefficient=1,
                stream=0,mode='train',steps=16,horizons=[16],repeat=0,pinned_gpu=g,priority=0))
        self.drain(benches)
        import torch
        equivalence={}
        for d in DATASETS:
            for source in SOURCES:
                old=self.reg['engineering_reference'][f'{d}_{source}']
                new=read_json(self.out/'bundles'/f'bench_{d}_{source}.json')['endpoints'][0]
                if old['source_state_sha256']!=new['source_state_sha256']:raise RuntimeError('Engineering source states differ')
                a=torch.load(old['state_file'],map_location='cpu',weights_only=False)['student']
                b=torch.load(new['state_file'],map_location='cpu',weights_only=False)['student']
                error=max(float((a[k].double()-b[k].double()).abs().max()) for k in a)
                equivalence[f'{d}_{source}']=error
                if error>1e-4:raise RuntimeError('16-step engineering equivalence exceeds predeclared tolerance')
                del a,b
        write_json(self.out/'engineering/gradient_check_equivalence.json',dict(status='PASS',max_state_absolute_errors=equivalence,tolerance=1e-4,steps=16,activation_pin_memory=self.reg['activation_pin_memory'],at=now()))
        pre=[self.baseline(d,1234,c) for d in DATASETS for c in SOURCES]
        self.state['phase']='native_score_preflight';self.drain(pre)
        # CE baselines share exact P15 source weights/pool and primary reductions.
        parent=Path(self.reg['parent_registration']).parent
        errors={}
        for d in DATASETS:
            a=read_json(self.out/'baselines'/(state_id(d,1234,'ce')+'.json'))['scores']
            b=read_json(parent/'baselines'/d/'1234_20000_B.json')['scores']
            errors[d]={m:abs(a[m]-b[m]) for m in ('pixel_ce','class_ce','soft_miou','hard_miou')}
            if any(v>1e-5 for v in errors[d].values()):raise RuntimeError('P2 baseline scoring differs from frozen P15')
        write_json(self.out/'preflight.json',dict(status='PASS',gpus=[0,1,2,3],full_batch=True,steps_per_benchmark=16,primary_baseline_errors=errors,gradient_check_equivalence=equivalence,at=now()))
        self.estimate()

    def estimate(self):
        rates={};score={}
        for d in DATASETS:
            rates[d]=statistics.median(read_json(self.out/'bundles'/f'bench_{d}_{s}.json')['step_seconds']['steady_median'] for s in SOURCES)
            score[d]=statistics.median(read_json(self.out/'baselines'/(state_id(d,1234,s)+'.json'))['seconds'] for s in SOURCES)
        # Equal dataset workloads; full val bound uses monitor seconds/image as a conservative upper estimate.
        train_hours=sum(rates[d]*(90540/2) for d in DATASETS)/3600
        score_hours=sum(score[d]*(540/2+12/2) for d in DATASETS)/3600
        val_upper=sum(score[d]/256*self.reg['datasets'][d]['validation']['count']*18 for d in DATASETS)/3600
        ideal=(train_hours+score_hours+val_upper)/4
        write_json(self.out/'eta.json',dict(at=now(),step_seconds=rates,monitor_256_image_seconds=score,
            training_gpu_hours=train_hours,monitoring_gpu_hours=score_hours,validation_gpu_hours_upper_estimate=val_upper,
            ideal_four_card_hours=ideal,planning_wall_hours_range=[ideal*1.2,ideal*1.7],
            limitation='Benchmark extrapolation; capacity/keepalive competition and archive latency vary. Official val has not been scored before frozen choices.'))

    def probes(self):
        updates,_=tasks()
        for d in DATASETS:
            for s in SEEDS:
                for source in SOURCES:self.add(self.baseline(d,s,source))
        for t in updates:self.add(dict(t,priority=1))
        added=set()
        while True:
            for t in updates:
                if t['id'] in added or t['id'] not in self.finished:continue
                endpoints=read_json(self.out/'bundles'/(t['id']+'.json'))['endpoints']
                self.add(dict(id='score_'+t['id'],kind='score',phase='probe_score',dataset=t['dataset'],seed=t['seed'],source=t['source'],endpoints=endpoints,priority=3))
                added.add(t['id'])
            if len(added)==len(updates) and not self.pending and not self.busy:break
            self.tick();time.sleep(3)

    def freeze_choices(self):
        if list((self.out/'validation').glob('*.json')) and not (self.out/'choices.json').exists():raise RuntimeError('Validation observed before frozen choices')
        rows=[read_json(p) for p in (self.out/'scores').glob('*.json')]
        if len(rows)!=540:raise RuntimeError('Incomplete probe scoring')
        grouped=defaultdict(list)
        for r in rows:
            if r['status']!='PASS' or r['registration_sha256']!=self.state['registration_sha256']:raise RuntimeError('Probe provenance differs')
            grouped[state_id(r['dataset'],r['seed'],r['source'])].append(r)
        result={}
        for d in DATASETS:
            for s in SEEDS:
                for source in SOURCES:
                    sid=state_id(d,s,source);result[sid]=decisions(grouped[sid],self.reg['p0_bands'],d,source)
        payload=dict(status='FROZEN_BEFORE_VALIDATION',at=now(),registration_sha256=self.state['registration_sha256'],decisions=result,
            score_sha256={str(p.relative_to(self.out)):sha256(p) for p in sorted((self.out/'scores').glob('*.json'))})
        path=self.out/'choices.json'
        if path.exists():
            prior=read_json(path)
            if prior['decisions']!=result or prior['score_sha256']!=payload['score_sha256']:raise RuntimeError('Previously frozen choices differ')
        else:write_json(path,payload)
        data=path.read_bytes();target=self.durable/'blind_choices'/(sha256(path)+'.json');store(target,data)
        write_json(self.out/'choices_backup.json',dict(status='PASS',object=str(target),sha256=sha256(path),at=now()))
        self.backup()

    def forks(self):
        _,jobs=tasks();choice_sha=sha256(self.out/'choices.json');added=set()
        for t in jobs:self.add(dict(t,choices_sha256=choice_sha,priority=1))
        while True:
            for t in jobs:
                if t['id'] in added or t['id'] not in self.finished:continue
                endpoints=read_json(self.out/'bundles'/(t['id']+'.json'))['endpoints']
                self.add(dict(id='validate_'+t['id'],kind='score',phase='validation',dataset=t['dataset'],seed=t['seed'],source=t['source'],
                    endpoints=endpoints,validation=True,choices_sha256=choice_sha,priority=3))
                added.add(t['id'])
            if len(added)==len(jobs) and not self.pending and not self.busy:break
            self.tick();time.sleep(3)

    def summarize(self):
        choices=read_json(self.out/'choices.json');outcomes=[read_json(p) for p in (self.out/'validation').glob('*.json')]
        if len(outcomes)!=36:raise RuntimeError('Incomplete full validation coverage')
        rows=[];reductions={d:[] for d in DATASETS}
        for d in DATASETS:
            for s in SEEDS:
                differences=[]
                for source in SOURCES:
                    sid=state_id(d,s,source);values={r['coefficient']:r['scores']['hard_miou'] for r in outcomes if (r['dataset'],r['seed'],r['source'])==(d,s,source)}
                    if set(values)!={0,1,4}:raise RuntimeError('Missing fork coefficient')
                    best=max(values.values());rules=choices['decisions'][sid]['rules'];regrets={name:100*(best-values[r['coefficient']]) for name,r in rules.items()}
                    differences.append(regrets['train_pixel_ce_1']-regrets['train_hard_miou_128_consistent'])
                    rows.append(dict(dataset=d,seed=s,source=source,validation_miou=values,regret_pp=regrets,choices=rules))
                reductions[d].append(statistics.mean(differences))
        result=dict(status='PASS',registration_sha256=self.state['registration_sha256'],choices_sha256=sha256(self.out/'choices.json'),
            counts=self.reg['counts'],states=rows,gate=gate(reductions),inference_unit='3 training seeds; CE/KD source states and streams nested within seed',
            source_checkpoint_writes=0,keepalive_processes_signalled=0,old_server_model_commands=0,full_80k_trajectories=0)
        write_json(self.out/'scientific_completion.json',result);return result

    def run(self):
        lock=(self.out/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        thread=threading.Thread(target=self.background,daemon=True);thread.start()
        try:
            self.engineering();self.state.update(status='RUNNING',phase='probes');self.probes()
            self.state['phase']='freeze_choices';self.freeze_choices()
            self.state['phase']='forks_and_validation';self.forks();self.summarize()
            self.state.update(status='SCIENTIFIC_COMPLETE',phase='p2_done',running={},pending=0)
        except Exception as e:self.state.update(status='FAILED',error=repr(e));raise
        finally:
            for r in self.busy.values():
                if r['child'].poll() is None:r['child'].terminate()
            self.state['updated_at']=now();write_json(self.out/'workflow.json',self.state)
            self.stop.set();thread.join(timeout=2)
            try:
                self.state['durable_backup']=self.backup()
                if self.state['status']=='SCIENTIFIC_COMPLETE':self.state['status']='COMPLETED'
            except Exception as e:self.state['backup_pending']=repr(e)
            write_json(self.out/'workflow.json',self.state);lock.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--policy',type=Path,required=True);p.add_argument('--durable',type=Path,required=True);a=p.parse_args()
    from .register import register
    register(a.root,a.out,Path(__file__).resolve().parent.parent,a.policy)
    Queue(a.out,a.durable).run()

if __name__=='__main__':main()
