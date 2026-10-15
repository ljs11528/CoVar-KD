"""Durable P0 -> calibration -> P1 scheduler; no P2 or full-training path."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import fcntl
from pathlib import Path
import os
import subprocess
import sys
import threading
import time

from utility_study.io import read_json,sha256,write_json
from .contracts import calibration,PRIMARY
from .resources import inventory,admissible,available_host_mib


def now():return datetime.now(timezone.utc).isoformat()


class Queue:
    def __init__(self,out,durable):
        self.out,self.durable=Path(out),Path(durable);self.reg=read_json(self.out/"registration.json")
        self.busy={};self.pending={};self.finished=set();self.attempts={}
        self.state=dict(status="PREPARING",phase="preflight",pid=os.getpid(),started_at=now(),gpus=[0,1,2,3],
                        registration_sha256=sha256(self.out/"registration.json"),keepalive_processes_signalled=0,
                        old_server_model_commands=0,full_training_runs_launched=0)
        self.stop_backup=threading.Event();self.backup_lock=threading.Lock()

    def add(self,job):
        identity=job["id"]
        if identity in self.pending or identity in self.finished or any(v["job"]["id"]==identity for v in self.busy.values()):return
        path=self.out/"jobs"/(identity+".json")
        if path.exists() and read_json(path)!=job:raise RuntimeError("Task definition changed: "+identity)
        if not path.exists():write_json(path,job)
        done=self.out/"done"/(identity+".json")
        if done.exists():
            receipt=read_json(done)
            if receipt["status"]!="PASS" or receipt["registration_sha256"]!=sha256(self.out/"registration.json") or receipt["job_sha256"]!=sha256(path):
                raise RuntimeError("Completed job status or provenance differs")
            self.finished.add(identity);return
        self.pending[identity]=job

    def backup(self):
        from .archive import snapshot
        with self.backup_lock:return snapshot(self.out,self.durable)

    def background_backup(self):
        while not self.stop_backup.wait(120):
            try:self.backup()
            except Exception as error:write_json(self.out/"backup_failure.json",dict(at=now(),error=repr(error)))

    def tick(self):
        for gpu,row in list(self.busy.items()):
            code=row["child"].poll()
            if code is None:continue
            row["log"].close();job=row["job"];del self.busy[gpu]
            if code==0:
                done=self.out/"done"/(job["id"]+".json")
                if not done.exists():raise RuntimeError("Worker exited without completion receipt")
                self.finished.add(job["id"])
            elif code==75:
                # Retry only resource failures of the identical registered task.
                self.pending[job["id"]]=job
                self.attempts[job["id"]]=self.attempts.get(job["id"],0)+1
            else:raise RuntimeError(f"Worker failed ({code}): {job['id']}; see its preserved log/attempt")
        cards=inventory()
        for gpu in self.reg["allowed_gpus"]:
            if gpu in self.busy:continue
            candidates=[j for j in self.pending.values() if j.get("pinned_gpu",gpu)==gpu and admissible(cards[gpu],self.reg["resource_policy"],"score" if j["kind"]=="score" else "update")]
            active_updates=sum(r["job"]["kind"]=="update" for r in self.busy.values())
            required=self.reg["resource_policy"]["host_minimum_available_mib"]+(active_updates+1)*self.reg["resource_policy"]["host_reservation_per_update_mib"]
            if available_host_mib()<required:candidates=[j for j in candidates if j["kind"]!="update"]
            if not candidates:continue
            # Keep large-memory lanes generating endpoints until the small queue is ready.
            candidates.sort(key=lambda j:(j["kind"]=="score",j.get("priority",10),j["id"]))
            job=candidates[0];del self.pending[job["id"]]
            logpath=self.out/"logs"/(job["id"]+".log");logpath.parent.mkdir(parents=True,exist_ok=True)
            log=logpath.open("ab",buffering=0)
            cmd=[sys.executable,"-u","-m","probe_validity.worker","--out",str(self.out),"--job",str(self.out/"jobs"/(job["id"]+".json")),"--gpu",str(gpu)]
            child=subprocess.Popen(cmd,cwd=self.reg["code"],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            self.busy[gpu]=dict(job=job,child=child,log=log,started_at=now())
        compact={str(g):dict(job=r["job"]["id"],kind=r["job"]["kind"],pid=r["child"].pid,started_at=r["started_at"]) for g,r in self.busy.items()}
        self.state.update(updated_at=now(),running=compact,pending=len(self.pending),completed_jobs=len(self.finished),
                          gpu_memory={str(g):{k:v for k,v in c.items() if k!="processes"} for g,c in cards.items()})
        write_json(self.out/"workflow.json",self.state)

    def drain(self,jobs):
        targets=[j["id"] for j in jobs]
        for job in jobs:self.add(job)
        while not set(targets)<=self.finished:self.tick();time.sleep(3)

    def preparation(self):
        self.state["phase"]="teacher_cache"
        jobs=[dict(id=f"teacher_{d}_{pool}",kind="teacher",dataset=d,pool=pool,priority=0) for d in ("voc","cityscapes") for pool in ("A","B")]
        self.drain(jobs)
        pre=[]
        from .contracts import branch
        for d in ("voc","cityscapes"):
            branches=[]
            for bn in ("running","batch"):
                for drop in (False,True):
                    for storage in (False,True):
                        branches.append(branch(f"check_{bn}_drop{int(drop)}_"+("offload" if storage else "native"),bn,drop,"shared_c1",
                                               buffer_policy="native",engineering_batch2=True,activation_offload=storage))
            for bn,drop in (("running",False),("batch",True)):
                branches.append(branch(f"check_full_{bn}",bn,drop,"shared_c1",buffer_policy="native",activation_offload=True))
            pre.append(dict(id="preflight_update_"+d,kind="update",phase="engineering",dataset=d,seed=1234,stage=20000,stream="legacy",branches=branches,monitor="A"))
        self.state["phase"]="update_preflight";self.drain(pre)
        bases=[dict(id="preflight_base_"+d,kind="score",dataset=d,seed=1234,stage=20000,pool="A",baseline=True,preflight=True) for d in ("voc","cityscapes")]
        self.state["phase"]="score_preflight_baselines";self.drain(bases)
        city=read_json(self.out/"bundles/preflight_update_cityscapes.json")
        endpoint=next(e for e in city["endpoints"] if e["branch"]["name"]=="check_full_batch")
        jobs=[dict(id=f"preflight_score_gpu{g}",kind="score",dataset="cityscapes",seed=1234,stage=20000,pool="A",preflight=True,
                   endpoints=[dict(endpoint,id=endpoint["id"]+f"__preflight_gpu{g}")],pinned_gpu=g) for g in self.reg["allowed_gpus"]]
        self.state["phase"]="four_gpu_score_preflight";self.drain(jobs)
        scores=[read_json(self.out/"scores"/(j["endpoints"][0]["id"]+".json")) for j in jobs]
        ranges={m:max(r["scores"][m] for r in scores)-min(r["scores"][m] for r in scores) for m in PRIMARY}
        if any(v>(1e-4 if m=="hard_miou" else 1e-5) for m,v in ranges.items()):raise RuntimeError("Cross-card engineering replay exceeds predeclared tolerance")
        write_json(self.out/"preflight.json",dict(status="PASS",gpus=[0,1,2,3],native_cityscapes_images=2,cross_card_ranges=ranges,
                   allocation="One full-batch update lane wherever memory permits; all four eligible scoring lanes",engineering_excluded_from_primary=True,
                   score_memory_cap_mib=6144,update_memory_cap_mib=6144,activation_storage_checks=city["activation_storage_checks"],keepalive_processes_signalled=0))
        self.backup()

    def phase(self,phase,selected=None):
        tasks=[t for t in self.reg["tasks"] if t["phase"]==phase and (selected is None or selected(t))]
        ids=set(t["id"] for t in tasks)
        bases=[]
        for d,s,t,pool in sorted({(j["dataset"],j["seed"],j["stage"],j["monitor"]) for j in tasks}):
            base=dict(id=f"base_{d}_{s}_{t}_{pool}",kind="score",dataset=d,seed=s,stage=t,pool=pool,baseline=True,priority=0)
            self.add(base);bases.append(base)
        for row in tasks:self.add(dict(row,kind="update",priority=1))
        scores_added=set()
        while True:
            for row in tasks:
                if row["id"] in scores_added or row["id"] not in self.finished:continue
                base_id=f"base_{row['dataset']}_{row['seed']}_{row['stage']}_{row['monitor']}"
                if base_id not in self.finished:continue
                endpoints=read_json(self.out/"bundles"/(row["id"]+".json"))["endpoints"]
                if phase=="p0":
                    chunks=[[e for e in endpoints if e["condition"]==c] for c in ("ce","shared_c1","shared_c4")]
                else:chunks=[endpoints[i:i+4] for i in range(0,len(endpoints),4)]
                for i,chunk in enumerate(chunks):
                    self.add(dict(id=f"score_{row['id']}_{i}",kind="score",dataset=row["dataset"],seed=row["seed"],stage=row["stage"],pool=row["monitor"],endpoints=chunk))
                if phase=="p0" and row["mode"]=="train":
                    endpoint=next(e for e in endpoints if e["condition"]=="shared_c1" and e["repeat"]==0)
                    for g in self.reg["allowed_gpus"]:
                        self.add(dict(id=f"cross_{row['dataset']}_{row['stage']}_gpu{g}",kind="score",dataset=row["dataset"],seed=row["seed"],stage=row["stage"],pool="A",endpoints=[endpoint],cross_gpu=True,pinned_gpu=g))
                scores_added.add(row["id"])
            if scores_added==ids and not self.pending and not self.busy:break
            self.tick();time.sleep(3)

    def run(self):
        lock=(self.out/"controller.lock").open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        backup_thread=threading.Thread(target=self.background_backup,daemon=True);backup_thread.start()
        try:
            self.preparation()
            self.state.update(status="RUNNING",phase="p0");self.phase("p0")
            p0=[read_json(p) for p in (self.out/"scores").glob("*.json") if p.name.startswith("p0_")]
            cross=[read_json(p) for p in (self.out/"cross_gpu").glob("*.json")]
            gate=calibration(p0,cross);write_json(self.out/"p0_calibration.json",gate)
            if not gate["pass"]:
                self.state.update(status="CALIBRATION_UNRESOLVED",phase="p0",reason="Preserve all replays; inspect deterministic-kernel feasibility before P1")
                return
            self.state["phase"]="p1_discovery";self.phase("p1",lambda x:x["discovery"])
            self.state["phase"]="p1_confirmation";self.phase("p1",lambda x:not x["discovery"])
            from .summarize import summarize
            result=summarize(self.out)
            write_json(self.out/"scientific_completion.json",result)
            self.state.update(status="SCIENTIFIC_COMPLETE",phase="p0_p1_done",result_counts=result["counts"])
        except Exception as error:
            self.state.update(status="FAILED",error=repr(error),at=now())
            raise
        finally:
            # Only children belonging to this exact controller may be stopped.
            for row in self.busy.values():
                if row["child"].poll() is None:row["child"].terminate()
            self.state["updated_at"]=now();write_json(self.out/"workflow.json",self.state)
            self.stop_backup.set();backup_thread.join(timeout=2)
            try:
                receipt=self.backup();self.state["durable_backup"]=receipt
                if self.state["status"]=="SCIENTIFIC_COMPLETE":self.state["status"]="COMPLETED"
            except Exception as error:self.state["backup_pending"]=repr(error)
            write_json(self.out/"workflow.json",self.state);lock.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True);parser.add_argument("--durable",type=Path,required=True)
    args=parser.parse_args();Queue(args.out,args.durable).run()


if __name__=="__main__":main()
