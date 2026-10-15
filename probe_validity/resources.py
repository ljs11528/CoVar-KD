"""Read-only GPU admission; never signal another process."""
from pathlib import Path
import fcntl
import os
import subprocess
import time


def inventory():
    raw=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    cards={}
    for row in raw.splitlines():
        i,u,used,free,util=[v.strip() for v in row.split(",")]
        cards[int(i)]={"gpu":int(i),"uuid":u,"used_mib":int(used),"free_mib":int(free),"utilization":int(util),"processes":[]}
    apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader,nounits"],text=True)
    byuuid={v["uuid"]:v for v in cards.values()}
    for line in apps.splitlines():
        if not line.strip():continue
        pid,u,m=[v.strip() for v in line.split(",")];proc=Path("/proc")/pid
        try:
            cmd=(proc/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace").strip()
            cwd=os.readlink(proc/"cwd")
            start=(proc/"stat").read_text().rsplit(") ",1)[1].split()[19]
            uid=(proc/"status").read_text().split("Uid:",1)[1].split()[0]
        except (OSError,ProcessLookupError):continue
        byuuid[u]["processes"].append(dict(pid=int(pid),command=cmd[:1000],cwd=cwd,starttime=start,uid=int(uid),used_mib=m))
    return cards


def keeper(process,policy):
    cmd,cwd=process["command"],process["cwd"]
    for rule in policy["keeper_rules"]:
        if any(cmd.startswith(p) for p in rule["command_prefixes"]) and any(cwd==p or (p.endswith("/") and cwd.startswith(p)) for p in rule["cwd_prefixes"]):
            return process["uid"]==rule["uid"]
    return False


def admissible(card,policy,kind):
    minimum=policy["score_minimum_free_mib"] if kind=="score" else policy["update_minimum_free_mib"]
    return card["free_mib"]>=minimum and all(keeper(p,policy) for p in card["processes"])


def available_host_mib():
    value=next(row.split()[1] for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemAvailable:'))
    return int(value)//1024


def acquire(gpu,policy,kind):
    handles=[]
    try:
        for name in (f"/tmp/covar-kd-gpu-{gpu}.lock",f"/tmp/covar-p13-gpu-{gpu}.lock"):
            handle=open(name,"a+");handles.append(handle);fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        samples=[]
        for _ in range(2):
            card=inventory()[gpu];samples.append(card)
            if not admissible(card,policy,kind):raise RuntimeError("RESOURCE_UNAVAILABLE: "+str(card))
            if len(samples)==1:time.sleep(2)
        return handles,samples
    except Exception:
        for handle in handles:handle.close()
        raise
