"""Complete coverage and paired P1 controls, never treating streams as seeds."""
import csv
from collections import Counter
from pathlib import Path
import statistics

from utility_study.io import read_json,sha256,write_json
from .contracts import PRIMARY


def summarize(out):
    out=Path(out);reg=read_json(out/"registration.json")
    records=[read_json(p) for p in (out/"scores").glob("*.json")]
    p0=[r for r in records if r.get("phase")=="p0"]
    p1=[r for r in records if r.get("phase")=="p1"]
    assert len(p0)==72 and len(p1)==912
    assert all(r["monitor_images"]==256 and r["status"]=="PASS" and r["monitor_state_unchanged"] for r in p0+p1)
    assert all(r["registration_sha256"]==sha256(out/"registration.json") for r in p0+p1)
    bytask={}
    for r in p1:bytask.setdefault(r["task"],{})[r["id"].split("__",1)[1]]=r
    assert len(bytask)==36
    contrasts=[];directional=[]
    for task,items in sorted(bytask.items()):
        first=next(iter(items.values()))
        for bn in ("running","batch"):
            for drop in (0,1):
                policies=("native",) if bn=="running" else ("original","updated")
                for policy in policies:
                    a=items[f"factor_{bn}_drop{drop}_shared_c4__{policy}"]
                    b=items[f"factor_{bn}_drop{drop}_shared_c1__{policy}"]
                    for metric in PRIMARY:
                        contrasts.append(dict(dataset=first["dataset"],seed=first["seed"],stage=first["stage"],stream=first["stream"],
                                              bn=bn,dropout=drop,buffers=policy,metric=metric,c4_minus_c1=a["scores"][metric]-b["scores"][metric]))
        if first["seed"]==1234:
            for mode in ("eval","train"):
                bn,drop,policy=("running",0,"native") if mode=="eval" else ("batch",1,"original")
                center=items[f"factor_{bn}_drop{drop}_ce__{policy}"]
                plus,minus=items[f"direction_{mode}_plus"],items[f"direction_{mode}_minus"]
                for metric in ("pixel_ce","class_ce","soft_miou"):
                    a,b,c=plus["scores"][metric],minus["scores"][metric],center["scores"][metric]
                    directional.append(dict(dataset=first["dataset"],seed=first["seed"],stage=first["stage"],stream=first["stream"],mode=mode,
                                            metric=metric,coefficient_epsilon=.1,central_slope=(a-b)/.2,central_curvature=(a-2*c+b)/.01))
    for name,rows in (("factorial_contrasts.csv",contrasts),("directional_response.csv",directional)):
        with (out/name).open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Three flows form one within-training-seed observation for each stage/control.
    groups={}
    for row in contrasts:
        key=tuple(row[k] for k in ("dataset","seed","stage","bn","dropout","buffers","metric"))
        groups.setdefault(key,[]).append(row["c4_minus_c1"])
    nested=[]
    for key,values in sorted(groups.items()):
        assert len(values)==3
        nested.append(dict(zip(("dataset","seed","stage","bn","dropout","buffers","metric"),key),
                           stream_mean=statistics.mean(values),stream_min=min(values),stream_max=max(values),stream_sample_sd=statistics.stdev(values)))
    write_json(out/"seed_level_controls.json",nested)
    generated=[read_json(p) for p in (out/"generated").glob("p1_*/*.json")]
    optimizer_updates=sum(not r["spec"].get("buffer_only",False) for r in generated)
    assert optimizer_updates==576
    counts=dict(p0_primary=72,p0_cross_gpu=len(list((out/"cross_gpu").glob("*.json"))),p1_endpoints=912,
                p1_optimizer_update_observations=optimizer_updates,p1_buffer_only=72,p1_directional=48,
                by_dataset_phase=dict(Counter(r["dataset"]+":"+r["phase"] for r in p0+p1)))
    assert counts["p0_cross_gpu"]==16
    return dict(status="PASS",counts=counts,registration_sha256=sha256(out/"registration.json"),
                p0=read_json(out/"p0_calibration.json"),inference_unit="3 training seeds; 3 streams and 2 stages are nested repeated measures",
                keepalive_processes_signalled=0,old_server_experiments=0,full_training_runs_launched=0,
                limits="P0/P1 audit measurement and mechanisms; no result here establishes long-horizon predictive accuracy or a selector gain")
