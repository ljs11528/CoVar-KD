"""Audit and summarize frozen P14 diagnostic JSON; no training or GPU access.

Usage: python tools/analyze_diagnostics_324.py --raw /path/to/extracted/archive
       --out reports/cvpr2027/diagnostics_324_20261015
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics as st

DATASETS = ("voc", "cityscapes")
SEEDS = (1234, 2025, 3407)
STAGES = (20000, 40000, 60000)
MODES = ("legacy_eval", "matched_eval", "matched_train")
CONDITIONS = ("ce", "shared_c1", "shared_c4", "teacher_c1", "teacher_c4", "shared_norm")
METRICS = ("pixel_ce", "class_ce", "soft_miou", "hard_miou")
CONTRASTS = (("shared_c4", "shared_c1"), ("teacher_c4", "teacher_c1"),
             ("teacher_c1", "shared_norm"), ("teacher_c1", "shared_c1"),
             ("teacher_c4", "shared_c4")) + tuple((c, "ce") for c in CONDITIONS[1:])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def describe(values):
    values = list(values)
    return dict(n=len(values), mean=st.mean(values), sample_sd=st.stdev(values),
                median=st.median(values), minimum=min(values), maximum=max(values),
                positive=sum(x > 0 for x in values), negative=sum(x < 0 for x in values))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def export_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def coefficient(scores, metric):
    best = min(scores.values()) if metric.endswith("ce") else max(scores.values())
    tied = [c for c, value in scores.items() if abs(value-best) <= 1e-6*max(1., abs(best))]
    return 1 if 1 in tied else min(tied)


def audit(raw):
    download = read(raw/"download_manifest.json")
    for item in download["files"]:
        path = raw/item["path"]
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
    registration = read(raw / "registration.json")
    reg_sha = sha(raw / "registration.json")
    parent_sha = sha(raw / "inheritance/parent_registration.json")
    inherited = read(raw / "inheritance/records.json")["files"]
    assert parent_sha == registration["continuation"]["parent_registration_sha256"]
    assert sha(raw / "inheritance/records.json") == registration["continuation"]["records_manifest_sha256"]
    paths = sorted((raw / "one_step").glob("*/*/*/*/*/*.json"))
    records = [read(p) for p in paths]
    expected = set(itertools.product(DATASETS, ("ce",), SEEDS, STAGES, MODES, CONDITIONS))
    actual = {(r["dataset"], r["cohort"], r["seed"], r["checkpoint"], r["mode"], r["condition"]) for r in records}
    assert len(records) == len(expected) == len(actual) == 324 and actual == expected
    largest_metric_error = 0.
    lineage = Counter()
    for path, r in zip(paths, records):
        relative = str(path.relative_to(raw))
        expected_path = f"one_step/{r['dataset']}/ce/{r['seed']}/{r['checkpoint']}/{r['mode']}/{r['condition']}.json"
        assert relative == expected_path
        assert r["status"] == "PASS" and r["state_restored"] and r["horizon"] == 1
        assert r["failed_virtual_step"] is None and r["monitor_images"] == 256
        assert r["batch_size"] == (2 if r["mode"] == "legacy_eval" else registration["datasets"][r["dataset"]]["matched_batch"])
        assert r["bn_counter_delta"] == (53 if r["mode"] == "matched_train" else 0)
        if r["registration_sha256"] == parent_sha:
            assert inherited[relative] == sha(path)
            lineage["inherited_byte_verified"] += 1
        else:
            assert r["registration_sha256"] == reg_sha and relative not in inherited
            lineage["r3_new"] += 1
        config = registration["datasets"][r["dataset"]]
        run = next(x for x in config["runs"] if x["condition"] == "ce" and x["seed"] == r["seed"])
        assert r["checkpoint_sha256"] == config["input_sha256"][run["checkpoints"][str(r["checkpoint"])]]
        for score in (r["before"], r["after"]["1"]):
            assert score["all_classes_present"] and sum(score["gt_counts"]) == score["valid_pixels"]
            n = len(score["gt_counts"])
            count = score["gt_counts"]
            confusion = score["confusion"]
            assert [sum(x) for x in confusion] == count
            pixel = sum(x*y for x, y in zip(count, score["class_ce_values"])) / sum(count)
            soft = [score["soft_intersection"][k] / (count[k]+score["soft_prediction"][k]-score["soft_intersection"][k]) for k in range(n)]
            hard = [confusion[k][k] / (sum(confusion[k])+sum(x[k] for x in confusion)-confusion[k][k]) for k in range(n)]
            reconstructed = dict(pixel_ce=pixel, class_ce=st.mean(score["class_ce_values"]), soft_miou=st.mean(soft), hard_miou=st.mean(hard))
            for metric, value in reconstructed.items():
                error = abs(score[metric]-value)
                largest_metric_error = max(error, largest_metric_error)
                assert error < 1e-12
    index = {(r["dataset"], r["seed"], r["checkpoint"], r["mode"], r["condition"]): r for r in records}
    largest_gradient_relative_error = 0.
    largest_local_ce_spread = 0.
    largest_norm_relative_error = 0.
    for d, s, t in itertools.product(DATASETS, SEEDS, STAGES):
        group = [index[d,s,t,m,c] for m,c in itertools.product(MODES,CONDITIONS)]
        assert all(r["before"] == group[0]["before"] for r in group)
        for m in MODES:
            local = [index[d,s,t,m,c] for c in CONDITIONS]
            local_losses = [r["local_ce_losses"][0] for r in local]
            spread = max(local_losses)-min(local_losses)
            largest_local_ce_spread = max(spread, largest_local_ce_spread)
            assert spread < 1e-5*max(1., max(local_losses))
            for a,b in (("shared_c1","shared_c4"),("teacher_c1","teacher_c4")):
                ga,gb = index[d,s,t,m,a]["first_gradient"],index[d,s,t,m,b]["first_gradient"]
                for k in ("ce_norm","kd_norm","kd_ce_cosine"):
                    relative = abs(ga[k]-gb[k])/max(abs(ga[k]),abs(gb[k]),1e-30)
                    largest_gradient_relative_error = max(relative, largest_gradient_relative_error)
                    assert relative < 1e-5
            norm = index[d,s,t,m,"shared_norm"]
            ref = index[d,s,t,m,"teacher_c1"]
            target = ref["first_gradient"]["kd_norm"]
            error = abs(norm["coefficients"][0]*norm["first_gradient"]["kd_norm"]-target)/target
            largest_norm_relative_error = max(error,largest_norm_relative_error)
            assert error < 1e-5
        streams = registration["datasets"][d]["updates"]
        ev,tr,leg = [streams[f"{s}_{t}_{m}"] for m in ("matched_eval","matched_train","legacy_eval")]
        assert ev == tr
        assert ev[0]["augmentation_seed"] == leg[0]["augmentation_seed"]
        assert ev[0]["indices"][:2] == leg[0]["indices"]
    return registration, records, index, dict(downloaded_files_sha_verified=len(download["files"]),
            records=324, cases=18, branches_per_case=18,
            lineage=dict(lineage), registration_sha256=reg_sha, parent_registration_sha256=parent_sha,
            complete_expected_factorial=True, before_identical_per_case=True,
            input_stream_pairing_verified=True, condition_gradient_replay_within_relative_1e_5=True,
            maximum_gradient_replay_relative_error=largest_gradient_relative_error,
            maximum_local_ce_replay_spread=largest_local_ce_spread,
            maximum_norm_control_relative_error=largest_norm_relative_error,
            norm_control_verified=True, state_restore_assertions=324,
            maximum_metric_reconstruction_error=largest_metric_error)


def analyze(raw, out):
    registration, records, index, checks = audit(raw)
    out.mkdir(parents=True, exist_ok=True)
    case_rows, stage_rows, seed_rows = [], [], []
    for d,m,(a,b),metric in itertools.product(DATASETS,MODES,CONTRASTS,METRICS):
        for s,t in itertools.product(SEEDS,STAGES):
            value = index[d,s,t,m,a]["after"]["1"][metric]-index[d,s,t,m,b]["after"]["1"][metric]
            case_rows.append(dict(dataset=d,mode=m,contrast=f"{a}-{b}",metric=metric,seed=s,stage=t,difference=value))
        for t in STAGES:
            values={str(s):index[d,s,t,m,a]["after"]["1"][metric]-index[d,s,t,m,b]["after"]["1"][metric] for s in SEEDS}
            stage_rows.append(dict(dataset=d,mode=m,contrast=f"{a}-{b}",metric=metric,stage=t,
                                   **values,mean=st.mean(values.values()),sample_sd=st.stdev(values.values())))
        for s in SEEDS:
            values = [index[d,s,t,m,a]["after"]["1"][metric]-index[d,s,t,m,b]["after"]["1"][metric] for t in STAGES]
            seed_rows.append(dict(dataset=d,mode=m,contrast=f"{a}-{b}",metric=metric,seed=s,stage_average=st.mean(values)))
    export_csv(out/"case_contrasts.csv",case_rows)
    export_csv(out/"stage_contrasts.csv",stage_rows)
    export_csv(out/"seed_stage_averages.csv",seed_rows)
    pooled=[]
    for d,m,c in itertools.product(DATASETS,MODES,CONDITIONS):
        group=[index[d,s,t,m,c] for s,t in itertools.product(SEEDS,STAGES)]
        gradients={k:describe(r["first_gradient"][k] for r in group) for k in group[0]["first_gradient"]}
        effect={metric:describe(r["after"]["1"][metric]-index[d,r["seed"],r["checkpoint"],m,"ce"]["after"]["1"][metric] for r in group) for metric in METRICS}
        drift={metric:describe(r["after"]["1"][metric]-r["before"][metric] for r in group) for metric in METRICS}
        pooled.append(dict(dataset=d,mode=m,condition=c,gradient=gradients,effect_vs_ce=effect,
                           drift_vs_before=drift,alpha=describe(r["coefficients"][0] for r in group)))
    mode_comparison=[]
    metric_comparison=[]
    selections=[]
    for d in DATASETS:
        entries=[]
        for s,t in itertools.product(SEEDS,STAGES):
            values={m:{metric:index[d,s,t,m,"shared_c4"]["after"]["1"][metric]-index[d,s,t,m,"shared_c1"]["after"]["1"][metric] for metric in METRICS} for m in MODES}
            entries.append(dict(seed=s,stage=t,values=values))
        mode_comparison.append(dict(dataset=d,case_count=9,
            eval_train_pixel_sign_flips=sum(e["values"]["matched_eval"]["pixel_ce"]*e["values"]["matched_train"]["pixel_ce"]<0 for e in entries),
            cases=entries))
        for m in MODES:
            group=[e["values"][m] for e in entries]
            metric_comparison.append(dict(dataset=d,mode=m,cases=9,
                ce_hard_disagreements=sum(x["pixel_ce"]*x["hard_miou"]>0 for x in group),
                ce_soft_disagreements=sum(x["pixel_ce"]*x["soft_miou"]>0 for x in group),
                classce_hard_disagreements=sum(x["class_ce"]*x["hard_miou"]>0 for x in group),
                soft_hard_disagreements=sum(x["soft_miou"]*x["hard_miou"]<0 for x in group)))
    for d,s,t,m in itertools.product(DATASETS,SEEDS,STAGES,MODES):
        choices={metric:coefficient({c:index[d,s,t,m,"ce" if c==0 else f"shared_c{c}"]["after"]["1"][metric] for c in (0,1,4)},metric) for metric in METRICS}
        selections.append(dict(dataset=d,seed=s,stage=t,mode=m,**choices))
    selection_summary=[]
    for d,m in itertools.product(DATASETS,MODES):
        group=[x for x in selections if x["dataset"]==d and x["mode"]==m]
        selection_summary.append(dict(dataset=d,mode=m,
            choice_counts={metric:dict(Counter(x[metric] for x in group)) for metric in METRICS},
            ce_soft_differ=sum(x["pixel_ce"]!=x["soft_miou"] for x in group),
            soft_hard_differ=sum(x["soft_miou"]!=x["hard_miou"] for x in group)))
    class_rows=[]
    for d,s,t in itertools.product(DATASETS,SEEDS,STAGES):
        a,b=[index[d,s,t,"matched_train",c]["after"]["1"] for c in ("shared_c4","shared_c1")]
        n=len(a["gt_counts"])
        for k in range(n):
            ce=a["class_ce_values"][k]-b["class_ce_values"][k]
            weight=a["gt_counts"][k]/sum(a["gt_counts"])
            class_rows.append(dict(dataset=d,seed=s,stage=t,class_id=k,pixel_fraction=weight,
                class_ce_difference=ce,pixel_ce_contribution=ce*weight,
                soft_iou_difference=a["soft_iou"][k]-b["soft_iou"][k],
                hard_iou_difference=a["hard_iou"][k]-b["hard_iou"][k]))
    export_csv(out/"matched_train_per_class.csv",class_rows)
    class_summary=[]
    for d,t in itertools.product(DATASETS,STAGES):
        group=[r for r in class_rows if r["dataset"]==d and r["stage"]==t]
        for k in range(registration["datasets"][d]["classes"]):
            subset=[r for r in group if r["class_id"]==k]
            class_summary.append(dict(dataset=d,stage=t,class_id=k,pixel_fraction=subset[0]["pixel_fraction"],
                **{key:st.mean(r[key] for r in subset) for key in ("class_ce_difference","pixel_ce_contribution","soft_iou_difference","hard_iou_difference")}))
    export_csv(out/"matched_train_class_stage_means.csv",class_summary)
    export_csv(out/"one_step_choices.csv",selections)
    # A descriptive rank comparison, never 324 independent replicates.
    from scipy.stats import spearmanr
    ranks=[]
    for d,s,t,m in itertools.product(DATASETS,SEEDS,STAGES,MODES):
        values={k:[index[d,s,t,m,c]["after"]["1"][k]*(-1 if k.endswith("ce") else 1) for c in CONDITIONS] for k in METRICS}
        ranks.append(dict(dataset=d,seed=s,stage=t,mode=m,
            pixelce_soft_spearman=float(spearmanr(values["pixel_ce"],values["soft_miou"]).statistic),
            pixelce_hard_spearman=float(spearmanr(values["pixel_ce"],values["hard_miou"]).statistic),
            soft_hard_spearman=float(spearmanr(values["soft_miou"],values["hard_miou"]).statistic)))
    export_csv(out/"candidate_rank_correlations.csv",ranks)
    time_summary={d:dict(sum_branch_seconds=sum(r["seconds"] for r in records if r["dataset"]==d),
        median_branch_seconds=st.median(r["seconds"] for r in records if r["dataset"]==d)) for d in DATASETS}
    summary=dict(audit=checks,old_gate=read(raw/"one_step_gate.json"),workflow=read(raw/"workflow.json"),
        units="CE raw loss; IoU fractions, multiply by 100 for percentage points. Seed is the replication unit; stages and modes are repeated.",
        descriptive_9case_summaries="9 combines 3 repeated stages per training seed, not 9 independent runs; no significance claim.",
        pooled_conditions=pooled,mode_comparison=mode_comparison,metric_comparison=metric_comparison,
        selection_summary=selection_summary,branch_times=time_summary)
    dump(out/"summary.json",summary)
    figures(index,out)
    print(json.dumps(dict(audit=checks,mode_comparison=[{k:v for k,v in x.items() if k!='cases'} for x in mode_comparison],
                        metric_comparison=metric_comparison,selection_summary=selection_summary,
                        times=time_summary),indent=2))


def figures(index,out):
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"covar-kd-analysis-matplotlib"))
    out=out/"figures"
    out.mkdir(exist_ok=True)
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                        "savefig.dpi":180,"pdf.fonttype":42})
    colors={"legacy_eval":"#a44935","matched_eval":"#b68d20","matched_train":"#206c9e"}
    fig,axes=plt.subplots(2,3,figsize=(12,6.8))
    for i,d in enumerate(DATASETS):
        for j,metric in enumerate(("pixel_ce","soft_miou","hard_miou")):
            ax=axes[i,j];scale=100 if "miou" in metric else 1
            for mo,m in enumerate(MODES):
                means=[]
                for ti,t in enumerate(STAGES):
                    values=[scale*(index[d,s,t,m,"shared_c4"]["after"]["1"][metric]-index[d,s,t,m,"shared_c1"]["after"]["1"][metric]) for s in SEEDS]
                    means.append(st.mean(values))
                    ax.scatter(ti+np.array([-.07,0,.07])+(mo-1)*.13,values,s=19,alpha=.7,color=colors[m])
                ax.plot(np.arange(3)+(mo-1)*.13,means,label=m,color=colors[m],marker="_",lw=1.5)
            ax.axhline(0,color="gray",lw=.8)
            ax.set_yscale("symlog",linthresh=1e-3 if metric=="pixel_ce" else .1)
            ax.set_xticks(range(3),["20k","40k","60k"])
            ax.set_title(d.upper()+": "+metric)
            ax.set_ylabel("c4 minus c1 (loss)" if metric=="pixel_ce" else "c4 minus c1 (pp)")
            ax.grid(axis="y",alpha=.17)
            ax.margins(y=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc="lower center",frameon=False)
    fig.suptitle("Same CE checkpoints: update mode changes the local verdict\nPoints: 3 training seeds; lines: seed means. Symmetric-log axes show both scales.",fontsize=12)
    fig.tight_layout(rect=(0,.06,1,.97))
    for ext in ("png","pdf"):fig.savefig(out/f"state_contrasts.{ext}")
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6.6))
    for i,d in enumerate(DATASETS):
        for j,metrics in enumerate((("pixel_ce","class_ce"),("soft_miou","hard_miou"))):
            ax=axes[i,j];scale=1 if j==0 else 100
            for mi,metric in enumerate(metrics):
                means=[];color=("#246b91","#b34e3a")[mi]
                for ti,t in enumerate(STAGES):
                    vals=[scale*(index[d,s,t,"matched_train","shared_c4"]["after"]["1"][metric]-index[d,s,t,"matched_train","shared_c1"]["after"]["1"][metric]) for s in SEEDS]
                    means.append(st.mean(vals));ax.scatter(ti+np.array([-.06,0,.06])+(mi-.5)*.12,vals,s=23,color=color,alpha=.65)
                ax.plot(np.arange(3)+(mi-.5)*.12,means,"o-",color=color,label=metric,ms=4)
            ax.axhline(0,color="gray",lw=.8);ax.set_xticks(range(3),["20k","40k","60k"])
            ax.set_title(d.upper()+": matched_train");ax.set_ylabel("c4 minus c1 (loss)" if j==0 else "c4 minus c1 (pp)")
            ax.legend(frameon=False);ax.grid(axis="y",alpha=.17)
    fig.suptitle("Metric differences remain after matching the update protocol\nAll seed observations retained; stage means do not imply significance.",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.98))
    for ext in ("png","pdf"):fig.savefig(out/f"matched_metrics.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    analyze(args.raw,args.out)
