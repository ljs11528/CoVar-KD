"""Sequential, content-addressed OSS snapshots with readback verification."""
from datetime import datetime,timezone
import hashlib
import io
from pathlib import Path
import tarfile

from utility_study.io import read_json,sha256,write_json


def store(path,data):
    h=hashlib.sha256(data).hexdigest()
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("wb") as handle:handle.write(data)
    if sha256(path)!=h:raise RuntimeError("Persistent readback mismatch: "+str(path))
    return h


def snapshot(out,durable):
    out,durable=Path(out),Path(durable)
    samples=[]
    for path in (out/"baselines").rglob("*.samples.pt"):
        receipt=path.with_name(path.name.removesuffix(".samples.pt")+".json")
        if not receipt.exists():continue
        key=sha256(path)
        if key!=read_json(receipt)["samples_sha256"]:raise RuntimeError("Baseline sample differs from completed receipt")
        target=durable/"objects"/key[:2]/key
        if not target.exists():
            data=path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=key:raise RuntimeError("Baseline sample changed during backup")
            store(target,data)
        samples.append(dict(relative=str(path.relative_to(out)),sha256=key,object=str(target)))
    manifest=[];buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode="w:gz",compresslevel=1) as tar:
        paths=list(out.rglob("*.json"))
        paths += [out/"source_bundle.tar.gz"] if (out/"source_bundle.tar.gz").exists() else []
        for path in sorted(paths):
            if path.name.startswith("backup_"):continue
            data=path.read_bytes();relative=str(path.relative_to(out));h=hashlib.sha256(data).hexdigest()
            member=tarfile.TarInfo(relative);member.size=len(data);tar.addfile(member,io.BytesIO(data))
            manifest.append(dict(relative=relative,sha256=h,bytes=len(data)))
    data=buffer.getvalue();key=hashlib.sha256(data).hexdigest();target=durable/"snapshots"/(key+".tar.gz")
    store(target,data)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    import json
    index=dict(at=stamp,snapshot=str(target),snapshot_sha256=key,json_and_source_files=manifest,baseline_sample_objects=samples,
               excludes="Ephemeral virtual endpoint states; source checkpoints are separately preserved")
    index_data=(json.dumps(index,indent=2)+"\n").encode();idx=durable/"indexes"/(stamp+".json")
    ih=store(idx,index_data)
    receipt=dict(status="PASS",at=stamp,index=str(idx),index_sha256=ih,snapshot=str(target),snapshot_sha256=key,
                 scientific_json_count=sum(x["relative"].endswith(".json") for x in manifest),baseline_sample_objects=len(samples))
    write_json(out/"backup_receipt.json",receipt);return receipt
