"""Maker-only CPU verification, immutable registration, then P0/P1 execution."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import os
import subprocess
import sys

from utility_study.io import sha256,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--durable",type=Path,required=True);parser.add_argument("--policy",type=Path,required=True)
    args=parser.parse_args();code=Path(__file__).resolve().parent.parent
    if args.root.resolve()!=Path("/tmp/liujinshi-covar-kd-20261015"):raise RuntimeError("Maker execution root required")
    args.out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
    checked=subprocess.run([sys.executable,"-m","pytest","-q","tests/test_probe_validity.py","tests/test_utility_study.py"],cwd=code,env=env,
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    receipt=dict(status="PASS" if checked.returncode==0 else "FAILED",returncode=checked.returncode,output=checked.stdout,
                 at=datetime.now(timezone.utc).isoformat(),location="Maker CPU; CUDA disabled",source_bundle_sha256=sha256(args.out/"source_bundle.tar.gz"))
    write_json(args.out/"engineering"/("cpu_"+receipt["source_bundle_sha256"]+".json"),receipt)
    write_json(args.out/"cpu_verification.json",receipt)
    print(checked.stdout,flush=True);checked.check_returncode()
    from .register import main as register
    register(args.root,args.out,code,args.policy)
    from .workflow import Queue
    Queue(args.out,args.durable).run()


if __name__=="__main__":main()
