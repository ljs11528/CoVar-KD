"""Construct exactly the frozen original models and preprocessing on Maker."""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch

from utility_study.core import isolated_rng
from utility_study.io import sha256
from utility_study.register import CITY_IDS,validate_checkpoint_metadata


class Runtime:
    def __init__(self,reg,dataset,worker,role):
        self.reg,self.dataset=reg,dataset;self.cfg=reg["datasets"][dataset]
        for path,h in reg["source_files"].items():
            if sha256(path)!=h:raise RuntimeError("Registered source changed: "+path)
        self.config=self.cfg
        self.runtime=Path(self.cfg["runtime"]);sys.path.insert(0,str(self.runtime))
        os.chdir(self.runtime)
        for p,h in self.cfg["input_sha256"].items():
            if Path(p).suffix==".py" and sha256(p)!=h:raise RuntimeError("Frozen runtime changed")
        if sha256(self.cfg["teacher"])!=self.cfg["input_sha256"][self.cfg["teacher"]]:raise RuntimeError("Teacher bytes changed")
        spec=importlib.util.spec_from_file_location("p15_frozen_trainer",self.runtime/"train_kd_3090.py")
        mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
        mod.logger=logging.getLogger("p15")
        ref=next(r for r in self.cfg["runs"] if r["seed"]==1234)
        args=argparse.Namespace(**ref["args"])
        args.data,args.teacher_pretrained=self.cfg["data"],self.cfg["teacher"]
        args.batch_size=self.cfg["matched_batch"];args.workers=0;args.max_iterations=1;args.resume=None
        args.student_pretrained_base=args.student_pretrained="None"
        args.log_dir=str(worker/"constructor");args.save_dir=str(worker/"unused-checkpoints")
        args.device=args.device_type="cuda";args.distributed=False;args.num_gpus=1;args.local_rank=0
        trainer=mod.Trainer(args);trainer.t_model.eval()
        self.trainer,self.module=trainer,mod
        self.student,self.teacher,self.optimizer=trainer.s_model,trainer.t_model,trainer.optimizer
        for name in ("criterion_cwd","criterion_fitnet"):getattr(trainer,name).eval()
        if role=="score":
            self.teacher=None;self.optimizer=None
            del trainer.t_model,trainer.optimizer,trainer.criterion_cwd,trainer.criterion_fitnet
            torch.cuda.empty_cache()
        if role=="teacher":
            self.student=None;self.optimizer=None
            del trainer.s_model,trainer.optimizer,trainer.criterion_cwd,trainer.criterion_fitnet
            torch.cuda.empty_cache()
        self.lookup=np.full(256,-1,dtype=np.int64)
        for k,v in enumerate(CITY_IDS):self.lookup[v]=k
        self.data=None;self.args=args

    def source(self,seed,stage):
        run=next(r for r in self.cfg["runs"] if r["seed"]==seed)
        path=run["checkpoints"][str(stage)]
        if sha256(path)!=self.cfg["input_sha256"][path]:raise RuntimeError("Original checkpoint changed")
        state=torch.load(path,map_location="cpu",weights_only=False)
        validate_checkpoint_metadata(state,run,stage)
        self.student.load_state_dict(state["student"],strict=True);self.student.eval()
        if self.optimizer is not None:
            for name in ("criterion_cwd","criterion_fitnet"):
                getattr(self.trainer,name).load_state_dict(state[name],strict=True)
            self.optimizer.load_state_dict(state["optimizer"])
            self.module.move_optimizer_state_to_device(self.optimizer,self.trainer.device)
            self.optimizer.zero_grad(set_to_none=True)
        return run,path

    def update_batch(self,entry):
        if self.data is None:
            if self.dataset=="voc":
                from dataset.voc import VOCDataTrainSet
                self.data=VOCDataTrainSet(self.cfg["data"],self.cfg["train_list"],crop_size=tuple(self.args.crop_size),scale=True,mirror=True,ignore_label=-1)
            else:
                from dataset.cityscapes import CSTrainValSet
                self.data=CSTrainValSet(self.cfg["data"],self.cfg["train_list"],crop_size=tuple(self.args.crop_size),scale=True,mirror=True,ignore_label=-1)
        with isolated_rng(entry["augmentation_seed"]):items=[self.data[i] for i in entry["indices"]]
        x=torch.stack([torch.from_numpy(i[0]) for i in items]).cuda()
        y=torch.stack([torch.from_numpy(i[1]).long() for i in items]).cuda()
        return x,y

    def monitor_image(self,item,verify=True):
        if verify and (sha256(item["img"])!=item["image_sha256"] or sha256(item["label"])!=item["label_sha256"]):raise RuntimeError("Monitor bytes changed")
        image=cv2.imread(item["img"],cv2.IMREAD_COLOR)
        if image is None:raise ValueError("Unreadable monitoring image")
        image=image.astype(np.float32)-np.array([104.00698793,116.66876762,122.67891434])
        image=np.ascontiguousarray(image.transpose(2,0,1),dtype=np.float32)
        with Image.open(item["label"]) as f:label=np.asarray(f).astype(np.int64)
        if self.dataset=="cityscapes":label=self.lookup[label]
        else:label[label==255]=-1
        return torch.from_numpy(image)[None].cuda(),torch.from_numpy(label)[None].cuda()
