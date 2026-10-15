"""Independent training-state interventions and pooled P15 measurements."""
from __future__ import annotations

import hashlib
import json
import math
import random

import torch
from torch import nn
import torch.nn.functional as F

from utility_study.core import PooledMetrics, resized_logits

BN=nn.modules.batchnorm._BatchNorm
DROPOUT=(nn.Dropout,nn.Dropout1d,nn.Dropout2d,nn.Dropout3d,nn.AlphaDropout,nn.FeatureAlphaDropout)


def module_inventory(model):
    result={"batchnorm":[],"dropout":[],"other_stochastic":[]}
    for name,module in model.named_modules():
        if isinstance(module,BN):
            if not module.track_running_stats:raise ValueError("Running-stat intervention needs stored BN statistics")
            result["batchnorm"].append({"name":name,"type":type(module).__name__,"momentum":module.momentum})
        elif isinstance(module,DROPOUT):
            result["dropout"].append({"name":name,"type":type(module).__name__,"p":module.p})
        elif "drop" in type(module).__name__.lower() or "stochastic" in type(module).__name__.lower():
            result["other_stochastic"].append({"name":name,"type":type(module).__name__})
    if result["other_stochastic"]:raise ValueError("Unregistered stochastic module: "+str(result["other_stochastic"]))
    if not result["batchnorm"] or not result["dropout"]:raise ValueError("Expected both BN and dropout in this frozen model")
    return result


def set_update_mode(model,bn,dropout):
    if bn not in ("batch","running"):raise ValueError("Unknown BN intervention")
    model.eval()
    for module in model.modules():
        if isinstance(module,BN):module.training=bn=="batch"
        elif isinstance(module,DROPOUT):module.training=bool(dropout)


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def fingerprint(value):
    h=hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            x=v.detach().cpu().contiguous()
            h.update(str((str(x.dtype),tuple(x.shape))).encode());h.update(x.numpy().tobytes())
        elif isinstance(v,dict):
            for key in sorted(v,key=lambda x:(type(x).__name__,str(x))):h.update(repr(key).encode());add(v[key])
        elif isinstance(v,(tuple,list)):
            h.update(type(v).__name__.encode())
            for x in v:add(x)
        else:h.update(repr(v).encode())
    add(value);return h.hexdigest()


def split_hashes(state,parameter_names):
    parameters={k:v for k,v in state.items() if k in parameter_names}
    buffers={k:v for k,v in state.items() if k not in parameter_names}
    return {"parameters_sha256":fingerprint(parameters),"buffers_sha256":fingerprint(buffers),
            "full_state_sha256":fingerprint(state)}


def old_buffers(state,baseline,parameter_names):
    return {k:(v if k in parameter_names else baseline[k]).clone() for k,v in state.items()}


def gradient_stats(named,ce_grad,kd_grad,alpha,momentum):
    rows=[];dot=0.;cn=0.;kn=0.
    for (name,p),a,b in zip(named,ce_grad,kd_grad):
        av=None if a is None else a.detach().double().cpu()
        bv=None if b is None else b.detach().double().cpu()
        aa=0. if av is None else float(av.square().sum());bb=0. if bv is None else float(bv.square().sum())
        ab=0. if av is None or bv is None else float((av*bv).sum())
        cn+=aa;kn+=bb;dot+=ab
        norm=float(p.detach().double().square().sum().sqrt())
        mv=momentum.get(p,{}).get("momentum_buffer")
        rows.append({"name":name,"parameter_norm":norm,"ce_norm":math.sqrt(aa),"kd_norm":math.sqrt(bb),
                     "kd_ce_cosine":ab/max(1e-30,math.sqrt(aa*bb)),
                     "saved_momentum_norm":0. if mv is None else float(mv.detach().double().square().sum().sqrt())})
    return {"ce_norm":math.sqrt(cn),"kd_norm":math.sqrt(kn),"kd_ce_cosine":dot/max(1e-30,math.sqrt(cn*kn)),
            "effective_kd_ce_norm_ratio":alpha*math.sqrt(kn)/max(1e-30,math.sqrt(cn)),"per_parameter":rows}


class DenseMetrics(PooledMetrics):
    """Same primary reductions as P14; extra calibration/teacher statistics."""
    def __init__(self,classes):
        super().__init__(classes)
        self.entropy=torch.zeros(classes,dtype=torch.float64)
        self.brier=torch.zeros(classes,dtype=torch.float64)
        self.teacher_wrong=torch.zeros(classes,dtype=torch.int64)
        self.correct_on_teacher_wrong=torch.zeros(classes,dtype=torch.int64)
        self.mimic_on_teacher_wrong=torch.zeros(classes,dtype=torch.int64)
        self.correct_on_teacher_right=torch.zeros(classes,dtype=torch.int64)
        self.sample_logp=[];self.sample_kl=[];self.teacher_measured=False

    @torch.no_grad()
    def add(self,logits,target,teacher=None,reference=None,sample_seed=0):
        z=resized_logits(logits,target)
        if z.shape[1]!=self.classes or not bool(torch.isfinite(z).all()):raise FloatingPointError("Invalid monitoring logits")
        valid=target!=self.ignore;y=target[valid].long()
        if not y.numel():raise ValueError("No valid image pixels")
        if bool(((y<0)|(y>=self.classes)).any()):raise ValueError("Invalid label")
        logp=F.log_softmax(z,dim=1).permute(0,2,3,1)[valid].double().cpu();y=y.cpu();p=logp.exp()
        py=p.gather(1,y[:,None])[:,0]
        self.ce+=torch.bincount(y,weights=-logp.gather(1,y[:,None])[:,0],minlength=self.classes)
        self.gt+=torch.bincount(y,minlength=self.classes)
        self.soft_intersection+=torch.bincount(y,weights=py,minlength=self.classes)
        self.soft_prediction+=p.sum(0)
        predicted=logp.argmax(1)
        self.confusion+=torch.bincount(self.classes*y+predicted,minlength=self.classes**2).reshape(self.classes,self.classes)
        self.entropy+=torch.bincount(y,weights=-(p*logp).sum(1),minlength=self.classes)
        self.brier+=torch.bincount(y,weights=p.square().sum(1)-2*py+1,minlength=self.classes)
        if teacher is not None:
            q=teacher.reshape(target.shape)[valid.cpu()].long().cpu()
            wrong=q!=y;correct=predicted==y
            self.teacher_wrong+=torch.bincount(y[wrong],minlength=self.classes)
            self.correct_on_teacher_wrong+=torch.bincount(y[wrong&correct],minlength=self.classes)
            self.mimic_on_teacher_wrong+=torch.bincount(y[wrong&(predicted==q)],minlength=self.classes)
            self.correct_on_teacher_right+=torch.bincount(y[(~wrong)&correct],minlength=self.classes)
            self.teacher_measured=True
        positions=random.Random(sample_seed).sample(range(y.numel()),min(256,y.numel()))
        sample=logp[positions].clone();sample-=torch.logsumexp(sample,dim=1,keepdim=True)
        self.sample_logp.append(sample)
        if reference is not None:
            if reference.shape!=sample.shape:raise ValueError("Mismatched function samples")
            self.sample_kl.append(float((reference.exp()*(reference-sample)).sum(1).mean()))

    def result(self):
        result=super().result();n=result["valid_pixels"]
        result.update(pixel_entropy=float(self.entropy.sum()/n),pixel_brier=float(self.brier.sum()/n),
                      class_entropy=(self.entropy/self.gt.clamp_min(1)).tolist(),
                      class_brier=(self.brier/self.gt.clamp_min(1)).tolist(),
                      teacher_measured=self.teacher_measured,teacher_wrong_counts=self.teacher_wrong.tolist(),
                      correct_on_teacher_wrong_counts=self.correct_on_teacher_wrong.tolist(),
                      mimic_on_teacher_wrong_counts=self.mimic_on_teacher_wrong.tolist(),
                      correct_on_teacher_right_counts=self.correct_on_teacher_right.tolist(),
                      sampled_function_kl=(sum(self.sample_kl)/len(self.sample_kl) if self.sample_kl else None),
                      function_sampling="256 uniform valid pixels/image; equal-image mean; float64 normalized log probabilities")
        return result
