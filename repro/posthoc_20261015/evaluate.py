#!/usr/bin/env python3
"""Full validation of frozen checkpoints; compact teacher prediction reuse."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/data1/youwu/jingshi/kd/CoVar-KD-3090')
OUT=ROOT/'reports/P13_CrossDataset_Posthoc_20261015'

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def gpu_setup(gpu):
    assert gpu in [0,2]
    lock=open(f'/tmp/covar-p13-gpu-{gpu}.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    selected=next([int(v) for v in line.split(',')] for line in rows.splitlines() if int(line.split(',')[0])==gpu)
    if selected[1]>=1024 or selected[2]>10:raise RuntimeError(f'GPU {gpu} is busy: {selected}')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    for k in ['WORLD_SIZE','RANK','LOCAL_RANK']:os.environ.pop(k,None)
    return lock

def run(dataset,gpu):
    import numpy as np
    import torch
    import torch.nn.functional as F
    registration=json.loads((OUT/'registration.json').read_text());ds=registration['datasets'][dataset]
    work=OUT/'inference'/dataset;work.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(20261014)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    runtime=Path(ds['runtime']);sys.path.insert(0,str(runtime));os.chdir(runtime)
    from models.model_zoo import get_segmentation_model
    if dataset=='voc':
        from dataset.voc import VOCDataValSet
        data=VOCDataValSet(ds['data'],ds['val_list'],ignore_label=-1)
    else:
        from dataset.cityscapes import CSTrainValSet
        data=CSTrainValSet(ds['data'],ds['val_list'],crop_size=(1024,2048),scale=False,mirror=False,ignore_label=-1)
    assert len(data)==ds['val_count'];classes=ds['classes']
    names=[x['name'] for x in data.files];assert len(set(names))==len(names)
    loader=torch.utils.data.DataLoader(data,batch_size=1,shuffle=False,num_workers=4,pin_memory=True)
    cache=work/'teacher_predictions';cache.mkdir(exist_ok=True)
    started=time.monotonic();completed=[]
    jobs=[dict(id='teacher',checkpoint=ds['teacher'])]+[dict(id=r['id'],checkpoint=r['checkpoints']['80000'],run=r) for r in ds['runs']]

    def measure(matrix):
        inter=np.diag(matrix);union=matrix.sum(0)+matrix.sum(1)-inter
        iou=np.divide(inter,union,out=np.zeros(classes,dtype=np.float64),where=union>0)
        return dict(miou_percent=float(iou.mean()*100),class_iou_percent=(iou*100).tolist(),pixacc_percent=float(inter.sum()/matrix.sum()*100))

    # A synthetic ignored-label case checks the independent count implementation.
    target=np.array([-1,0,1,2]);pred=np.array([0,0,2,2]);keep=target>=0
    cm=np.bincount(classes*target[keep]+pred[keep],minlength=classes**2).reshape(classes,classes)
    assert cm.sum()==3 and np.trace(cm)==2 and cm[1,2]==1
    for job in jobs:
        dest=work/(job['id']+'.json');statfile=work/(job['id']+'.npz')
        if dest.exists():
            existing=json.loads(dest.read_text())
            assert existing['status']=='PASS' and existing['checkpoint_sha256']==sha(job['checkpoint'])
            assert existing['statistics_sha256']==sha(statfile)
            if job['id']=='teacher':assert all((cache/f'{i:04d}.npz').exists() for i in range(len(data)))
            completed.append(job['id']);continue
        write(work/'state.json',dict(status='RUNNING',job=job['id'],completed=completed,gpu=gpu,images_done=0,total_images=len(data),pid=os.getpid()))
        teacher=job['id']=='teacher';ckpt=torch.load(job['checkpoint'],map_location='cpu')
        model=get_segmentation_model('deeplabv3' if teacher else 'deeplabv3_mobilenet_ssseg',
            backbone='resnet101' if teacher else 'mobilenetv3_large',local_rank=None,
            pretrained='None',pretrained_base='None',aux=teacher,norm_layer=torch.nn.BatchNorm2d,num_class=classes)
        model.load_state_dict(ckpt if teacher else ckpt['student'],strict=True);del ckpt
        model.eval().cuda().requires_grad_(False)
        matrices=[];regional=[];label_hashes=[];prediction_hashes=[]
        tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for i,(images,labels,_) in enumerate(loader):
            with torch.inference_mode():
                logits=F.interpolate(model(images.cuda(non_blocking=True))[0],labels.shape[-2:],mode='bilinear',align_corners=True)
                assert logits.shape[1]==classes and bool(torch.isfinite(logits).all())
                predicted=logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            target=labels[0].numpy().astype(np.int64);keep=(target>=0)&(target<classes)
            assert bool(keep.any()) and bool(((target==-1)|keep).all())
            matrices.append(np.bincount(classes*target[keep]+predicted[keep],minlength=classes**2).reshape(classes,classes))
            label_digest=hashlib.sha256(target.tobytes()).hexdigest();label_hashes.append(label_digest)
            if teacher:
                np.savez_compressed(cache/f'{i:04d}.npz',prediction=predicted,label_sha256=np.asarray(label_digest),name=np.asarray(names[i]))
                prediction_hashes.append(sha(cache/f'{i:04d}.npz'))
            else:
                with np.load(cache/f'{i:04d}.npz',allow_pickle=False) as saved:
                    assert str(saved['name'])==names[i] and str(saved['label_sha256'])==label_digest
                    t=saved['prediction'];assert t.shape==target.shape
                wrong=keep&(t!=target);correct=keep&(t==target)
                # Columns: teacher-wrong pixels, corrected, copied wrong teacher,
                # teacher-correct pixels, retained correct. Rows: GT classes.
                counts=np.zeros((classes,5),dtype=np.int64)
                for j,mask in enumerate([wrong,wrong&(predicted==target),wrong&(predicted==t),correct,correct&(predicted==target)]):
                    counts[:,j]=np.bincount(target[mask],minlength=classes)
                assert np.all(counts[:,1]+counts[:,2]<=counts[:,0]);assert np.all(counts[:,4]<=counts[:,3])
                regional.append(counts)
            del logits,images,labels
            if (i+1)%100==0 or i+1==len(data):
                write(work/'state.json',dict(status='RUNNING',job=job['id'],completed=completed,gpu=gpu,
                    images_done=i+1,total_images=len(data),elapsed_seconds=time.monotonic()-started,pid=os.getpid()))
                print('EVAL',dataset,job['id'],i+1,'/',len(data),flush=True)
        matrices=np.stack(matrices);matrix=matrices.sum(0);metrics=measure(matrix)
        if not teacher:
            delta=metrics['miou_percent']-job['run']['final_miou_percent']
            if abs(delta)>.001:raise RuntimeError(f'Final mIoU reproduction failed: {job["id"]}, {delta} pp')
        else:delta=None
        np.savez_compressed(statfile,confusion=matrices,regional_counts=np.asarray(regional,dtype=np.int64),names=np.asarray(names),label_sha256=np.asarray(label_hashes))
        extra={}
        if not teacher:
            total=np.asarray(regional).sum((0,1))
            extra=dict(teacher_wrong_pixels=int(total[0]),corrected_pixels=int(total[1]),wrong_imitation_pixels=int(total[2]),
                teacher_correct_pixels=int(total[3]),retained_correct_pixels=int(total[4]),
                correction_rate=float(total[1]/total[0]),wrong_imitation_rate=float(total[2]/total[0]),teacher_correct_retention=float(total[4]/total[3]))
        result=dict(status='PASS',id=job['id'],dataset=dataset,images=len(data),checkpoint=job['checkpoint'],
            checkpoint_sha256=sha(job['checkpoint']),statistics_sha256=sha(statfile),source_sha256=sha(__file__),
            registration_sha256=sha(OUT/'registration.json'),confusion_matrix=matrix.tolist(),**metrics,**extra,
            reproduction_delta_pp=delta,seconds=time.monotonic()-tick,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
        if teacher:result['prediction_file_sha256']=prediction_hashes
        else:result.update(seed=job['run']['seed'],condition=job['run']['condition'])
        write(dest,result);completed.append(job['id'])
        print('EVAL_PASS',dataset,job['id'],metrics['miou_percent'],'delta',delta,flush=True)
        del model;torch.cuda.empty_cache()
    write(work/'state.json',dict(status='PASS',completed=completed,gpu=gpu,elapsed_seconds=time.monotonic()-started,completed_at=dt.datetime.now(dt.timezone.utc).isoformat()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['voc','cityscapes'],required=True);p.add_argument('--gpu',type=int,required=True)
    a=p.parse_args();lock=gpu_setup(a.gpu)
    try:run(a.dataset,a.gpu)
    except Exception as e:
        write(OUT/'inference'/a.dataset/'failure.json',dict(status='FAILED',error=repr(e),source_sha256=sha(__file__)))
        raise
