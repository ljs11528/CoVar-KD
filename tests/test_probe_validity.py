import copy
import math

import pytest
import torch
from torch import nn

from probe_validity.contracts import tasks,expected_counts,stream_entry,calibration
from probe_validity.core import DenseMetrics,module_inventory,set_update_mode,cpu_state,old_buffers,fingerprint,split_hashes
from probe_validity.resources import admissible,keeper
from utility_study.core import PooledMetrics,Snapshot,isolated_rng,ce_loss


@pytest.mark.parametrize("bn",["running","batch"])
@pytest.mark.parametrize("drop",[False,True])
def test_bn_and_dropout_are_separate_interventions_and_snapshot_restores_them(bn,drop):
    model=nn.Sequential(nn.Conv2d(3,4,1),nn.BatchNorm2d(4),nn.Dropout(.5),nn.Conv2d(4,2,1))
    opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9)
    x=torch.randn(4,3,5,5);y=torch.randint(0,2,(4,5,5))
    saved=Snapshot.capture(model,opt);inventory=module_inventory(model)
    assert len(inventory['batchnorm'])==len(inventory['dropout'])==1
    set_update_mode(model,bn,drop)
    assert model[1].training==(bn=='batch') and model[2].training==drop
    ce_loss(model(x),y).backward();opt.step()
    assert int(model[1].num_batches_tracked)==(1 if bn=='batch' else 0)
    saved.restore(model,opt);saved.verify(model,opt)


def test_buffer_transplant_keeps_every_updated_parameter_and_original_buffer():
    model=nn.Sequential(nn.Conv2d(2,2,1),nn.BatchNorm2d(2),nn.Dropout(.1))
    before=cpu_state(model);names=set(dict(model.named_parameters()))
    model(torch.randn(3,2,4,4))
    with torch.no_grad():
        for p in model.parameters():p.add_(.25)
    after=cpu_state(model);mixed=old_buffers(after,before,names)
    assert split_hashes(mixed,names)['parameters_sha256']==split_hashes(after,names)['parameters_sha256']
    assert split_hashes(mixed,names)['buffers_sha256']==split_hashes(before,names)['buffers_sha256']


def test_p15_primary_sufficient_statistics_match_p14_exactly_with_ignore_pixels():
    torch.manual_seed(813)
    logits=torch.randn(2,3,3,4,dtype=torch.float64);y=torch.randint(0,3,(2,6,8));y[0,:2]=-1
    base,extended=PooledMetrics(3),DenseMetrics(3)
    base.add(logits,y);extended.add(logits,y)
    a,b=base.result(),extended.result()
    for key in a:assert a[key]==b[key]


def test_extra_metrics_match_analytic_uniform_probabilities_and_teacher_regions():
    z=torch.zeros(1,2,2,2,dtype=torch.float64)
    y=torch.tensor([[[0,0],[1,-1]]]);teacher=torch.tensor([[[1,0],[0,1]]])
    m=DenseMetrics(2);m.add(z,y,teacher);r=m.result()
    assert r['pixel_entropy']==pytest.approx(math.log(2),abs=1e-14)
    assert r['pixel_brier']==pytest.approx(.5,abs=1e-14)
    assert r['teacher_wrong_counts']==[1,1]
    assert r['correct_on_teacher_wrong_counts']==[1,0]
    assert r['mimic_on_teacher_wrong_counts']==[0,1]
    assert r['correct_on_teacher_right_counts']==[1,0]


def test_function_kl_is_sampled_deterministically_and_detects_confidence_change():
    y=torch.zeros(1,20,20,dtype=torch.long)
    z=torch.zeros(1,2,20,20,dtype=torch.float64);z[:,0]=1.
    a=DenseMetrics(2);a.add(z,y,sample_seed=70)
    b=DenseMetrics(2);b.add(z,y,reference=a.sample_logp[0],sample_seed=70)
    assert a.sample_logp[0].shape==(256,2)
    assert abs(b.result()['sampled_function_kl'])<1e-14
    c=DenseMetrics(2);c.add(z*2,y,reference=a.sample_logp[0],sample_seed=70)
    assert c.result()['sampled_function_kl']>0
    assert a.result()['hard_miou']==c.result()['hard_miou']


def test_declared_task_counts_and_reused_full_lr_controls():
    rows=tasks();counts=expected_counts();p0=[j for j in rows if j['phase']=='p0'];p1=[j for j in rows if j['phase']=='p1']
    assert len({j['id'] for j in rows})==len(rows)
    assert sum(len(j['branches']) for j in p0)==72
    assert sum(not b.get('buffer_only',False) for j in p1 for b in j['branches'])==576
    endpoints=sum(2 if b['bn']=='batch' and b['buffer_policy']=='both' else 1 for j in p1 for b in j['branches'])
    assert endpoints+sum(4 for j in p1 if j['directions'])==counts['p1_total_endpoints']==912
    assert sum(j['discovery'] for j in p1)==4
    assert sum(b.get('buffer_only',False) for j in p1 for b in j['branches'])==72


def test_streams_exclude_both_pools_and_are_identical_when_replayed():
    eligible=list(range(512,3000))
    a=stream_entry(eligible,16,'voc',1234,20000,0)
    assert a==stream_entry(eligible,16,'voc',1234,20000,0)
    assert a!=stream_entry(eligible,16,'voc',1234,20000,1)
    assert len(a['indices'])==len(set(a['indices']))==16 and min(a['indices'])>=512


def p0_scores():
    rows=[];cross=[]
    for d in ('voc','cityscapes'):
        for stage in (20000,60000):
            for mode in ('eval','train'):
                for c,v in [('ce',.8),('shared_c1',.9),('shared_c4',1.)]:
                    for repeat in range(3):rows.append(dict(dataset=d,stage=stage,mode=mode,condition=c,scores={m:v+repeat*1e-9 for m in ('pixel_ce','class_ce','soft_miou','hard_miou')}))
            for gpu in range(4):cross.append(dict(dataset=d,stage=stage,score_gpu=gpu,scores={m:.9+gpu*1e-9 for m in ('pixel_ce','class_ce','soft_miou','hard_miou')}))
    return rows,cross


def test_calibration_requires_all_anchors_and_measured_cross_gpu_ranges():
    rows,cross=p0_scores();r=calibration(rows,cross)
    assert r['pass'] and not r['p2_or_full_training_authorized']
    with pytest.raises(ValueError):calibration(rows[:-1],cross)
    with pytest.raises(ValueError):calibration(rows,cross[:-1])
    for row in rows:
        if row['condition']=='shared_c4':row['scores']['pixel_ce']=.9+1e-8
    assert not calibration(rows,cross)['pass']


def test_memory_admission_preserves_known_keepers_and_refuses_unknown_jobs():
    policy={'score_minimum_free_mib':8192,'update_minimum_free_mib':28672,
            'keeper_rules':[{'uid':0,'command_prefixes':['ray::DataWorker'],'cwd_prefixes':['/root/maker/junkang/ray_tmp/']}]}
    p={'command':'ray::DataWorker.update_cached','cwd':'/root/maker/junkang/ray_tmp/session_x/resources','uid':0}
    card={'free_mib':9000,'processes':[p]}
    assert keeper(p,policy) and admissible(card,policy,'score') and not admissible(card,policy,'update')
    assert not keeper(dict(p,command='python unrelated.py'),policy)
    assert not admissible(dict(card,processes=[dict(p,command='python unrelated.py')]),policy,'score')


def test_endpoint_serialization_preserves_float_parameters_and_bn_bytes(tmp_path):
    model=nn.Sequential(nn.Conv2d(3,4,1),nn.BatchNorm2d(4),nn.Dropout(.2))
    state=cpu_state(model);path=tmp_path/'virtual.pth';torch.save({'student':state},path)
    loaded=torch.load(path,map_location='cpu',weights_only=False)['student']
    assert fingerprint(state)==fingerprint(loaded)


def test_backup_requires_complete_sample_receipt_and_rejects_corruption(tmp_path):
    from probe_validity.archive import snapshot
    from utility_study.io import write_json,sha256
    out,durable=tmp_path/'out',tmp_path/'durable'
    sample=out/'baselines/voc/a.samples.pt';sample.parent.mkdir(parents=True)
    sample.write_bytes(b'not-published-until-complete')
    first=snapshot(out,durable)
    assert first['baseline_sample_objects']==0
    write_json(sample.with_name('a.json'),{'samples_sha256':sha256(sample)})
    second=snapshot(out,durable)
    assert second['baseline_sample_objects']==1
    sample.write_bytes(b'corrupted-after-completion')
    with pytest.raises(RuntimeError,match='differs from completed receipt'):snapshot(out,durable)


def test_cpu_saved_activation_storage_preserves_gradients_and_buffers():
    model=nn.Sequential(nn.Conv2d(3,4,1),nn.BatchNorm2d(4),nn.Dropout(.2),nn.Conv2d(4,2,1))
    opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9)
    x=torch.randn(4,3,5,5);y=torch.randint(0,2,(4,5,5));snap=Snapshot.capture(model,opt);results=[]
    from contextlib import nullcontext
    for offload in (False,True):
        snap.restore(model,opt)
        with isolated_rng(873):
            with (torch.autograd.graph.save_on_cpu(pin_memory=False) if offload else nullcontext()):loss=ce_loss(model(x),y)
            loss.backward();opt.step();results.append(cpu_state(model))
    assert fingerprint(results[0])==fingerprint(results[1])


def test_frozen_teacher_eval_is_independent_of_chunk_size():
    teacher=nn.Sequential(nn.Conv2d(3,4,1),nn.BatchNorm2d(4),nn.Dropout(.4),nn.Conv2d(4,2,1)).eval()
    x=torch.randn(4,3,5,5)
    with torch.no_grad():
        whole=teacher(x);chunk=torch.cat([teacher(x[i:i+2]) for i in (0,2)])
    torch.testing.assert_close(whole,chunk,atol=1e-6,rtol=1e-6)
