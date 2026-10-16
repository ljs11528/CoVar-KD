"""P2 inference-unit, blind-decision and restart contracts; run on Maker CPU."""
import copy
import hashlib
import inspect
import json
import pytest
import torch

from horizon_validation.contracts import stream,tasks,counts,decisions,best_candidate,gate
from horizon_validation.worker import save,load_checked,publish_endpoint,finite_gradients
from utility_study.core import Snapshot,rng_state,restore_rng,tree_equal

def fixtures():
    rows=[]
    for c in (0,1,4):
        for flow in range(3):
            for h in (1,16,128):
                rows.append(dict(mode='train',coefficient=c,stream=flow,repeat=0,horizon=h,
                    scores=dict(pixel_ce=1-c*.001,soft_miou=.5+c*.001,hard_miou=.5+c*.001,class_ce=1-c*.001)))
            rows.append(dict(mode='eval',coefficient=c,stream=flow,repeat=0,horizon=1,
                scores=dict(pixel_ce=1+c*.001,soft_miou=.5,hard_miou=.5,class_ce=1.)))
        for h in (1,16,128):
            rows.append(dict(mode='train',coefficient=c,stream=0,repeat=1,horizon=h,
                scores=dict(pixel_ce=1-c*.001,soft_miou=.5+c*.001,hard_miou=.5+c*.001,class_ce=1-c*.001)))
    bands={f'voc:{mode}:{m}':1e-6 for mode in ('train','eval') for m in ('pixel_ce','class_ce','soft_miou','hard_miou')}
    return rows,bands

def test_stream_pairing_and_exclusion():
    eligible=list(range(100,200));a=stream(eligible,16,'voc',1234,0,128)
    assert a==stream(eligible,16,'voc',1234,0,128)
    assert all(len(set(r['indices']))==16 and set(r['indices'])<=set(eligible) for r in a)
    assert a!=stream(eligible,16,'voc',1234,1,128)
    assert a!=stream(eligible,16,'voc',2025,0,128)
    assert a!=stream(eligible,16,'voc',1234,3,128)

def test_rollout_cost_and_prefix_contract():
    probes,forks=tasks();c=counts()
    assert len(probes)==252 and len(forks)==36
    assert sum(len(t['horizons']) for t in probes)==540
    assert sum(t['steps'] for t in probes)==18540 and sum(t['steps'] for t in forks)==72000
    assert sum(t['repeat']==1 for t in probes)==36
    assert all(t['stream']==3 and t['steps']==2000 for t in forks)
    assert len({t['id'] for t in probes+forks})==288 and c['full_80k_trajectories']==0

def test_blind_rules_can_disagree():
    rows,bands=fixtures();r=decisions(rows,bands,'voc','ce')['rules']
    assert r['eval_pixel_ce_1']['coefficient']==0
    assert r['train_pixel_ce_1']['coefficient']==4
    assert r['train_soft_miou_1']['coefficient']==4
    assert r['train_hard_miou_128_consistent']['coefficient']==4
    assert 'validation' not in inspect.signature(decisions).parameters

def test_one_disagreeing_stream_forces_abstention():
    rows,bands=fixtures()
    for r in rows:
        if r['mode']=='train' and r['horizon']==128 and r['stream']==2:r['scores']['hard_miou']=.5-r['coefficient']*.001
    r=decisions(rows,bands,'voc','shared_c1')['rules']['train_hard_miou_128_consistent']
    assert r['coefficient']==1 and r['abstained']

def test_numerical_replay_band_can_prevent_switch():
    rows,bands=fixtures()
    for r in rows:
        if r['repeat']==1 and r['horizon']==128:r['scores']['hard_miou']+=.002
    r=decisions(rows,bands,'voc','ce')
    assert r['numerical_bands']['128:hard_miou']==pytest.approx(.01)
    assert r['rules']['train_hard_miou_128_consistent']['coefficient']==0

def test_missing_repeat_and_duplicate_are_rejected():
    rows,bands=fixtures()
    with pytest.raises(KeyError):decisions(rows[:-1],bands,'voc','ce')
    with pytest.raises(ValueError):decisions(rows+[rows[0]],bands,'voc','ce')

def test_ties_keep_current_then_choose_smaller():
    assert best_candidate({0:1.,1:1.,4:1.},'pixel_ce',1,1e-6)==1
    assert best_candidate({0:1.,1:2.,4:1.},'pixel_ce',1,1e-6)==0
    with pytest.raises(ValueError):best_candidate({0:1.,1:float('nan'),4:1.},'pixel_ce',1,1e-6)

def test_gate_uses_pp_both_datasets_and_three_seeds():
    assert gate({'voc':[.11,.12,.13],'cityscapes':[.11,.12,.13]})['passed']
    assert not gate({'voc':[.11,.12,.13],'cityscapes':[0.,0.,0.]})['passed']
    assert not gate({'voc':[.5,-.01,-.01],'cityscapes':[.11,.12,.13]})['passed']
    with pytest.raises(ValueError):gate({'voc':[.11]*6,'cityscapes':[.11]*3})

def test_serialized_checkpoint_tamper_detection(tmp_path):
    path=tmp_path/'state.pth';digest=save(path,{'student':{'w':torch.ones(2)}})
    assert torch.equal(load_checked({'state_file':str(path),'state_file_sha256':digest})['student']['w'],torch.ones(2))
    path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(RuntimeError):load_checked({'state_file':str(path),'state_file_sha256':digest})

def test_resource_retry_preserves_prior_endpoint_and_model_bytes(tmp_path):
    records=[]
    for i in (1,2):
        path=tmp_path/'states'/str(i)/'h1.pth'
        e=dict(task='probe_same',horizon=1,registration_sha256='r',job_sha256='j',state_file=str(path),state_file_sha256=save(path,{'student':{'w':torch.tensor([float(i)])}}))
        publish_endpoint(tmp_path,e);records.append(e)
    history=list((tmp_path/'endpoint_history'/'probe_same').glob('*.json'))
    assert len(history)==1 and json.loads(history[0].read_text())==records[0]
    assert load_checked(records[0])['student']['w'].item()==1.
    assert load_checked(records[1])['student']['w'].item()==2.

def test_batched_gradient_validation_is_exact_and_read_only():
    model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Linear(4,2))
    for value in (0.,1e30,float('inf'),float('nan')):
        for p in model.parameters():p.grad=torch.full_like(p,2.)
        next(model.parameters()).grad.flatten()[0]=value
        before=[p.grad.clone() for p in model.parameters()]
        expected=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        assert finite_gradients(model)==expected
        for a,p in zip(before,model.parameters()):assert torch.equal(torch.isnan(a),torch.isnan(p.grad)) and torch.equal(torch.nan_to_num(a),torch.nan_to_num(p.grad))

def test_optimizer_rng_bn_resume_matches_continuous_cpu():
    torch.manual_seed(42)
    model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.BatchNorm1d(4),torch.nn.Dropout(.2),torch.nn.Linear(4,2))
    opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=1e-4)
    def advance(m,o,n):
        m.train()
        for _ in range(n):
            x=torch.randn(8,3);y=torch.randn(8,2);o.zero_grad(set_to_none=True)
            (m(x)-y).square().mean().backward();o.step()
    advance(model,opt,8)
    saved=copy.deepcopy((model.state_dict(),opt.state_dict()));random_state=rng_state()
    advance(model,opt,8);expected=copy.deepcopy((model.state_dict(),opt.state_dict()))
    restored=copy.deepcopy(model);newopt=torch.optim.SGD(restored.parameters(),lr=.01,momentum=.9,weight_decay=1e-4)
    restored.load_state_dict(saved[0]);newopt.load_state_dict(saved[1]);restore_rng(random_state)
    advance(restored,newopt,8)
    assert tree_equal(restored.state_dict(),expected[0]) and tree_equal(newopt.state_dict(),expected[1])

def test_cpu_saved_activations_pin_option_preserves_gradients_and_rng():
    torch.manual_seed(37)
    model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.BatchNorm1d(4),torch.nn.Dropout(.2),torch.nn.Linear(4,2))
    initial=copy.deepcopy(model.state_dict());initial_rng=rng_state();x=torch.randn(8,3);y=torch.randn(8,2)
    results=[]
    for pinned in (False,True):
        model.load_state_dict(initial);model.zero_grad(set_to_none=True);restore_rng(initial_rng)
        with torch.autograd.graph.save_on_cpu(pin_memory=pinned):loss=(model(x)-y).square().mean()
        loss.backward()
        results.append((copy.deepcopy(model.state_dict()),[p.grad.clone() for p in model.parameters()],rng_state()))
    assert tree_equal(results[0],results[1])

def test_snapshot_restores_training_flags_gradients_and_optimizer():
    m=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.BatchNorm1d(4),torch.nn.Dropout(.2));o=torch.optim.SGD(m.parameters(),lr=.01,momentum=.9)
    snap=Snapshot.capture(m,o);m.train();m(torch.randn(8,3)).sum().backward();o.step()
    snap.restore(m,o);snap.verify(m,o)
