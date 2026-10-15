#!/usr/bin/env python3
"""Verify all approved analysis outputs and unchanged original artifacts."""
import datetime as dt
import json
from pathlib import Path
import math
from evaluate import ROOT,OUT,write,sha

def main():
    import numpy as np
    reg=json.loads((OUT/'registration.json').read_text())
    checked={};inference=[];probe_checks=[]
    for path,expected in reg['source_sha256'].items():
        assert sha(path)==expected,path
        checked[path]=expected
    for dataset,ds in reg['datasets'].items():
        for path,expected in ds['input_sha256'].items():
            assert sha(path)==expected,path
            checked[path]=expected
        p=OUT/'inference'/dataset
        assert json.loads((p/'state.json').read_text())['status']=='PASS'
        for run in ['teacher']+[r['id'] for r in ds['runs']]:
            r=json.loads((p/(run+'.json')).read_text());f=p/(run+'.npz')
            assert r['status']=='PASS' and r['images']==ds['val_count'] and sha(f)==r['statistics_sha256']
            with np.load(f,allow_pickle=False) as a:
                matrix=a['confusion'].sum(0)
                assert np.array_equal(matrix,np.asarray(r['confusion_matrix']))
                assert len(a['names'])==ds['val_count']
                if run!='teacher':
                    total=a['regional_counts'].sum((0,1))
                    assert int(total[0]+total[3])==int(matrix.sum())
                    assert int(total[1]+total[4])==int(np.trace(matrix))
            if run=='teacher':
                for i,digest in enumerate(r['prediction_file_sha256']):assert sha(p/'teacher_predictions'/f'{i:04d}.npz')==digest
            else:assert abs(r['reproduction_delta_pp'])<=.001
            inference.append(dict(dataset=dataset,id=run,passed=True))
        work=OUT/'probe'/dataset;summary=json.loads((work/'summary.json').read_text())
        assert summary['status']=='PASS' and summary['rows']==1440 and summary['source_sha256']==sha(OUT/'scripts/probe.py')
        pairs=list((work/'pairs').glob('*.json'));assert len(pairs)==144
        seen=set();norm_errors=[]
        for file in pairs:
            result=json.loads(file.read_text())
            assert result['status']=='PASS' and result['exact_parameter_buffer_optimizer_restore']
            assert result['source_sha256']==summary['source_sha256'] and len(result['rows'])==10 and len(result['baselines'])==2
            for row in result['rows']:
                key=(row['seed'],row['checkpoint'],row['pair'],row['lr_multiplier'],row['condition'])
                assert key not in seen;seen.add(key)
                assert all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
            for lm in [1.0,.1]:
                a=next(r for r in result['rows'] if r['condition']=='teacher_c1' and r['lr_multiplier']==lm)
                b=next(r for r in result['rows'] if r['condition']=='shared_norm' and r['lr_multiplier']==lm)
                error=abs(a['effective_kd_ce_norm_ratio']-b['effective_kd_ce_norm_ratio'])
                assert error<=1e-10*max(1,a['effective_kd_ce_norm_ratio']);norm_errors.append(error)
        assert len(seen)==1440
        sampled=[i for values in ds['samples'].values() for p in values for i in p['same']+p['next']]
        assert len(sampled)==len(set(sampled))==192
        train=Path(ds['train_list']).read_text().splitlines();val=Path(ds['val_list']).read_text().splitlines()
        if dataset=='voc':assert not {train[i] for i in sampled}&set(val)
        else:assert not {Path(train[i].split()[0]).name for i in sampled}&{Path(x.split()[0]).name for x in val}
        probe_checks.append(dict(dataset=dataset,pairs=144,rows=1440,unique_train_images=192,
            training_and_validation_images_disjoint=True,max_norm_match_error=max(norm_errors),
            exact_parameter_buffer_optimizer_restore=True))
    assert json.loads((OUT/'training_log_audit.json').read_text())['status']=='PASS'
    assert json.loads((OUT/'analysis_summary.json').read_text())['status']=='PASS'
    assert not list(OUT.rglob('*.pth')), 'Analysis must not write model checkpoints'
    report=dict(status='PASS',checked_at=dt.datetime.now(dt.timezone.utc).isoformat(),unchanged_input_files=len(checked),
        unchanged_input_sha256=checked,inference=inference,probe_checks=probe_checks,
        full_training_runs_launched=0,student_final_evaluations=30,teacher_evaluations=2,formal_virtual_steps=3456,
        checkpoint_files_written=0,analysis_output_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
        analysis_sources={p.name:sha(p) for p in (OUT/'scripts').glob('*.py')},
        note='3456 formal virtual steps = 2 datasets × 3 seeds × 3 stages × 16 pairs × 2 LR multipliers × 6 conditions; two smoke pairs excluded.')
    write(OUT/'final_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['unchanged_input_sha256','inference','analysis_sources']},indent=2))

if __name__=='__main__':main()
