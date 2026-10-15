#!/usr/bin/env python3
"""Read existing logs only; check finite losses and recovery continuity."""
import json
import math
from pathlib import Path
import re
import statistics
from evaluate import OUT,write,sha

def main():
    reg=json.loads((OUT/'registration.json').read_text());results=[]
    pattern=re.compile(r'Iters:\s*(\d+)/80000.*?Task Loss:\s*([^\s|]+).*?KD Loss:\s*([^\s|]+)')
    for dataset,ds in reg['datasets'].items():
        for row in ds['runs']:
            text=Path(row['log']).read_text();audit=json.loads(Path(row['audit']).read_text())
            if 'continuation' in audit:
                info=audit['continuation'];prefix=Path(info['prefix_root'])
                candidates=[p for p in prefix.rglob('*') if p.is_file() and p.suffix in ['.txt','.log'] and sha(p)==info['prefix_log_sha256']]
                assert candidates
                lines=[]
                for line in candidates[0].read_text().splitlines(keepends=True):
                    m=re.search(r'Iters:\s*(\d+)/80000',line)
                    if m and int(m.group(1))>info['iteration']:break
                    lines.append(line)
                text=''.join(lines)+'\n'+text
            values=[(int(m.group(1)),float(m.group(2)),float(m.group(3))) for m in pattern.finditer(text)]
            assert [v[0] for v in values]==list(range(20,80001,20)),row['id']
            assert all(math.isfinite(x) for v in values for x in v[1:]),row['id']
            windows=[]
            for low,high in [(0,1000),(1000,20000),(20000,40000),(40000,60000),(60000,80000)]:
                chosen=[v for v in values if low<v[0]<=high]
                windows.append(dict(start_exclusive=low,end_inclusive=high,records=len(chosen),
                    task_loss_mean=statistics.mean(v[1] for v in chosen),task_loss_max=max(v[1] for v in chosen),
                    logged_kd_loss_mean=statistics.mean(v[2] for v in chosen),logged_kd_loss_max=max(v[2] for v in chosen)))
            results.append(dict(dataset=dataset,id=row['id'],condition=row['condition'],seed=row['seed'],
                logged_steps=4000,all_logged_losses_finite=True,complete_20_step_log_sequence=True,
                windows=windows,validations=row['validations']))
    write(OUT/'training_log_audit.json',dict(status='PASS',runs=results,logged_steps=120000,
        note='Logged losses are rounded observations every 20 steps, not a guarantee about every intervening tensor. Recovery prefixes are trimmed at the saved step.'))
    print('LOG_AUDIT_PASS',len(results),'runs',120000,'logged steps')

if __name__=='__main__':main()
