"""Run the Maker preparation, fixed-input registration and bounded diagnostic queue."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import threading
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def backup(root, out, durable, copied):
    # Content-addressed objects avoid rename/mtime/random-write operations on OSS.
    inputs = list(out.rglob('*.json')) + list((root/'migration').glob('*.json'))
    inputs += list((root/'checkpoints').rglob('*.pth'))
    inputs += [root/name for name in ['fast_data_receipt.json', 'cpu_verification.json'] if (root/name).is_file()]
    changed = False
    for source in inputs:
        identity = (source.stat().st_size, source.stat().st_mtime_ns)
        if copied.get(str(source), {}).get('identity') == identity:
            continue
        # Records are atomically replaced; read a complete immutable byte value.
        data = source.read_bytes()
        key = hashlib.sha256(data).hexdigest()
        rel = str(source.relative_to(root))
        target = durable/'objects'/key[:2]/key
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('wb') as handle:
                handle.write(data)
            if digest(target) != key:
                raise RuntimeError('Persistent backup readback mismatch: '+rel)
        copied[str(source)] = {'identity': identity, 'relative': rel, 'sha256': key, 'object': str(target)}
        changed = True
    if changed:
        now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        target = durable/'indexes'/('index-'+now+'.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps({'created_at': now, 'files': list(copied.values())}, indent=2).encode()
        with target.open('wb') as handle:
            handle.write(data)
        if digest(target) != hashlib.sha256(data).hexdigest():
            raise RuntimeError('Persistent index readback mismatch')
        write(root/'backup_receipt.json', {'status': 'PASS', 'files': len(copied), 'index': str(target), 'at': now})


def main(args):
    root, out, durable = Path(args.root).resolve(), Path(args.out).resolve(), Path(args.durable).resolve()
    package_root = Path(__file__).resolve().parents[1]
    if not (package_root/'utility_study').is_dir():
        package_root = root
    sys.path.insert(0, str(package_root))
    gpus = [int(x) for x in args.gpus.split(',')]
    out.mkdir(parents=True, exist_ok=True)
    state = {'status': 'PREPARING', 'pid': os.getpid(), 'gpus': gpus,
             'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'root': str(root), 'out': str(out), 'durable': str(durable)}
    state_file = root/'preparation.json'
    write(state_file, state)
    stopped, copied = threading.Event(), {}

    def backups():
        while not stopped.wait(60):
            try:
                backup(root, out, durable, copied)
            except Exception as error:
                write(root/'backup_failure.json', {'status': 'FAILED', 'error': repr(error)})

    worker = threading.Thread(target=backups, daemon=True)
    worker.start()
    try:
        started = time.monotonic()
        while True:
            ready = root/'fast_data_receipt.json'
            log = root/'environment_setup.log'
            env_ready = log.exists() and 'ENVIRONMENT_PASS' in log.read_text(errors='replace')
            if ready.exists() and json.loads(ready.read_text())['status'] == 'PASS' and env_ready:
                break
            if time.monotonic()-started > 4*3600:
                raise TimeoutError('Environment/data preparation did not finish within four hours')
            launch = root/'environment_launch.json'
            if not env_ready and launch.exists():
                pid = json.loads(launch.read_text())['pid']
                if not Path('/proc', str(pid)).exists():
                    raise RuntimeError('Environment installation exited before its success marker')
            time.sleep(5)
        python = str(root/'env/bin/python')
        cpu = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        test = subprocess.run([python, '-m', 'pytest', '-q', 'tests/test_utility_study.py'],
                              cwd=package_root, env=cpu, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        write(root/'cpu_verification.json', {'returncode': test.returncode, 'output': test.stdout,
                                           'status': 'PASS' if test.returncode == 0 else 'FAILED'})
        print(test.stdout, flush=True)
        test.check_returncode()
        if args.resume_from:
            from utility_study.continuation import amend
            policy = json.loads(Path(args.resource_policy).read_text()) if args.resource_policy else {}
            amend(args.resume_from, out, gpus, policy)
        else:
            subprocess.run([python, '-u', '-m', 'utility_study.migrate', '--origin-registration',
                            str(root/'migration/origin_registration.json'), '--transfer-manifest',
                            str(root/'migration/checkpoint_transfer_manifest.json'), '--root', str(root),
                            '--out', str(out), '--gpus', args.gpus, '--defer-checkpoints'], cwd=package_root, env=cpu, check=True)
        if args.resource_preflight:
            state.update(status='RESOURCE_PREFLIGHT')
            write(state_file, state)
            children = []
            try:
                for index, gpu in enumerate(gpus):
                    dataset = ('voc', 'cityscapes')[index % 2]
                    scratch = out/'resource_preflight'/('gpu'+str(gpu))
                    scratch.mkdir(parents=True, exist_ok=True)
                    command = [python, '-u', '-m', 'utility_study.runner', '--registration', str(out/'registration.json'),
                               '--out', str(scratch), '--dataset', dataset, '--gpu', str(gpu), '--phase', 'smoke',
                               '--cohorts', 'ce', '--seeds', '1234', '--stages', '20000']
                    with (scratch/'worker.log').open('a') as log:
                        child = subprocess.Popen(command, cwd=package_root, stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=subprocess.STDOUT)
                    children.append((gpu, dataset, scratch, child))
                results = []
                while children:
                    for entry in list(children):
                        gpu, dataset, scratch, child = entry
                        if child.poll() is None:
                            continue
                        if child.returncode:
                            raise RuntimeError(f'Resource preflight failed on GPU{gpu}: {scratch}/worker.log')
                        summary = json.loads((scratch/'smoke'/dataset/'summary_ce_seed1234_stage20000.json').read_text())
                        results.append(summary)
                        children.remove(entry)
                    if children:
                        time.sleep(2)
                write(out/'resource_preflight.json', {'status': 'PASS', 'gpus': gpus, 'results': results,
                      'excluded_from_scientific_observations': True, 'keepalive_processes_signalled': 0})
            finally:
                for _, _, _, child in children:
                    if child.poll() is None:
                        child.terminate()
        state.update(status='RUNNING_DIAGNOSTICS')
        write(state_file, state)
        subprocess.run([python, '-u', '-m', 'utility_study.workflow', '--root', str(root),
                        '--out', str(out), '--gpus', args.gpus], cwd=package_root, check=True)
        state.update(status=json.loads((out/'workflow.json').read_text())['status'])
    except Exception as error:
        state.update(status='FAILED', error=repr(error))
        raise
    finally:
        write(state_file, state)
        stopped.set()
        worker.join(timeout=60)
        backup(root, out, durable, copied)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--durable', required=True)
    parser.add_argument('--gpus', default='2,3')
    parser.add_argument('--resume-from', help='Completed Maker observations to preserve byte for byte')
    parser.add_argument('--resource-policy', help='Explicit shared-GPU admission policy JSON')
    parser.add_argument('--resource-preflight', action='store_true', help='Excluded smoke jobs on every allocated GPU')
    main(parser.parse_args())
