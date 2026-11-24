"""Finite repairs of frozen failures; a separate immutable directory per round."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, re, shutil, signal, subprocess, time
from contextlib import ExitStack
from pathlib import Path
from lharness.scheduler import _atomic_json, _idle_service_policy, _process_start_ticks, gpu_snapshot
from lharness.capacity_scheduler import (capacity_settings, compute_memory_snapshot,
    host_available_mib, process_rss_mib, admission_capacity)
ROOT = Path(__file__).resolve().parent


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _target_gpu_overrides(config, target_count, gpus):
    supplied = config.get('target_gpu_overrides', {})
    if not isinstance(supplied, dict):
        raise ValueError('target_gpu_overrides must map target indexes to configured GPU IDs')
    result = {}
    for target, gpu in supplied.items():
        if (not isinstance(target, str) or not re.fullmatch(r'0|[1-9][0-9]*', target)
                or int(target) >= target_count):
            raise ValueError('target_gpu_overrides requires existing canonical target indexes')
        if type(gpu) not in (str, int) or str(gpu) not in gpus:
            raise ValueError('target_gpu_overrides GPU must belong to config.gpu_ids')
        result[target] = str(gpu)
    return result


def _process_identity(pid, start_ticks):
    """Distinguish an exited process from an unreadable or reused identity."""
    if type(pid) is not int or pid <= 0 or type(start_ticks) is not int or start_ticks <= 0:
        return 'unknown'
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if int(fields[19]) != start_ticks:
            return 'mismatch'
        return 'exited' if fields[0] in {'Z', 'X'} else 'matching'
    except FileNotFoundError:
        return 'exited'
    except (OSError, ValueError, IndexError):
        return 'unknown'


def _draining_parent(directory, selected, expected_identity=None):
    """Read-only admission proof for a frozen dispatcher that has no work to launch."""
    parent = read_json(directory/'dispatch.json')
    if not isinstance(parent, dict):
        raise ValueError('Parent dispatch is unreadable')
    identity = (parent.get('pid'), parent.get('start_ticks'))
    if expected_identity is not None and identity != expected_identity:
        raise ValueError('Parent dispatcher identity changed')
    process_state = _process_identity(*identity)
    if process_state in {'unknown', 'mismatch'}:
        raise ValueError('Parent dispatcher PID/start_ticks cannot be confirmed')
    jobs = parent.get('jobs')
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('Parent jobs are unreadable')
    by_index = {}
    permitted = {'running', 'complete', 'unknown', 'deferred_launch_window'}
    for job in jobs:
        if not isinstance(job, dict) or job.get('status') not in permitted:
            raise ValueError('Parent is still admitting or has an unsupported job state')
        index = job.get('target_index')
        if type(index) is not int or index < 0 or index in by_index:
            raise ValueError('Parent target indexes are invalid or duplicated')
        by_index[index] = job
    for index in selected:
        job = by_index.get(index, {})
        if job.get('status') not in {'deferred_launch_window', 'unknown'}:
            raise ValueError('Followup targets must be parent deferred or terminal unknown; running/complete overlap is forbidden')
        if job['status'] == 'deferred_launch_window' and 'pid' in job:
            raise ValueError('Deferred parent target has unexpected process ownership')
        if job['status'] == 'unknown' and type(job.get('returncode')) is not int:
            raise ValueError('Parent unknown target is not a committed terminal attempt')
    terminal = parent.get('status') in {'finished', 'finished_with_unknown'}
    if not terminal and parent.get('status') not in {'running', 'waiting_for_capacity'}:
        raise ValueError('Parent dispatcher state is not a confirmed drain or terminal state')
    if process_state == 'exited' and not terminal:
        raise ValueError('Parent exited without a terminal dispatch record')
    live_jobs = []
    for job in jobs:
        if job['status'] == 'deferred_launch_window':
            continue
        state = _process_identity(job.get('pid'), job.get('start_ticks'))
        if state == 'unknown':
            raise ValueError('Parent worker identity is unreadable')
        if job['status'] == 'running':
            if terminal or state == 'mismatch':
                raise ValueError('Parent running worker has inconsistent identity/state')
            if state == 'matching':
                if type(job.get('gpu_id')) not in (str, int) or not isinstance(job.get('output'), str):
                    raise ValueError('Parent running worker resource metadata is invalid')
                live_jobs.append(dict(job, gpu_id=str(job['gpu_id'])))
        elif state == 'matching':
            raise ValueError('Parent terminal worker is still alive')
    return dict(identity=identity, status=parent['status'], jobs=by_index,
                live_jobs=live_jobs, terminal=terminal, process_state=process_state)


def summarize(directory):
    status, groups = read_json(directory/'dispatch.json'), {}
    for job in status.get('jobs', []):
        row = read_json(Path(job['output'])/'outcome.json')
        c = groups.setdefault(job['group'], dict(selected=0, rescued=0, failed=0, unknown=0, pending=0))
        c['selected'] += 1
        valid = (row.get('status') == 'completed' and row.get('repair_eligible') is True
                 and row.get('prefix_reproduced') is True and row.get('policy_success') is False)
        if valid and type(row.get('native_success')) is bool:
            c['rescued' if row['native_success'] else 'failed'] += 1
        elif row or job['status'] in {'unknown', 'interrupted'}:
            c['unknown'] += 1
        else:
            c['pending'] += 1
    for c in groups.values():
        c['completed'] = c['rescued']+c['failed']
        c['rescue_rate_on_completed_original_failures'] = c['rescued']/c['completed'] if c['completed'] else None
    return dict(evaluation_mode='targeted_native_failure_repair', groups=groups,
        dispatch_status=status.get('status'), fresh_full_cohort_evaluation=False,
        note='Repair attempts on recorded policy failures; the original scores are unchanged.')


def snapshot_source(directory, config_path, manifest_path):
    source = directory/'source'
    source.mkdir()
    paths = [ROOT/'repair_failed_case.py', ROOT/'repair_dispatch.py', ROOT/'run.py',
             ROOT/'verify_native_artifacts.py', ROOT/'verify_repair_round.py']
    for sub, glob in [('lharness','*.py'), ('skills','*.json'), ('memory/general','*.json'),
                      ('memory/experiences','*.json')]:
        paths += sorted((ROOT/sub).glob(glob))
    files = []
    for path in paths:
        dest = source/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        files.append(dict(path=str(dest.relative_to(source)), sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    for path, name in [(config_path, 'config.json'), (manifest_path, 'targets.json')]:
        shutil.copyfile(path, source/name)
        files.append(dict(path=name, sha256=hashlib.sha256((source/name).read_bytes()).hexdigest()))
    config = read_json(config_path)
    check = '''import sys
sys.path.insert(0,sys.argv[1])
from lharness.model_api import query_model  # Imports the frozen model connection; no request.
'''
    subprocess.run([config['runtime_python'], '-I', '-B', '-c', check, str(source)],
                   cwd=source, check=True, timeout=30, capture_output=True, text=True)
    _atomic_json(directory/'source_manifest.json', dict(files=files, frozen_at_unix=time.time()))
    return source


def launch(args):
    config, manifest = read_json(args.config), read_json(args.manifest)
    selected = list(range(len(manifest['targets']))) if args.indices == 'all' else [int(i) for i in args.indices.split(',')]
    if not selected or len(selected) != len(set(selected)) or any(i < 0 or i >= len(manifest['targets']) for i in selected):
        raise ValueError('Select distinct existing target indexes')
    gpus = [str(i) for i in config['gpu_ids']]
    overrides = _target_gpu_overrides(config, len(manifest['targets']), gpus)
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.round):
        raise ValueError('Use a plain round name')
    output = Path(config['output_root']).resolve()
    if output.is_relative_to(Path(config['reference_output']).resolve()):
        raise ValueError('Baseline must remain read-only')
    directory = output/args.round
    jobs = [dict(target_index=i, group=manifest['targets'][i]['group'],
                 condition=manifest['targets'][i]['condition'],
                 output=str(directory/'cases'/f'target_{i:03d}'), status='pending') for i in selected]
    for job in jobs:
        if str(job['target_index']) in overrides:
            job['required_gpu_id'] = overrides[str(job['target_index'])]
    parent_directory = getattr(args, 'after_draining_round', None)
    parent_initial = None
    if parent_directory is not None:
        parent_directory = Path(parent_directory).resolve()
        if parent_directory.parent != output or parent_directory == directory:
            raise ValueError('Parent must be a different round in the same output_root')
        parent_initial = _draining_parent(parent_directory, selected)
        parent_targets = read_json(parent_directory/'source/targets.json').get('targets', [])
        if any(i >= len(parent_targets) or parent_targets[i] != manifest['targets'][i] for i in selected):
            raise ValueError('Followup target manifest differs from the frozen parent')
    if not args.execute:
        print(json.dumps(dict(status='prepared', jobs=jobs, gpu_started=False,
                             after_draining_round=str(parent_directory) if parent_directory else None), indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    allowed, _ = _idle_service_policy(config)
    settings = capacity_settings(config)
    with ExitStack() as locks:
        lock_names = ['repair_followup.lock']
        if parent_directory is None:
            lock_names.append('repair_dispatch.lock')
        for name in lock_names:
            lock = locks.enter_context((output/name).open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if parent_initial is not None:
            _draining_parent(parent_directory, selected, parent_initial['identity'])
        directory.mkdir(exist_ok=False)
        (directory/'logs').mkdir()
        source = snapshot_source(directory, args.config, args.manifest)
        status = dict(status='running', started_at_unix=time.time(), jobs=jobs,
            max_live=args.max_live, launch_window_seconds=args.launch_window_seconds,
            selected_indices=selected, source=str(source), pid=os.getpid(),
            start_ticks=_process_start_ticks(os.getpid()))
        if parent_initial is not None:
            status['after_draining_round'] = str(parent_directory)
            status['parent_identity'] = dict(pid=parent_initial['identity'][0], start_ticks=parent_initial['identity'][1])
        pending, live = jobs[:], []
        deadline = time.monotonic()+args.launch_window_seconds
        def flush():
            status['updated_at_unix'] = time.time()
            _atomic_json(directory/'dispatch.json', status)
            _atomic_json(directory/'summary.json', summarize(directory))
        try:
            while pending or live:
                for item in list(live):
                    rc = item['proc'].poll()
                    if rc is None:
                        continue
                    job = item['job']
                    row = read_json(Path(job['output'])/'outcome.json')
                    ok = (row.get('status') == 'completed' and row.get('repair_eligible') is True
                          and row.get('prefix_reproduced') is True and type(row.get('native_success')) is bool)
                    job.update(status='complete' if ok else 'unknown', returncode=rc,
                               finished_at_unix=time.time(), native_success=row.get('native_success'))
                    live.remove(item)
                if time.monotonic() >= deadline:
                    for job in pending:
                        job['status'] = 'deferred_launch_window'
                    pending.clear()
                if pending and (not args.max_live or len(live) < args.max_live):
                    parent_live = []
                    parent_blocked = False
                    if parent_initial is not None:
                        try:
                            parent = _draining_parent(parent_directory, selected, parent_initial['identity'])
                            parent_live = parent['live_jobs']
                            status['parent_admission'] = dict(admit=True, status=parent['status'],
                                live_workers=len(parent_live), process_state=parent['process_state'])
                        except ValueError as exc:
                            parent_blocked = True
                            status['parent_admission'] = dict(admit=False, reason=str(exc))
                    if parent_blocked:
                        status['status'] = 'waiting_for_parent_confirmation'
                        flush()
                        time.sleep(settings['poll_seconds'])
                        continue
                    if config.get('coexisting_repair_rounds'):
                        try:
                            from lharness.coexisting_repairs import coexisting_workers
                            coexisting = coexisting_workers(config,
                                [manifest['targets'][index] for index in selected],
                                process_reader=_process_identity)
                            parent_live += coexisting['jobs']
                            status['coexisting_admission'] = dict(admit=True,
                                rounds=coexisting['evidence'])
                        except (OSError, ValueError, KeyError, TypeError) as exc:
                            status['coexisting_admission'] = dict(admit=False, reason=str(exc))
                            status['status'] = 'waiting_for_external_confirmation'
                            flush()
                            time.sleep(settings['poll_seconds'])
                            continue
                    memory, snapshot = compute_memory_snapshot(), gpu_snapshot()
                    resource_jobs = [x['job'] for x in live]+parent_live
                    rss = {job['pid']: process_rss_mib(job['pid']) for job in resource_jobs}
                    ready = {}
                    for job in resource_jobs:
                        marker = read_json(Path(job['output'])/'model_ready.json')
                        ready[job['pid']] = (job['_coexisting_ready'] if '_coexisting_ready' in job else
                            marker.get('ready') is True and marker.get('pid') == job['pid'])
                    ram = host_available_mib()
                    while pending and (not args.max_live or len(live) + len(parent_live) < args.max_live):
                        owned = [dict(pid=x['job']['pid'], start_ticks=x['job']['start_ticks'],
                                      gpu_id=x['job']['gpu_id'], ready=ready.get(x['job']['pid'], False)) for x in live]
                        owned += [dict(pid=job['pid'], start_ticks=job['start_ticks'], gpu_id=job['gpu_id'],
                                       ready=ready.get(job['pid'], False)) for job in parent_live]
                        capacity = admission_capacity(gpus, snapshot, memory, ram, owned, allowed, settings,
                                                      rss_reader=lambda pid: rss.get(pid, 0.0))
                        status['capacity'] = capacity
                        choices = [(v['projected_free_mib'], g) for g,v in capacity['gpus'].items() if v['admit']]
                        if not choices:
                            break
                        for pending_index, candidate in enumerate(pending):
                            required = candidate.get('required_gpu_id')
                            eligible = [c for c in choices if required is None or c[1] == required]
                            if eligible:
                                break
                        else:
                            break
                        empty = [c for c in eligible if capacity['gpus'][c[1]]['owned_workers'] == 0]
                        _, gpu = max(empty or eligible)
                        job = pending.pop(pending_index)
                        command = [config['runtime_python'], str(source/'repair_failed_case.py'),
                            '--config', str(source/'config.json'), '--manifest', str(source/'targets.json'),
                            '--target-index', str(job['target_index']), '--gpu-id', gpu, '--output', job['output']]
                        log = directory/'logs'/f"target_{job['target_index']:03d}.log"
                        with log.open('xb') as stream:
                            proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                cwd=source, start_new_session=True,
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
                        job.update(status='running', pid=proc.pid, start_ticks=_process_start_ticks(proc.pid),
                                   gpu_id=gpu, log=str(log), command=command, started_at_unix=time.time())
                        live.append(dict(job=job, proc=proc))
                        flush()
                status['status'] = ('needs_attention' if status.get('infrastructure_circuit', {}).get('status') == 'open'
                    else 'running' if live else 'waiting_for_capacity' if pending else 'finished')
                flush()
                if live or pending:
                    time.sleep(settings['poll_seconds'])
        except BaseException:
            for item in live:
                if item['proc'].poll() is None:
                    os.killpg(item['proc'].pid, signal.SIGINT)
                    item['job']['status'] = 'interrupted'
            for item in live:
                try:
                    item['proc'].wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(item['proc'].pid, signal.SIGKILL)
                    item['proc'].wait()
            status['status'] = 'interrupted'
            flush()
            raise
        status['status'] = ('needs_attention' if status.get('infrastructure_circuit', {}).get('status') == 'open'
            else 'finished_with_unknown' if any(j['status']=='unknown' for j in jobs) else 'finished')
        flush()
        print(json.dumps(summarize(directory), indent=2))
        return 0 if all(j['status']=='complete' for j in jobs) else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'launch', 'summarize'])
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--indices', default='all')
    p.add_argument('--round', default='round_01')
    p.add_argument('--max-live', type=int, default=0, help='0 fills measured capacity; a positive value caps live workers')
    p.add_argument('--launch-window-seconds', type=int, default=1800)
    p.add_argument('--after-draining-round', type=Path,
                   help='Join a frozen round after all its pending jobs have been deferred; use a disjoint followup round')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if args.max_live < 0 or args.launch_window_seconds <= 0:
        p.error('Invalid capacity/window')
    if args.command == 'summarize':
        c = read_json(args.config)
        print(json.dumps(summarize(Path(c['output_root'])/args.round), indent=2))
        return 0
    if args.command == 'prepare':
        args.execute = False
    return launch(args)
