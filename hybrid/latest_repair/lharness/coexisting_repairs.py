"""Read-only ownership proof for fully admitted repair rounds.

This is not a general process allowlist. A foreign round with pending/deferred
work is refused because its uncoordinated future starts cannot be reserved.
Overlap remains forbidden unless explicitly pinned to retired unknown/nativefalse
attempts for one new attempt. No service/process actions or automatic retries.
"""
import hashlib
import json
from pathlib import Path

from .capacity_scheduler import capacity_settings


def _json(path):
    return json.loads(Path(path).read_text())


def _command(pid):
    return Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')


def _identity(target):
    fields = ('group', 'condition', 'task_id', 'init_id', 'seed')
    if not isinstance(target, dict) or any(field not in target for field in fields):
        raise ValueError('Missing frozen target identity')
    return tuple(target[field] for field in fields)


def _source(directory, expected_hash):
    manifest_path = directory/'source_manifest.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError('Coexisting source manifest changed')
    source = directory/'source'
    found = set()
    for row in _json(manifest_path)['files']:
        name = row['path']
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts or name in found:
            raise ValueError('Invalid coexisting source pin')
        path = source/name
        if path.is_symlink() or not path.resolve().is_relative_to(source) or not path.is_file():
            raise ValueError('Missing or escaping coexisting source')
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Coexisting frozen source changed')
        found.add(name)
    if not {'config.json','targets.json','repair_dispatch.py','repair_failed_case.py',
            'lharness/capacity_scheduler.py'} <= found:
        raise ValueError('Coexisting manifest misses its resource/worker source')
    return source


def _terminal_overlap_pins(spec):
    if 'terminal_overlap' not in spec:
        return {}
    option = spec['terminal_overlap']
    if (not isinstance(option, dict) or set(option) != {'mode', 'cases'}
            or option['mode'] != 'single_attempt' or not isinstance(option['cases'], list)
            or not option['cases']):
        raise ValueError('Terminal overlap requires an explicit pinned single_attempt list')
    pins = {}
    for row in option['cases']:
        if not isinstance(row, dict) or set(row) != {'target_index', 'outcome_sha256'}:
            raise ValueError('Terminal overlap requires target index and outcome SHA')
        index, digest = row['target_index'], row['outcome_sha256']
        if (type(index) is not int or index < 0 or index in pins or not isinstance(digest, str)
                or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('Invalid or duplicate terminal overlap pin')
        pins[index] = digest
    return pins


def _terminal_overlap_evidence(job, target, output, digest, state):
    if job.get('status') not in {'complete', 'unknown'} or state != 'exited':
        raise ValueError('Overlapping case must be committed terminal and its exact worker exited')
    rc = job.get('returncode')
    if type(rc) is not int or (job['status'] == 'complete' and rc != 0) or (job['status'] == 'unknown' and rc == 0):
        raise ValueError('Overlapping worker has no consistent terminal return code')
    path = output/'outcome.json'
    if path.is_symlink() or path.resolve().parent != output or not path.is_file() or path.stat().st_size > 1024*1024:
        raise ValueError('Missing or escaping overlapping outcome')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Pinned overlapping outcome SHA changed')
    row = json.loads(raw)
    if _identity(row) != _identity(target) or 'native_success' not in row:
        raise ValueError('Overlapping outcome case identity differs')
    allowed = ((job['status'] == 'unknown' and row.get('status') == 'unknown' and row['native_success'] is None)
               or (job['status'] == 'complete' and row.get('status') == 'completed' and row['native_success'] is False))
    if not allowed or job.get('native_success') is True:
        raise ValueError('Overlap permits only committed unknown or nativefalse, never success')
    return dict(target_index=job['target_index'], outcome=str(path), outcome_sha256=digest,
                outcome_status=row['status'], native_success=row['native_success'],
                pid=job['pid'], start_ticks=job['start_ticks'], worker_identity='exited',
                authorization='explicit_single_attempt_terminal_overlap')


def coexisting_workers(config, selected_targets, *, process_reader, command_reader=_command):
    specs = config.get('coexisting_repair_rounds', [])
    if not isinstance(specs, list):
        raise ValueError('coexisting_repair_rounds must be a pinned list')
    selected = {_identity(target) for target in selected_targets}
    local_settings = capacity_settings(config)
    jobs, evidence, seen_pids, seen_rounds = [], [], set(), set()
    for spec in specs:
        required = {'round_dir','pid','start_ticks','source_manifest_sha256'}
        if not isinstance(spec, dict) or set(spec) not in (required, required | {'terminal_overlap'}):
            raise ValueError('Coexisting round requires exact path, PID/start_ticks and source pin')
        overlap_pins = _terminal_overlap_pins(spec)
        directory = Path(spec['round_dir']).resolve()
        if directory in seen_rounds or directory.parent == Path(config['output_root']).resolve():
            raise ValueError('Coexisting rounds must be distinct external repair roots')
        seen_rounds.add(directory)
        source = _source(directory, spec['source_manifest_sha256'])
        external_config = _json(source/'config.json')
        if overlap_pins and any('terminal_overlap' in item for item in external_config.get('coexisting_repair_rounds', [])):
            raise ValueError('Recursive terminal overlap rounds are forbidden')
        if Path(external_config['output_root']).resolve() != directory.parent:
            raise ValueError('Coexisting output root is not source-bound')
        external_settings = capacity_settings(external_config)
        for key in ('worker_peak_memory_mib','startup_host_memory_mib','steady_host_memory_mib'):
            if external_settings[key] > local_settings[key]:
                raise ValueError('Local reservations are smaller than external worker peaks')
        targets = _json(source/'targets.json')['targets']
        dispatch = _json(directory/'dispatch.json')
        identity = (dispatch.get('pid'), dispatch.get('start_ticks'))
        if identity != (spec['pid'], spec['start_ticks']) or Path(dispatch.get('source','')).resolve() != source:
            raise ValueError('Coexisting dispatcher/source identity changed')
        owner_state = process_reader(*identity)
        terminal = dispatch.get('status') in {'finished','finished_with_unknown'}
        if (owner_state not in {'matching','exited'} or (owner_state == 'exited' and not terminal)
                or dispatch.get('status') not in {'running','finished','finished_with_unknown'}):
            raise ValueError('Coexisting dispatcher identity or drain status unconfirmed')
        found, live_count, allowed_overlaps = set(), 0, []
        rows = dispatch.get('jobs')
        if not isinstance(rows, list) or not rows:
            raise ValueError('Missing coexisting worker records')
        for job in rows:
            index = job.get('target_index')
            if type(index) is not int or not 0 <= index < len(targets) or index in found:
                raise ValueError('Invalid or duplicated coexisting target index')
            found.add(index)
            overlaps = _identity(targets[index]) in selected
            if overlaps and index not in overlap_pins:
                raise ValueError('Selected case overlaps a coexisting round')
            if job.get('status') not in {'running','complete','unknown'}:
                raise ValueError('Coexisting round may still admit pending/deferred work')
            output = directory/'cases'/f'target_{index:03d}'
            if Path(job.get('output','')).resolve() != output:
                raise ValueError('Coexisting case output mismatch')
            pid, ticks = job.get('pid'), job.get('start_ticks')
            state = process_reader(pid,ticks)
            if state not in {'matching','exited'}:
                raise ValueError('Coexisting worker PID identity unconfirmed')
            if overlaps:
                allowed_overlaps.append(_terminal_overlap_evidence(
                    job, targets[index], output, overlap_pins[index], state))
            if job['status'] != 'running':
                if state != 'exited':
                    raise ValueError('Coexisting terminal worker still alive')
                continue
            if terminal:
                raise ValueError('Terminal dispatcher retains running workers')
            if state == 'exited':
                continue
            gpu = str(job.get('gpu_id'))
            if gpu not in [str(g) for g in config['gpu_ids']] or pid in seen_pids:
                raise ValueError('Unrecognized GPU or duplicated worker identity')
            expected = [external_config['runtime_python'],str(source/'repair_failed_case.py'),
                        '--config',str(source/'config.json'),'--manifest',str(source/'targets.json'),
                        '--target-index',str(index),'--gpu-id',gpu,'--output',str(output)]
            if job.get('command') != expected:
                raise ValueError('Coexisting recorded worker command differs')
            actual = command_reader(pid)
            script = expected[1]
            if not isinstance(actual,list) or actual.count(script) != 1 or actual[actual.index(script):] != expected[1:]:
                raise ValueError('Live worker command is not the pinned native case')
            ready = False
            try:
                marker = _json(output/'model_ready.json')
                ready = (marker.get('ready') is True and marker.get('pid') == pid
                         and marker.get('start_ticks') == ticks and str(marker.get('gpu_id')) == gpu)
            except (OSError,ValueError):
                pass
            jobs.append(dict(job, gpu_id=gpu, _coexisting_ready=ready))
            seen_pids.add(pid)
            live_count += 1
        listed = dispatch.get('selected_indices')
        if not isinstance(listed,list) or len(set(listed)) != len(listed) or set(listed) != found:
            raise ValueError('Coexisting selected indices differ from worker records')
        if set(overlap_pins) != {item['target_index'] for item in allowed_overlaps}:
            raise ValueError('Terminal overlap pins must match exactly the selected overlapping cases')
        evidence.append(dict(round_dir=str(directory),pid=identity[0],start_ticks=identity[1],
                             live_workers=live_count,source_manifest_sha256=spec['source_manifest_sha256'],
                             no_future_admission=True, terminal_overlap_allowed=allowed_overlaps))
    return dict(jobs=jobs,evidence=evidence)
