"""Opt-in finite task shards, admitted against measured GPU and host capacity."""
from __future__ import annotations
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from .scheduler import _atomic_json, _idle_service_policy, _process_start_ticks, gpu_snapshot
from .shards import plan_jobs
from .worker_environment import worker_environment

DEFAULTS = dict(worker_peak_memory_mib=8704, startup_host_memory_mib=22528,
                steady_host_memory_mib=6144, max_concurrent_starts=8, poll_seconds=2)


def capacity_settings(config):
    settings = {**DEFAULTS, **config.get('adaptive_capacity', {})}
    for key in DEFAULTS:
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid adaptive capacity setting: {key}')
    if type(settings['max_concurrent_starts']) is not int or settings['max_concurrent_starts'] > 8:
        raise ValueError('At most eight concurrent model starts are permitted')
    return settings


def compute_memory_snapshot():
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                 '--format=csv,noheader,nounits'], capture_output=True,
                                text=True, check=True, timeout=15)
        memory = {}
        for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
            if len(row) != 2:
                raise ValueError('Unknown compute memory record')
            pid, used = int(row[0]), float(row[1])
            if pid <= 0 or not math.isfinite(used) or used < 0:
                raise ValueError('Invalid compute memory record')
            memory[pid] = memory.get(pid, 0) + used
        return memory
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def host_available_mib():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return float(line.split()[1])/1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def process_rss_mib(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return float(line.split()[1])/1024
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def admission_capacity(gpu_ids, snapshot, process_memory, mem_available, owned, allowed, settings,
                       ticks_reader=_process_start_ticks, rss_reader=process_rss_mib):
    """Pure resource accounting apart from injected PID identity/RSS readers.

    Each live worker reserves its full peak minus memory already observed by
    NVML. Host reservations likewise subtract RSS already reflected in available
    RAM. A not-yet-allocated child therefore consumes its full reservation.
    """
    report = dict(gpus={}, loading=sum(not p['ready'] for p in owned), host_available_mib=mem_available)
    if snapshot is None or process_memory is None or mem_available is None:
        report['reason'] = 'resource_status_unknown'
        return report
    metrics, pids = snapshot
    host_reserved = sum(max((settings['steady_host_memory_mib'] if p['ready'] else
                             settings['startup_host_memory_mib']) - rss_reader(p['pid']), 0) for p in owned)
    host_free = mem_available - host_reserved
    report['host_projected_free_mib'] = host_free
    host_ok = host_free >= settings['startup_host_memory_mib'] and report['loading'] < settings['max_concurrent_starts']
    for gpu in gpu_ids:
        metric = metrics.get(gpu, {})
        total, used, util = (metric.get(k, math.nan) for k in ('memory_total_mib', 'memory_mib', 'utilization'))
        local = [p for p in owned if p['gpu_id'] == gpu]
        owned_ids = {p['pid']: p['start_ticks'] for p in local}
        reason = None
        if not all(math.isfinite(x) and x >= 0 for x in (total, used, util)) or used > total:
            reason = 'gpu_metrics_unknown'
        for pid in pids.get(gpu, []):
            expected = owned_ids.get(pid, allowed.get(pid))
            if expected is None or ticks_reader(pid) != expected:
                reason = 'foreign_or_unidentified_process'
                break
        if not local and util >= 5:
            reason = reason or 'unowned_gpu_busy'
        reserved = sum(max(settings['worker_peak_memory_mib'] - process_memory.get(p['pid'], 0), 0) for p in local)
        committed = sum(max(settings['worker_peak_memory_mib'], process_memory.get(p['pid'], 0)) for p in local)
        foreign_used = sum(process_memory.get(pid, 0) for pid in pids.get(gpu, []) if pid not in owned_ids)
        free = min(total - used - reserved, total - committed - foreign_used)
        if free < settings['worker_peak_memory_mib']:
            reason = reason or 'gpu_capacity_reserved'
        if not host_ok:
            reason = reason or 'host_capacity_or_start_limit'
        report['gpus'][gpu] = dict(admit=reason is None, reason=reason,
                                   actual_free_mib=total-used, projected_free_mib=free,
                                   unallocated_reservations_mib=reserved, owned_workers=len(local))
    return report


def _ready(job):
    try:
        marker = json.loads(Path(job['model_ready']).read_text())
        return marker.get('ready') is True and marker.get('pid') == job['pid']
    except (OSError, ValueError):
        return False


def run_capacity(config_path, output=None, *, execute=False, resume=False,
                 wait_for_idle=False, once=False, shard_count=5, admission_seconds=None):
    if admission_seconds is not None and (
            isinstance(admission_seconds, bool) or not isinstance(admission_seconds, (int, float))
            or not math.isfinite(admission_seconds) or admission_seconds <= 0):
        raise ValueError('admission_seconds must be a positive finite duration')
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if 'adaptive_capacity' not in config:
        raise ValueError('Adaptive admission requires measured policy GPU/host capacity settings')
    settings = capacity_settings(config)
    allowed, _ = _idle_service_policy(config)
    output = Path(output or config['output_root']).expanduser().resolve()
    gpu_ids = [str(g) for g in config.get('gpu_ids', range(8))]
    if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8:
        raise ValueError('Eight unique GPU identifiers required')
    jobs = plan_jobs(config, shard_count)
    status = dict(status='prepared', execute=execute, resume=resume, adaptive_shards=True,
                  capacity_settings=settings, output=str(output), jobs=jobs,
                  admission_window=dict(seconds=admission_seconds, started_monotonic=None,
                                        deadline_monotonic=None, remaining_seconds=admission_seconds,
                                        state='disabled' if admission_seconds is None else 'not_started'))
    for job in jobs:
        job['status'] = 'pending' if execute else 'queued'
    output.mkdir(parents=True, exist_ok=True)
    if not execute:
        _atomic_json(output/'queue_preview.json', status)
        return status
    status_path = output/'campaign_status.json'
    with (output/'campaign.lock').open('a') as campaign_lock:
        fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_id = f'{time.time_ns()}-{os.getpid()}'
        if status_path.exists():
            history = output/'scheduler_history'
            history.mkdir(exist_ok=True)
            shutil.copy2(status_path, history/f'campaign_status-{run_id}.json')
        status['status'] = 'preparing_shards'
        _atomic_json(status_path, status)
        from .evaluation import prepare_condition_shards
        prepared = {}
        for job in jobs:
            key = (job['group'], job['condition'])
            if key not in prepared:
                rows = prepare_condition_shards(config, *key, output, shard_count, resume)
                prepared[key] = {(r['task_id'], r['init_id']) for r in rows}
            if all(tuple(k) in prepared[key] for k in job['case_keys']):
                job['status'] = 'complete_skipped'
        pending = [j for j in jobs if j['status'] == 'pending']
        live, launched = [], 0
        window = status['admission_window']
        if admission_seconds is not None:
            window['started_monotonic'] = time.monotonic()
            window['deadline_monotonic'] = window['started_monotonic'] + admission_seconds

        def admission_open():
            if admission_seconds is None:
                return True
            window['remaining_seconds'] = max(0.0, window['deadline_monotonic'] - time.monotonic())
            window['state'] = 'accepting' if window['remaining_seconds'] > 0 else 'expired'
            return window['remaining_seconds'] > 0

        try:
            while pending or live:
                for item in list(live):
                    proc, job = item['proc'], item['job']
                    rc = proc.poll()
                    if rc is None:
                        continue
                    try:
                        summary = json.loads(Path(job['shard_summary']).read_text())
                        complete = (summary.get('status') == 'complete' and summary.get('unknown') == 0
                                    and summary.get('shard_id') == job['shard_id']
                                    and summary.get('group') == job['group'] and summary.get('condition') == job['condition'])
                    except (OSError, ValueError):
                        complete = False
                    job.update(status='complete' if complete else 'failed_unknown', returncode=rc)
                    _atomic_json(Path(job['attempt_record']), job)
                    live.remove(item)
                accepting = admission_open()
                if not accepting and not live:
                    break
                process_memory = compute_memory_snapshot() if pending and accepting else None
                snapshot = gpu_snapshot() if pending and accepting else None
                rss_snapshot = {item['job']['pid']: process_rss_mib(item['job']['pid']) for item in live}
                ready_snapshot = {item['job']['pid']: _ready(item['job']) for item in live}
                available_ram = host_available_mib() if pending and accepting else None
                while pending and (not once or launched < len(gpu_ids)) and admission_open():
                    owned = [dict(pid=item['job']['pid'], start_ticks=item['job']['start_ticks'],
                                  gpu_id=item['job']['gpu_id'], ready=ready_snapshot.get(item['job']['pid'], False))
                             for item in live]
                    capacity = admission_capacity(gpu_ids, snapshot, process_memory, available_ram,
                                                  owned, allowed, settings,
                                                  rss_reader=lambda pid: rss_snapshot.get(pid, 0.0))
                    status['capacity'] = capacity
                    choices = [(data['projected_free_mib'], gpu) for gpu, data in capacity['gpus'].items() if data['admit']]
                    if not choices:
                        break
                    empty = [choice for choice in choices if capacity['gpus'][choice[1]]['owned_workers'] == 0]
                    if empty:
                        choices = empty
                    _, gpu = max(choices)
                    job = pending[0]
                    shard_dir = output/job['group']/job['condition']/'shards'/f'shard-{job["shard_id"]:02d}'
                    attempt_dir = output/'attempts'/job['group']/job['condition']/f'shard-{job["shard_id"]:02d}'
                    attempt_dir.mkdir(parents=True, exist_ok=True)
                    command = [config['runtime_python'], str(Path(__file__).resolve().parents[1]/'run.py'),
                               'worker', '--config', str(config_path), '--output', str(output),
                               '--group', job['group'], '--condition', job['condition'],
                               '--shard-id', str(job['shard_id']), '--shard-count', str(shard_count)]
                    env = worker_environment(config, CUDA_VISIBLE_DEVICES=gpu, MUJOCO_EGL_DEVICE_ID=gpu,
                        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', PYTHONDONTWRITEBYTECODE='1')
                    log_path = attempt_dir/f'worker-{run_id}.log'
                    if not admission_open():
                        break
                    with log_path.open('xb') as log:
                        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    pending.pop(0)
                    job.update(status='running', pid=proc.pid, start_ticks=_process_start_ticks(proc.pid),
                        gpu_id=gpu, command=command, log=str(log_path), model_ready=str(shard_dir/'model_ready.json'),
                        shard_summary=str(shard_dir/'summary.json'),
                        attempt_record=str(attempt_dir/f'attempt-{run_id}.json'))
                    live.append(dict(proc=proc, job=job))
                    launched += 1
                    status['status'] = 'running'
                    _atomic_json(Path(job['attempt_record']), job)
                    _atomic_json(status_path, status)
                if once and launched >= len(gpu_ids):
                    pending = []
                if pending and not live and not wait_for_idle:
                    break
                accepting = admission_open()
                status['status'] = ('draining_admission_window' if live and not accepting else
                                    'running' if live else 'waiting_for_capacity')
                status['pending_shards'] = sum(j['status'] == 'pending' for j in jobs)
                _atomic_json(status_path, status)
                if live or (pending and accepting):
                    time.sleep(settings['poll_seconds'])
            status['status'] = 'finished' if all(j['status'] in ('complete', 'complete_skipped') for j in jobs) else 'needs_attention'
            status['pending_shards'] = sum(j['status'] == 'pending' for j in jobs)
            if not admission_open() and status['pending_shards']:
                status['completion_reason'] = 'admission_window_completed'
        except BaseException:
            status['status'] = 'interrupted_unknown'
            _atomic_json(status_path, status)
            raise
        _atomic_json(status_path, status)
        return status
