"""Policy prefix -> same-state public recovery -> private native first-hit result."""
from pathlib import Path
import fcntl
import json
import os
import shutil
import time
import traceback
import numpy as np
from .controller import PublicController, EpisodeStop
from .campaign import PROTOCOL, model_path, summarize


def _policy_tail_after_recovery(controller, receipt, output, *, recovery_returned):
    """Evaluator-owned continuation; never add actions outside recovery's cap."""
    before = controller.evaluator_record()
    remaining = controller.get_budget()['remaining_steps']
    receipt.update(recovery_steps_before=before['recovery_steps'], remaining_steps_before=remaining)
    if before['native_success']:
        receipt['status'] = 'skipped_native_success'
        return
    if remaining == 0:
        receipt['status'] = 'skipped_no_budget'
        return
    if not recovery_returned:
        receipt['status'] = 'skipped_harness_stop'
        return
    receipt.update(status='running', requested_steps=remaining)
    _atomic_write(Path(output)/'policy_tail.private.json', json.dumps(receipt, indent=2)+'\n')
    try:
        controller.run_policy(remaining)
        receipt['status'] = 'budget_exhausted'
    except EpisodeStop as stop:
        if stop.reason not in {'native_success', 'action_budget'}:
            raise
        receipt['status'] = 'native_success' if stop.reason == 'native_success' else 'budget_exhausted'
    finally:
        receipt['executed_steps'] = controller.evaluator_record()['recovery_steps'] - before['recovery_steps']


def evaluate_episode(session, policy, config, case, output, *, repair_fn=None, recorder=None):
    if repair_fn is None:
        from .repair import recover
        repair_fn = recover
    start = time.monotonic()
    controller = None
    tail = None
    row = dict(protocol=PROTOCOL, **case, status='unknown', native_success=None,
               policy_success=None, harness_called=False, tail_steps=0)
    try:
        tail_enabled = config.get('policy_tail_after_harness', False)
        if type(tail_enabled) is not bool:
            raise ValueError('policy_tail_after_harness must be Boolean')
        if tail_enabled:
            tail = dict(schema='libero.policy_tail.v1', enabled=True, status='not_reached',
                        requested_steps=0, executed_steps=0)
            if type(config['recovery_steps']) is not int or not 0 <= config['recovery_steps'] <= 400:
                raise ValueError('Policy tail requires the existing recovery cap of at most 400 actions')
        raw = session.reset()
        policy.reset(case['seed'])
        cap = config['policy_caps'][case['condition'].split('_')[0]]
        controller = PublicController(session, policy, raw, cameras=config['cameras'],
            policy_cap=cap, recovery_cap=config['recovery_steps'],
            wall_seconds=config['case_wall_seconds'], recorder=recorder)
        if getattr(session, 'initial_native_success', False):
            row.update(status='completed', native_success=True, policy_success=True,
                       stopped_at='initial_native_success')
            return row
        try:
            controller.run_policy(cap)
        except EpisodeStop as stop:
            if stop.reason != 'native_success':
                raise
        if controller.evaluator_record()['native_success']:
            row.update(status='completed', native_success=True, policy_success=True)
            return row
        row['policy_success'] = False
        row['harness_called'] = True
        controller.begin_recovery()
        recovery_returned = False
        try:
            repair_fn(controller, {**config, 'case_deadline_monotonic': controller._deadline}, output)
            recovery_returned = True
        except EpisodeStop as stop:
            if stop.reason not in {'native_success', 'action_budget'}:
                raise
        if tail is not None:
            _policy_tail_after_recovery(controller, tail, output, recovery_returned=recovery_returned)
        row.update(status='completed', native_success=controller.evaluator_record()['native_success'])
    except KeyboardInterrupt:
        if tail is not None:
            tail.update(status='infrastructure_error', error_type='InterruptedForInfrastructureRepair')
        row.update(status='unknown', native_success=None, policy_success=None,
                   error_type='InterruptedForInfrastructureRepair')
        (Path(output)/'infrastructure_error.txt').write_text(traceback.format_exc())
    except Exception as exc:
        if tail is not None:
            tail.update(status='infrastructure_error', error_type=type(exc).__name__)
        row.update(status='unknown', native_success=None, error_type=type(exc).__name__)
        (Path(output)/'infrastructure_error.txt').write_text(traceback.format_exc())
    finally:
        if controller is not None:
            private = controller.evaluator_record()
            private.pop('native_success')
            row.update(private)
        if tail is not None:
            if tail['status'] == 'not_reached' and row['native_success'] is True:
                tail['status'] = 'skipped_native_success'
            after = row.get('recovery_steps', 0)
            recovery_cap = config.get('recovery_steps')
            remaining = max(0, recovery_cap - after) if type(recovery_cap) is int else None
            tail.setdefault('recovery_steps_before', after)
            tail.setdefault('remaining_steps_before', remaining)
            tail.update(recovery_steps_after=after, remaining_steps_after=remaining)
            row['tail_steps'] = tail['executed_steps']
            _atomic_write(Path(output)/'policy_tail.private.json', json.dumps(tail, indent=2)+'\n')
        row['wall_seconds'] = time.monotonic() - start
    return row


class CompactRecorder:
    def __init__(self, output, config):
        self.output = Path(output)
        self.config = config
        self.actions, self.states, self.stages = [], [], []
        self.writer = None

    def __call__(self, obs, action, stage):
        self.actions.append(np.asarray(action).copy())
        self.states.append(np.r_[obs['robot_cartesian_pos'], obs['robot_eef_quat'], obs['robot_gripper_pos']])
        self.stages.append(stage)
        if self.config.get('save_video', True):
            import imageio.v2 as imageio
            if self.writer is None:
                self.writer = imageio.get_writer(str(self.output/'episode.mp4'), fps=self.config['video_fps'],
                    codec='libx264', quality=7, macro_block_size=None)
            panels = [obs[c]['images']['rgb'] for c in self.config['cameras']]
            self.writer.append_data(np.concatenate(panels, axis=1))

    def close(self):
        if self.writer:
            self.writer.close()
        np.savez_compressed(self.output/'actions_proprioception.npz',
            actions=np.asarray(self.actions), states=np.asarray(self.states), stages=np.asarray(self.stages))


def _atomic_write(path, text):
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(text)
    os.replace(temporary, path)


def _resume_rows(directory, config, group, condition):
    """Validate everything first; then preserve the old journal and retry evidence."""
    journal = directory/'episodes.jsonl'
    rows, original_lines = {}, {}
    expected = {(t, i) for t in config['task_ids'] for i in config['init_ids']}
    def validate(row):
        if not isinstance(row, dict):
            raise ValueError('Resume outcome must be an object')
        summarize([row], group, condition)
        key = (row['task_id'], row['init_id'])
        if (type(key[0]) is not int or type(key[1]) is not int or key not in expected
                or row.get('seed') != config['seed'] + key[0]*1000 + key[1]):
            raise ValueError('Resume outcome cohort or seed mismatch')
        return key
    if journal.exists():
        for line in journal.read_text().splitlines(keepends=True):
            row = json.loads(line)
            key = validate(row)
            if key in rows:
                raise ValueError('Duplicate resume outcome')
            rows[key], original_lines[key] = row, line if line.endswith('\n') else line + '\n'
    for tid, init_id in sorted(expected):
        outcome = directory/f'task{tid:02d}_init{init_id:02d}'/'outcome.json'
        if not outcome.exists():
            continue
        row = json.loads(outcome.read_text())
        key = validate(row)
        if key != (tid, init_id) or (key in rows and rows[key] != row):
            raise ValueError('Conflicting per-case and journal outcomes')
        rows[key] = row
    kept = [row for row in rows.values() if row['status'] == 'completed']
    summarize(kept, group, condition)
    known = {(row['task_id'], row['init_id']) for row in kept}
    retry = [(tid, init_id) for tid, init_id in sorted(expected) if (tid, init_id) not in known
             and ((tid, init_id) in rows or (directory/f'task{tid:02d}_init{init_id:02d}').exists())]
    history = directory/'retry_history'/f'{time.time_ns()}-{os.getpid()}'
    history.mkdir(parents=True, exist_ok=False)
    manifest = dict(protocol=PROTOCOL, group=group, condition=condition,
                    kept_completed=len(kept), retries=[], journal_present=journal.exists())
    for tid, init_id in retry:
        name = f'task{tid:02d}_init{init_id:02d}'
        manifest['retries'].append(dict(task_id=tid, init_id=init_id,
            original_row=rows.get((tid, init_id)), directory_present=(directory/name).exists()))
    (history/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for name in ('episodes.jsonl', 'summary.json', 'binding.json'):
        if (directory/name).exists():
            shutil.copy2(directory/name, history/name)
    for tid, init_id in retry:
        name = f'task{tid:02d}_init{init_id:02d}'
        if (directory/name).exists():
            shutil.move(str(directory/name), str(history/name))
    text = ''.join(original_lines.get((row['task_id'], row['init_id']), json.dumps(row)+'\n') for row in kept)
    _atomic_write(journal, text)
    return kept


def _validate_resume_binding(directory, config, group, condition):
    if not (directory/'binding.json').exists():
        return
    old = json.loads((directory/'binding.json').read_text())
    if any(old.get(k) != v for k, v in dict(group=group, condition=condition,
            model_path=str(model_path(config, group, condition)), protocol=PROTOCOL).items()):
        raise ValueError('Resume binding mismatch')
    for key in ('protocol', 'seed', 'task_ids', 'init_ids', 'policy_caps', 'recovery_steps',
                'repair_rounds', 'case_wall_seconds', 'cameras', 'baseline_path', 'ppo_root'):
        if old.get('config', {}).get(key) != config.get(key):
            raise ValueError(f'Resume configuration changed: {key}')


def prepare_condition_shards(config, group, condition, output, shard_count, resume):
    """The dispatcher alone reconciles a condition before any shard starts."""
    from .shards import merge_condition, shard_keys
    directory = Path(output)/group/condition
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'worker.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _validate_resume_binding(directory, config, group, condition)
        existing = ((directory/'episodes.jsonl').exists() or (directory/'shards').exists()
                    or (directory/'binding.json').exists() or any(directory.glob('task*_init*')))
        if existing and not resume:
            raise FileExistsError('Existing outcomes require explicit --resume')
        merged, _ = merge_condition(directory, group, condition)
        if (directory/'shards').exists():
            history = directory/'retry_history'/f'shard-migration-{time.time_ns()}'
            history.mkdir(parents=True)
            if (directory/'episodes.jsonl').exists():
                shutil.copy2(directory/'episodes.jsonl', history/'episodes.jsonl')
            _atomic_write(directory/'episodes.jsonl', ''.join(json.dumps(r)+'\n' for r in merged))
            shutil.move(str(directory/'shards'), str(history/'shards'))
        rows = _resume_rows(directory, config, group, condition)
        checkpoint = model_path(config, group, condition)
        _atomic_write(directory/'binding.json', json.dumps(dict(group=group, condition=condition,
            model_path=str(checkpoint), object_task_baseline=(group=='ppo' and condition=='object_task'),
            config=config, protocol=PROTOCOL), indent=2)+'\n')
        shards = directory/'shards'
        shards.mkdir()
        _atomic_write(shards/'plan.json', json.dumps(dict(group=group, condition=condition,
            protocol=PROTOCOL, shard_count=shard_count,
            case_keys={str(s): shard_keys(config, s, shard_count) for s in range(shard_count)}), indent=2)+'\n')
        _atomic_write(directory/'summary.json', json.dumps(summarize(rows, group, condition), indent=2)+'\n')
        return rows


def run_shard(config, group, condition, output, shard_id, shard_count):
    from .runtime_adapter import NativeSession, NativePolicy
    from .shards import shard_keys, read_rows, merge_condition
    directory = Path(output)/group/condition
    selected = shard_keys(config, shard_id, shard_count)
    shard_dir = directory/'shards'/f'shard-{shard_id:02d}'
    with (directory/'worker.lock').open('a') as condition_guard:
        fcntl.flock(condition_guard, fcntl.LOCK_SH | fcntl.LOCK_NB)
        plan = json.loads((directory/'shards'/'plan.json').read_text())
        if (plan.get('shard_count') != shard_count or plan.get('group') != group
                or plan.get('condition') != condition or plan.get('protocol') != PROTOCOL
                or plan['case_keys'].get(str(shard_id)) != [list(k) for k in selected]):
            raise ValueError('Shard plan mismatch; dispatcher reconciliation required')
        shard_dir.mkdir(exist_ok=True)
        with (shard_dir/'worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (shard_dir/'episodes.jsonl').exists():
                raise FileExistsError('Shard attempt exists; use dispatcher --resume')
            base = read_rows(directory/'episodes.jsonl')
            summarize(base, group, condition)
            if any(r['status'] != 'completed' for r in base):
                raise ValueError('Unreconciled base outcomes')
            known = {(r['task_id'], r['init_id']) for r in base}
            remaining = [key for key in selected if key not in known]
            rows = []
            _atomic_write(shard_dir/'episodes.jsonl', '')
            binding = dict(group=group, condition=condition, protocol=PROTOCOL, shard_id=shard_id,
                           shard_count=shard_count, case_keys=selected, config=config)
            _atomic_write(shard_dir/'binding.json', json.dumps(binding, indent=2)+'\n')
            policy = None
            try:
                if remaining:
                    policy = NativePolicy(str(model_path(config, group, condition)), condition,
                                          config['baseline_path'], 'cuda:0', shard_dir/'policy_runtime')
                    _atomic_write(shard_dir/'model_ready.json', json.dumps(dict(pid=os.getpid(), ready=True))+'\n')
                for tid, init_id in remaining:
                    case = dict(group=group, condition=condition, task_id=tid, init_id=init_id,
                                seed=config['seed']+tid*1000+init_id)
                    case_dir = directory/f'task{tid:02d}_init{init_id:02d}'
                    case_dir.mkdir(exist_ok=False)
                    recorder, session = CompactRecorder(case_dir, config), None
                    try:
                        session = NativeSession(condition, tid, init_id, case['seed'], case_dir/'runtime',
                                                device='cuda:0', cameras=config['cameras'],
                                                sensor_time_calibration=config.get('sensor_time_calibration', False),
                                recovery_extra_view=config.get('recovery_extra_view', False))
                        row = evaluate_episode(session, policy, config, case, case_dir, recorder=recorder)
                    except (Exception, KeyboardInterrupt) as exc:
                        (case_dir/'infrastructure_error.txt').write_text(traceback.format_exc())
                        row = dict(protocol=PROTOCOL, **case, status='unknown', native_success=None,
                            policy_success=None, error_type='InterruptedForInfrastructureRepair'
                            if isinstance(exc, KeyboardInterrupt) else type(exc).__name__)
                    finally:
                        try:
                            recorder.close()
                        finally:
                            if session is not None:
                                session.close()
                    _atomic_write(case_dir/'outcome.json', json.dumps(row, indent=2)+'\n')
                    rows.append(row)
                    _atomic_write(shard_dir/'episodes.jsonl', ''.join(json.dumps(r)+'\n' for r in rows))
                    with (directory/'summary.lock').open('a') as summary_lock:
                        fcntl.flock(summary_lock, fcntl.LOCK_EX)
                        _, merged_summary = merge_condition(directory, group, condition)
                        _atomic_write(directory/'summary.json', json.dumps(merged_summary, indent=2)+'\n')
                    print(f'{group}/{condition}/shard-{shard_id} {len(rows)}/{len(remaining)} status={row["status"]}', flush=True)
                    if row.get('error_type') == 'InterruptedForInfrastructureRepair':
                        break
            finally:
                if policy is not None:
                    policy.close()
            complete = len(rows) == len(remaining) and all(r['status'] == 'completed' for r in rows)
            result = dict(status='complete' if complete else 'incomplete', protocol=PROTOCOL,
                          group=group, condition=condition, shard_id=shard_id, planned_episodes=len(selected),
                          episodes=len(rows), skipped_completed=len(selected)-len(remaining),
                          unknown=sum(r['status'] != 'completed' for r in rows))
            _atomic_write(shard_dir/'summary.json', json.dumps(result, indent=2)+'\n')
            return result


def run_condition(config, group, condition, output, *, resume=False, shard_id=None, shard_count=5):
    if shard_id is not None:
        return run_shard(config, group, condition, output, shard_id, shard_count)
    from .runtime_adapter import NativeSession, NativePolicy
    directory = Path(output)/group/condition
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Condition already running')
        if (directory/'episodes.jsonl').exists() and not resume:
            raise FileExistsError('Existing outcomes preserved; explicit infra retry requires a new attempt')
        checkpoint = model_path(config, group, condition)
        if resume:
            _validate_resume_binding(directory, config, group, condition)
        rows = _resume_rows(directory, config, group, condition) if resume else []
        known = {(row['task_id'], row['init_id']) for row in rows}
        summary = summarize(rows, group, condition)
        _atomic_write(directory/'summary.json', json.dumps(summary, indent=2)+'\n')
        if all((t, i) in known for t in config['task_ids'] for i in config['init_ids']):
            return summary
        (directory/'binding.json').write_text(json.dumps(dict(group=group, condition=condition,
            model_path=str(checkpoint), object_task_baseline=(group=='ppo' and condition=='object_task'),
            config=config, protocol=PROTOCOL), indent=2)+'\n')
        policy = NativePolicy(str(checkpoint), condition, config['baseline_path'],
                              'cuda:0', directory/'policy_runtime')
        try:
            with (directory/'episodes.jsonl').open('a' if resume else 'x') as stream:
                for tid, init_id in ((t, i) for t in config['task_ids'] for i in config['init_ids']):
                    if (tid, init_id) in known:
                        continue
                    seed = config['seed'] + tid*1000 + init_id
                    case_dir = directory/f'task{tid:02d}_init{init_id:02d}'
                    case_dir.mkdir(exist_ok=False)
                    case = dict(group=group, condition=condition, task_id=tid, init_id=init_id, seed=seed)
                    session = None
                    recorder = CompactRecorder(case_dir, config)
                    try:
                        session = NativeSession(condition, tid, init_id, seed, case_dir/'runtime',
                                                device='cuda:0', cameras=config['cameras'],
                                                sensor_time_calibration=config.get('sensor_time_calibration', False),
                                recovery_extra_view=config.get('recovery_extra_view', False))
                        row = evaluate_episode(session, policy, config, case, case_dir, recorder=recorder)
                    except (Exception, KeyboardInterrupt) as exc:
                        (case_dir/'infrastructure_error.txt').write_text(traceback.format_exc())
                        row = dict(protocol=PROTOCOL, **case, status='unknown', native_success=None,
                                   policy_success=None, error_type=('InterruptedForInfrastructureRepair'
                                       if isinstance(exc, KeyboardInterrupt) else type(exc).__name__))
                    finally:
                        try:
                            recorder.close()
                        finally:
                            if session:
                                session.close()
                    _atomic_write(case_dir/'outcome.json', json.dumps(row, indent=2)+'\n')
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                    rows.append(row)
                    summary = summarize(rows, group, condition)
                    _atomic_write(directory/'summary.json', json.dumps(summary, indent=2)+'\n')
                    print(f'{group}/{condition} {len(rows)}/100 success={summary["successes"]} unknown={summary["unknown"]}', flush=True)
                    if row.get('error_type') == 'InterruptedForInfrastructureRepair':
                        break
        finally:
            policy.close()
    return summary
