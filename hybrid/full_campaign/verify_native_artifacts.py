"""Incremental, read-only checks of committed evaluation artifacts (no GPU/API)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from lharness.campaign import aggregate, GROUPS, CONDITIONS, PROTOCOL
from lharness.coding import ROBOT_FIELDS
from lharness.evaluation import _validate_resume_binding
from lharness.shards import merge_condition

CONTEXT_KEYS = {
    'instruction', 'cameras', 'robot', 'remaining_steps', 'remaining_revisions',
    'skills', 'general_lessons', 'previous_public_execution', 'reusable_experiences',
}


def completion_status(receipt):
    """Validate the full denominator from counters, never from a cached flag.

    Partial receipts remain useful to callers; missing/malformed group or
    condition structures cannot certify a complete 1600-case campaign.
    """
    errors = []
    totals = dict(native_completed=0, unknown=0, missing=0)
    structure_valid = True

    def check(test, message):
        if not test:
            errors.append(message)

    def count(value):
        return type(value) is int and value >= 0

    aggregate_receipt = receipt.get('aggregate') if isinstance(receipt, dict) else None
    groups = aggregate_receipt.get('groups') if isinstance(aggregate_receipt, dict) else None
    if not isinstance(groups, dict) or set(groups) != set(GROUPS):
        errors.append('Exactly two configured policy groups are required')
        structure_valid = False
    for group in GROUPS:
        record = groups.get(group) if isinstance(groups, dict) else None
        if not isinstance(record, dict):
            structure_valid = False
            continue
        check(type(record.get('planned_episodes')) is int and record['planned_episodes'] == 800,
              f'{group}: planned denominator must be 800')
        conditions = record.get('conditions')
        if (not isinstance(conditions, list) or len(conditions) != len(CONDITIONS)
                or any(not isinstance(c, dict) or not isinstance(c.get('condition'), str) for c in conditions)
                or {c.get('condition') for c in conditions if isinstance(c, dict)} != set(CONDITIONS)):
            errors.append(f'{group}: exactly eight distinct configured conditions are required')
            structure_valid = False
            continue
        subtotal = dict(native_completed=0, unknown=0, missing=0)
        valid_counts = True
        for condition in conditions:
            label = f'{group}/{condition["condition"]}'
            check(condition.get('group') == group and condition.get('protocol') == PROTOCOL,
                  label + ': condition identity differs')
            check(type(condition.get('planned_episodes')) is int and condition['planned_episodes'] == 100,
                  label + ': planned denominator must be 100')
            fields = ('episodes', 'native_completed', 'unknown', 'missing')
            if not all(count(condition.get(name)) for name in fields):
                errors.append(label + ': valid integer episode counters required')
                valid_counts = False
                structure_valid = False
                continue
            native, unknown, missing = (condition[name] for name in ('native_completed', 'unknown', 'missing'))
            check(condition['episodes'] == native + unknown and condition['episodes'] + missing == 100,
                  label + ': episode counters do not match the 100-case denominator')
            check(native == 100 and unknown == 0 and missing == 0 and condition.get('status') == 'complete',
                  label + ': native outcomes remain unknown or missing')
            for name in subtotal:
                subtotal[name] += condition[name]
        if valid_counts:
            for name in totals:
                totals[name] += subtotal[name]
            for name in ('unknown', 'missing'):
                check(count(record.get(name)) and record[name] == subtotal[name],
                      f'{group}: {name} total differs from its conditions')
        check(record.get('status') == 'complete', f'{group}: group is not complete')
    if not structure_valid:
        totals = dict.fromkeys(totals)
    check(isinstance(receipt, dict) and type(receipt.get('checked_completed')) is int
          and receipt['checked_completed'] == 1600 and totals['native_completed'] == 1600,
          'Exactly 1600 completed native cases must have artifact checks')
    check(isinstance(receipt, dict) and type(receipt.get('failed')) is int
          and receipt['failed'] == 0 and receipt.get('failures') == {},
          'Artifact verification failures must be zero')
    return dict(all_completed_verified=not errors, planned_total_episodes=1600,
                **totals, completion_errors=errors)


def verification_exit_code(receipt, require_complete=False):
    if require_complete:
        return 0 if completion_status(receipt)['all_completed_verified'] else 1
    return 1 if receipt['failed'] else 0


def fingerprint(directory, row, config):
    paths = [directory/'outcome.json', directory/'actions_proprioception.npz',
             directory/'episode.mp4']
    paths += sorted(directory.glob('coding/request_*/transport/api/messages.json'))
    metadata = [(str(p.relative_to(directory)), p.stat().st_size, p.stat().st_mtime_ns)
                if p.exists() else (str(p.relative_to(directory)), None, None) for p in paths]
    value = {'files': metadata, 'row': row, 'config': config,
             'checker': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(), paths[3:]


def check_case(directory, row, config, messages):
    def require(test, message):
        if not test:
            raise ValueError(message)

    require(json.loads((directory/'outcome.json').read_text()) == row,
            'Canonical outcome differs from journal')
    p, r = row['policy_steps'], row['recovery_steps']
    require(type(p) is int and type(r) is int and min(p, r) >= 0, 'Invalid action counts')
    require(p <= config['policy_caps'][row['condition'].split('_')[0]], 'Policy budget exceeded')
    require(r <= config['recovery_steps'], 'Recovery budget exceeded')
    require(row['seed'] == config['seed'] + 1000*row['task_id'] + row['init_id'], 'Seed differs')
    if row['policy_success']:
        require(row['native_success'] is True and not row['harness_called'] and r == 0,
                'Policy success did not stop before harness')
        require(not messages, 'Policy success unexpectedly has planner requests')
    with np.load(directory/'actions_proprioception.npz', allow_pickle=False) as z:
        actions, states, stages = z['actions'], z['states'], z['stages']
        if p+r:
            require(actions.shape == (p+r, 7) and states.shape == (p+r, 9),
                    'Action/proprioception length or shape differs')
            require(np.isfinite(actions).all() and np.isfinite(states).all(), 'Non-finite trajectory')
            require(np.max(np.abs(actions)) <= 1, 'Action exceeds normalized bounds')
        else:
            require(len(actions) == len(states) == 0, 'Nonempty zero-step trajectory')
        require(stages.tolist() == ['policy']*p + ['recovery']*r, 'Recorded stage counts/order differ')
    if p+r and config.get('save_video', True):
        video = cv2.VideoCapture(str(directory/'episode.mp4'))
        try:
            require(video.isOpened(), 'Video header is unreadable')
            require(int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == p+r, 'Video/action frame count differs')
            require(abs(video.get(cv2.CAP_PROP_FPS) - config['video_fps']) < .01, 'Video FPS differs')
            width, height = video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            require(height > 0 and width == height*len(config['cameras']), 'Camera panel layout differs')
        finally:
            video.release()
    requests = set()
    for path in messages:
        payload = json.loads(path.read_text())
        require([m['role'] for m in payload] == ['system', 'user'], 'Unexpected actor message roles')
        context = json.loads(payload[1]['content'][0]['text'])
        require(set(context) == CONTEXT_KEYS, 'Unexpected actor context fields')
        require(set(context['robot']) <= set(ROBOT_FIELDS), 'Unexpected robot fields')
        require(set(context['cameras']) == set(config['cameras']), 'Actor camera list differs')
        require(0 <= context['remaining_steps'] <= config['recovery_steps'], 'Actor action budget differs')
        require(1 <= context['remaining_revisions'] <= config['repair_rounds'], 'Actor revision budget differs')
        for name, values in context['robot'].items():
            a = np.asarray(values)
            require(a.shape == (ROBOT_FIELDS[name],) and np.isfinite(a).all(), 'Robot field shape/value differs')
        requests.add(path.parents[3].name)
    require(len(requests) <= config['repair_rounds'], 'Too many planner revisions')
    return {'actions': p+r, 'planner_requests': len(requests)}


def verify(output, config, destination):
    report = aggregate(output)
    destination.mkdir(parents=True, exist_ok=True)
    cache_path = destination/'artifact_checks.cache.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cases = []
    for group in GROUPS:
        for condition in CONDITIONS:
            directory = output/group/condition
            _validate_resume_binding(directory, config, group, condition)
            rows, _ = merge_condition(directory, group, condition)
            cases.extend((directory/f'task{r["task_id"]:02d}_init{r["init_id"]:02d}', r)
                         for r in rows if r['status'] == 'completed')

    def one(item):
        directory, row = item
        key = str(directory.relative_to(output))
        try:
            digest, messages = fingerprint(directory, row, config)
            old = cache.get(key, {})
            if old.get('fingerprint') == digest and old.get('status') == 'passed':
                return key, old, True
            result = check_case(directory, row, config, messages)
            return key, dict(status='passed', fingerprint=digest, **result), False
        except Exception as exc:
            return key, dict(status='failed', error_type=type(exc).__name__, error=str(exc)), False

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, cases))
    current = {key: value for key, value, _ in results}
    failures = {k: v for k, v in current.items() if v['status'] != 'passed'}
    result = dict(checked_at_unix=time.time(), source=str(output), checked_completed=len(cases),
                  reused_cached=sum(cached for _, _, cached in results), failed=len(failures),
                  failures=failures, aggregate=report,
                  all_native_outcomes_present=all(g['status'] == 'complete' for g in report['groups'].values()),
                  scope='Journal/canonical equality, fixed binding, seed/budgets, finite actions and states, '
                        'stage order, video header frame count/FPS/panels, actor request field allowlist. '
                        'Does not decode every video frame or prove image interpretation correctness.')
    result.update(completion_status(result))
    for path, data in [(cache_path, current), (destination/'artifact_checks.actual.json', result)]:
        temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
        temp.replace(path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true',
                        help='Exit zero only for all 1600 native cases with zero unknown/missing/check failures')
    args = parser.parse_args(argv)
    result = verify(args.output, json.loads(args.config.read_text()), args.destination)
    print(json.dumps({k: v for k, v in result.items() if k not in {'aggregate', 'failures'}}, ensure_ascii=False))
    return verification_exit_code(result, args.require_complete)


if __name__ == '__main__':
    raise SystemExit(main())
