"""Fixed paired cohort and strict result accounting; imports no simulator."""
from pathlib import Path
import json

PROTOCOL = 'libero_pro_public_harness_v1'
GROUPS = ('sft_v8', 'ppo')
CONDITIONS = tuple(f'{suite}_{axis}' for suite in ('spatial', 'object', 'goal', 'long')
                   for axis in ('position', 'task'))
ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config['protocol'] != PROTOCOL or config['init_ids'] != list(range(10, 20)) or config['task_ids'] != list(range(10)):
        raise ValueError('Expected fixed 800-case cohort per policy')
    if config['policy_caps'] != dict(spatial=220, object=280, goal=300, long=520):
        raise ValueError('Policy prefix differs from original evaluation caps')
    if config['cameras'] != ['agentview', 'robot0_eye_in_hand', 'sideview']:
        raise ValueError('Explicit camera contract required')
    if config['recovery_steps'] != 400 or config['repair_rounds'] != 3:
        raise ValueError('Evaluation recovery protocol changed')
    return config


def model_path(config, group, condition):
    if group not in GROUPS or condition not in CONDITIONS:
        raise ValueError('Unknown policy group/condition')
    if group == 'sft_v8' or condition == 'object_task':
        return Path(config['baseline_path'])
    return Path(config['ppo_root']) / condition


def expected_keys():
    return {(t, i) for t in range(10) for i in range(10, 20)}


def summarize(rows, group, condition):
    keys = [(r['task_id'], r['init_id']) for r in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= expected_keys():
        raise ValueError('Duplicate or out-of-cohort outcomes')
    for r in rows:
        if (r.get('group'), r.get('condition'), r.get('protocol')) != (group, condition, PROTOCOL):
            raise ValueError('Outcome group/condition/protocol mismatch')
        if r.get('status') not in {'completed', 'unknown'}:
            raise ValueError('Invalid outcome status')
        if r['status'] == 'completed' and type(r.get('native_success')) is not bool:
            raise ValueError('Completed outcome requires native Boolean')
        if r['status'] == 'unknown' and r.get('native_success') is not None:
            raise ValueError('Infrastructure outcome must not be a native failure')
        if r.get('policy_success') is True and r.get('native_success') is not True:
            raise ValueError('First success cannot be revoked')
    known = [r for r in rows if r['status'] == 'completed']
    unknown = len(rows) - len(known)
    direct = sum(r.get('policy_success') is True for r in known)
    recovered = sum(r['native_success'] and r.get('policy_success') is False for r in known)
    complete = set(keys) == expected_keys() and unknown == 0
    return dict(protocol=PROTOCOL, group=group, condition=condition,
                status='complete' if complete else 'incomplete', episodes=len(rows),
                planned_episodes=100, native_completed=len(known), unknown=unknown,
                missing=100-len(rows), policy_direct=direct, harness_recovered=recovered,
                successes=direct+recovered, failures=sum(not r['native_success'] for r in known),
                success_rate=(direct+recovered)/100 if complete else None,
                policy_prefix_rate=direct/100 if complete else None,
                extra_recovery_actions=400)


def aggregate(output):
    output = Path(output)
    groups = {}
    for group in GROUPS:
        reports = []
        for cond in CONDITIONS:
            from .shards import merge_condition
            _, report = merge_condition(output/group/cond, group, cond)
            reports.append(report)
        complete = all(r['status'] == 'complete' for r in reports)
        successes = sum(r['successes'] for r in reports)
        groups[group] = dict(status='complete' if complete else 'incomplete',
            planned_episodes=800, successes=successes,
            policy_direct=sum(r['policy_direct'] for r in reports),
            harness_recovered=sum(r['harness_recovered'] for r in reports),
            unknown=sum(r['unknown'] for r in reports), missing=sum(r['missing'] for r in reports),
            success_rate=successes/800 if complete else None, conditions=reports)
    return dict(protocol=PROTOCOL, groups=groups,
                same_policy_prefix_cohort=True, policy_harness_extra_action_budget=400)
