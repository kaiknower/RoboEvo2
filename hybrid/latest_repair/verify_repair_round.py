"""Validate saved-prefix repair artifacts without invoking GPU or planner."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from repair_dispatch import read_json, summarize
from verify_native_artifacts import check_case, planner_message_paths
from lharness.scheduler import _atomic_json


def verify(directory):
    config = read_json(directory/'source/config.json')
    manifest = read_json(directory/'source/targets.json')
    status = read_json(directory/'dispatch.json')
    checks = []
    for job in status['jobs']:
        path = Path(job['output'])
        row = read_json(path/'outcome.json')
        if not row:
            checks.append(dict(target_index=job['target_index'], status='pending'))
            continue
        if row.get('status') != 'completed':
            checks.append(dict(target_index=job['target_index'], status='infrastructure_unknown',
                               error_type=row.get('error_type')))
            continue
        try:
            target = manifest['targets'][job['target_index']]
            assert row['status'] == 'completed' and row['prefix_reproduced'] and row['repair_eligible']
            assert row['policy_success'] is False and row['harness_called'] is True
            assert all(row[k] == target[k] for k in ('group','condition','task_id','init_id','seed'))
            source = Path(target['source_case'])
            assert hashlib.sha256((source/'outcome.json').read_bytes()).hexdigest() == target['source_outcome_sha256']
            assert hashlib.sha256((source/'actions_proprioception.npz').read_bytes()).hexdigest() == target['source_actions_sha256']
            prefix = read_json(path/'prefix_reproduction.private.json')
            assert prefix['matched'] and prefix['checked_prefix_actions'] == row['policy_steps']
            assert prefix['max_robot_state_abs_error'] <= 1e-6
            if config.get('sensor_time_calibration'):
                from lharness.prefix_sensor_evidence import load_public_snapshot, CAMERAS
                public = load_public_snapshot(path/'prefix_public_snapshot.npz')
                calibration = read_json(path/'sensor_time_calibration.private.json')
                assert calibration['schema'] == 'libero.sensor_time_calibration.v1'
                assert calibration['additional_render_calls'] == calibration['additional_observable_updates'] == 0
                for camera in CAMERAS:
                    sample = calibration['cameras'][camera]
                    assert sample['sampled_sensor_calls'] > 0
                    assert np.array_equal(public[camera]['intrinsics'], sample['sample_intrinsics'])
                    assert np.array_equal(public[camera]['extrinsics'], sample['sample_extrinsics'])
                    raw_rgb = public[camera]['images']['rgb'][::-1].copy()
                    assert hashlib.sha256(raw_rgb.tobytes()).hexdigest() == sample['raw_rgb_sha256']
            if prefix.get('match_type') == 'sensor_confirmed_wrist_roundoff':
                from lharness.prefix_sensor_evidence import load_public_snapshot
                from repair_failed_case import check_handoff
                from types import SimpleNamespace
                assert config.get('prefix_sensor_confirmed_wrist_roundoff') is True
                assert row.get('prefix_validation_mode') == prefix['match_type']
                public = load_public_snapshot(path/'prefix_public_snapshot.npz')
                computed = check_handoff(public, source, SimpleNamespace(seen=row['policy_steps'],
                    prefix_steps=row['policy_steps'], max_state_error=prefix['max_robot_state_abs_error'], tolerance=1e-6),
                    config['cameras'], rgb_max_abs_error=config.get('prefix_rgb_max_abs_error',0),
                    rgb_changed_fraction_limit=config.get('prefix_rgb_changed_fraction_limit',0),
                    sensor_confirmed_wrist_roundoff=True)
                assert computed['match_type'] == 'sensor_confirmed_wrist_roundoff' and computed['matched']
                assert computed['sensor_evidence'] == prefix['sensor_evidence']
                assert computed['cameras'] == prefix['cameras']
            else:
                assert all(x['max_abs_error'] <= config.get('prefix_rgb_max_abs_error',0)
                           and x.get('changed_fraction',0) <= config.get('prefix_rgb_changed_fraction_limit',0)
                           for x in prefix['cameras'].values())
            with np.load(path/'actions_proprioception.npz',allow_pickle=False) as current, np.load(source/'actions_proprioception.npz',allow_pickle=False) as original:
                assert np.array_equal(current['actions'][:row['policy_steps']], original['actions'][:row['policy_steps']])
                assert np.allclose(current['states'][:row['policy_steps']], original['states'][:row['policy_steps']],
                                   rtol=0, atol=1e-6, equal_nan=False)
            messages = planner_message_paths(path)
            details = check_case(path, row, config, messages)
            assert messages, 'Repair has no preserved planner requests'
            cards = {c['id']:c for c in read_json(directory/'source/memory/experiences/public_index.json')['experiences']}
            fields = ('id','title','when_public','action_template','recipe_ids','limits')
            ids = set()
            for m in messages:
                context = json.loads(read_json(m)[1]['content'][0]['text'])
                for card in context['reusable_experiences']:
                    assert card == {k:cards[card['id']][k] for k in fields}
                    ids.add(card['id'])
            checks.append(dict(target_index=job['target_index'],status='passed',native_success=row['native_success'],
                               prefix_validation_mode=prefix.get('match_type'),
                               experience_ids=sorted(ids),**details))
        except Exception as exc:
            checks.append(dict(target_index=job['target_index'],status='failed',error_type=type(exc).__name__,error=str(exc)))
    result = dict(checks=checks, summary=summarize(directory),
        failed=sum(c['status']=='failed' for c in checks), pending=sum(c['status']=='pending' for c in checks),
        infrastructure_unknown=sum(c['status']=='infrastructure_unknown' for c in checks),
        scope='Frozen baseline hashes, identical saved policy actions, public robot/RGB handoff, budgets, video header and exact public experience cards; not full video semantics.')
    _atomic_json(directory/'artifact_checks.actual.json',result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    args=p.parse_args()
    result=verify(args.directory)
    print(json.dumps(result,indent=2))
    raise SystemExit(1 if result['failed'] or result['pending'] else 0)
